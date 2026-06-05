# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import haiku as hk
import jax
import jax.numpy as jnp


class BlockV1(hk.Module):
    def __init__(self, num_channels, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class SelfAttentionBlock(hk.Module):
    """Multi-head self-attention over the spatial grid, used as a final block.

    The (H, W) spatial dimensions are flattened into a length H*W sequence,
    self-attention is applied across that sequence, and the result is reshaped
    back to the original (H, W, C) grid and added residually.
    """

    def __init__(self, num_channels, num_heads, name="SelfAttentionBlock"):
        super(SelfAttentionBlock, self).__init__(name=name)
        self.num_channels = num_channels
        self.num_heads = num_heads

    def __call__(self, x, is_training, test_local_stats):
        i = x
        b, h, w, c = x.shape
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        # Flatten the spatial grid into a sequence of length H*W.
        seq = x.reshape(b, h * w, c)
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=max(c // self.num_heads, 1),
            model_size=c,
            w_init=hk.initializers.VarianceScaling(1.0),
        )(seq, seq, seq)
        attn = attn.reshape(b, h, w, c)
        return attn + i


class AZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        num_heads: int = -1,
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        # num_heads > 0 replaces the final convolution block with a
        # multi-head self-attention block (using that many heads).
        self.num_heads = num_heads
        self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(jnp.float32)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            is_last = i == self.num_blocks - 1
            if self.num_heads > 0 and is_last:
                x = SelfAttentionBlock(self.num_channels, self.num_heads, name=f"block_{i}")(
                    x, is_training, test_local_stats
                )
            else:
                x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                    x, is_training, test_local_stats
                )

        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # policy head
        logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
        logits = jax.nn.relu(logits)
        logits = hk.Flatten()(logits)
        logits = hk.Linear(self.num_actions)(logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v

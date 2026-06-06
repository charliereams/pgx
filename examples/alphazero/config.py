# Shared AlphaZero training configuration.
#
# Defined in its own module (rather than in train.py's __main__) so that the
# same Config class can be imported by both the trainer and run_tournament.py,
# and so checkpoints pickle/unpickle against a stable module path (config.Config).

import pgx
from pydantic import BaseModel, model_validator


class Config(BaseModel):
    env_id: pgx.EnvId = "g_hex"
    seed: int = 0
    max_num_iters: int = 400
    # path to a .ckpt file to resume training from (empty = start from scratch)
    resume_from: str = ""
    # wandb run id to resume logging into. If empty when resuming, the id stored
    # in the checkpoint (if any) is used so the original run continues.
    wandb_run_id: str = ""
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # num_heads > 0 replaces the final conv block with a multi-head
    # self-attention block using that many heads; -1 keeps an all-conv net.
    num_heads: int = -1
    # Number of self-attention blocks applied at the end (after the conv blocks),
    # used only when num_heads > 0. Must not exceed num_layers.
    num_attention_layers: int = 1
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    @model_validator(mode="after")
    def _check_attention_layers(self):
        if self.num_attention_layers > self.num_layers:
            raise ValueError(
                f"num_attention_layers ({self.num_attention_layers}) cannot exceed "
                f"num_layers ({self.num_layers})."
            )
        return self

    class Config:
        extra = "forbid"

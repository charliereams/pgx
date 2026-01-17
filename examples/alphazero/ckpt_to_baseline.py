# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chex
import datetime
import os
import pickle
import time
import random
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from pgx.experimental import act_randomly
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "domineering"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)
baseline = pgx.make_baseline_model(config.env_id + "_v0")


def forward_fn(x, is_eval=True):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=False, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


@jax.pmap
def evaluate(rng_key, my_model):
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        (my_logits, _), _ = forward.apply(
            my_model_params, my_model_state, state.observation, is_eval=True
        )
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


if __name__ == "__main__":
    with open("checkpoints/domineering_20260112173305/001400.ckpt", "rb") as f:
      ckpt = pickle.load(f)
      model = ckpt["model"]
      opt_state = ckpt["opt_state"]
    print(f"loaded_model={model.__repr__} state={type(opt_state)}")


    class MctsConfig(NamedTuple):
        env_id: pgx.EnvId = "domineering"
        seed: int = 0
        num_simulations: int = 100 #1_000
        batch_size: int = 1

    mcts_config = MctsConfig()
    env = pgx.make(mcts_config.env_id)
    root_key = jax.random.PRNGKey(mcts_config.seed)

    def policy_fn(state):
        """Return the logits of random policy. -Inf is set to illegal actions."""

        legal_action_mask = state.legal_action_mask
        chex.assert_shape(legal_action_mask, (env.num_actions,))

        (logits, act) = baseline(jnp.array([state.observation]))
        # logits = legal_action_mask.astype(jnp.float32)
        logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        #logits = jnp.where(legal_action_mask, gmz_policy.action_weights, jnp.finfo(logits.dtype).min)
        return logits[0]


    def value_fn(key, state):
        """Return the value based on random rollout."""
        chex.assert_rank(state.current_player, 0)

        (logits, value) = baseline(jnp.array([state.observation]))
        return value[0]

    def recurrent_fn(params, rng_key, action, state):
        del params
        current_player = state.current_player
        state = env.step(state, action)
        logits = policy_fn(state)
        value = value_fn(rng_key, state)
        reward = state.rewards[current_player]
        value = jax.lax.select(state.terminated, 0.0, value)
        discount = jax.lax.select(state.terminated, 0.0, -1.0)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return recurrent_fn_output, state


    def run_mcts(key, state):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, mcts_config.batch_size)
        key, subkey = jax.random.split(key)

        root = mctx.RootFnOutput(
            prior_logits=jax.vmap(policy_fn)(state),
            value=jax.vmap(value_fn)(keys, state),
            embedding=state
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=subkey,
            root=root,
            invalid_actions=~state.legal_action_mask,
            recurrent_fn=jax.vmap(recurrent_fn, in_axes=(None, None, 0, 0)),
            num_simulations=mcts_config.num_simulations,
            max_depth=env.observation_shape[0] * env.observation_shape[1],  # set for each game
            qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),  # could be fixed
            #dirichlet_fraction=0.0
            gumbel_scale=1.0,
        )
        return policy_output

    def vs_human(is_human_first=True):
        assert mcts_config.batch_size == 1
        init_fn = jax.jit(jax.vmap(env.init))
        step_fn = jax.jit(jax.vmap(env.step))

        key, subkey = jax.random.split(root_key)
        keys = jax.random.split(subkey, mcts_config.batch_size)
        state: pgx.State = init_fn(keys)

        is_human_turn = is_human_first
        hmove = 0
        while True:
            #print(state._x.board.reshape(9, 9))
            b = state._x.board.reshape(8, 8)
            print ("\n".join(
                "".join("_" if cell else "X" for cell in row)
                for row in b
            ))
            print("")

            if state.terminated.all():
                break
            if is_human_turn:
                #action = int(input("Your action: "))
                #action = -1
                #while action < 0 or not state.legal_action_mask[action].any():
                #     action = random.randint(0, 62)  #len(state.legal_action_mask) - 1)
                #action = jnp.int32(action)
                action = jnp.int32([hmove])
                while not state.legal_action_mask[0][hmove]:
                  hmove = hmove + 1
                  action = jnp.int32([hmove])
            else:
                policy_output = jax.jit(run_mcts)(key, state)
                action = policy_output.action
            state = step_fn(state, action)
            is_human_turn = not is_human_turn


    vs_human()

    # Initialize model and opt_state
    #dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    #dummy_input = dummy_state.observation
    #model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    #opt_state = optimizer.init(params=model[0])
    # replicates to all devices
    #model, opt_state = jax.device_put_replicated((model, opt_state), devices)

            # Store checkpoints
#            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
#            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
#                dic = {
#                    "config": config,
#                    "rng_key": rng_key,
#                    "model": jax.device_get(model_0),
#                    "opt_state": jax.device_get(opt_state_0),
#                    "iteration": iteration,
#                    "frames": frames,
#                    "hours": hours,
#                    "pgx.__version__": pgx.__version__,
#                    "env_id": env.id,
#                    "env_version": env.version,
#                }
#                pickle.dump(dic, f)


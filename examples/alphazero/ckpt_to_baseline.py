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
import pgx
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
    num_channels: int = 256
    num_layers: int = 8
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


if __name__ == "__main__":
    with open("checkpoints/domineering_20260112173305/001000.ckpt", "rb") as f:
      ckpt = pickle.load(f)
      model = ckpt["model"]
      opt_state = ckpt["opt_state"]
    if True:
      model_params, model_state = model
      print(f"loaded_model={model_params.shape()} state={type(opt_state)}")

    class MctsConfig(NamedTuple):
        env_id: pgx.EnvId = "domineering"
        seed: int = 4199
        num_simulations: int = 10_000 #1_000
        batch_size: int = 1

    mcts_config = MctsConfig()
    env = pgx.make(mcts_config.env_id)
    root_key = jax.random.PRNGKey(mcts_config.seed)

    def recurrent_fn(params, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model: params
        # state: embedding
        del params
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

    def run_mcts(key, state):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, mcts_config.batch_size)
        key, subkey = jax.random.split(key)

        model_params, model_state = model
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(
            prior_logits=logits, #jax.vmap(policy_fn)(state),
            value=value, #jax.vmap(value_fn)(keys, state),
            embedding=state
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=subkey,
            root=root,
            invalid_actions=~state.legal_action_mask,
            recurrent_fn=recurrent_fn, #jax.vmap(recurrent_fn, in_axes=(None, None, 0, 0)),
            num_simulations=mcts_config.num_simulations,
            max_depth=32, #env.observation_shape[0] * env.observation_shape[1],  # set for each game
            qtransform=mctx.qtransform_completed_by_mix_value, #partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),  # could be fixed
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
            b = state._x.board.reshape(8, 8)
            print ("\n".join(
                "".join("_" if cell else "X" for cell in row)
                for row in b
            ))
            print("Human to play..." if is_human_turn else "AI to play...")
            print("")

            if state.terminated.all():
                print("Game over!")
                break
            if is_human_turn:
                #action = int(input("Your action: "))
                #while action < 0 or not state.legal_action_mask[action].any():
                #     action = random.randint(0, 62)  #len(state.legal_action_mask) - 1)
                while not state.legal_action_mask[0][hmove]:
                  hmove = hmove + 1
                action = jnp.int32([hmove])
            else:
                policy_output = jax.jit(run_mcts)(key, state)
                action_weights = policy_output.action_weights.reshape(8, 8)
                print("\n".join(
                  "".join(f"{w:.4f}  " for w in w_row)
                  for w_row in action_weights
                ))
                print("")
                action = policy_output.action
            state = step_fn(state, action)
            is_human_turn = not is_human_turn


    vs_human()

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


class MctsConfig(NamedTuple):
    seed: int = 7386708
    num_simulations: int = 100
    batch_size: int = 1


class Config(BaseModel):
    env_id: pgx.EnvId = "domineering"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
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


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # Mask first, then scale for numerical stability.
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)

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


def action_from_square(square_code):
    try:
        col = ord(square_code[0]) - ord('a')
        row = int(square_code[1]) - 1
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            return None
        return row * 8 + col
    except Exception as e:
        return None


def run_mcts(model, key, state, debug=False):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, mcts_config.batch_size)
    key, subkey = jax.random.split(key)

    model_params, model_state = model
    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )

    if debug:
      logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
      logits = logits - jnp.max(logits, axis=-1, keepdims=True)
      action_weights = jax.scipy.special.softmax(logits.reshape(8, 8))
      print("\n".join(
                   "".join(f"{100*w:5.2f}%  " for w in w_row)
                 for w_row in action_weights
                ))
      print(f"value={value}")
      return None

    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=state
    )
    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=subkey,
        root=root,
        invalid_actions=~state.legal_action_mask,
        recurrent_fn=recurrent_fn,
        num_simulations=mcts_config.num_simulations,
        max_depth=32,
        #qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=0.0,
    )
    return policy_output


if __name__ == "__main__":
    #with open("checkpoints/domineering_20260112173305/000000.ckpt", "rb") as f:
    #  ckpt = pickle.load(f)
    #  model2 = ckpt["model"]
    with open("checkpoints/domineering_20260118213906/000150.ckpt", "rb") as f:
      ckpt = pickle.load(f)
      model1 = ckpt["model"]

    mcts_config = MctsConfig()
    env = pgx.make(config.env_id)

    print("\n\nLet's play!\n\n\n")

    def vs_human(game_num, is_human_first=True):
        assert mcts_config.batch_size == 1
        init_fn = jax.jit(jax.vmap(env.init))
        step_fn = jax.jit(jax.vmap(env.step))

        root_key = jax.random.PRNGKey(mcts_config.seed ^ game_num)
        key, subkey = jax.random.split(root_key)
        keys = jax.random.split(subkey, mcts_config.batch_size)
        state: pgx.State = init_fn(keys)

        is_human_turn = is_human_first
        while True:
            print ("   abcdefgh")
            print ("\n".join(
                f"{idx+1} |" + "".join("·" if cell else "■" for cell in row) + "|"
                for idx, row in enumerate(state._x.board.reshape(8, 8))
            ))
            print("")
            print(("Human to play..." if is_human_turn else "AI to play...") + f" ({'V' if state._x.color else 'H'}) c_p={state.current_player}")
            print("", flush=True)

            if state.terminated.all():
                print("Game over!")
                break
            if is_human_turn:
                # Human interactive version
                action_i = None
                while action_i is None or not state.legal_action_mask[0][action_i]:
                  action_i = action_from_square(input("move="))
                action = jnp.int32([action_i])

                # Random valid move
                #hmove = None
                #while hmove is None or not state.legal_action_mask[0][hmove]:
                #  hmove = random.randint(0, 62)
                #action = jnp.int32([hmove])

                # Use another model!
                #policy_output = jax.jit(run_mcts)(model1, key, state)
                #action = policy_output.action
            else:
                run_mcts(model1, key, state, debug=True)
                policy_output = jax.jit(run_mcts)(model1, key, state)
                action_weights = policy_output.action_weights.reshape(8, 8)
                print("\n".join(
                  "".join(f"{100*w:6.2f}%  " for w in w_row)
                  for w_row in action_weights
                ))
                print("")
                action = policy_output.action
            print(f"Played {'abcdefgh'[action[0] % 8]}{1 + (action[0] // 8)}\n")
            state = step_fn(state, action)
            is_human_turn = not is_human_turn

        print("Human wins!" if (state._x.winner == 0) == is_human_first else "AI wins!")
        return state._x.winner == 0

    human_wins = 0
    for game_num in range(0, 5000):
        is_human_first = ((game_num % 2) == 0)
        p1_won = vs_human(game_num, is_human_first=is_human_first)
        if is_human_first == p1_won:
            human_wins = human_wins + 1
            break
        print(f"Human win rate = {100 * human_wins / (1 + game_num)}%", flush=True)
        time.sleep(10)

# Copyright 2026 The Pgx Authors. All Rights Reserved.
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
import re
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import mctx
import pgx
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pgx.g_hex import black, white
from pydantic import BaseModel

from network import AZNet
from abc import ABC, abstractmethod


class MctsConfig(NamedTuple):
    seed: int = 7386708
    num_simulations: int = 30


class Agent(ABC):
    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def getAction(self, key, state):
        pass


class KeyboardAgent(Agent):
    def __init__(self, env_id):
        self.env_id = env_id

    def getName(self):
        return "Human"

    def getAction(self, key, state):
        action_i = None
        while action_i is None or not state.legal_action_mask[0][action_i]:
            if self.env_id == "domineering":
                action_i = self._action_from_square(input("move="))
            elif self.env_id == "g_hex":
                action_i = self._action_from_ghex_name(input("move="))
            else
                raise Exception(f"Keyboard parsing not implemented for {self.env_id}")
            #print(f"action_i={action_i} legal={state.legal_action_mask[0][action_i]}")
        return jnp.int32([action_i])

    def _action_from_square(self, square_code):
        try:
            col = ord(square_code[0]) - ord('a')
            row = int(square_code[1]) - 1
            if col < 0 or col >= 8 or row < 0 or row >= 8:
                return None
            return row * 8 + col
        except Exception as e:
            return None

    def _action_from_ghex_name(self, name):
        try:
            m = re.fullmatch("([0-9]+) on ([0-9]+)", name)
            if m is None:
                return None
            tile = int(m.group(1))
            triangle = int(m.group(2))
            return black(tile, triangle)
        except Exception as e:
            print(e)
            return None


class RandomAgent(Agent):
    def getName(self):
        return "Rando"

    def getAction(self, key, state):
        action_i = None
        while action_i is None or not state.legal_action_mask[0][action_i]:
            action_i = random.randint(0, 21 * 10) # TODO: select based on env or something
        return jnp.int32([action_i])



class ModelAgent(Agent):
    def __init__(self, env_id, mcts_config: MctsConfig, model):
        self.env_id = env_id
        self.mcts_config = mcts_config
        self.model = model
        self.mcts_jit = jax.jit(partial(ModelAgent._run_mcts, mcts_config))

    def getName(self):
        return f"Model[sims={self.mcts_config.num_simulations}]"

    def getAction(self, key, state):
        ModelAgent._run_mcts(self.mcts_config, self.model, key, state, debug_domineering=(self.env_id == "domineering"), debug_g_hex=(self.env_id == "g_hex"))

        policy_output = jax.jit(partial(ModelAgent._run_mcts, self.mcts_config))(self.model, key, state)
        if self.env_id == "domineering":
            action_weights = policy_output.action_weights.reshape(8, 8)
            print("\n".join(
                        "".join(f"{100*w:6.2f}%  " for w in w_row)
                        for w_row in action_weights
                    ))
            print("")
        elif self.env_id == "g_hex":
            action_weights = policy_output.action_weights.reshape(21, 10)
            print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" for w in tri_row])
                             for tri_i, tri_row in enumerate(action_weights)]))
            print("")

        return policy_output.action

    @staticmethod  # Static for JITting.
    def _run_mcts(mcts_config, model, key, state, debug_domineering=False, debug_g_hex=False):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 1)  # Batch size must be 1
        key, subkey = jax.random.split(key)

        model_params, model_state = model
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        if debug_domineering or debug_g_hex:
            logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            if debug_domineering:
                action_weights = jax.scipy.special.softmax(logits.reshape(8, 8))
                print("\n".join(
                       "".join(f"{100*w:5.2f}%  " for w in w_row)
                         for w_row in action_weights
                      ))
            elif debug_g_hex:
                action_weights = jax.scipy.special.softmax(logits.reshape(21, 10))
                print("Action weights:")
                print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" for w in tri_row])
                                 for tri_i, tri_row in enumerate(action_weights)]))
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
            num_simulations=round(mcts_config.num_simulations),
            #max_depth=32,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=0.0,
        )
        return policy_output

devices = jax.local_devices()
num_devices = len(devices)


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
    #logits = logits - jnp.max(logits, axis=-1, keepdims=True)

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


def pretty_tiles(tiles):
  return "  ".join([(f"{val:2}" if tiles[i] else "  ")
                   for i, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])

def load_from_checkpoint(path):
  with open(f"checkpoints/{path}", "rb") as f:
      ckpt = pickle.load(f)
      return ckpt["model"]

if __name__ == "__main__":
    #model_domineering = load_from_checkpoint("domineering_20260122174624/001100.ckpt")
    model1 = load_from_checkpoint("g_hex_20260125182112/000100.ckpt")

    env = pgx.make(config.env_id)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    print("\n\nLet's play!\n\n\n")

    def run_game(game_num, agents):
        root_key = jax.random.PRNGKey(9237823 ^ game_num)
        key, subkey = jax.random.split(root_key)
        keys = jax.random.split(subkey, 1)  # Batch size must be 1.
        state: pgx.State = init_fn(keys)

        p1_to_play = True
        while True:
            if config.env_id == "domineering":
              print("   abcdefgh")
              print("\n".join(
                  f"{idx+1} |" + "".join("·" if cell else "■" for cell in row) + "|"
                  for idx, row in enumerate(state._x.board.reshape(8, 8))
              ))
              print("")
            elif config.env_id == "g_hex":
              _TILE_VAL = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32)
              print(env.pretty_game(state))
              print("")
              print(f"Black tiles remaining: {pretty_tiles(state._x.tiles[0][0])}")
              print(f"White tiles remaining: {pretty_tiles(state._x.tiles[0][1])}")
              print("")

            if state.terminated.all():
                print("Game over!")
                return state._x.winner

            agent = agents[0 if p1_to_play else 1]
            print(f"{agent.getName()} to play...", flush=True)
            action = agent.getAction(key, state)
            if config.env_id == "domineering":
                print(f"{agent.getName()} played {'abcdefgh'[action[0] % 8]}{1 + (action[0] // 8)}\n")
            if config.env_id == "g_hex":
                print(f"{agent.getName()} played the {1 + (action[0] % 10)} on triangle {action[0] // 10}\n")
            state = step_fn(state, action)
            p1_to_play = not p1_to_play


    agents = [
        #RandomAgent(),
        #RandomAgent(),
        ModelAgent(config.env_id, MctsConfig(seed=33910238, num_simulations=1), model1),
        ModelAgent(config.env_id, MctsConfig(seed=24364321, num_simulations=1000), model1),
        #KeyboardAgent(config.env_id),
    ]
    wins = np.array([0, 0])
    for game_num in range(0, 4):
        agent_1_first = (game_num % 2) == 0
        game_agents = [agents[0], agents[1]] if agent_1_first else [agents[1], agents[0]]
        print(f"Game {game_num}: {game_agents[0].getName()} vs {game_agents[1].getName()}")

        winner = run_game(game_num, game_agents)
        if winner == (0 if agent_1_first else 1):
            wins[0] += 1
        elif winner == (1 if agent_1_first else 0):
            wins[1] += 1
        win_rates = 100 * wins / (1 + game_num)
        print(f"""Win rates after {1 + game_num} games:
                  {agents[0].getName():>20}: {win_rates[0]:6.2f}% ({wins[0]})
                  {agents[1].getName():>20}: {win_rates[1]:6.2f}% ({wins[1]})
               """, flush=True)
        time.sleep(10)

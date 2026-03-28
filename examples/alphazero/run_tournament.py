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

import os
import pickle
import random
import re
import time
from functools import partial
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import mctx
import pgx
import pgx.heckmeck as Heckmeck
from omegaconf import OmegaConf
from pgx.g_hex import black
from pgx.g_hex2 import black2
from pydantic import BaseModel
from network import AZNet
from abc import ABC, abstractmethod


class TourneyConfig(BaseModel):
    env_id: pgx.EnvId = "g_hex"
    seed: int = 49064405
    games: int = 256


class Config(BaseModel):
    env_id: pgx.EnvId = "g_hex"
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


class MctsConfig(NamedTuple):
    num_simulations: int = 128
    max_num_considered_actions: int = 16


class Cli(ABC):
    @abstractmethod
    def getActionId(self):
        pass

    @abstractmethod
    def display(self, state):
        pass

    @abstractmethod
    def loadModelBasedAgent(self, player_num):
        pass


class DomineeringCli(Cli):
    def getActionId(self):
        square_code = input("Move: ")
        col = ord(square_code[0]) - ord('a')
        row = int(square_code[1]) - 1
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            return None
        return row * 7 + col

    def display(self, state):
        print("   abcdefgh")
        print("\n".join(
              f"{idx+1} |" + "".join("·" if cell else "■" for cell in row) + "|"
              for idx, row in enumerate(readable_domineering_board(state))
        ))
        print("")


    def loadModelBasedAgent(self, player_num):
        config, model = load_from_checkpoint("domineering_20260131044700/000125.ckpt")
        return ModelAgent("vDOM", "domineering", MctsConfig(), config, model)


class GHexCli(Cli):
    def getActionId(self):
        name = input("Move: ")
        m = re.fullmatch("([0-9]+) on ([0-9]+)", name)
        if m is None:
            return None
        tile = int(m.group(1))
        triangle = int(m.group(2))
        return black(tile, triangle)

    def display(self, state):
        print(env.pretty_game(state))
        print("")
        print(f"Black tiles remaining: {self._pretty_tiles(state._x.tiles[0][0])}")
        print(f"White tiles remaining: {self._pretty_tiles(state._x.tiles[0][1])}")
        print("")

    def _pretty_tiles(self, tiles):
      return "  ".join([(f"{val:2}" if tiles[i] else "  ")
                       for i, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])

    def loadModelBasedAgent(self, player_num):
        config, model = load_from_checkpoint("g_hex_20260125182112/000100.ckpt" if (player_num % 2) == 0 else "g_hex_20260125222445/000050.ckpt")
        return ModelAgent(f"v{player_num}", "g_hex", MctsConfig(), config, model)


class GHex2Cli(Cli):
    def getActionId(self):
        name = input("Move: ")
        m = re.fullmatch("([0-9]+) on ([0-9]+)", name)  # TODO: allow * as tile
        if m is None:
            return None
        tile = int(m.group(1))
        triangle = int(m.group(2))
        return black2(tile, triangle)

    def display(self, state):
        print(env.pretty_game(state))
        print("")
        print(f"Black tiles remaining: {self._pretty_tiles2(state._x.tiles[0][0])}")
        print(f"White tiles remaining: {self._pretty_tiles2(state._x.tiles[0][1])}")
        print("")

    def _pretty_tiles2(self, tiles):
      return "  ".join([(f"{val:2}" if tiles[i] else "  ")
                       for i, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])])

    def loadModelBasedAgent(self, player_num):
        v_num = player_num % 2
        config, model = load_from_checkpoint("g_hex2_20260327165407/000300.ckpt" if v_num == 0 else "g_hex2_20260327165407/000050.ckpt")
        return ModelAgent(f"v{"PRO!" if v_num == 0 else "WEAK"}", "g_hex2", MctsConfig(), config, model)


#class HeckmeckCli(Cli):
#    def getActionId(self):
#
#    def display(self, state):
#        print(f"To play: P{state.current_player[0]}")
#        print(f"Grill: {state._x.grill[0] * HECKMECK_TILE_VALS}")
#        print(f"Player stacks: P0={state._x.stacks[0][0]}")
#        print(f"               P1={state._x.stacks[0][1]}")
#        print(f"               P2={state._x.stacks[0][2]}")
#        print(f"Dice taken:  {state._x.dice_taken[0]}")
#        print(f"Dice rolled: {state._x.dice_rolled[0]}")


def GetCli(env_id: pgx.EnvId) -> Cli:
    match env_id:
        case "domineering":
            return DomineeringCli()
        case "g_hex":
            return GHexCli()
        case "g_hex2":
            return GHex2Cli()
        #case "heckmeck":
        #    return HeckmeckCli()
        case _:
            raise f"No CLI support for f{env_id}"


class Agent(ABC):
    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def getAction(self, key, state):
        pass


class KeyboardAgent(Agent):
    def __init__(self, cli):
        self.cli = cli

    def getName(self):
        return "Human"

    def getAction(self, key, state):
        action_i = None
        while action_i is None or not state.legal_action_mask[0][action_i]:
            try:
                action_i = self.cli.getActionId()
            except Exception as e:
                print(f"fail: {e}")
                action_i = None
        return jnp.int32([action_i])


class RandomAgent(Agent):
    def getName(self):
        return "Rando"

    def getAction(self, key, state):
        action_i = None
        while action_i is None or not state.legal_action_mask[0][action_i]:
            action_i = random.randint(0, len(state.legal_action_mask[0]) - 1)
        print(f"  Picking randomly! Selected action index {action_i} for no particular reason.")
        return jnp.int32([action_i])



class ModelAgent(Agent):
    def __init__(self, name_prefix, env_id, mcts_config: MctsConfig, config: Config, model):
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

        self.name_prefix = name_prefix
        self.env_id = env_id
        self.mcts_config = mcts_config
        self.mcts = partial(ModelAgent._run_mcts, forward, recurrent_fn, mcts_config, model) # Unjitted, for debugging.
        self.mcts_jit = jax.jit(self.mcts)

    def getName(self):
        return f"Model[{self.name_prefix}:sims={self.mcts_config.num_simulations}]"

    def getAction(self, key, state):
        # Debug view into the policy evaluation: (slow)
        # self.mcts(key, state, debug_domineering=(self.env_id == "domineering"), debug_g_hex=(self.env_id == "g_hex"))
        start_time = time.perf_counter()
        policy_output = self.mcts_jit(key, state)
        print(f"Thought for {time.perf_counter() - start_time:.1f} seconds.")

        # Print some game-specific debug info.
        if self.env_id == "domineering":
            action_weights = jnp.hstack([
                policy_output.action_weights.reshape(8, 7),
                jnp.zeros((8, 1), dtype=jnp.float32),
            ])
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
        elif self.env_id == "g_hex2":
            action_weights = policy_output.action_weights.reshape(23, 11)
            print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" for w in tri_row])
                             for tri_i, tri_row in enumerate(action_weights)]))
            print("")

        return policy_output.action

    @staticmethod  # Static for JITting.
    def _run_mcts(forward, recurrent_fn, mcts_config, model, key, state, debug_domineering=False, debug_g_hex=False):
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
            #elif debug_g_hex2:
            #    action_weights = jax.scipy.special.softmax(logits.reshape(23, 11))
            #    print("Action weights:")
            #    print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" for w in tri_row])
            #                     for tri_i, tri_row in enumerate(action_weights)]))
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
            max_num_considered_actions=mcts_config.max_num_considered_actions,
            #qtransform=mctx.qtransform_completed_by_mix_value, # TODO: optimize? https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
            gumbel_scale=0.0,
        )
        return policy_output


def load_from_checkpoint(path):
  with open(f"checkpoints/{path}", "rb") as f:
      ckpt = pickle.load(f)
      return ckpt["config"], ckpt["model"]

def readable_domineering_board(state):
  board = state._x.board[0]
  return jax.lax.select(state.current_player[0] == 0, board, board.transpose())


#HECKMECK_TILE_VALS = jnp.int32([
#    0,
#    21, 22, 23, 24,
#    25, 26, 27, 28,
#    29, 30, 31, 32,
#    33, 34, 35, 36,])

#HECKMECK_WORM_VALS = jnp.int32([
#    0,
#    1, 1, 1, 1,
#    2, 2, 2, 2,
#    3, 3, 3, 3,
#    4, 4, 4, 4])

HECKMECK_TILE_VALS = jnp.int32([
    0,
    11, 12,
    13, 14,
    15, 16,
    17, 18,
])

HECKMECK_WORM_VALS = jnp.int32([
    0,
    1, 1, 2, 2,
    3, 3, 4, 4,
])

_HECKMECK_ACTION_NAMES = [
    "Took worms and kept rolling",
    "Took 1s and kept rolling",
    "Took 2s and kept rolling",
    "Took 3s and kept rolling",
    "Took 4s and kept rolling",
    "Took 5s and kept rolling",
    "Took worms and ended turn",
    "Took 1s and ended turn",
    "Took 2s and ended turn",
    "Took 3s and ended turn",
    "Took 4s and ended turn",
    "Took 5s and ended turn",
    "Busted",
]

if __name__ == "__main__":
    tourney_conf_dict = OmegaConf.from_cli()
    tourney_config: TourneyConfig = TourneyConfig(**tourney_conf_dict)
    print(tourney_config)

    cli = GetCli(tourney_config.env_id)

    devices = jax.local_devices()

    env = pgx.make(tourney_config.env_id)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    print("\n\nLet's play!\n\n\n")

    def run_game(game_num, agents, start_depth):
        key = jax.random.PRNGKey(tourney_config.seed ^ game_num)
        key, subkey = jax.random.split(key)
        state: pgx.State = init_fn(jax.random.split(subkey, 1))

        # Play random moves up to start_depth to determine the initial position.
        # These must be the same for each minimatch so we divide by the number of agents.
        random.seed(77659 ^ (game_num // len(agents)))
        for move_i in range(0, start_depth):
            action_i = None
            while action_i is None or not state.legal_action_mask[0][action_i]:
                action_i = random.randint(0, len(state.legal_action_mask[0]) - 1)
            state = step_fn(state, jnp.int32([action_i]), jax.random.split(subkey, 1))

        turn_num = 1 + start_depth
        while True:
            cli.display(state)

            if state.terminated.all():
                print(f"Game over! winner={state._x.winner} rewards={state.rewards}")
                return state._x.winner

            agent = agents[state.current_player[0]]
            print(f"Turn {turn_num}: {agent.getName()} to play because current_player={state.current_player[0]}...", flush=True)
            action = agent.getAction(key, state)
            # TODO: absorb this into the CLI.
            if tourney_config.env_id == "domineering":
                print(f"{agent.getName()} played {'abcdefgh'[action[0] % 7]}{1 + (action[0] // 7)}\n")
            elif tourney_config.env_id == "g_hex":
                print(f"{agent.getName()} played the {1 + (action[0] % 10)} on triangle {action[0] // 10}\n")
            elif tourney_config.env_id == "g_hex2":
                print(f"{agent.getName()} played the {1 + (action[0] % 11)} on triangle {action[0] // 11}\n")  # TODO: *
            elif tourney_config.env_id == "heckmeck":
                print(f"{agent.getName()} took action {action[0]}: {_HECKMECK_ACTION_NAMES[action[0]]}\n")

            key, subkey = jax.random.split(key)
            state = step_fn(state, action, jax.random.split(subkey, 1))
            turn_num += 1


    agents = [
        #RandomAgent(),
        KeyboardAgent(cli),
        cli.loadModelBasedAgent(0),
        #cli.loadModelBasedAgent(1),
    ]
    wins = np.zeros_like(agents)
    for game_num in range(0, tourney_config.games):
        rotation_pos = game_num % len(agents)
        game_agents = agents[rotation_pos:] + agents[:rotation_pos]
        print(f"Game {game_num} of {tourney_config.games}: {" vs ".join([agent.getName() for agent in game_agents])}")

        winner = run_game(game_num, game_agents, 0)  # TODO: more interesting start depths
        for w in range(0, len(agents)):
            if (winner + rotation_pos) % len(agents) == w:
                wins[w] += 1
        win_rates = 100 * wins / (1 + game_num)
        print(f"""#######################################################################
                  Win rates after {1 + game_num} {'game' if game_num == 0 else 'games'}:""")
        for w in range(0, len(agents)):
            print(f"       {agents[w].getName():>20}: {win_rates[w]:6.2f}% ({wins[w]})", flush=True)

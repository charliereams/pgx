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

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    """Internal state for the game Domineering on an 8x8 board."""

    color: Array = jnp.int32(0)
    # 8x8 board
    # [[ 0,  1,  2,  3,  4,  5,  6,  7],
    #  [ 8,  9, 10, 11, 12, 13, 14, 15],
    #  [16, 17, 18, 19, 20, 21, 22, 23],
    #  [24, 25, 26, 27, 28, 29, 30, 31],
    #  [32, 33, 34, 35, 36, 37, 38, 39],
    #  [40, 41, 42, 43, 44, 45, 46, 47],
    #  [48, 49, 50, 51, 52, 53, 54, 55],
    #  [56, 57, 58, 59, 60, 61, 62, 63]]
    board: Array = jnp.ones(64, jnp.bool_)  # True (available), False (occupied)
    winner: Array = jnp.int32(-1)


class Game:
    """The game representation of Domineering on an 8x8 board."""

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        """Performs a step in the Domineering game.

        Args:
          state: The current game state.
          action: The chosen action, representing the index of the top/left
            square of the domino to be placed.

        Returns:
          The new game state after the action has been applied.
        """
        new_board = state.board.at[jnp.array([action, action + jax.lax.select(state.color == 0, 1, 8)])].set(False)

        def can_play(move_mask):
            return (new_board & move_mask).sum(axis=0) == 2

        # Game is over if the player next to play has no legal moves.
        has_next_move = jax.vmap(can_play)(jax.lax.select(state.color == 0, MASK_CACHE_V, MASK_CACHE_H)).any()

        return state._replace(  # type: ignore
            color=1 - state.color,
            board=new_board,
            winner=jax.lax.select(has_next_move, -1, state.color),
        )

    def observe(self, state: GameState, _: Optional[Array] = None) -> Array:
        return state.board.reshape(8, 8)

    def legal_action_mask(self, state: GameState) -> Array:
        # To be legal, a move have its own square and a neighbour free, and not be
        # on the edge of the board. The relevant definition of neighbour and edge
        # depends on the player's direction.
        return state.board & jax.lax.select(
            state.color == 0,
            EDGE_EXCLUDER_H & jnp.roll(state.board.reshape(8, 8), shift=-1, axis=1).flatten(),
            EDGE_EXCLUDER_V & jnp.roll(state.board, shift=-8, axis=0),
        )

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0  # Game always ends with a winner.

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )


def _make_mask_cache_horizontal():
    move_masks = []
    for x in range(7):
        for y in range(8):
            move_masks.append(jnp.zeros(64, jnp.bool_).at[jnp.array([y * 8 + x, y * 8 + x + 1])].set(True))
    return jnp.array(move_masks)


def _make_mask_cache_vertical():
    move_masks = []
    for x in range(8):
        for y in range(7):
            move_masks.append(jnp.zeros(64, jnp.bool_).at[jnp.array([y * 8 + x, y * 8 + x + 8])].set(True))
    return jnp.array(move_masks)


# Precomputed masks for required empty squares for each possible move.
MASK_CACHE_H = _make_mask_cache_horizontal()
MASK_CACHE_V = _make_mask_cache_vertical()

# Blockers for moves along the (player-appropriate) edge.
EDGE_EXCLUDER_H = jnp.tile(jnp.ones(8, jnp.bool_).at[7].set(False), 8)
EDGE_EXCLUDER_V = jnp.append(jnp.tile(jnp.ones(8, jnp.bool_), 7), jnp.zeros(8, jnp.bool_))

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
import functools
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

class GameState(NamedTuple):
    """Internal state for the game Heckmeck."""

    color: Array = jnp.int32(0)
    grill: Array = jnp.ones(17, jnp.bool_)  # True (available), False (occupied)
    stacks: Array = jnp.zeros((3, 16), jnp.int32)
    dice_rolled: Array = jnp.zeros(6, jnp.int32)
    dice_taken: Array = jnp.zeros(6, jnp.int32)
    winner: Array = jnp.int32(-1)


def _winner(grill: Array, stacks: Array) -> Array:
    worm_totals = WORM_VALS[stacks].sum(axis=1)
    best_tiles = stacks.max(axis=1)

    return jax.lax.select(grill.sum() == 0,
                          jnp.argmax(worm_totals + best_tiles / 64),
                          jnp.int32(-1))

# Which dice a player gets to keep, based on how many they already rolled.
_DICE_MASK = jnp.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

TILE_VALS = jnp.int32([
    0,
    21, 22, 23, 24,
    25, 26, 27, 28,
    29, 30, 31, 32,
    33, 34, 35, 36,])

WORM_VALS = jnp.int32([
    0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4])

_DICE_VALS = jnp.int32([5, 1, 2, 3, 4, 5])

def _roll(dice_taken: Array, key: PRNGKey) -> Array:
    assert isinstance(key, Array)
    roll = jax.random.randint(key, shape=(8), minval=1, maxval=7, dtype=jnp.int32)
    roll_kept = roll * _DICE_MASK[dice_taken.sum()]
    return jnp.int32([
        (roll_kept == 6).sum(),
        (roll_kept == 1).sum(),
        (roll_kept == 2).sum(),
        (roll_kept == 3).sum(),
        (roll_kept == 4).sum(),
        (roll_kept == 5).sum(),
    ])


def _step_take_and_roll(die, state: GameState, action: Array, key: PRNGKey) -> GameState:
    dice_taken = state.dice_taken.at[action].set(state.dice_rolled[action])
    return state._replace(
        dice_rolled = _roll(dice_taken, key),
        dice_taken = dice_taken,
    )


def _step_take_and_stop(die, state: GameState, action: Array, key: PRNGKey) -> GameState:
    dice_taken = state.dice_taken.at[action - 6].set(state.dice_rolled[action - 6])
    total_made = (_DICE_VALS * dice_taken).sum()
    tile_index_taken = jnp.argmax(state.grill * TILE_VALS * (TILE_VALS <= total_made))

    new_grill = state.grill.at[tile_index_taken].set(False)
    new_stacks = state.stacks.at[state.color].set(
        jnp.roll(state.stacks[state.color], 1).at[0].set(tile_index_taken))
    fresh_dice_taken = jnp.zeros(6, dtype=jnp.int32)

    return state._replace(
        color = (state.color + 1) % 3,
        grill = new_grill,
        stacks = new_stacks,
        dice_rolled = _roll(fresh_dice_taken, key),
        dice_taken = fresh_dice_taken,
        winner = _winner(new_grill, new_stacks),
    )


def _step_steal(from_player, state: GameState, action: Array, key: PRNGKey) -> GameState:
    return state  # TODO


def _step_bust(state: GameState, action: Array, key: PRNGKey) -> GameState:
    assert isinstance(key, Array)

    returned_tile = state.stacks[state.color][0]

    # Apply the flip first, then return the tile.
    # TODO: this is wrong in the case where the returned tile is the new highest.
    grill_with_flip = state.grill.at[len(state.grill) - 1 - jnp.argmax(state.grill[::-1])].set(False)
    new_grill = jax.lax.select(returned_tile == 0,
                               state.grill,
                               grill_with_flip.at[returned_tile].set(True))
    new_stacks = state.stacks.at[state.color].set(jnp.pad(state.stacks[state.color][1:], (0,1)))

    fresh_dice_taken = jnp.zeros(6, jnp.int32)

    return state._replace(  # type: ignore
        color = (state.color + 1) % 3,
        grill = new_grill,
        stacks = new_stacks,
        dice_rolled = _roll(fresh_dice_taken, key),
        dice_taken = fresh_dice_taken,
        winner = _winner(new_grill, new_stacks)
    )


class Game:
    """The game representation of Heckmeck."""

    def init(self, key: PRNGKey) -> GameState:
        state = GameState()
        return state._replace(
            dice_rolled = _roll(state.dice_taken, key),
        )

    def step(self, state: GameState, action: Array, key: PRNGKey) -> GameState:
        return jax.lax.switch(action,
          [
             functools.partial(_step_take_and_roll, 0),
             functools.partial(_step_take_and_roll, 1),
             functools.partial(_step_take_and_roll, 2),
             functools.partial(_step_take_and_roll, 3),
             functools.partial(_step_take_and_roll, 4),
             functools.partial(_step_take_and_stop, 5),
             functools.partial(_step_take_and_stop, 0),
             functools.partial(_step_take_and_stop, 1),
             functools.partial(_step_take_and_stop, 2),
             functools.partial(_step_take_and_stop, 3),
             functools.partial(_step_take_and_stop, 4),
             functools.partial(_step_take_and_stop, 5),
             _step_bust,
             # TODO: Steal from 1 and 2.
          ],
          state, action, key,
        )

    def observe(self, state: GameState) -> Array:
        stack_state = jnp.array([
            state.stacks[state.color],
            state.stacks[(state.color + 1) % 3],
            state.stacks[(state.color + 2) % 3],
        ])

        global_state = jnp.hstack([
            jnp.ones(16, dtype=jnp.int32) * state.grill[1:],
            state.dice_rolled,
            state.dice_taken,
        ])

        return jnp.dstack([
            stack_state,
            jnp.tile(global_state, 3 * 16).reshape(3, 16, -1),
        ])


    def legal_action_mask(self, state: GameState) -> Array:
        has_worm_already = state.dice_taken[0] > 0
        dice_taken_already = state.dice_taken.sum()
        total_already = (_DICE_VALS * state.dice_taken).sum()
        picks_taken = (state.dice_taken > 0).sum()

        can_take_0 = (state.dice_taken[0] == 0) & (state.dice_rolled[0] > 0)
        can_take_1 = (state.dice_taken[1] == 0) & (state.dice_rolled[1] > 0)
        can_take_2 = (state.dice_taken[2] == 0) & (state.dice_rolled[2] > 0)
        can_take_3 = (state.dice_taken[3] == 0) & (state.dice_rolled[3] > 0)
        can_take_4 = (state.dice_taken[4] == 0) & (state.dice_rolled[4] > 0)
        can_take_5 = (state.dice_taken[5] == 0) & (state.dice_rolled[5] > 0)

        grill_values = state.grill * TILE_VALS
        smallest_grill_value = jnp.min(jnp.where(grill_values == 0, 999, grill_values))

        legal_moves = jnp.bool_([
            can_take_0 & (dice_taken_already + state.dice_rolled[0] < 8) & (picks_taken < 6),
            can_take_1 & (dice_taken_already + state.dice_rolled[1] < 8) & (picks_taken < 6),
            can_take_2 & (dice_taken_already + state.dice_rolled[2] < 8) & (picks_taken < 6),
            can_take_3 & (dice_taken_already + state.dice_rolled[3] < 8) & (picks_taken < 6),
            can_take_4 & (dice_taken_already + state.dice_rolled[4] < 8) & (picks_taken < 6),
            can_take_5 & (dice_taken_already + state.dice_rolled[5] < 8) & (picks_taken < 6),
            can_take_0 & (total_already + state.dice_rolled[0] * 5 >= smallest_grill_value),
            can_take_1 & (total_already + state.dice_rolled[0] * 1 >= smallest_grill_value) & has_worm_already,
            can_take_2 & (total_already + state.dice_rolled[0] * 2 >= smallest_grill_value) & has_worm_already,
            can_take_3 & (total_already + state.dice_rolled[0] * 3 >= smallest_grill_value) & has_worm_already,
            can_take_4 & (total_already + state.dice_rolled[0] * 4 >= smallest_grill_value) & has_worm_already,
            can_take_5 & (total_already + state.dice_rolled[0] * 5 >= smallest_grill_value) & has_worm_already,
        ])
        return jnp.hstack([legal_moves, (~legal_moves).all()])


    def is_terminal(self, state: GameState) -> Array:
        return state.grill.sum() == 0


    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1, -1]).at[state.winner].set(1.0),
            jnp.zeros(3, jnp.float32),
        )

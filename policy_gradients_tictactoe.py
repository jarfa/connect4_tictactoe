"""
Build a super dumb supervised learning approach to tictactoe.
"""
import sys
import random
import numpy as np
from collections import deque
from tictactoe_env import TictactoeEnv
from utils import random_opponent, show_board, IllegalMoveError

import logging

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y.%m.%d %I:%M:%S",
    level=logging.DEBUG
)

def forward(board, weights):
    """predict next move"""
    # TODO: I didn't bother with biases yet... I should do that.
    product = np.dot(board.reshape((1, -1)), weights).ravel()
    exp_product = np.exp(product)
    return exp_product / sum(exp_product)  # softmax
    

def backward(board, diff):
    return np.dot(
        board.reshape((-1, 1)),
        diff.reshape((1, -1))
    )


def discount_rewards(r, gamma=0.8):
    """ take 1D float array of rewards and compute discounted reward
    """
    discounted_r = np.zeros(len(r))
    game_reward = r[-1]
    discounted_r[0] = 1.0
    for i in range(1, len(r)):
        discounted_r[i] = discounted_r[i - 1] * gamma

    # The total reward for each game should sum to the final reward. Otherwise,
    # the agent would prefer longer wins over shorter wins, or shorter losses over
    # longer losses, which would lead to weird behavior
    discounted_r = discounted_r[::-1]
    discounted_r = discounted_r * game_reward / sum(discounted_r)
    return list(discounted_r)


env = TictactoeEnv()

N_games = int(1e6)

minibatch = 2500
learning_rate = 0.1
lr_decay = 0.99
# l2_reg = 0.01
init_weight_sd = 0.1  #0.5

np.random.seed(5)
weights = np.random.randn(18, 9) * init_weight_sd

reward_window = deque(maxlen=5000)

rewards, states, diffs = [], [], []
move_choices = [(r, c) for r in range(3) for c in range(3)]

max_win_rate = 0.0

for g in range(1, 1 + N_games):
    env.reset()

    for i in range(1, 101):
        # my_move = random.choice(env.legal_moves())  # take a random action
        board_repr = env.flat_board()

        states.append(board_repr)

        pred = forward(board_repr, weights)
        my_move = move_choices[np.random.choice(len(move_choices), p=pred)]

        max_p = max(pred)
        y = np.array(pred == max_p, dtype=np.float)  #one-hot encoding of the move we made
        diffs.append(y - pred)        
        # if my move is an illegal space, that's a forfiet
        forfeit = False
        try:
            _, reward, done, info = env.step(my_move, random_opponent)
        except IllegalMoveError:
            reward = -2
              #illegal moves are embarrassing, should be worse than losing
            done = True
            forfeit = True
        if done:
            episode_rewards = [reward] * i
            rewards.extend(discount_rewards(episode_rewards))
            break
    
    reward_window.append(
        'w' if reward == 1 else ('t' if reward == 0 else (
            'f' if forfeit else 'l'))
    )
    if len(reward_window) == reward_window.maxlen:
        recent_rewards = np.array(reward_window)
        def outcome_rate(outcome):
            return np.mean(recent_rewards == outcome)
            
        logging.info(
            "Episode %d. W: %.3f  L: %.3f  F: %.3f  T: %.3f  Wnorm: %.1e (lr=%.1e)",
            g, outcome_rate("w"), outcome_rate("l"), outcome_rate("f"), outcome_rate("t"),
            np.linalg.norm(weights, "fro"),
            learning_rate
        )
        max_win_rate = max(max_win_rate, outcome_rate("w"))
        reward_window.clear()

    if g % minibatch == 0:
        # standardize the rewards
        rewards = np.array(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        gradients = [backward(s, r * d) for r, s, d in zip(rewards, states, diffs)]
        weights += learning_rate * np.sum(gradients, axis=0)
        # TODO: l2 norm regularization???
        # weights -= 2.0 * l2_reg * weights

        learning_rate *= lr_decay

        rewards, states, diffs = [], [], []

print("Max win rate: %.3f" % max_win_rate)
from collections import deque
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import random_opponent, IllegalMoveError, discount_rewards


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
    level=logging.DEBUG
)

EPSILON = np.finfo(np.float32).eps.item()


def model_num_weights(policy):
    return sum([
        len(l.view(-1)) for l in policy.parameters()
    ])


class BasePolicy(nn.Module):
    def __init__(self, gpu):
        super(BasePolicy, self).__init__()

        self.gpu = gpu
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, only_best=False):
        tensor_state = torch.from_numpy(state).float()

        if self.gpu:
            tensor_state = tensor_state.cuda()

        probs = self.__call__(tensor_state)
        if only_best:
            # Return the best action, don't randomly choose from a distribution.
            # There won't be any training, so no need to save the log probs
            return np.argmax(probs)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).unsqueeze(0))
        return int(action.item())

    def finish_episode(self):
        """Unlike in many RL problems, our episodes never contain more than one
        reward.
        """
        rewards_to_learn = torch.tensor(self.rewards)
        if self.gpu:
            rewards_to_learn = rewards_to_learn.cuda()

        rewards_to_learn = (rewards_to_learn - rewards_to_learn.mean()
                            ) / (rewards_to_learn.std() + EPSILON)
        policy_loss = torch.dot(
            torch.cat(self.saved_log_probs).squeeze(),
            rewards_to_learn
        )
        final_loss = policy_loss.sum()
        return final_loss

    def cleanup(self):
        del self.rewards[:]
        del self.saved_log_probs[:]

    @staticmethod
    def repr_input(env):
        "Each policy could need a different input format"
        raise NotImplementedError()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))



class FlatConnectPolicy(BasePolicy):
    "Works for any size of board, connect 4 or tictactoe"
    def __init__(self, board_size, H, double=False, gpu=False):
        # `double` states whether the input is the same size as the board
        # with with value {+1,0,-1} or 2x the size of the board and puts
        # a 1 down for the x's and o's in separate cells.
        super(FlatConnectPolicy, self).__init__(gpu=gpu)

        board_units = np.product(board_size)
        self.double = double
        input_units = board_units * (2 if double else 1)
        self.hidden = nn.Linear(input_units, H)
        self.out = nn.Linear(H, board_units)

    def forward(self, x):
        h = F.relu(self.hidden(x))
        logp = self.out(h)
        return F.softmax(logp, dim=0)

    def repr_input(self, env):
        if self.double:
            return env.flat_board_double()
        return env.flat_board()


class ConvConnectPolicy(BasePolicy):
    "Kind of a kitchen sink approach to the problem"
    def __init__(self, board_size, conv_channels, H1, H2, gpu=False):
        super(ConvConnectPolicy, self).__init__(gpu=gpu)
        board_units = np.product(board_size)
        conv2d_output_size = (board_size[0] - 1) * (board_size[1] - 1)
        conv2_output_n = conv2d_output_size * conv_channels

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=2)
        self.fc1 = nn.Linear(conv2_output_n + board_units, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, board_units)

    def forward(self, x):
        formatted_x = x.unsqueeze(0).unsqueeze(0)
        # concat the convolutional results with the raw input
        conv_results = (F.relu(self.conv2d(formatted_x).view(-1)),
                        formatted_x.view(-1))
        hidden1 = self.fc1(torch.cat(conv_results, 0))
        hidden2 = self.fc2(F.relu(hidden1))
        logp = self.fc3(F.relu(hidden2))
        return F.softmax(logp, dim=0)

    @staticmethod
    def repr_input(env):
        return env.square_board()



def translate_outcome(reward, forfeit):
    return 'f' if forfeit else ('w' if reward == 1 else (
        't' if reward == 0 else 'l'))


def play_and_train(
    env,
    policy,
    optimizer,
    N_games,
    minibatch,
    forfeit_reward=-2,
    opponent_fn=random_opponent,
    show_moves_freq=None
    ):
    reward_window = deque(maxlen=max(int(1e4), minibatch))
    max_win_rate = 0.0

    logging.info(
        "Playing %.1e games with a minibatch of %d and an initial learning rate of %.1e",
        N_games, minibatch, optimizer.param_groups[0]["lr"]
    )

    logging.info("Weights to train: %d", model_num_weights(policy))

    for g in range(1, 1 + N_games):
        env.reset()
        show_moves = show_moves_freq is not None and g % show_moves_freq == 0
        if show_moves:
            print("Game %d" % g)
        for i in range(1, 1000):
            forfeit = False  # if my move is an illegal space, that's a forfeit
            # my_move = random.choice(env.legal_moves())  # take a random legal action
            action_ix = policy.select_action(policy.repr_input(env))
            my_move = env.move_choices[action_ix]
            try:
                _, reward, done, info = env.step(my_move, opponent_fn, show_moves=show_moves)
            except IllegalMoveError:
                reward = forfeit_reward
                done = forfeit = True
            if done:
                episode_rewards = discount_rewards(-1.0 * reward, i)
                policy.rewards.extend(episode_rewards)
                assert len(policy.rewards) == len(policy.saved_log_probs), (
                    "Something is wrong: rewards len={}, logprobs len={}".format(
                        len(policy.rewards), len(policy.saved_log_probs)
                    ))
                break


        reward_window.append(translate_outcome(reward, forfeit))

        if len(reward_window) == reward_window.maxlen:
            recent_rewards = np.array(reward_window)
            def outcome_rate(outcome):
                return np.mean(recent_rewards == outcome)

            logging.info(
                "Episode %s;   W: %.3f  L: %.3f  F: %.3f  T: %.3f",
                "{0:>7}".format(g), outcome_rate("w"), outcome_rate("l"),
                outcome_rate("f"), outcome_rate("t")
            )

            max_win_rate = max(max_win_rate, outcome_rate("w"))
            reward_window.clear()

        if g % minibatch == 0:
            loss = policy.finish_episode()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            policy.cleanup()

    logging.info("Max win rate: %.3f", max_win_rate)
    return policy

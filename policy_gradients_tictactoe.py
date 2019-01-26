"""
Build a super dumb supervised learning approach to tictactoe.
"""
import numpy as np
import torch
from torch import optim

from tictactoe_env import TictactoeEnv
from c4_env import Connect4Env
from policy import (
    FlatConnectPolicy,
    ConvConnectPolicy,
    play_and_train,
)
from utils import random_opponent, make_next_stop_opponent

import logging

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
    level=logging.DEBUG
)

torch.manual_seed(5)

policy = ConvConnectPolicy(conv_channels=7, H1=100, H2=50, board_size=(3, 3))

half_smart_opponent = make_next_stop_opponent(0.5)

trained_policy = play_and_train(
    TictactoeEnv(),
    policy,
    optim.Adam(policy.parameters(), lr=1e-3),
    N_games=int(6e6),
    minibatch=int(5e4),
    forfeit_reward=-3,
    opponent_fn=half_smart_opponent,
)



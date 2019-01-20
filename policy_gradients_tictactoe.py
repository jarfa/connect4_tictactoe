"""
Build a super dumb supervised learning approach to tictactoe.
"""
import numpy as np
from torch import optim

from tictactoe_env import TictactoeEnv
from c4_env import Connect4Env
from policy import (
    FlatConnectPolicy,
    SimpleConvPolicy,
    ConvPlusHiddenPolicy,
    play_and_train,
    play_and_test
)
from utils import random_opponent, check_next_step_opponent

torch.manual_seed(5)

policy = ConvPlusHiddenPolicy(12, board_size=(3, 3))

trained_policy = play_and_train(
    TictactoeEnv(),
    policy,
    optim.Adam(policy.parameters(), lr=0.025),
    N_games=int(5e6),
    minibatch=int(2e4),
    forfeit_reward=-5,
    opponent_fn=check_next_step_opponent,
    test_every=int(5e5),
    test_N=int(5e4)
)

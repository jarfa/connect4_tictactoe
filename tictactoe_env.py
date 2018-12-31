import gym
from gym import error, spaces, utils
from gym.utils import seeding

from itertools import product

from utils import connected_rows, BaseConnectEnv

class TictactoeEnv(BaseConnectEnv):
    rows = 3
    columns = 3
    my_team = "x"
    their_team = "o"
    num_to_connect = 3

    def __init__(self):
        super().__init__()

    def legal_moves(self):
        "I don't like gym's action_space API, I'll do it my own way"
        return self.empty_spaces()

    def make_move(self, action, mark):
        r, c = action
        self.board[r][c] = mark

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from utils import BaseConnectEnv, array_connected


def connect4_rows(board):
    for row in board[::-1]:
        if array_connected(row, num_to_connect=4):
            return True
    return False


class Connect4Env(BaseConnectEnv):
    rows = 6
    columns = 7
    my_team = "r"
    their_team = "b"
    num_to_connect = 4

    def __init__(self):
        super().__init__()

    def lowest_empty(self):
        """For each column, what's the ix of the next empty row? -1 for all full
        We'll assume that [0,0] is the bottom-left
        """
        results = []
        all_empties = self.empty_spaces()
        for col in range(self.columns):
            rows = [r for r, c in all_empties if c == col]
            if len(rows) == 0:
                results.append(-1)
            else:
                results.append(min(rows))

        return results

    def legal_moves(self):
        "A list of which columns aren't yet full"
        cols = list(set(
           [c for _, c in self.empty_spaces()]
        ))
        return sorted(cols)

    def make_move(self, action, mark):
        c = action
        r = self.lowest_empty()[c]
        self.board[r][c] = mark


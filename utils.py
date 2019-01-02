import gym
from itertools import product
import numpy as np
import random

def show_board(board):
    "We'll assume that [0,0] is the bottom-left"
    for r in reversed(range(len(board))):
        pretty_row = " ".join(str(e) for e in board[r])
        print(pretty_row)

def random_opponent(env):
    "An opponent that chooses squares randomly"
    options = env.legal_moves()
    return random.choice(options)


def array_connected(row, num_to_connect):
    assert len(row) >= num_to_connect
    for i in range(1 + len(row) - num_to_connect):
        subset = row[i:(i + num_to_connect)]
        # check that they're all the same, but not empty
        unique_vals = set(subset)
        if len(unique_vals) == 1 and 0 not in unique_vals:
            return True

    return False


def connected_rows(board, num_to_connect):
    for row in board[::-1]:
        if array_connected(row, num_to_connect):
            return True
    return False


def connected_diagonals(board, num_to_connect):
    """This only looks at upward-sloping diagonals, with board cell [0][0]
    being the bottom left. Flip the board  horizontally to look at both directions.
    """
    column_lengths = [len(x) for x in board]
    if len(set(column_lengths)) > 1:
        raise ValueError("Not a legal board. Column lengths: %s" % column_lengths)

    rows = len(board)
    columns = len(board[0])
    if num_to_connect > min(rows, columns):
        raise ValueError("Diagonal length (%d) is less than the rows(%d) and columns(%d)" %
                         (num_to_connect, rows, columns))
    # To cover all upward-sloping diagonals, we need to pad the board with extra rows
    # on the bottom.
    rows_padding = columns - num_to_connect
    board = (rows_padding * [[0] * columns]) + board
    rows += rows_padding

    for starting_row in range(0, 1 + rows - num_to_connect):
        diag = []
        r = starting_row
        c = 0
        while r < rows and c < columns:
            diag.append(board[r][c])
            r += 1
            c += 1
    
        if array_connected(diag, num_to_connect):
            return True
    
    return False


class IllegalMoveError(Exception):
    pass


class BaseConnectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    rows = None
    columns = None
    num_to_connect = None
    my_team = None
    their_team = None

    def __init__(self):
        self.board_init()

    def board_init(self):
        self.board = [
            [0 for _ in range(self.columns)]
            for _ in range(self.rows)
        ]

    def render(self, mode='human', close=False):
        show_board(self.board)

    def flat_board(self):
        # concat the board row-wise and player-wise, convert to a numeric array
        numeric_board = []
        for mark in (self.my_team, self.their_team):
            for row in self.board:
                numeric_board += [1.0 if x == mark else 0.0 for x in row]

        assert len(numeric_board) == (self.rows * self.columns * 2)
        return np.array(numeric_board)

    def success(self):
        # check the rows
        if connected_rows(self.board, self.num_to_connect):
            return True
        # check the columns
        transposed_board = [
            [self.board[r][c] for r in range(self.rows)]
            for c in range(self.columns)
        ]
        if connected_rows(transposed_board, self.num_to_connect):
            return True
        # upward-sloping diagonals
        if connected_diagonals(self.board, self.num_to_connect):
            return True
        # flip the board to check the other direction of diagonals
        flipped_board = [row[::-1] for row in self.board]
        if connected_diagonals(flipped_board, self.num_to_connect):
            return True 
        return False

    def empty_spaces(self):
        empties = []
        for r, c in product(range(self.rows), range(self.columns)):
            if self.board[r][c] == 0:
                empties.append((r, c))
    
        return empties

    def full_board_tie(self, verbose=False):
        if len(self.empty_spaces()) == 0:
            if verbose:
                print("No more moves, tie game!!!")
            return True

        return False

    def legal_moves(self):
        raise NotImplementedError()

    def make_move(self, action, mark):
        """TicTacToe takes in a row/column pair, Connect4 just a column.
        Let the subclasses handle it.
        """
        raise NotImplementedError()

    def step(self, action, opponent_fn):
        options = self.legal_moves()
        # print("options: %s" % options)
        if action not in options:
            raise IllegalMoveError(
                "Bad move: {action} is not in {options}".format(
                    action=action, options=options))

        
        self.make_move(action, self.my_team)
        reward = 0
        full = False
        
        my_win = self.success()
        if my_win:
            reward = 1
        else:
            full = self.full_board_tie()
            if not full:
                self.make_move(opponent_fn(self), self.their_team)
                their_win = self.success()
                if their_win:
                    reward = -1
                else:
                    full = self.full_board_tie()
        
        observation = self.board
        done = reward != 0 or full
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.board_init()

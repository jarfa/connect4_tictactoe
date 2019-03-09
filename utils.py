from itertools import product
import random
import numpy as np
import gym


def show_board(board):
    "We'll assume that [0,0] is the bottom-left"
    for r in reversed(range(len(board))):
        pretty_row = " ".join(str(e) for e in board[r])
        print(pretty_row)


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
    for row in board:
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


def board_success(board, num_to_connect):
    # check the rows
    if connected_rows(board, num_to_connect):
        return True
    # check the columns
    rows = len(board)
    columns = len(board[0])
    transposed_board = [
        [board[r][c] for r in range(rows)]
        for c in range(columns)
    ]
    if connected_rows(transposed_board, num_to_connect):
        return True
    # upward-sloping diagonals
    if connected_diagonals(board, num_to_connect):
        return True
    # flip the board to check the other direction of diagonals
    flipped_board = [row[::-1] for row in board]
    if connected_diagonals(flipped_board, num_to_connect):
        return True
    return False


def random_opponent(env):
    "An opponent that chooses squares randomly"
    options = env.legal_moves()
    return random.choice(options)


def _try_moves(legal_options, board, num_to_connect, team_mark):
    for mv in random.sample(legal_options, len(legal_options)):
        r, c = mv
        tmp_board = list(board)  #copy it
        tmp_board[r][c] = team_mark
        if board_success(tmp_board, num_to_connect):
            return mv

    return None

def make_next_stop_opponent(random_only_fraction):
    "Just a factory to make it easy to create easy to hard opponents"
    def check_next_step_opponent(env):
        """Checks to see if the next step for either them or me leads to a win
        Moves randomly otherwise
        """
        options = env.legal_moves()
        if random.random() < random_only_fraction:  # sometimes move randomly
            return random.choice(options)

        # Check to see if any of their moves lead directly to a win, then check
        # to see if any of my moves lead directly to a win - block if so.
        for team in (env.their_team, env.my_team):
            mv = _try_moves(options, env.board, env.num_to_connect, team)
            if mv is not None:
                return mv

        return random.choice(options)
    return check_next_step_opponent


def printable_action(action):
    if len(action) == 1:
        return str(action)
    return ",".join(str(x) for x in action)


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
        # 0 will denote an empty cell
        self.board = [
            [0 for _ in range(self.columns)]
            for _ in range(self.rows)
        ]

    def render(self, mode='human', close=False):
        show_board(self.board)

    def flat_board_double(self):
        """Concat the board row-wise and player-wise, convert to a numeric array"""
        numeric_board = []
        for mark in (self.my_team, self.their_team):
            for row in self.board:
                numeric_board += [1.0 if x == mark else 0.0 for x in row]

        assert len(numeric_board) == (self.rows * self.columns * 2)
        return np.array(numeric_board)

    def square_board(self):
        """Unlike the flat board, we'll code up different marks as 1 and -1, respectively"""
        def translator(x):
            return {self.my_team: 1.0, self.their_team: -1.0}.get(x, 0.0)

        return np.array([[translator(x) for x in row] for row in self.board])

    def flat_board(self):
        return self.square_board().ravel()

    def empty_spaces(self):
        empties = []
        for r, c in product(range(self.rows), range(self.columns)):
            if self.board[r][c] == 0:
                empties.append((r, c))

        return empties

    def full_board_tie(self, verbose=False):
        if not self.empty_spaces():
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

    def step(self, action, opponent_fn, show_moves=False):
        if show_moves:
            print("=" * 50)
            print("X: " + printable_action(action))
        options = self.legal_moves()
        if action not in options:
            raise IllegalMoveError(
                "Bad move: {action} is not in {options}".format(
                    action=action, options=options))

        self.make_move(action, self.my_team)
        if show_moves:
            show_board(self.board)
        reward = 0
        full = False

        my_win = board_success(self.board, self.num_to_connect)
        if my_win:
            reward = 1
        else:
            full = self.full_board_tie()
            if not full:
                their_move = opponent_fn(self)
                self.make_move(their_move, self.their_team)
                if show_moves:
                    print("-" * 40)
                    print("O: " + printable_action(their_move))
                    show_board(self.board)
                their_win = board_success(self.board, self.num_to_connect)
                if their_win:
                    reward = -1
                else:
                    full = self.full_board_tie()

        observation = self.board
        done = reward != 0 or full
        if show_moves:
            print("Reward: {:d}{}".format(reward, " & Done!" if done else ""))
            print("=" * 50)
        return observation, reward, done, {}  # the last item is expected by the gym API

    def reset(self):
        self.board_init()


def discount_rewards(game_reward, n_turns, gamma=0.8):
    """ take 1D float array of rewards and compute discounted reward
    """
    discounted_r = np.zeros(n_turns)
    discounted_r[0] = 1.0
    for i in range(1, n_turns):
        discounted_r[i] = discounted_r[i - 1] * gamma

    # The total reward for each game should sum to the final reward. Otherwise,
    # the agent would prefer longer wins over shorter wins, or shorter losses over
    # longer losses, which would lead to weird behavior
    discounted_r = discounted_r[::-1]
    discounted_r = discounted_r * game_reward / sum(discounted_r)
    return list(discounted_r)

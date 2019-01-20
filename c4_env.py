from utils import BaseConnectEnv


class Connect4Env(BaseConnectEnv):
    rows = 6
    columns = 7
    my_team = "r"
    their_team = "b"
    num_to_connect = 4

    def __init__(self):
        super().__init__()
        self.move_choices = list(range(self.columns))

    def lowest_empty(self):
        """For each column, what's the ix of the next empty row? -1 for all full
        We'll assume that [0,0] is the bottom-left
        """
        results = []
        all_empties = self.empty_spaces()
        for col in range(self.columns):
            rows = [r for r, c in all_empties if c == col]
            if rows:
                results.append(min(rows))
            else:
                results.append(-1)

        return results

    def legal_moves(self):
        "A list of which columns aren't yet full"
        cols = list(set(c for _, c in self.empty_spaces()))
        return sorted(cols)

    def make_move(self, action, mark):
        c = action
        r = self.lowest_empty()[c]
        self.board[r][c] = mark

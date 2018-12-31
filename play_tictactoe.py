import sys
import random
from tictactoe_env import TictactoeEnv
from utils import random_opponent

env = TictactoeEnv()

N_games = 10 ** 4


wins = losses = 0

for _ in range(N_games):
    env.reset()
    # env.render(); print()

    for i in range(1000):
        # if i == 0:  #playing with how much the center piece matters
        #     my_move = (1, 1)
        # else:
        my_move = random.choice(env.legal_moves())  # take a random action

        _, reward, done, info = env.step(my_move, random_opponent)
        # env.render()
        # print(reward)
        # print()
        if done:
            if reward == 1:
                wins += 1
            if reward == -1:
                losses += 1
            break

print(
    "Wins: {w:.3f}, Losses: {l:.3f}, Ties: {t:.3f}".format(
        w=wins / N_games,
        l=losses / N_games,
        t=(N_games - wins - losses) / N_games
    )
)
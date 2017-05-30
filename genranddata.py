# Generate and store random goal states given a certain state.

import random
import numpy as np
from tetris import TetrisApp

# Are we training or generating training data?
TRAINING = False
NUM_EPOCHS = 100

sign = lambda x: x and (1, -1)[x < 0]

def generate_data(game):
    for i in range(NUM_EPOCHS):
        while not game.gameover:
            # Choose random goal state
            translate = random.randint(-5, 5)
            rotate = random.randint(-1, 2)
            # Carry out goal state
            for _ in range(abs(translate)):
                game.move(sign(translate))
            for _ in range(abs(rotate)):
                game.rotate_stone(sign(rotate))
            game.step()
        game.start_game()

def main():
    game = TetrisApp()
    generate_data(game)

if __name__ == '__main__':
    main()
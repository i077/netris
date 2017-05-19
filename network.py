import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

LR = 1e-3 # Learning rate
score_threshold = 1 # Number of lines needed to clear to learn
initial_games = 1000 # Number of games to train on

# Reads board and returns as 1D list
def readboard(board):
    new_board = []
    for i, row in enumerate(board[2:-1]):
        for j, cell in enumerate(row[1:-1]):
            if cell:
                if cell == 10 :
                    new_val = 1
                else:
                    new_val = 0.5
            else:
                new_val = 0
            new_board.append(new_val)
    return new_board

# Returns a one_hot list with 1 at index one_index.
def gen_onehot(one_index):
    if one_index > 4:
        one_index = 4
    onehot = [0, 0, 0, 0, 0]
    onehot[one_index] = 1
    return onehot

def create_training_data(tetris_app):
    observation = readboard(tetris_app.prep_current_board())

    if len(tetris_app.prev_observation) > 0:
        tetris_app.game_memory.append([tetris_app.prev_observation, tetris_app.action])

    tetris_app.prev_observation = observation
    
def on_gameover(tetris_app):
    if tetris_app.lines >= score_threshold:
        tetris_app.accepted_scores.append(tetris_app.lines)
        tetris_app.training_data.append(tetris_app.game_memory)

    tetris_app.scores.append(tetris_app.lines)
    print("Finished game", tetris_app.number_of_games, "of", initial_games, "Score:", tetris_app.lines)

    if tetris_app.number_of_games >= initial_games:
        np.save('data.npy', np.array(tetris_app.training_data))
        tetris_app.quit()
    else:
        tetris_app.start_game()



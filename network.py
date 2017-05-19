import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3 # Learning rate
score_threshold = 5 # Number of lines needed to clear to learn
initial_games = 10 # Number of games to train on

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
    onehot = [0, 0, 0, 0, 0]
    onehot[one_index] = 1
    return onehot

    

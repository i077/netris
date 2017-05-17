import tensorflow as tf

# Reads board and returns as 1D list
def readboard(board):
    new_board = []
    for row in board[:-1]:
        new_board.extend(row[1:-1]) # Takes off first and last columns 
    return new_board

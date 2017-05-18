import tensorflow as tf
# Reads board and returns as 1D list
def readboard(board):
    new_board = []
    for i, row in enumerate(board[2:-1]):
        for j, cell in enumerate(row[1:-1]):
            if cell:
                if(cell == 10):
                    new_val = 1
                else:
                    new_val = 0.5
            else:
                new_val = 0
            new_board.append(new_val)
    return new_board

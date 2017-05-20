import tetris
from network import readboard
from tetris import *
import tflearn
from numpy import argmax

game = TetrisApp()

def output_to_one_hot(action):
    big_index = argmax(action)
    one_hot = [0, 0, 0, 0, 0]
    one_hot[big_index] = 1
    return one_hot

if __name__ == "__main__":
    game.run()

    input_layer = tflearn.input_data(shape=[None, 200])
    layer1 = tflearn.fully_connected(input_layer, 64, activation='relu')
    dropout1 = tflearn.dropout(layer1, 0.8)
    
    layer2 = tflearn.fully_connected(dropout1, 128, activation='relu')
    dropout2 = tflearn.dropout(layer2, 0.8)

    layer3 = tflearn.fully_connected(dropout2, 64, activation='relu')
    dropout3 = tflearn.dropout(layer3, 0.8)

    output = fully_connected(dropout3, 5, activation='softmax')
    net = regression(output, optimizer='adam', learning_rate=0.8)

    model = tflearn.DNN(net)

    while True:
        observation = readboard(game.prep_current_board())
        action = model.predict(observation)
        action = output_to_one_hot(action)
        tx.one_hot_to_inputs(action)
        game.step()


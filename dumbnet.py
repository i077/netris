import tetris
from network import readboard
from tetris import *
import tflearn
from numpy import argmax, argmin, array, reshape

game = TetrisApp()

def output_to_one_hot(action):
    big_index = argmax(action)
    one_hot = [0, 0, 0, 0, 0]
    one_hot[big_index] = 1
    return one_hot

if __name__ == "__main__":
    game.run()

    net = tflearn.input_data(shape=[None, 200])
    net = tflearn.fully_connected(net, 200, activation='sigmoid')
    # net = tflearn.dropout(net, 0.7)
    
    net = tflearn.fully_connected(net, 100, activation='tanh')
    # net = tflearn.dropout(net, 0.7)

    net = tflearn.fully_connected(net, 10, activation='sigmoid')
    # net = tflearn.dropout(net, 0.7)

    net = fully_connected(net, 5, activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.01)

    model = tflearn.DNN(net)

    while True:
        observation = readboard(game.prep_current_board())
        observation = reshape(observation,(1, 200))
        # observation = tflearn.reshape(array(observation), [200])
        action = model.predict([observation][0])
        print(action)
        action = output_to_one_hot(action)
        print(action)
        game.one_hot_to_inputs(action)
        game.step()

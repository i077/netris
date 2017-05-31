import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#mnist = input_data.read_data_sets("mnist-data/", one_hot=True)


data = open('playerdata.dat', 'r')
lnum = 0
states = []
actions = []
for line in data:
    l = line.lstrip('[').rstrip(']\n')
    sp = l.split(', ')
    #print(sp)
    #break
    if lnum % 2 == 0:
        s = list(map(float, sp))
        states.append(s)
    elif lnum % 2 == 1:
        a = list(map(float, sp))
        actions.append(a)
    lnum += 1
# print(states[1], actions[1])
data.close()

n_nodes_hl1 = 512
n_nodes_hl2 = 1024
n_nodes_hl3 = 512

n_classes = 5
batch_size = 100

x = tf.placeholder('float', [None, 200])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([200, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 50

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            batch_index = 0
            # for _ in range(int(mnist.train.num_examples / batch_size)):
            #     epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            for _ in range(int(len(states) / batch_size)):
                epoch_x = states[batch_index:batch_index+batch_size]
                # print(epoch_x)
                epoch_y = actions[batch_index:batch_index+batch_size]
                batch_index+=batch_size
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print('Accuracy:', accuracy.eval({x: states, y: actions}))

train_neural_network(x)

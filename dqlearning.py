import os
import random
import tensorflow as tf
import numpy as np
import tetris
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TRAINING VARIABLES
TRAINING = True # Are we training or testing?
ACTIONS = 5 # Number of actions game has
GAMMA = 0.9 # Discount of future reward values
TRAIN_STEP = 10000 # Step to start training
LEARNING_RATE = 1e-6
EPSILON_ANNEAL = 4e5 # Frames to anneal epsilon towards final value
EPSILON_FINAL = 0.05
EPSILON_INITIAL = 1.0
REPLAY_MEMORY = 6e5 # Number of previous transitions to remember
BATCH = 1000 # Size of replay minibatch
CHECKPOINT_DIR = './learningdata/'
CHECKPOINT_INTERVAL = 10000

# Helper functions for building the network
def tf_weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def tf_bias_var(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

def tf_conv2D(input, filter, stride):
    return tf.nn.conv2d(input, filter, [1, stride, stride, 1], padding="SAME")

def tf_max_pool2x2(input):
    return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

def build_network():
    # Define weights and biases
    conv1_W = tf_weight_var([8, 8, 4, 32])
    conv1_b = tf_bias_var([32])

    conv2_W = tf_weight_var([4, 4, 32, 64])
    conv2_b = tf_bias_var([64])

    conv3_W = tf_weight_var([3, 3, 64, 64])
    conv3_b = tf_bias_var([64])
    
    fc1_W = tf_weight_var([128, 512])
    fc1_b = tf_bias_var([512])

    fc2_W = tf_weight_var([512, ACTIONS])
    fc2_b = tf_bias_var([ACTIONS])

    # Input layer
    st = tf.placeholder("float", [None, 20, 10, 4])

    # Hideden layers
    conv1 = tf.nn.relu(tf_conv2D(st, conv1_W, 4) + conv1_b)
    conv1_pool = tf_max_pool2x2(conv1)

    conv2 = tf.nn.relu(tf_conv2D(conv1_pool, conv2_W, 2) + conv2_b)

    conv3 = tf.nn.relu(tf_conv2D(conv2, conv3_W, 1) + conv3_b)
    conv3_flat = tf.reshape(conv3, [-1, 128])

    fc1 = tf.nn.relu(tf.matmul(conv3_flat, fc1_W) + fc1_b)

    # Output layer
    out = tf.matmul(fc1, fc2_W) + fc2_b

    return st, out

def train(st, out, sess):
    # Cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    out_action = tf.reduce_sum(tf.multiply(out, a), 1)
    cost = tf.reduce_mean(tf.square(y - out_action))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Create new game environment
    game = tetris.TetrisApp()

    # Store previous observations in replay memory
    D = deque()

    # TODO Tensorboard summaries

    # Get initial state
    s, r_0, terminal = game.step_act([0,0,0,0,1])
    s_t = np.stack((s, s, s, s), 2)

    # Load and save networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(sess, checkpoint_path)
        print("Loaded {} from disk.".format(checkpoint_path))
    else:
        print("Couldn't find model on disk.")

    epsilon = EPSILON_INITIAL
    t = 0
    while True:
        # Choose an action by epsilon-greedy policy
        out_t = out.eval(feed_dict = {st : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= TRAIN_STEP:
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(out_t)
        a_t[action_index] = 1

        # Anneal epsilon
        if epsilon > EPSILON_FINAL and t > TRAIN_STEP:
            epsilon -= (EPSILON_INITIAL - EPSILON_FINAL) / EPSILON_ANNEAL

        # Take action and observe reward and next state
        s_t1, r_t, terminal = game.step_act(a_t)
        s_t1 = np.reshape(s_t1, (20, 10, 1))
        s_t1 = np.append(s_t1, s_t[:,:,0:3], 2)

        # Store transition in replay memory
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        # Update network
        if t > TRAIN_STEP:
            # Get sample of replay memory
            minibatch = random.sample(D, BATCH)
            
            # Get batch variables
            s_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s1_batch = [d[3] for d in minibatch]

            y_batch = []
            out_batch = out.eval(feed_dict = {st : [s1_batch[3]]})
            for i in range(0, len(minibatch)):
                # Get reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(out_batch[-1]))
            
            # Apply gradients
            optimizer.run(feed_dict = {
                y: y_batch,
                a: a_batch,
                st: s_batch
            })

        # Update counter and state
        t += 1
        s_t = s_t1

        # Save progress
        if t % CHECKPOINT_INTERVAL == 0:
            saver.save(sess, CHECKPOINT_DIR+"tetris-training", global_step=t)
        
        # Print out info
        state = ""
        if t < TRAIN_STEP:
            state = "o"
        elif t <= TRAIN_STEP + EPSILON_ANNEAL:
            state = "e"
        else:
            state = "t"
        if t % 50 == 0:
            print("T: {},\t S: {},\t e: {:.4f},\t A: {},\t R: {:.4f},\t Q_MAX: {:.4e}"
                .format(t, state, epsilon, action_index, r_t, np.max(out_t)))
        
        # Start new game if terminal
        if terminal:
            game.start_game()

def main():
    if TRAINING:
        sess = tf.InteractiveSession()
        s, out = build_network()
        train(s, out, sess)

if __name__ == "__main__":
    main()
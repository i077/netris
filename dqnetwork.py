"""
Using Deep-Q learning to train a machine to play Tetris.
"""
import tetris
import tensorflow as tf
import tflearn
import numpy as np
import time
import random

# Initialize game environment
game = tetris.TetrisApp()
num_actions = 5

# TRAINING PARAMETERS
# -------------------
# Are we training or testing?
test = False
# Path to saved network model
model_path = 'dqn_model.tflearn.ckpt'
# Number of actor-learner threads
n_threads = 8
# Training steps
T_MAX = 8e8
# Current step
T = 0
# Size of minibatch when recalling from experience replay
replay_minibatch = 4
# Gradient update freq for each thread
grad_update_freq = 5
# Step to reset target network
target_reset_freq = 40000
# Learning rate
learning_rate = 1e-3
# Discount rate of future rewards
y = 0.9
# Number of steps to change ϵ to final val
anneal_epsilon_timesteps = 4e5
# Checkpoint data
checkpoint_interval = 2000
checkpoint_path = 'dqn_learning.tflearn.ckpt'

# DEEP Q NETWORK
# --------------
# Building the Deep Q Network using TFLearn. The network should return
# a list of q-values for each action.
def build_model():
    # Establish inputs
    inputs = tf.placeholder(tf.float32, [None, 20, 10, replay_minibatch])
    # Build network
    net = tf.fully_connected(inputs, 128, activation='relu')
    net = tf.fully_connected(inputs, 256, activation='relu')
    net = tf.fully_connected(inputs, 128, activation='relu')
    q_vals = tflearn.fully_connected(net, num_actions)
    # Return inputs and q-values
    return inputs, q_vals

# Q-LEARNING
# ----------
# Determine final ϵ-value from list of possible values and probabilities.
# Values taken from https://arxiv.org/pdf/1602.01783v1.pdf
def get_final_epsilon():
    final_vals = np.array([1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random_choice(final_vals, 1, p=list(probabilities))[0]

# Async actor-learner thread, 1-step Q-learning
def actor_learner_thread(thread_id, env, session, graph_ops, saver):
    # Step counter and threshold should be global
    global T_MAX, T
    s, q_vals, target_s, target_q_vals, reset_target_network_params, a, y, grad_update = graph_ops

    # Initialize Q network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = get_final_epsilon()
    start_epsilon = 1.0
    epsilon = start_epsilon

    print("Thread {0} - Final epsilon: {1}".format(str(thread_id), str(final_epsilon)))
    time.sleep(3 * thread_id)
    t = 0

    while T < T_MAX:
        # Get initial state
        s_t = env.readboard(env.prep_current_board())
        state_terminal = False

        # Set per-episode counters
        ep_r = 0
        ep_avg_max_q = 0
        ep_t = 0

        # Run the current episode until terminal state reached
        while True:
            # Get Q-values from network
            q_vals_t = q_vals.eval(session=session, feed_dict={s: [s_t]})
            # Choose action based on ϵ-greedy policy
            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(q_vals_t)
            a_t[action_index] = 1

            # Reduce ϵ
            if epsilon > final_epsilon:
                epsilon -= (start_epsilon - final_epsilon) / anneal_epsilon_timesteps
            
            # Take action and recieve new state and reward
            s_t1, r_t, ep, state_terminal = env.step_act(a_t)

            # Accumulate gradients
            target_q_vals_t = target_q_vals.eval(session=session, feed_dict={s: [s_t1]})
            if state_terminal:
                y_batch.append(r_t)
            else:
                y_batch.append(r_t + y * np.max(target_q_vals_t))

            s_batch.append(s_t)
            a_batch.append(a_t)

            # Update state and counters
            s_t = s_t1
            T += 1
            t += 1
            ep_t += 1
            ep_r += r_t
            ep_avg_max_q = np.max(readout_t)

            # Update target network if necessary
            if T % target_reset_freq == 0:
                session.run(reset_target_network_params)

            # Update current network if necessary
            if t % grad_update_freq == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        s: s_batch,
                                                        a: a_batch})
                # Clear gradients
                s_batch = []
                a_batch = []
                y_batch = []

            # Save model progress to disk
            if t % checkpoint_interval == 0:
                saver.save(session, checkpoint_path, global_step=t)

            # At end of episode, print stats
            if terminal:
                print("Thread {0} - T: {1}, R: {2}, Qmax: {3}, Eps = {4}, Eps progress: {5}"
                      .format(thread_id, t, ep_r, ep_avg_max_q / float(ep_t), epsilon,
                              t / float(anneal_epsilon_timesteps)))
                break

# Build networks and return operations
def build_graph():
    # Create a shared deep Q network
    s, q_net = build_model()
    network_params = tf.trainable_variables()
    q_vals = q_net

    # Create a target deep Q network
    target_s, target_q_net = build_model()
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_vals = target_q_net

    # Periodically update target network weights
    reset_target_network_params = \
            [target_network_params[i].assign(network_params[i])
             for i in range(len(target_network_params))]

    # Cost and gradient operations
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_vals = tf.reduce_sum(tf.multiply(q_vals, a), axis=1)
    cost = tflearn.mean_square(action_q_vals, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    update = optimizer.minimize(cost, var_list=network_params)

    return s, q_vals, target_s, target_q_vals, reset_target_network_params, a, y, grad_update

if __name__ == '__main__':
    game.run()

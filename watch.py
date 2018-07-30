import tensorflow as tf
import numpy as np
import gym
from collections import deque
from scipy import misc
import itertools as it

######################### MODEL LOCATION ########################
MODEL_LOCATION = "models/remake/model-5-495.ckpt"
#################################################################

# Initiate environment
env = gym.make('CarRacing-v0')
env = env.unwrapped

# List of discrete actions
possible_actions =  [[+.5, .1, 0], [-.5, .1, 0], [0,+1,0], [0,0,0.5]]

""" Hyperparameters """
action_size = len(possible_actions)     # Number of possible actions
max_episodes = 500                      # Run the game this much times
max_steps = 3000                        # Maximum number of steps per game
stack_size = 4                          # Image stack size
resized_image_res = (84, 96, 3)         # Size of the image after preprocessing
termination_reward = -25                # Terminate the episode if we reach this value ...
                                        # ... prevents us from losing time if the car gets stuck


""" Convenient functions are placed inside the helper class for better organization """
class Helper:
    def __init__(self):
        # Initialize deque with zero-images one array for each image. 'maxlen=' is very important here
        self.stacked_frames = deque([np.zeros(resized_image_res, dtype=np.int) for i in range(stack_size)], maxlen=4)

    def preprocess_frame(self, frame):
        """ Preprocessing
            Take a frame.
            Resize it.
                __________________
                |                 |
                |                 |
                |                 |
                |                 |
                |_________________|
                to
                _____________
                |            |
                |            |
                |            |
                |____________|
            Normalize it.
            return preprocessed_frame """
        # Greyscale frame
        #img = np.mean(frame,-1)

        # Remove black bar at the bottom
        img = frame[:-12, :, :]

        # Resize
        # img = misc.imresize(img, (resized_image_res))

        # Crop the screen (remove the roof because it contains no information) (not necessary here)
        cropped_frame = img

        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0

        return normalized_frame

    def stack_frames(self, state):
        """ Stacks frames to give the AI a notion of movement """
        # Preprocess frame
        frame = self.preprocess_frame(state)

        # Append frame to deque, automatically removes the oldest frame
        self.stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

""" Network """
tf.reset_default_graph() # Reset graph to avoid bugs
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, 84, 96, 12], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    # Input: NONE x 84x96x12
    conv1 = tf.layers.conv2d(inputs = input_,
                                 filters = 32,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv1")

    conv1_batchnorm = tf.layers.batch_normalization(conv1,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm1')

    conv1_out = tf.nn.elu(conv1_batchnorm, name="conv1_out")
    ## --> [41, 47, 32]

    conv2 = tf.layers.conv2d(inputs = conv1_out,
                         filters = 64,
                         kernel_size = [3,3],
                         strides = [2,2],
                         padding = "VALID",
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                         name = "conv2")

    conv2_batchnorm = tf.layers.batch_normalization(conv2,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm2')

    conv2_out = tf.nn.relu(conv2_batchnorm, name="conv2_out")
    ## --> [20, 23, 64]

    conv3 = tf.layers.conv2d(inputs = conv2_out,
                                     filters = 128,
                                     kernel_size = [5,5],
                                     strides = [3,3],
                                     padding = "VALID",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv3")

    conv3_batchnorm = tf.layers.batch_normalization(conv3,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm3')

    conv3_out = tf.nn.elu(conv3_batchnorm, name="conv3_out")
    ## --> [5, 6, 128]

    flatten = tf.layers.flatten(conv3_out)
    ## --> [3840]

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = flatten,
                                                num_outputs = 240,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                num_outputs = 400,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                num_outputs = action_size,
                                                activation_fn= None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)

""" Train """
# Initialize Helper and Saver
helper = Helper()
saver = tf.train.Saver()

# Avoid crashes by using GPUOptions
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    # Restore model
    saver.restore(sess, MODEL_LOCATION)

    for episode in range(max_episodes):
        # Start the game and stack initial frame
        state = helper.stack_frames(env.reset()) # stacked frames

        for i in range(max_steps+1):
            env.render()

            # Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 84, 96, 12])})

            # Select best action
            action_index = np.amax(action_probability_distribution.ravel())
            action = possible_actions[action_index]

            # Perform action & get next data
            new_state, reward, done, info = env.step(action)
            new_state = helper.stack_frames(new_state)

            # Once the we reach the maximum number of steps, done=true
            if i == max_steps: done = True
            if done:
                break

    env.close()

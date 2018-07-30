import tensorflow as tf
import numpy as np
import gym
from collections import deque# Ordered collection with ends
from scipy import misc
import itertools as it

# Initiate environment
env = gym.make('CarRacing-v0')
env = env.unwrapped
env.seed(100) # Seed so we can compare

# List of discrete actions
possible_actions =  [[+.5, .1, 0], [-.5, .1, 0], [0,+1,0], [0,0,0.5]]

""" Hyperparameters """
action_size = len(possible_actions)     # Number of possible actions
max_episodes = 500                      # Run the game this much times
max_steps = 1300                        # Maximum number of steps (prevents memory issues)
learning_rate = 0.0005                  # Learning rate / Alpha
gamma = 0.95                            # Discount rate
stack_size = 4                          # Image stack size
resized_image_res = (84, 96)         # Size of the image after preprocessing
termination_reward = -25                # Terminate the episode if we reach this value ...
                                        # ... prevents us from losing time if the car gets stuck

""" Convenient functions are placed inside the helper class for better organization """
class Helper:
    def __init__(self):
        # Initialize deque with zero-images one array for each image. 'maxlen=' is very important here
        self.stacked_frames = deque([np.zeros(resized_image_res, dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Get list of trainable variables in both networks
        self.trainables = tf.trainable_variables()

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
        img = np.mean(frame,-1)

        # Remove black bar at the bottom
        cropped_img = img[:-12, :]

        # Normalize Pixel Values
        normalized_frame = cropped_img/255.0

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

    def discount_and_normalize_rewards(self, episode_rewards):
        """ Returns list of discounted rewards. Rewards closer at the beginning are more
            important so they are very high. The last reward is equal to 1 (before normalizing)
            so the first reward has a huge value (before normalizing). Try printing it to see."""
        # Get empty array with the same size as the rewards array
        discounted_episode_rewards = np.zeros_like(episode_rewards)

        # Variable that stores value of the discounted reward being calculated by the loop
        current_reward = 0.0
        # Loop that does the magic
        for i in reversed(range(len(episode_rewards))):
            # Calculate the discounted reward
            current_reward = current_reward * gamma + episode_rewards[i]
            # Store it in the array
            discounted_episode_rewards[i] = current_reward

        # Normalize.
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

""" Network """
tf.reset_default_graph()
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, *resized_image_res, stack_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    # Input: NONE x 84x96x4
    conv1 = tf.layers.conv2d(inputs = input_,
                                 filters = 32,
                                 kernel_size = [8,8],
                                 strides = [4,4],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv1")

    conv1_batchnorm = tf.layers.batch_normalization(conv1,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm1')

    conv1_out = tf.nn.elu(conv1_batchnorm, name="conv1_out")
    ## --> [20, 22, 32]

    conv2 = tf.layers.conv2d(inputs = conv1_out,
                         filters = 64,
                         kernel_size = [4,4],
                         strides = [2,2],
                         padding = "VALID",
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                         name = "conv2")

    conv2_batchnorm = tf.layers.batch_normalization(conv2,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm2')

    conv2_out = tf.nn.relu(conv2_batchnorm, name="conv2_out")
    ## --> [9, 10, 64]

    conv3 = tf.layers.conv2d(inputs = conv2_out,
                                     filters = 128,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv3")

    conv3_batchnorm = tf.layers.batch_normalization(conv3,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm3')

    conv3_out = tf.nn.elu(conv3_batchnorm, name="conv3_out")

    flatten = tf.layers.flatten(conv3_out)
    ## --> [1536]

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = flatten,
                                                num_outputs = 512,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc1,
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
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)


    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

""" Tensorboard """
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/PolicyGradient-CarRacing-Remake/1")
tf.summary.scalar("Loss", loss) # Write loss
tf.summary.scalar("Mean Reward", mean_reward_) # Write mean reward

# Define the operation
write_op = tf.summary.merge_all()

""" Train """
all_rewards = []
total_rewards = 0
max_reward_recorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

# Initialize Helper and Saver
helper = Helper()
saver = tf.train.Saver()

# Avoid crashes by using GPUOptions
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(max_episodes):
        episode_rewards_sum = 0

        # Start the game and stack initial frame
        state = helper.stack_frames(env.reset()) # stacked frames

        for i in range(max_steps+1):
            # Render game. Remove this line if you don't want to see the game.
            env.render()

            # Choose action. Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, *state.shape])})

            if i%100==0:
                print(action_probability_distribution)

            # Select action based on the the actions probability distribution . Ravel() flattens the array (2D -> 1D).
            action_index = np.random.choice(range(action_size), p=action_probability_distribution.ravel())
            action = possible_actions[action_index]

            # Perform action & get next data
            new_state, reward, done, info = env.step(action)
            new_state = helper.stack_frames(new_state)

            # Store state and reward
            episode_states.append(state.reshape([1, *resized_image_res, stack_size]))
            episode_rewards.append(reward)

            # In order to pass the action_ mask placeholder we must first one hot enconde the action,
            #   since the 'action' is not onehot encoded yet. PS: [1,0]=left and [0,1]=right
            action_ = np.zeros(action_size)
            action_[action_index] = 1
            # Store one_hot encoded action
            episode_actions.append(action_)

            # Once the we reach the maximum number of steps or the reward gets very low, done=true
            if i == max_steps or (np.sum(episode_rewards)<termination_reward): done = True

            if done:
                # Calculate sum of rewards (reward shown on screen)
                episode_rewards_sum = np.sum(episode_rewards)

                # Append the reward of the episode to all_rewards so we can visualize the progress.
                all_rewards.append(episode_rewards_sum)

                # Mean reward from games so far
                mean_reward = np.mean(all_rewards)
                # Maximum reward from games so far
                max_reward_recorded = np.amax(all_rewards)

                print("==========================================")
                print("Number of episodes so far: ", episode)
                print("Mean Reward: ", episode_rewards_sum)
                print("Average Reward of all training: ", mean_reward)
                print("Max reward for a batch so far: ", max_reward_recorded)

                # Save Model if it performed well. OBS: we must save before training, otherwise we will save a different model.
                if all_rewards[-1] == max_reward_recorded:
                    save_path = saver.save(sess, f"./models/remake/model-{int(episode)}-{int(max_reward_recorded)}.ckpt")
                    print("Model Saved")

                # Calculate discounted reward
                discounted_episode_rewards = helper.discount_and_normalize_rewards(episode_rewards)

                # Loss: the softmax_cross_entropy between the results from the last dense layer vs the onehot-encoded action
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards
                                                                })



                # Write TensorBoard summary
                summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards,
                                                                    mean_reward_: mean_reward
                                                                })


                writer.add_summary(summary, episode)
                writer.flush()



                # Reset episode data
                episode_states, episode_actions, episode_rewards = [],[],[]

                break

            state = new_state
    env.close()

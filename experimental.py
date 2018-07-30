import tensorflow as tf
import numpy as np
import gym
from collections import deque # List of limited length.
from scipy import misc
import matplotlib.pyplot as plt
import time

# Create env and close the window that shows.
env = gym.make('CarRacing-v0')
env = env.unwrapped
env.frameskip = 1

# List of possible actions (alternatively, you could use a network with cintinous output)
possible_actions =  [[+1, .1, 0], [-1, .1, 0], [0,+1,0], [0,0,0.5]]

action_size = len(possible_actions)     # Number of possible actions
gamma = 0.99                            # Discount rate
stack_size = 4                          # Image stack size
resized_image_res = (84,96)             # Size of the image after preprocessing
n_epochs = 3000                         # Number of epochs to train
batch_size = 1                          # Number of games to feed into the network at once (careful here)
alpha_start = 0.0300                    # Learning rate at the start
alpha_stop = 0.0001                     # Minimum learning rate
learning_rate = alpha_start             # Initial learning rate
decay_rate = 0.01                       # Exponential decay rate for learning rate
min_reward = -75                        # End game is the reward is less than this
max_consecutive_failure = 450           # If the car doesn't find a positive reward in 450 steps (-0.1 for each step), done=true.


""" Convenient functions are placed inside the helper class for better organization """
class Helper:
    def __init__(self):
        # Initialize deque with zero-images one array for each image. 'maxlen=' is very important here
        self.stacked_frames = deque([np.zeros(resized_image_res, dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Get list of trainable variables in both networks
        self.trainables = tf.trainable_variables()

    def preprocess_frame(self, frame):
        """ Resize, grayscale and normalize frame. """

        # Greyscale frame
        img = np.mean(frame,-1)

        # Normalize Pixel Values
        img = img/255.0

        # Remove black bar at the bottom
        img = img[:-12]

        # Resize
        #img = misc.imresize(img, (resized_image_res))

        return img

    def stack_frames(self, state, reset=False):
        """ Stacks frames so the AI can have a notion of movement """
        if reset:
            # Create deque to stack the frames. It has a fixed size of 4
            self.stacked_frames = deque([np.zeros(resized_image_res, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

        # Preprocess frame
        frame = self.preprocess_frame(state)

        # Append frame to deque, automatically removes the oldest frame
        self.stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

def discount_and_normalize_rewards(episode_rewards):
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

#### NOTE: Using separate batches is not necessary. It's just an attempt to improve the algorithm.
def make_batch(batch_size, sess):
    """ Runs N (batch_size) games and returns the states, actions, rewards and
        the list of discounted rewards for that game."""
    # Initialize empty lists
    batch_states, batch_actions, batch_rewards, batch_discountedRewards = [], [], [], []

    for episode in range(batch_size):
        # Initialize lists: states, actions, rewards, discountedRewards
        ep_states, ep_actions, ep_rewards, ep_discountedRewards = [], [], [], []

        # Launch the game
        state = env.reset()
        state = helper.stack_frames(state, reset=True) # stacked frames

        # Update learning rate
        learning_rate = alpha_stop + (alpha_start - alpha_stop) * np.exp(-decay_rate * episode)

        step = 0
        while True:
            step += 1
            if episode == 0: env.render()
            # Choose action a. Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape(1, *state.shape)})

            # Adds another level of randomness. 10% chance to perform a completly random action.
            # This is useful because in this game the network oftem gets stuck on a 99.99% possibility of doing a certain movement and stops progressing.
            if step%100==0: print(action_probability_distribution)
            # if np.amax(action_probability_distribution) >= 0.98:
            #     action_index = np.random.choice(range(action_size))
            # else:
            #     # Select action based on the the actions probability distribution . Ravel() flattens the array (2D -> 1D).
            #
            action_index = np.random.choice(range(action_size), p=action_probability_distribution.ravel())

            action = possible_actions[action_index]

            # Perform action & get next data
            new_state, reward, done, info = env.step(action)
            new_state = helper.stack_frames(new_state)

            # Store s, a, r
            ep_states.append(state)
            ep_rewards.append(reward)

            # In order to pass the action_ mask placeholder we must first one hot enconde the action,
            #   since the 'action' is not onehot encoded yet. PS: [1,0]=left and [0,1]=right
            action_ = np.zeros(action_size)
            action_[action_index] = 1
            ep_actions.append(action_)

            # If the reward is less then
            if (np.sum(ep_rewards)<-75): done = True
            #If the car doesn't grab a positive reward in 'max_consecutive_failure' steps, done=true.
            if step>max_consecutive_failure and (np.sum(ep_rewards[step-max_consecutive_failure:step]) < (max_consecutive_failure/10.5)): done = True

            # Once the whole game is done we can calculate the discounted rewards and append the lists
            if done:
                # Calculate discounted reward
                ep_discountedRewards = discount_and_normalize_rewards(ep_rewards)

                # Append episode data to batch
                batch_states.append(ep_states)
                batch_actions.append(ep_actions)
                batch_rewards.append(ep_rewards)
                batch_discountedRewards.append(ep_discountedRewards)

                print("Done with batch")

                break

            env.viewer.window.dispatch_events() # Fix bug with CarRacing-v0
            state = new_state

    return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_discountedRewards)

""" Network """
tf.reset_default_graph()
with tf.name_scope("inputs"):
    ######### Placeholders #########
    input_ = tf.placeholder(tf.float32, [None, *resized_image_res, stack_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_epoch_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    ######### Layers #########
    # Convolutional > ELU > BatchNormalization >
    # Convolutional > ELU > BatchNormalization >
    # Convolutional > ELU > BatchNormalization >
    # Flatten > Dense > Dense
    # Input is 84x96.
    conv1 = tf.layers.conv2d(inputs = input_,
                                 filters = 32,
                                 kernel_size = [8,8],
                                 strides = [4,4],
                                 padding = "VALID",
                                 name = "conv1")

    conv1_batchnorm = tf.layers.batch_normalization(conv1,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm1')

    conv1_out = tf.nn.elu(conv1_batchnorm, name="conv1_out")

    conv2 = tf.layers.conv2d(inputs = conv1,
                                 filters = 64,
                                 kernel_size = [5,5],
                                 strides = [3,3],
                                 padding = "VALID",
                                 name = "conv2")

    conv2_batchnorm = tf.layers.batch_normalization(conv2,
                                           training = True,
                                           epsilon = 1e-5,
                                           name = 'batch_norm2')

    conv2_out = tf.nn.elu(conv2_batchnorm, name="conv2_out")

    conv3 = tf.layers.conv2d(inputs = conv2_out,
                                     filters = 128,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "SAME",
                                     name = "conv3")

    conv3_batchnorm = tf.layers.batch_normalization(conv3,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                    name = 'batch_norm3')

    conv3_out = tf.nn.elu(conv3_batchnorm, name="conv3_out")

    flatten = tf.layers.flatten(conv3_out, name="flatten")

    fc1 = tf.layers.dense(inputs = flatten,
                          units = 512,
                          activation=tf.nn.elu,
                          name="fc1")

    fc2 = tf.layers.dense(inputs = fc1,
                          units = action_size,
                          activation=tf.nn.sigmoid,
                          name="fc2")

    # Probability of doing each action
    action_distribution = tf.nn.softmax(fc2)

    ######### Loss and training #########
    # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
    # If you have single-class labels, where an object can only belong to one class, you might now consider using
    # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc2, labels = actions)
    loss = tf.reduce_mean(neg_log_prob * discounted_epoch_rewards_)

    train_opt = tf.train.AdamOptimizer(learning_rate)

    # Separating the gradients from the optimizer is not necessary. You can just do train_opt.minimize(loss)
    gradient = train_opt.compute_gradients(loss)
    apply_op = train_opt.apply_gradients(gradient)


""" Tensorboard """
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("./tensorboard/")
## Losses
tf.summary.scalar("Loss", loss)
## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)
# Operation
write_op = tf.summary.merge_all()

""" Train """
# Statistics
rewards_from_batches = []
mean_batch_rewards = []
max_reward_recorded = 0

# Visualization (my tensorboard isn't working)
losses = []
durations = []
rewards = []

# Initialize Helper and Saver
helper = Helper()
saver = tf.train.Saver()

# Limit GPU memory to 75%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Load the model
    #saver.restore(sess, "E:/Code/Anaconda/Learning/RL/MCPG/CarRacing-v0/models/model1-90.ckpt")

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        timeit = time.time()

        # Run games
        batch_states, batch_actions, batch_rewards, batch_discounted_rewards = make_batch(batch_size, sess)

        # Calculate mean rewards from batch
        mean_reward_of_each_game = [np.sum(i) for i in batch_rewards]
        rewards_from_batches.extend(mean_reward_of_each_game)

        # Calculate the mean reward of all games in the batch
        mean_reward_of_that_batch = np.mean(mean_reward_of_each_game)
        mean_batch_rewards.append(mean_reward_of_that_batch)

        # Mean reward from a game so far
        mean_reward = np.mean(mean_batch_rewards)

        # Calculate maximum reward recorded
        max_reward_recorded = np.amax(rewards_from_batches)

        # Time to run batch
        time_to_run_batch = time.time()-timeit

        print("==========================================")
        print("Epoch: ", epoch, "/", n_epochs)
        print("Time to run batch (in seconds): {}".format(time_to_run_batch))
        print("-----------")
        print("Number of episodes so far: {}".format(batch_size*(epoch+1)))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(mean_reward))
        print("Max reward for a batch so far: {}".format(max_reward_recorded))

        # Instead of N separate arrays (one for each game) we now want only one sequential array for all games.
        epoch_states = np.array([item for sublist in batch_states for item in sublist])
        epoch_actions = np.array([item for sublist in batch_actions for item in sublist])
        epoch_discounted_rewards = np.array([item for sublist in batch_discounted_rewards for item in sublist])

        # Loss: the softmax_cross_entropy between the results from the last dense layer vs the onehot-encoded action
        loss_, _ = sess.run([loss, apply_op], feed_dict={input_: epoch_states,
                                                          actions: epoch_actions,
                                                          discounted_epoch_rewards_: epoch_discounted_rewards})
        print("Loss: ", loss_)

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={input_: epoch_states,
                                                actions: epoch_actions,
                                                discounted_epoch_rewards_: epoch_discounted_rewards,
                                                mean_reward_: mean_reward})

        writer.add_summary(summary, epoch)
        writer.flush()

        # Append data to lists
        losses.append(loss_)
        durations.append(time_to_run_batch)
        rewards.append(mean_reward_of_that_batch)

        # Save model every 5 epochs
        if rewards_from_batches[-1] == max_reward_recorded:
            save_path = saver.save(sess, f"./models/best_model-{epoch}-{max_reward_recorded}.ckpt")
            print("Model Saved")

        if epoch % 20 == 0 and epoch > 199:
            # Plot data (not necessary)
            plt.figure(1)
            plt.subplot(221)
            plt.plot(losses)
            plt.subplot(222)
            plt.plot(durations)
            plt.subplot(223)
            plt.plot(rewards)
            plt.show()

            # Save model
            save_path = saver.save(sess, f"./models/model-{epoch}.ckpt")
            print("Model Saved")

    env.close()

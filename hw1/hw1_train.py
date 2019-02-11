
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import train_test_split


tf.enable_eager_execution()

class StandardPolicyModel(tf.keras.Model):
    def __init__(self, hidden, actions, observations):
        super(StandardPolicyModel, self).__init__()
        self.observations = observations
        self.layer1 = tf.keras.layers.Dense(hidden, input_shape=[None, observations],
        use_bias=False, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.layer2 = tf.keras.layers.Dense(hidden, input_shape=[None, 50],
        use_bias=False, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.outputLayer = tf.keras.layers.Dense(actions, input_shape=[None, 50],
        use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    def call(self, indata):
        in_data = tf.reshape(indata, shape=[-1, self.observations])
        mid = self.layer1(in_data)
        mid2 = self.layer2(mid)
        outdata = self.outputLayer(mid2)
        outdataclipped = tf.clip_by_value(outdata, -1, 1)
        return outdataclipped

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print('creating model')
    hidden=50
    logdir = "tensorlogs"
    global_step=tf.train.get_or_create_global_step()  # return global step var

    print('loading training data')
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        data = pickle.loads(f.read())

    print('loaded')
    actions = data["actions"]
    observations = data["observations"].astype(np.float32)
    import gym
    env = gym.make(args.envname)
    num_epochs = 1
    num_observations=env.observation_space.shape[0]
    num_actions =  env.action_space.shape[0]
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    #Train model
    x_train, x_test, y_train, y_test = train_test_split(observations, actions, test_size=0.2)
    y_train = np.reshape(y_train, [-1, num_actions])
    y_test = np.reshape(y_test, [-1, num_actions])
    model = StandardPolicyModel(50, num_actions, num_observations)
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
        best_validation = 20000
        for epoch in range(num_epochs):        
            print("epoch: ", epoch)
            epoch_loss = 0
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.batch(32)
            for x in tfe.Iterator(train_data):
                global_step.assign_add(1)
                x_batch = x[0]
                y_batch = x[1]
                with tf.GradientTape() as tape:
                    output = model(x_batch)
                    loss = tf.reduce_mean(tf.reduce_sum((output - y_batch)**2, axis=1))
                    epoch_loss += loss
                    tf.contrib.summary.scalar('loss', loss)
                    if global_step % 10 == 0:
                        writer.add_summary(loss_summary, global_step)
                    grads = tape.gradient(loss, model.variables)
                model.optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
            tf.contrib.summary.scalar('epoch_loss', epoch_loss, step=epoch)
            # Validate
            test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_data = test_data.batch(32)
            validation_loss = 0
            for x in tfe.Iterator(test_data):
                x_batch = x[0]
                y_batch = x[1]
                output = model(x_batch)
                loss = tf.reduce_mean(tf.reduce_sum((output - y_batch)**2, axis=1))
                validation_loss += loss
            tf.contrib.summary.scalar('validation_loss', validation_loss, step=epoch)
            if validation_loss < best_validation:
                print("new best, saving model at epoch", epoch)
                saver = tfe.Saver(model.variables)
                saver.save("imitation_data/" + args.envname + ".pkl", epoch)
                best_validation = validation_loss
              
if __name__ == '__main__':
    main()

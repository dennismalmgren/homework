
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
    global_step=tf.train.get_or_create_global_step()  # return global step var

    import gym
    env = gym.make(args.envname)
    num_observations=env.observation_space.shape[0]
    num_actions =  env.action_space.shape[0]

    #Create model
    checkpoint = tf.train.latest_checkpoint("imitation_data")
    model = StandardPolicyModel(50, num_actions, num_observations)
    init_sample = np.zeros((1, num_observations), dtype=np.float32)
    model(init_sample)
    saver = tfe.Saver(model.variables)
    saver.restore(checkpoint)

    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(20):
        print('iter', i)
        obs = env.reset().astype(np.float32)
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            obs = obs.astype(np.float32)
            totalr += r
            steps += 1
            #if args.render:
                
             #   env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()

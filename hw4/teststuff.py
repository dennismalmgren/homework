import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


arr1 = np.array([10, 1, 1, 1, 1])
arr2 = np.array([-10, -1, -1, -1, -1])
        #    [self._num_random_action_selection, self._action_dim])
num_random_action_selection = 12
action_dim = 5
horizon = 15
actions = tf.random_uniform([num_random_action_selection, horizon, action_dim])
print(arr1 - arr2)

sampled_actions = actions * (arr1 - arr2) + arr2

print(actions)

print(sampled_actions)

#         tf.random_uniform(
#     shape,
#     minval=0,
#     maxval=None,
#     dtype=tf.float32,
#     seed=None,
#     name=None
# )

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

tf.enable_eager_execution()

val1 = [float('nan'), float('nan')]
valt = tf.constant(val1)
valt = tf.reshape(valt, shape=[2, 1])
sample = tf.multinomial(valt, 1)
print(sample)

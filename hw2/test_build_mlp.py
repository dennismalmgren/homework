from train_pg_f18 import build_mlp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np

def main():
    print('hello world')
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = build_mlp(x, 1, "", 1, 1024)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        rand_array = np.random.rand(1024, 1024)
        y_res = sess.run(y, feed_dict={x: rand_array})
        print(y_res)

if __name__ == "__main__":
    main()

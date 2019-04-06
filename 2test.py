'''

one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)

注意:   indices代表的事下标 ,从0开始,所以如果是1 的话,one_hot后为[0,1,0,0,0]
'''

import tensorflow as tf
import numpy as np

# a = np.random.randint(1,10,size=[3,3])
a = np.arange(9).reshape(3, 3)
print(a)
t = tf.one_hot(indices=a, depth=9, axis=2)
with tf.Session() as sess:
    sess.run(t)
    print(t.eval())

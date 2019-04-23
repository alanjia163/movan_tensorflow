import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#batch
batch_size =100
n_batch = mnist.train.num_examples // batch_size
keep_prob = tf.placeholder(tf.float32)
#placeholders
x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])

#w
def weigth_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)
#b
def bias_variable(shape):
    initial = tf.constant(0.1,dtype=tf.float32)
    return tf.Variable(initial)

#conv
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [ 1,1,1,1],padding = 'SAME')
#pooling
def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='same')

#input_reshape
x_image = tf.reshape(x,[-1,28,28,1])

#add_layers

#conv1,pool1
#W1_and_b1
W_conv1 = weigth_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1))
h_pool1 = max_pooling(h_conv1)

#W2_and_b2
#conv2,pool2
W_conv2 =weigth_variable([5,5,32,43])
b_conv2 = max_pooling([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2))
h_pool2 = max_pooling(h_conv2)

#fully_connect_layer
W_fc1 = weigth_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout_layer
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#fully_connect
W_fc2 = weigth_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#loss and train_op
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(le-4).minimize(cross_entropy)

#correct_predition
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#train_step
with tf.Session() as sess:
    #init
    sess.run(tf.global_variables_initializer())
    #epoch
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print('iter'+str(epoch)+',test_accuracy=',str(acc))
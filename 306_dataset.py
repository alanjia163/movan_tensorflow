# import
import tensorflow as tf
import numpy as np



with tf.variable_scope('import_data'):
    # load data
    npx = np.random.uniform(-1, 1, (1000, 1))
    npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)
with tf.variable_scope('spilt_data'):
    npx_train, npx_test = np.split(npy, [800])
    npy_train, npy_test = np.split(npy, [800])

with tf.variable_scope('placeholiders'):
    # placeholders
    tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
    tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

with tf.variable_scope('dataloader'):
    # dataloader
    dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(3)
with tf.variable_scope('iterator'):
    iterator = dataset.make_initializable_iterator()

bx, by = iterator.get_next()

with tf.variable_scope('layers'):
    # add layers
    l1 = tf.layers.dense(bx, 10, tf.nn.relu)
    out = tf.layers.dense(l1, npy.shape[1])

with tf.variable_scope('train_op'):
    # loss and train_op
    loss = tf.losses.mean_squared_error(by, out)
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# with tf.variable_scope('Collection'):
#     tf.summary.scalar('loss',loss)
#     #tf.summary.scalar('out',out)
#     merged =tf.summary.merge_all()
saver =tf.train.Saver()

with tf.variable_scope('training'):
    # sess and init
    # train_op
    with tf.Session() as sess:
        saver.restore(sess,'./data/ckpt/')
        sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})
        for step in range(201):
            try:
                _, trainl = sess.run([train, loss])  # train
                # summary  = sess.run(merged)
                # filewriter = tf.summary.FileWriter('./data/summary/', graph=sess.graph)
                # filewriter.add_summary(summary, step)

                if step % 10 == 0:
                    testl = sess.run(loss, {bx: npx_test, by: npy_test})  # test
                    print('step: %i/200' % step, '|train loss:', trainl, '|test loss:', testl)
            except tf.errors.OutOfRangeError:  # if training takes more than 3 epochs, training will be stopped
                print('Finish the last epoch.')
                break

        # saver.save(sess,'./data/ckpt/')
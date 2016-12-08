import tensorflow as tf

LEARNING_RATE = 1e-4


# Helper functions to create tensors for the hidden layers
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Our convolutions use a stride of one and are zero padded so that the output is the same size as the input.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Our pooling is plain old max pooling over 2x2 blocks.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# We now define the graph representing the different layers of our CNN.
# It takes a batch of digits as input and outputs a batch of predictions. The second argument is the rate for dropout.
def inference(X, keep_prob):
    with tf.name_scope('reshape_input'):
        X_image = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        weights = weight_variable([5, 5, 1, 32])
        biases = bias_variable([32])

        conv1 = tf.nn.relu(conv2d(X_image, weights) + biases)
        pool1 = max_pool_2x2(conv1)

    with tf.name_scope('conv2'):
        weights = weight_variable([5, 5, 32, 64])
        biases = bias_variable([64])

        conv2 = tf.nn.relu(conv2d(pool1, weights) + biases)
        pool2 = max_pool_2x2(conv2)

    with tf.name_scope('fully_connected'):
        weights = weight_variable([7*7*64, 1024])
        biases = bias_variable([1024])

        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
        fully_connected = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    with tf.name_scope('dropout'):
        dropout = tf.nn.dropout(fully_connected, keep_prob)

    with tf.name_scope('softmax_linear'):
        weights = weight_variable([1024, 10])
        biases = bias_variable([10])

        y_pred = tf.nn.softmax(tf.matmul(dropout, weights) + biases)

    return y_pred


# Next we define the part of the graph which computes our loss function (cross-entropy).
def loss(y_pred, y_true):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]), name='xentropy')

    return cross_entropy


# We now need to create the training part of the graph.
def training(loss):
    tf.scalar_summary(loss.op.name, loss)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


# Compute the accuracy given the graph computing the predictions and the placeholder representing true labels.
def evaluation(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

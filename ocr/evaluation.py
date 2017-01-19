import os

import tensorflow as tf

from ocr.cnn import inference, accuracy_evaluation
from ocr.helpers import get_absolute_path

MODELS_DIR = get_absolute_path('../data/models')


# Operation to compute the precision with a given dataset
def do_eval(sess, eval_accuracy, X_placeholder, y_placeholder, keep_prob, dataset):
    feed_dict = {X_placeholder: dataset.images, y_placeholder: dataset.labels, keep_prob: 1.0}
    accuracy = sess.run(eval_accuracy, feed_dict=feed_dict)

    print('Accuracy: {:0.04f}'.format(accuracy))


def test_accuracy(model_file, dataset):
    with tf.Graph().as_default():  # Create a new graph and temporarily make it the default one
        X = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        y_pred = inference(X, keep_prob)

        eval_accuracy = accuracy_evaluation(y_pred, y_true)

        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess, os.path.join(MODELS_DIR, model_file))

        print('Initial test data eval:')
        do_eval(sess, eval_accuracy, X, y_true, keep_prob, dataset)

        sess.close()

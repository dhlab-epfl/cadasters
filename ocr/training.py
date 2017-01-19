import tensorflow as tf
from ocr.cnn import inference, loss_cross_entropy, training, accuracy_evaluation
from ocr.helpers import get_absolute_path
import os

BATCH_SIZE = 50
MAX_STEPS = 20000
MODELS_DIR = get_absolute_path('./data/models/')


def train(dataset_train, dataset_test, models_dir=MODELS_DIR, dataset_validation=None):

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784], name='image_flat')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        y_infer = inference(x, keep_prob)
        with tf.name_scope('entropy'):
            cross_entropy = loss_cross_entropy(y_infer, y_)
            tf.summary.scalar('loss-xentropy', cross_entropy)

        train_op = training(cross_entropy)

        with tf.name_scope('accuracy'):
            accuracy = accuracy_evaluation(y_infer, y_)
            tf.summary.scalar('accuracy', accuracy)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(models_dir, sess.graph)

        for step in range(MAX_STEPS):
            batch = dataset_train.next_batch(BATCH_SIZE)
            feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            sess.run(train_op, feed_dict=feed_dict)

            if step % 100 == 0:
                feed_dict[keep_prob] = 1.0
                loss_value, accuracy_value = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
                print('Training step {}: loss {:.2f}, accuracy {:.2f}'.format(step, loss_value, accuracy_value))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if step % 1000 == 0:
                saver.save(sess, os.path.join(models_dir, 'model'), global_step=step)

        saver.save(sess, os.path.join(models_dir, 'model'))
        accuracy_value = sess.run(accuracy,
                                  feed_dict={x: dataset_test.images, y_: dataset_test.labels, keep_prob: 1.0})
        print('Accuracy: {:0.04f}'.format(accuracy_value))
        sess.close()
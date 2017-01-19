import tensorflow as tf
from .cnn import inference, loss_cross_entropy, training, accuracy_evaluation
from .evaluation import do_eval
from ocr.helpers import get_absolute_path
import os

MODELS_DIR = get_absolute_path('../data/models/')
RESTORE_FILE = get_absolute_path('../data/models/mnist-net2/mnist-net2')
BATCH_SIZE = 50
MAX_STEPS = 10000


def finetune(dataset_train, dataset_test, restored_model_file=RESTORE_FILE, model_dir=MODELS_DIR):

    with tf.Graph().as_default():  # Create a new graph and temporarily make it the default one
        X = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        y_infer = inference(X, keep_prob)
        with tf.name_scope('entropy'):
            cross_entropy = loss_cross_entropy(y_infer, y_true)
            tf.summary.scalar('loss-xentropy', cross_entropy)

        train_op = training(cross_entropy)

        with tf.name_scope('accuracy'):
            accuracy = accuracy_evaluation(y_infer, y_true)
            tf.summary.scalar('accuracy', accuracy)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess, restored_model_file)

        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        print('Initial test data eval:')
        do_eval(sess, accuracy, X, y_true, keep_prob, dataset_test)

        # No early stopping, we need to stop manually (ctrl-C) when the validation error gets bigger!
        for step in range(MAX_STEPS):

            batch = dataset_train.next_batch(BATCH_SIZE)
            feed_dict = {X: batch[0], y_true: batch[1], keep_prob: 0.5}

            sess.run(train_op, feed_dict=feed_dict)

            if step % 100 == 0:
                feed_dict[keep_prob] = 1.0
                loss_value, accuracy_value = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
                print('Step {}: loss {:.2f}, accuracy {:.2f}'.format(step, loss_value, accuracy_value))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                saver.save(sess, os.path.join(model_dir, 'finetuned'), global_step=step)
                print('Test data eval:')
                do_eval(sess, accuracy, X, y_true, keep_prob, dataset_test)

        sess.close()

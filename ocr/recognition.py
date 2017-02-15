import tensorflow as tf
import cv2
import numpy as np
from .cnn import inference, inference_fc1
from ocr_image_processing import binarize_with_preprocess, preprocess_digit, segment_number
from .helpers import get_absolute_path

TF_MODEL = get_absolute_path('../data/models/net2/mixed-model-net2/mixed-net2')


# Currently we reconstruct the graph and reload the model each time we try te recognize a number, this is not efficient!
# For production we should construct the graph and load the model once, for instance when the Flask webserver starts
def recognize_digits(digits, tf_model):
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 784])
        keep_prob = tf.placeholder(tf.float32)

        if tf_model.find('net2') > 0:
            y_pred = inference(X, keep_prob)
        else:
            y_pred = inference_fc1(X, keep_prob)

        saver = tf.train.Saver()
        sess = tf.Session()
        # saver.restore(sess, os.path.join(MODELS_DIR, 'finetuned-final'))
        saver.restore(sess, tf_model)

        result = sess.run(y_pred, feed_dict={X: digits, keep_prob: 1.0})

        return result


def recognize_number(image, number_of_digits=None, tf_model=None):

    if tf_model is None:
        tf_model = TF_MODEL

    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_bin = binarize_with_preprocess(img, 23)

    digits = [preprocess_digit(digit) for digit in
              segment_number(img_bin, number_of_digits=number_of_digits) if digit.any()]
    if digits:
        flatten_digits = np.array(digits).reshape(len(digits), -1)

        results = recognize_digits(flatten_digits, tf_model=tf_model)
        prediction = ''.join(np.argmax(results, axis=1).astype(str))
        proba = np.prod(np.amax(results, axis=1)).astype(float)

        return prediction, proba
    else:
        return None, 0.0

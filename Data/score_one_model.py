from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


def score_one_model(model_file, data):
    tf.reset_default_graph()
    loader = tf.train.import_meta_graph(model_file + ".meta")
    with tf.Session() as sess:
        loader.restore(sess, model_file)
        try:
            accuracy_sum = add_acc(sess.graph.get_tensor_by_name('last/activations:0'))
        except KeyError as e:
            accuracy_sum = add_acc(sess.graph.get_tensor_by_name('activations:0'))
        acc = compute_acc(sess, data, accuracy_sum)
        return acc

def compute_acc(sess, data, accuracy_sum, batch_size = 10):
    tot_acc = 0.
    nb = 0
    epoch_number = data.epochs_completed
    while True:
        batch_xs, batch_ys = data.next_batch(batch_size)
        if data.epochs_completed != epoch_number: break
        try:
            tot_acc += sess.run(accuracy_sum, feed_dict={"input/x:0": batch_xs, "validation_accuracy/inputs_y:0": batch_ys})
        except KeyError as e:
            tot_acc += sess.run(accuracy_sum, feed_dict={"x:0": batch_xs, "validation_accuracy/inputs_y:0": batch_ys})
        nb += batch_size
    return tot_acc / nb

def add_acc(activations):
    with tf.name_scope("validation_accuracy"):
        inputs_y = tf.placeholder(tf.float32, [None, 10], name='inputs_y')
        correct_prediction = tf.cast(tf.equal(tf.argmax(activations, 1), tf.argmax(inputs_y, 1)), tf.float32)
        return tf.reduce_sum(correct_prediction, name="sum")

def main():
    data_dir = 'Ensai/data'
    model_dir = "Ensai/logs"
    data = input_data.read_data_sets(data_dir, validation_size=100, one_hot=True)
    accs = score_one_model(model_dir + "/model.ckpt", data.validation)
    print('Validation accuracy of hidden1:       %.3f' % accs)
    accs = score_one_model(model_dir + "/model.ckpt", data.validation)
    print("Validation accuracy of hidden1 (bis): %.3f" % accs)

if __name__ == "__main__":
    main()

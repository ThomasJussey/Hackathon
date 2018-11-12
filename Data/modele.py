# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CNN classifier
2 convolutionnal layers
1 fully connected hidden layer
1 fully connected last layer

The (training) data are perturbed with random noise

The 2 fully connected layers are separated with a drop-out layer during training phase.
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
#console : tensorboard --logdir "C:\Users\PLAFFITTE\Documents\Cours\python\NN"

# --- parameters ---
data_dir = 'NN/data'

stddev_train = 0.2
keep_prob_train = 0.9

train_size = 400
SGD__step_size = 0.001
nb_iter = 1000

log_dir = 'NN/models/model_tp'
model_file = log_dir + "/model.ckpt"

# --- model backbone ---
# Inputs
tf.reset_default_graph()
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y')

activations = tf.reshape(x, [-1, 28, 28, 1])

# add-noise Layer
with tf.name_scope("add-noise"):
    stddev = tf.constant(0., tf.float32, name='stddev')
    noise = tf.random_normal(shape=tf.shape(activations), mean=0.0, stddev=stddev, dtype=tf.float32, name="noise")
    tf.summary.histogram("noise", noise)
    activations = tf.add(activations, noise, name="after")

# Convolutional Layer #1
with tf.name_scope("conv1"):
    conv = tf.layers.conv2d(
        inputs=activations,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    activations = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

# Flatten tensor into a batch of vectors
activations = tf.reshape(activations, [-1, 7 * 7 * 64])


with tf.name_scope("hidden"):
    W = tf.Variable(tf.truncated_normal([7*7*64, 200], stddev=0.1), name = "W")
    tf.summary.histogram('W/histogram', W)
    b = tf.Variable(tf.zeros([200]), name = "b")
    tf.summary.histogram('b', b)
    activations = tf.nn.relu(tf.matmul(activations,W)+b)

    
# dense layer
with tf.name_scope("fc1"):
    activations = tf.layers.dense(inputs=activations, units=512, activation=tf.nn.relu)
    tf.summary.histogram('activations', activations)

# # dropout
with tf.name_scope("dropout"):
    keep_prob_def = tf.constant(1., tf.float32, name='keep_prob_def')
    keep_prob = tf.placeholder_with_default(keep_prob_def, [], name='keep_prob')
    activations = tf.nn.dropout(activations, keep_prob)

# (last) dense layer
with tf.name_scope("last"):
    activations = tf.layers.dense(inputs=activations, units=10)
    activations = tf.identity(activations, name="activations")
    tf.summary.histogram('activations', activations)

# --- loss and optimizer ---
with tf.name_scope("cross-entropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=activations), name="mean")
    tf.summary.scalar('batch/mean', cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(SGD__step_size).minimize(cross_entropy)

# --- accuracy ---
with tf.name_scope("accuracy"):
    correct_prediction = tf.cast(tf.equal(tf.argmax(activations, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy_sum = tf.reduce_sum(correct_prediction, name="sum")
    accuracy_mean = tf.reduce_mean(correct_prediction, name="mean")
    tf.summary.scalar('batch/mean', accuracy_mean)

# --- prepare run ---
sess = tf.InteractiveSession()

mnist = input_data.read_data_sets(data_dir, validation_size=500-train_size, one_hot=True)
mnist_eval = input_data.read_data_sets(data_dir, validation_size=500-train_size, one_hot=True)

tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(log_dir + '/validation')

# function to compute accuracy on the whole dataset
def compute_acc(data, batch_size = 10, max_ex=np.inf):
    tot_acc = 0.
    nb = 0
    epoch_number = data.epochs_completed
    while data.epochs_completed == epoch_number:
        data.next_batch(batch_size)
    epoch_number = data.epochs_completed
    while nb < max_ex:
        batch_xs, batch_ys = data.next_batch(batch_size)
        if data.epochs_completed != epoch_number: break
        tot_acc += sess.run(accuracy_sum, feed_dict={x: batch_xs, y_: batch_ys})
        nb += batch_size
    return tot_acc / nb

# Train
tf.global_variables_initializer().run()
for i in range(nb_iter):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 50 == 0 or i == nb_iter - 1:
        # Train and log params
        _, summary = sess.run([train_step, merged], feed_dict={x: batch_xs, y_: batch_ys, stddev: stddev_train, keep_prob: keep_prob_train})
        train_writer.add_summary(summary, i)
    else:
        # Train only
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, stddev: stddev_train, keep_prob: keep_prob_train})
    if i==0 or i==nb_iter-1 or round(np.log(i)/np.log(nb_iter)*10) != round(np.log(i+1)/np.log(nb_iter)*10):
        # Log train accuracy
        accuracy = compute_acc(mnist_eval.train)
        value = tf.Summary.Value(tag="accuracy", simple_value=accuracy)
        summary = tf.Summary(value=[value])
        train_writer.add_summary(summary, i)
        accuracy_train = accuracy
        # Log validation accuracy
        accuracy = compute_acc(mnist_eval.validation)
        value = tf.Summary.Value(tag="accuracy", simple_value=accuracy)
        summary = tf.Summary(value=[value])
        validation_writer.add_summary(summary, i)
        print("iteration %5d\ttraining accuracy:%.3f\tvalidation accuracy:%.3f" % (i, accuracy_train, accuracy))


# Test trained model

acc = compute_acc(mnist_eval.train)
print("Train accuracy: %.3f" % acc)

acc = compute_acc(mnist_eval.validation)
print("Validation accuracy: %.3f" % acc)


# --- save learned model ---
tf.train.Saver().save(sess, model_file)


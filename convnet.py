from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging, csv
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'update_ops'  # must be grouped with training op

activation = tf.nn.relu

def create_convnet(x, n_class, is_training, weights_seed=0):
    """
    Creates a new convolutional net for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param n_class: number of output labels
    :param is_training: boolean tf.Variable, true indicates training phase
    """

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    nx = 32
    ny = 32

    x_image = tf.reshape(x, tf.stack([-1, nx, ny, 1]))

    ouputs_per_layer = OrderedDict()
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = True  # if we use batch norm, this to False
    c['fc_units_out'] = n_class

    # conv1a
    with tf.variable_scope('conv1a'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x_image, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)  # tf.nn.leaky_relu(x, alpha=0.1)

    # conv1b
    with tf.variable_scope('conv1b'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # conv1c
    with tf.variable_scope('conv1c'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # pool1
    x = _max_pool(x, ksize=2, stride=2)

    # drop1
    x = control_flow_ops.cond(c['is_training'],
                              lambda: tf.nn.dropout(x, 0.5),
                              lambda: x)

    # conv2a
    with tf.variable_scope('conv2a'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # conv2b
    with tf.variable_scope('conv2b'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # conv2c
    with tf.variable_scope('conv2c'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # pool2
    x = _max_pool(x, ksize=2, stride=2)

    # drop2
    x = control_flow_ops.cond(c['is_training'],
                              lambda: tf.nn.dropout(x, 0.5),
                              lambda: x)

    # conv3a
    with tf.variable_scope('conv3a'):
        c['conv_filters_out'] = 512
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # conv3b
    with tf.variable_scope('conv3b'):
        c['conv_filters_out'] = 256
        c['ksize'] = 1
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # conv3c
    with tf.variable_scope('conv3c'):
        c['conv_filters_out'] = 128
        c['ksize'] = 1
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)
        #x = leaky_relu(x)

    # pool3
    x = _avg_pool2d(x, pool_size=[6, 6], strides=[1, 1])  # 6x6
    #x = tf.reshape(x, [-1, 128])
    x = tf.squeeze(x, axis=[1, 2])
    x_embedding = x

    # dense
    with tf.variable_scope('fc6'):
        x = fc(x, c, keep_prob=1., weights_seed=weights_seed)
    logits = x

    tf.summary.histogram("logits/activations", x)

    return logits, x_embedding


def leaky_relu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha=alpha)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    # x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c, keep_prob, weights_seed):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV, seed=weights_seed)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_DECAY)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)

    x = control_flow_ops.cond(
        c['is_training'], lambda: tf.nn.dropout(x, keep_prob),  # do dropout if training
        lambda: x)  # don't do dropout if val/test

    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c, weights_seed, padding='SAME'):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]

    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV, seed=weights_seed)

    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)


def _max_pool(x, ksize=3, stride=2, name=None):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


def _avg_pool2d(x, pool_size, strides, padding='VALID', name=None):
    return tf.layers.average_pooling2d(x,
                                       pool_size=pool_size,  # size of the pooling window
                                       strides=strides,
                                       padding=padding,
                                       name=name)


class ConvNet(object):
    """
    A ConvNet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=1, n_class=10, is_training=False, cost_name='baseline'):

        tf.reset_default_graph()

        self.n_class = n_class
        self.is_training = is_training
        self.cost_name = cost_name

        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, n_class])  # one-hot encoding for labels
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # self.variables contains weights and biases  # logits will be a dictionary now
        logits, x_embedding = create_convnet(self.x, n_class, self.is_training)

        self.cost, self.individual_losses = self._get_cost(logits, cost_name)
        self.predicter_embedding = x_embedding
        self.predicter = tf.nn.softmax(logits)
        self.predicter_logits = logits

        self.correct_pred = tf.equal(tf.argmax(self.predicter, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        individual_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=self.y)
        loss = tf.reduce_mean(individual_losses)

        return loss, individual_losses

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: output of the logits after the softmax
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test,
                                                             self.y: y_dummy,
                                                             self.keep_prob: 1.})
        return prediction

    def predict_embedding(self, model_path, x_test):
        """
                Uses the model to create a prediction for the given data

                :param model_path: path to the model checkpoint to restore
                :param x_test: Data to predict on. Shape [n, nx, ny, channels]
                :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
                """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter_embedding, feed_dict={self.x: x_test,
                                                                       self.y: y_dummy,
                                                                       self.keep_prob: 1.})
        return prediction

    def predict_logits(self, model_path, x_test):
        """
                Uses the model to create a prediction for the given data

                :param model_path: path to the model checkpoint to restore
                :param x_test: Data to predict on. Shape [n, nx, ny, channels]
                :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
                """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter_logits, feed_dict={self.x: x_test,
                                                                    self.y: y_dummy,
                                                                    self.keep_prob: 1.})
        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
    Trains a net instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    verification_batch_size = 4

    def __init__(self, net, optimizer="adam", batch_size=16, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, training_iters, global_step):

        loss_ = self.net.cost

        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.01)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.5)  # decay rate -- lr/2 every 30 epochs
            momentum = self.opt_kwargs.pop("momentum", 0.9)
            decay_steps = self.opt_kwargs.pop("decay_steps", 100)
            type_decay = self.opt_kwargs.pop("type_decay", 'exponential')

            if type_decay == 'exponential':
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                             **self.opt_kwargs)
            grads = opt.compute_gradients(loss_)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        elif self.optimizer == 'adam':
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.01)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.1)
            beta1 = self.opt_kwargs.pop("beta1", 0.9)
            beta2 = self.opt_kwargs.pop("beta2", 0.99999)
            decay_steps = self.opt_kwargs.pop("decay_steps", 100)
            type_decay = self.opt_kwargs.pop("type_decay", 'exponential')

            if type_decay == 'exponential':
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                         beta1=beta1,
                                         beta2=beta2)

            grads = opt.compute_gradients(loss_)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        else:
            print('Optimizer not available')

        return train_op

    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters, epochs=100, dropout=0.8, restore=False):
        """
        Launches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        """
        train_batch_size = self.batch_size

        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore)

        with tf.Session() as sess:
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            summary_writer_train = tf.summary.FileWriter(output_path + '/logs/train', graph=sess.graph)
            summary_writer_val = tf.summary.FileWriter(output_path + '/logs/val', graph=sess.graph)

            logging.info("Start optimization")

            # early stop
            best_loss_val = np.infty  # fpt: fraction of positive triplets
            wait_epochs = 20
            loss_val_epochs = np.zeros((1, wait_epochs), np.float32)
            best_epochs = []
            last_epoch = 0

            n_iterations_validation = data_provider.val.num_examples // train_batch_size
            n_iterations_per_epoch = training_iters

            for epoch in range(epochs):
                total_loss = 0.0

                if (epoch - last_epoch) > wait_epochs:
                    print('STOP TRAINING! DID NOT IMPROVE IN LAST EPOCHS')
                    break

                # for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                for step in range(1, n_iterations_per_epoch + 1):
                    # training samples
                    batch_x, batch_y = data_provider.train.next_batch(train_batch_size)
                    batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input

                    # Run optimization op (backprop)
                    _, loss, lr = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node),
                        feed_dict={self.net.x: batch_x,  # input  -> image
                                   self.net.y: batch_y,  # label
                                   self.net.keep_prob: dropout})  # self.net.w: batch_w # weights per label per batch

                    total_loss += loss

                # summary train
                self.output_minibatch_stats(sess, summary_writer_train, epoch, batch_x, batch_y, dropout, phase='Train')

                # if step % display_step == 0 and step != 0:
                loss_vals = []
                for step in range(1, n_iterations_validation + 1):
                    # validation samples
                    val_x, val_y = data_provider.val.next_batch(train_batch_size)
                    val_x = np.pad(val_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input

                    loss_val = sess.run(self.net.cost,
                                        feed_dict={self.net.x: val_x,
                                                   self.net.y: val_y,
                                                   self.net.keep_prob: 1.})

                    loss_vals.append(loss_val)

                # summary validation
                self.output_minibatch_stats(sess, summary_writer_val, epoch, val_x, val_y, dropout, phase='Val')

                loss_val = np.mean(loss_vals)

                if loss_val < best_loss_val:
                    best_epochs.append([epoch, loss_val])
                    last_epoch = epoch
                    save_path = self.net.save(sess, save_path)
                    best_loss_val = loss_val
                    print('SAVED AT EPOCH: {}, BEST_VAL_LOSS: {:.4f}'.format(epoch, loss_val))

            logging.info("Optimization Finished!")

            return save_path

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, dropout, phase):

        if phase == 'Train':
            # Calculate batch loss and accuracy
            summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: dropout})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info(
                "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                               loss,
                                                                                                               acc,
                                                                                                               error_rate(
                                                                                                                   predictions,
                                                                                                                   batch_y)))
        else:
            self.net.is_training = False
            # Calculate batch loss and accuracy
            summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
            self.net.is_training = True
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info(
                "Iter {:}, Minibatch Loss= {:.4f}, Validation Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                                 loss,
                                                                                                                 acc,
                                                                                                                 error_rate(
                                                                                                                     predictions,
                                                                                                                     batch_y)))


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            (predictions.shape[0]))

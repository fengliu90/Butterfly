# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:20:53 2019

@author: alexw
"""

import time
import tensorflow as tf
import numpy as np
import pickle
import argparse, sys
from utils import return_mnist, return_svhn, judge_func, weight_variable, bias_variable, max_pool_3x3, conv2d, \
    batch_norm_conv, batch_norm_fc, batch_generator, noisify_pairflip, noisify_multiclass_symmetric

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lambda_weight', type=float, default=0.01)
parser.add_argument('--Task_type', type=int, default=2)

args = parser.parse_args()
lr = args.lr
lambda_weight = args.lambda_weight

N_CLASS = 10
# load data
path_svhn_train = './data/synth_train_32x32.mat'
path_svhn_test = './data/synth_test_32x32.mat'
path_mnist_train = './data/train_mnist_32x32.npy'
path_mnist_test = './data/test_mnist_32x32.npy'
print('data loading...')

# Syn Tasks
if args.Task_type == 1:
    # M2S P20
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_pairflip(data_s_label, 0.20)
    print('current task: M2S P20')
    save_file_name = 'store_results_M2S_P20_MentorATDA.pckl'

if args.Task_type == 2:
    # M2S P45
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_pairflip(data_s_label, 0.45)
    print('current task: M2S P45')
    save_file_name = 'store_results_M2S_P45_MentorATDA.pckl'

if args.Task_type == 3:
    # M2S S20
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.2)
    print('current task: M2S S20')
    save_file_name = 'store_results_M2S_S20_MentorATDA.pckl'

if args.Task_type == 4:
    # M2S S45
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.45)
    print('current task: M2S S45')
    save_file_name = 'store_results_M2S_S45_MentorATDA.pckl'

if args.Task_type == 5:
    # S2M P20
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_pairflip(data_s_label, 0.20)
    print('current task: S2M P20')
    save_file_name = 'store_results_S2M_P20_MentorATDA.pckl'

if args.Task_type == 6:
    # S2M P45
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_pairflip(data_s_label, 0.45)
    print('current task: S2M P45')
    save_file_name = 'store_results_S2M_P45_MentorATDA.pckl'

if args.Task_type == 7:
    # S2M S20
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.2)
    print('current task: S2M S20')
    save_file_name = 'store_results_S2M_S20_MentorATDA.pckl'

if args.Task_type == 8:
    # S2M S45
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
    data_s_label_org = data_s_label
    data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.45)
    print('current task: S2M S45')
    save_file_name = 'store_results_S2M_S45_MentorATDA.pckl'

## Tasks for accuracy vs. noisy rate
# if args.Task_type == 1:
#     # S5 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.05)
#     print('current task: M2S S5')
#     save_file_name = 'AVN_store_results_M2S_S5_Butterfly.pckl'
# if args.Task_type == 2:
#     # S10 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.10)
#     print('current task: M2S S10')
#     save_file_name = 'AVN_store_results_M2S_S10_Butterfly.pckl'
# if args.Task_type == 3:
#     # S15 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.15)
#     print('current task: M2S S15')
#     save_file_name = 'AVN_store_results_M2S_S15_Butterfly.pckl'
# if args.Task_type == 4:
#     # S25 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.25)
#     print('current task: M2S S25')
#     save_file_name = 'AVN_store_results_M2S_S25_Butterfly.pckl'
# if args.Task_type == 5:
#     # S30 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.30)
#     print('current task: M2S S30')
#     save_file_name = 'AVN_store_results_M2S_S30_Butterfly.pckl'
#
#     # data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.2)
# if args.Task_type == 6:
#     # S40 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.40)
#     print('current task: M2S S40')
#     save_file_name = 'AVN_store_results_M2S_S40_Butterfly.pckl'
# if args.Task_type == 7:
#     # S50 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.50)
#     print('current task: M2S S50')
#     save_file_name = 'AVN_store_results_M2S_S50_Butterfly.pckl'
# if args.Task_type == 8:
#     # S60 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.60)
#     print('current task: M2S S60')
#     save_file_name = 'AVN_store_results_M2S_S60_Butterfly.pckl'
# if args.Task_type == 9:
#     # S70 - M2S
#     data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)
#     data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist(path_mnist_train, path_mnist_test)
#     data_s_label_org = data_s_label
#     data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, 0.70)
#     print('current task: M2S S70')
#     save_file_name = 'AVN_store_results_M2S_S70_Butterfly.pckl'


print('load finished')

# pre-processing
pixel_mean = np.vstack([data_t_im, data_s_im]).mean((0, 1, 2))
# num_test = 500
batch_size = 128


class SVHNModel(object):
    """SVHN domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, [None, N_CLASS])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.KK = tf.placeholder(tf.int32, [])
        all_labels = lambda: self.y
        source_labels = lambda: tf.slice(self.y, [0, 0], [int(batch_size / 2), -1])
        self.classify_labels = tf.cond(self.train, source_labels, all_labels)

        X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 64], stddev=0.01, name='W_conv0')
            b_conv0 = bias_variable([64], init=0.01, name='b_conv0')
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_3x3(h_conv0)

            W_conv1 = weight_variable([5, 5, 64, 64], stddev=0.01, name='W_conv1')
            b_conv1 = bias_variable([64], init=0.01, name='b_conv1')
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_3x3(h_conv1)

            W_conv2 = weight_variable([5, 5, 64, 128], stddev=0.01, name='W_conv2')
            b_conv2 = bias_variable([128], init=0.01, name='b_conv1')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_conv2 = batch_norm_conv(h_conv2, 128)

            h_fc1_drop = tf.nn.dropout(h_conv2, self.keep_prob)
            h_fc1_drop = tf.reshape(h_fc1_drop, [-1, 8192])

            W_fc_0 = weight_variable([8192, 3072], stddev=0.01, name='W_fc0')
            b_fc_0 = bias_variable([3072], init=0.01, name='b_fc0')
            h_fc_0 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc_0) + b_fc_0)
            self.feature = tf.nn.dropout(h_fc_0, self.keep_prob)

        with tf.variable_scope('label_predictor_1'):
            W_fc0 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0')
            b_fc0 = bias_variable([2048], init=0.01, name='b_fc0')
            h_fc0 = tf.nn.relu(batch_norm_fc(tf.matmul(self.feature, W_fc0) + b_fc0, 2048))
            h_fc0 = tf.nn.dropout(h_fc0, self.keep_prob)

            W_fc1 = weight_variable([2048, N_CLASS], stddev=0.01, name='W_fc1')
            b_fc1 = bias_variable([N_CLASS], init=0.01, name='b_fc1')
            logits = tf.matmul(h_fc0, W_fc1) + b_fc1

            all_logits = lambda: logits
            source_logits = lambda: tf.slice(logits, [0, 0], [int(batch_size / 2), -1])
            classify_logits = tf.cond(self.train, source_logits, all_logits)
            self.pred_1 = tf.nn.softmax(classify_logits)
            self.pred_loss_1_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                            labels=self.classify_labels)
            self.pred_loss_1, _ = tf.nn.top_k(-1 * self.pred_loss_1_Full, k=self.KK)
            self.pred_loss_1 = -1 * self.pred_loss_1

        with tf.variable_scope('label_predictor_2'):
            W_fc0_2 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_2')
            b_fc0_2 = bias_variable([2048], init=0.01, name='b_fc0_2')
            h_fc0_2 = tf.nn.relu(batch_norm_fc(tf.matmul(self.feature, W_fc0_2) + b_fc0_2, 2048))
            h_fc0_2 = tf.nn.dropout(h_fc0_2, self.keep_prob)

            W_fc1_2 = weight_variable([2048, 10], stddev=0.01, name='W_fc1_2')
            b_fc1_2 = bias_variable([10], init=0.01, name='b_fc1_2')
            logits2 = tf.matmul(h_fc0_2, W_fc1_2) + b_fc1_2

            all_logits_2 = lambda: logits2
            source_logits_2 = lambda: tf.slice(logits2, [0, 0], [int(batch_size / 2), -1])
            classify_logits_2 = tf.cond(self.train, source_logits_2, all_logits_2)

            self.pred_2 = tf.nn.softmax(classify_logits_2)
            self.pred_loss_2_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_2,
                                                                            labels=self.classify_labels)
            self.pred_loss_2, _ = tf.nn.top_k(-1 * self.pred_loss_2_Full, k=self.KK)
            self.pred_loss_2 = -1 * self.pred_loss_2

        with tf.variable_scope('label_predictor_target'):
            W_fc0_t = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_t')
            b_fc0_t = bias_variable([2048], init=0.01, name='b_fc0_t')
            h_fc0_t = tf.nn.relu(tf.matmul(self.feature, W_fc0_t) + b_fc0_t)
            h_fc0_t = tf.nn.dropout(h_fc0_t, self.keep_prob)
            W_fc1_t = weight_variable([2048, 10], stddev=0.01, name='W_fc1_t')
            b_fc1_t = bias_variable([10], init=0.01, name='b_fc1_t')
            logits_t = tf.matmul(h_fc0_t, W_fc1_t) + b_fc1_t

            all_logits_t = lambda: logits_t
            source_logits = lambda: tf.slice(logits_t, [0, 0], [int(batch_size / 2), -1])
            classify_logits_t = tf.cond(self.train, source_logits, all_logits_t)

            self.pred_t = tf.nn.softmax(classify_logits_t)
            self.pred_loss_t_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_t,
                                                                            labels=self.classify_labels)
            self.pred_loss_t, _ = tf.nn.top_k(-1 * self.pred_loss_t_Full, k=self.KK)
            self.pred_loss_t = -1 * self.pred_loss_t
        #            self.pred_loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
        #                                                                       labels=self.classify_labels)

        with tf.variable_scope('label_predictor_target2'):
            W_fc0_t2 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_t2')
            b_fc0_t2 = bias_variable([2048], init=0.01, name='b_fc0_t2')
            h_fc0_t2 = tf.nn.relu(tf.matmul(self.feature, W_fc0_t2) + b_fc0_t2)
            h_fc0_t2 = tf.nn.dropout(h_fc0_t2, self.keep_prob)
            W_fc1_t2 = weight_variable([2048, 10], stddev=0.01, name='W_fc1_t2')
            b_fc1_t2 = bias_variable([10], init=0.01, name='b_fc1_t2')
            logits_t2 = tf.matmul(h_fc0_t2, W_fc1_t2) + b_fc1_t2

            all_logits_t2 = lambda: logits_t2
            source_logits_t2 = lambda: tf.slice(logits_t2, [0, 0], [int(batch_size / 2), -1])
            classify_logits_t2 = tf.cond(self.train, source_logits_t2, all_logits_t2)

            self.pred_t2 = tf.nn.softmax(classify_logits_t2)
            self.pred_loss_t2_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_t2,
                                                                             labels=self.classify_labels)
            self.pred_loss_t2, _ = tf.nn.top_k(-1 * self.pred_loss_t2_Full, k=self.KK)
            self.pred_loss_t2 = -1 * self.pred_loss_t2
        #            self.pred_loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
        #                                                                       labels=self.classify_labels)

        temp_w = W_fc0
        temp_w2 = W_fc0_2
        weight_diff = tf.matmul(temp_w, temp_w2, transpose_b=True)
        weight_diff = tf.abs(weight_diff)
        weight_diff = tf.reduce_sum(weight_diff, 0)
        self.weight_diff = tf.reduce_mean(weight_diff)


graph = tf.get_default_graph()
with graph.as_default():
    model = SVHNModel()
    learning_rate = tf.placeholder(tf.float32, [])
    # exchange losses for Branch I
    temp = model.pred_loss_2
    model.pred_loss_2 = model.pred_loss_1
    model.pred_loss_1 = temp
    pred_lossF1 = tf.reduce_mean(model.pred_loss_1)
    pred_lossF2 = tf.reduce_mean(model.pred_loss_2)
    # exchange losses for Branch II
    temp_t = model.pred_loss_t2
    model.pred_loss_t2 = model.pred_loss_t
    model.pred_loss_t = temp_t
    pred_loss_Ftarget = tf.reduce_mean(model.pred_loss_t)
    pred_loss_Ftarget2 = tf.reduce_mean(model.pred_loss_t2)

    weight_diff = model.weight_diff
    pred_loss1 = pred_lossF1 + pred_lossF2 + lambda_weight * weight_diff
    pred_loss2 = pred_loss1 + pred_loss_Ftarget + pred_loss_Ftarget2
    target_loss = pred_loss_Ftarget + pred_loss_Ftarget2
    target_loss2 = pred_loss_Ftarget2
    total_loss = pred_loss1 + pred_loss2

    regular_train_op1 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss1)
    regular_train_op2 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss2)
    target_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(target_loss)
    target_train_op2 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(target_loss2)

    # Evaluation

    correct_label_pred1 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_1, 1))
    correct_label_pred2 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_2, 1))
    correct_label_pred_t = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_t, 1))
    correct_label_pred_t2 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_t2, 1))

    label_acc_t = tf.reduce_mean(tf.cast(correct_label_pred_t, tf.float32))
    label_acc_t2 = tf.reduce_mean(tf.cast(correct_label_pred_t2, tf.float32))
    label_acc1 = tf.reduce_mean(tf.cast(correct_label_pred1, tf.float32))
    label_acc2 = tf.reduce_mean(tf.cast(correct_label_pred2, tf.float32))
# Params
num_steps = 1000

T_t = np.zeros([30])
S_t1 = np.zeros([30])
S_t2 = np.zeros([30])
S_s = np.zeros([30])


def train_and_evaluate(graph, model, verbose=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        for t in range(30):
            print('phase:%d' % (t))
            forget_rate = 0 + min(0.4 * t / 5, 0.4)  # 0.2
            kk_c = int(batch_size * (1 - forget_rate))
            label_target = np.zeros((data_t_im.shape[0], N_CLASS))
            if t == 0:
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size)

            else:
                source_train = data_s_im
                source_label = data_s_label
                if new_data.shape[0] != 0:
                    source_train = np.r_[source_train, new_data]
                    new_label = new_label.reshape((new_label.shape[0], new_label.shape[2]))
                    source_label = np.r_[source_label, new_label]
                    gen_source_batch = batch_generator(
                        [source_train, source_label], int(batch_size / 2))
                    gen_new_batch = batch_generator(
                        [new_data, new_label], batch_size)
                    gen_source_only_batch = batch_generator(
                        [data_s_im, data_s_label], batch_size)
                else:
                    gen_source_batch = gen_source_only_batch
                    gen_new_batch = gen_source_only_batch
                    print('No candidate!')

            # Training loop
            for i in range(num_steps):
                # lr = FLAGS.learning_rate
                dropout = 0.5
                # Training step
                if t == 0:
                    X0, y0 = next(gen_source_only_batch)
                    _, _, batch_loss, w_diff, ploss, p_l1, p_l2, p_acc1, p_acc2 = \
                        sess.run([target_train_op, regular_train_op1, total_loss, weight_diff, total_loss, pred_loss1,
                                  pred_loss2, label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: dropout,
                                            model.KK: kk_c})
                    if verbose and i % 500 == 0:
                        print('loss: %f  w_diff: %f  p_l1: %f  p_l2: %f  p_acc1: %f p_acc2: %f' % \
                              (batch_loss, w_diff, p_l1, p_l2, p_acc1, p_acc2))

                if t >= 1:

                    # Here is different: new data is trained again.

                    X0, y0 = next(gen_source_batch)
                    _, batch_loss, w_diff, ploss, p_l1, p_l2, p_acc1, p_acc2 = \
                        sess.run([regular_train_op1, total_loss, weight_diff, total_loss, pred_loss1, pred_loss2,
                                  label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout, model.KK: int(kk_c / 2)})

                    X1, y1 = next(gen_new_batch)
                    if np.shape(y1)[0] < int(batch_size / 2):
                        kk_c_target = np.shape(y1)[0]
                    else:
                        #                        print('Butterfly comes.')
                        forget_rate_t = 0 + min(0.05 * t / 5, 0.05)  # 0.05
                        kk_c_target = int(np.shape(y1)[0] * (1 - forget_rate_t))
                    _, p_acc_t, p_acc_t2 = \
                        sess.run([target_train_op, label_acc_t, label_acc_t2],
                                 feed_dict={model.X: X1, model.y: y1, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout, model.KK: kk_c_target})

                    if verbose and i % 500 == 0:
                        print('loss: %f  w_diff: %f  loss1: %f  loss2: %f  acc1: %f acc2: %f acc_t: %f acc_t2: %f' % \
                              (batch_loss, w_diff, p_l1, p_l2, p_acc1, p_acc2, p_acc_t, p_acc_t2))
            # Attach Pseudo Label
            step = 0
            pred1_stack = np.zeros((0, N_CLASS))
            pred2_stack = np.zeros((0, N_CLASS))
            predt_stack = np.zeros((0, N_CLASS))
            stack_num = min(data_t_im.shape[0] / batch_size, 100 * (t + 1))
            # Shuffle pseudo labeled candidates
            perm = np.random.permutation(data_t_im.shape[0])
            gen_target_batch = batch_generator(
                [data_t_im[perm, :], label_target], batch_size, shuffle=False)
            while step < stack_num:
                if t == 0:
                    X1, y1 = next(gen_target_batch)
                    pred_1, pred_2 = sess.run([model.pred_1, model.pred_2],
                                              feed_dict={model.X: X1,
                                                         model.y: y1,
                                                         model.train: False,
                                                         model.keep_prob: 1,
                                                         model.KK: 128})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    step += 1
                else:
                    X1, y1 = next(gen_target_batch)

                    pred_1, pred_2, pred_t = sess.run([model.pred_1, model.pred_2, model.pred_t],
                                                      feed_dict={model.X: X1,
                                                                 model.y: y1,
                                                                 model.train: False,
                                                                 model.keep_prob: 1,
                                                                 model.KK: 128})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    predt_stack = np.r_[predt_stack, pred_t]
                    step += 1
            if t == 0:
                cand = data_t_im[perm, :]
                rate = max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 1000)
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],
                                                 num_class=N_CLASS)
            if t != 0:
                cand = data_t_im[perm, :]
                rate = min(max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 5000),
                           15000)  # always 20000 was best int(N_source*0.8)
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],
                                                 num_class=N_CLASS)

            # Evaluation
            gen_source_batch = batch_generator(
                [data_s_im, data_s_label], batch_size, test=True)
            gen_target_batch = batch_generator(
                [data_t_im_test, data_t_label_test], batch_size, test=True)
            num_iter = int(data_t_im_test.shape[0] / batch_size) + 1
            step = 0
            total_source = 0
            total_target = 0
            target_pred1 = 0
            target_pred2 = 0
            total_acc1 = 0
            total_acc2 = 0
            size_t = 0
            size_s = 0
            while step < num_iter:
                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                source_acc = sess.run(label_acc1,
                                      feed_dict={model.X: X0, model.y: y0,
                                                 model.train: False, model.keep_prob: 1, model.KK: 128})
                target_acc, target_acc2, t_acc1, t_acc2, = sess.run([label_acc_t, label_acc_t2, label_acc1, label_acc2],
                                                                    feed_dict={model.X: X1, model.y: y1,
                                                                               model.train: False,
                                                                               model.keep_prob: 1, model.KK: 128})
                total_source += source_acc * len(X0)
                total_target += target_acc * len(X1)
                total_acc1 += t_acc1 * len(X1)
                total_acc2 += t_acc2 * len(X1)
                size_t += len(X1)
                size_s += len(X0)
                step += 1
            T_t[t] = total_target / size_t
            S_t1[t] = total_acc1 / size_t
            S_t2[t] = total_acc2 / size_t
            S_s[t] = total_source / size_s
            print('train target', total_target / size_t, total_acc1 / size_t, total_acc2 / size_t,
                  total_source / size_s)
    return total_source / size_s, total_target / size_t, total_acc1 / size_t, total_acc2 / size_t, T_t, S_t1, S_t2, S_s


print('\nTraining Start')
all_source = 0
all_target = 0
NN = 5
all_source_list = np.zeros([NN])
all_target_list = np.zeros([NN])
T_t_M = np.zeros([NN, 30])
S_t1_M = np.zeros([NN, 30])
S_t2_M = np.zeros([NN, 30])
S_s_M = np.zeros([NN, 30])
for i in range(NN):
    print('Total rounds: ', i)
    start = time.time()
    source_acc, target_acc, t_acc1, t_acc2, T_t, S_t1, S_t2, S_s = train_and_evaluate(graph, model)
    end = time.time()
    all_source += source_acc
    all_target += target_acc
    all_source_list[i] = source_acc
    all_target_list[i] = target_acc
    T_t_M[i, :] = T_t
    S_t1_M[i, :] = S_t1
    S_t2_M[i, :] = S_t2
    S_s_M[i, :] = S_s
    print('Source accuracy:', source_acc)
    print('Target accuracy (Target Classifier):', target_acc)
    print('Target accuracy (Classifier1):', t_acc1)
    print('Target accuracy (Classifier2):', t_acc2)
    print('Time elapsed:', end - start)

print('Source accuracy:', all_source / NN)
print('Target accuracy:', all_target / NN)

f = open(save_file_name, 'wb')
pickle.dump([S_s_M, S_t1_M, S_t2_M, T_t_M, all_source_list, all_target_list], f)
f.close()
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:34:15 2019

@author: alexw
"""

import tensorflow as tf
import numpy as np
import pickle
from utils import return_Amazon, weight_variable, bias_variable, \
    batch_norm_fc, batch_generator, noisify_pairflip, noisify_multiclass_symmetric, \
    judge_func_amazon

flags = tf.app.flags
flags.DEFINE_float('lamda', 0.001, "value of lamda")  # 0.5
flags.DEFINE_float('learning_rate', 0.01, "value of learnin rage")  # 0.05
FLAGS = flags.FLAGS
N_CLASS = 2
path_amazon_data = './data/amazon.mat'
num_test = 500
batch_size = 24
Threshold_confidence = 0.9


class AmazonModel(object):
    """SVHN domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 5000])
        self.y = tf.placeholder(tf.float32, [None, N_CLASS])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.KK = tf.placeholder(tf.int32, [])
        all_labels = lambda: self.y
        source_labels = lambda: tf.slice(self.y, [0, 0], [int(batch_size / 2), -1])
        self.classify_labels = tf.cond(self.train, source_labels, all_labels)
        X_input = tf.cast(self.X, tf.float32)
        with tf.variable_scope('label_predictor_1'):
            W_fc0 = weight_variable([5000, 50], stddev=0.01, name='W_fc0')
            b_fc0 = bias_variable([50], init=0.01, name='b_fc0')
            h_fc0 = tf.nn.relu(batch_norm_fc(tf.matmul(X_input, W_fc0) + b_fc0, 50))
            h_fc0 = tf.nn.dropout(h_fc0, self.keep_prob)

            W_fc1 = weight_variable([50, N_CLASS], stddev=0.01, name='W_fc1')
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
            W_fc0_2 = weight_variable([5000, 50], stddev=0.01, name='W_fc0_2')
            b_fc0_2 = bias_variable([50], init=0.01, name='b_fc0_2')
            h_fc0_2 = tf.nn.relu(batch_norm_fc(tf.matmul(X_input, W_fc0_2) + b_fc0_2, 50))
            h_fc0_2 = tf.nn.dropout(h_fc0_2, self.keep_prob)

            W_fc1_2 = weight_variable([50, N_CLASS], stddev=0.01, name='W_fc1_2')
            b_fc1_2 = bias_variable([N_CLASS], init=0.01, name='b_fc1_2')
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
            W_fc0_t = weight_variable([5000, 50], stddev=0.01, name='W_fc0_t')
            b_fc0_t = bias_variable([50], init=0.01, name='b_fc0_t')
            h_fc0_t = tf.nn.relu(batch_norm_fc(tf.matmul(X_input, W_fc0_t) + b_fc0_t,50))
            h_fc0_t = tf.nn.dropout(h_fc0_t, self.keep_prob)
            W_fc1_t = weight_variable([50, N_CLASS], stddev=0.01, name='W_fc1_t')
            b_fc1_t = bias_variable([N_CLASS], init=0.01, name='b_fc1_t')
            logits_t = tf.matmul(h_fc0_t, W_fc1_t) + b_fc1_t

            all_logits_t = lambda: logits_t
            source_logits = lambda: tf.slice(logits_t, [0, 0], [int(batch_size / 2), -1])
            classify_logits_t = tf.cond(self.train, source_logits, all_logits_t)

            self.pred_t = tf.nn.softmax(classify_logits_t)
            self.pred_loss_t_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_t,
                                                                            labels=self.classify_labels)
            self.pred_loss_t, _ = tf.nn.top_k(-1 * self.pred_loss_t_Full, k=self.KK)
            self.pred_loss_t = -1 * self.pred_loss_t


        with tf.variable_scope('label_predictor_target2'):
            W_fc0_t2 = weight_variable([5000, 50], stddev=0.01, name='W_fc0_t2')
            b_fc0_t2 = bias_variable([50], init=0.01, name='b_fc0_t2')
            h_fc0_t2 = tf.nn.relu(batch_norm_fc(tf.matmul(X_input, W_fc0_t2) + b_fc0_t2,50))
            h_fc0_t2 = tf.nn.dropout(h_fc0_t2, self.keep_prob)
            W_fc1_t2 = weight_variable([50, N_CLASS], stddev=0.01, name='W_fc1_t2')
            b_fc1_t2 = bias_variable([N_CLASS], init=0.01, name='b_fc1_t2')
            logits_t2 = tf.matmul(h_fc0_t2, W_fc1_t2) + b_fc1_t2

            all_logits_t2 = lambda: logits_t2
            source_logits_t2 = lambda: tf.slice(logits_t2, [0, 0], [int(batch_size / 2), -1])
            classify_logits_t2 = tf.cond(self.train, source_logits_t2, all_logits_t2)

            self.pred_t2 = tf.nn.softmax(classify_logits_t2)
            self.pred_loss_t2_Full = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_t2,
                                                                             labels=self.classify_labels)
            self.pred_loss_t2, _ = tf.nn.top_k(-1 * self.pred_loss_t2_Full, k=self.KK)
            self.pred_loss_t2 = -1 * self.pred_loss_t2

        temp_w = W_fc0
        temp_w2 = W_fc0_2
        weight_diff = tf.matmul(temp_w, temp_w2, transpose_b=True)
        weight_diff = tf.abs(weight_diff)
        weight_diff = tf.reduce_sum(weight_diff, 0)
        self.weight_diff = tf.reduce_mean(weight_diff)


graph = tf.get_default_graph()
with graph.as_default():
    model = AmazonModel()
    learning_rate = tf.placeholder(tf.float32, [])
    temp = model.pred_loss_2
    model.pred_loss_2 = model.pred_loss_1
    model.pred_loss_1 = temp
    pred_lossF1 = tf.reduce_mean(model.pred_loss_1)
    pred_lossF2 = tf.reduce_mean(model.pred_loss_2)
    temp_t = model.pred_loss_t2
    model.pred_loss_t2 = model.pred_loss_t
    model.pred_loss_t = temp_t
    pred_loss_Ftarget = tf.reduce_mean(model.pred_loss_t)
    pred_loss_Ftarget2 = tf.reduce_mean(model.pred_loss_t2)

    weight_diff = model.weight_diff
    pred_loss1 = pred_lossF1 + pred_lossF2 + FLAGS.lamda * weight_diff
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
num_steps = 200
T_t = np.zeros([30])
S_t1 = np.zeros([30])
S_t2 = np.zeros([30])
S_s = np.zeros([30])


def train_and_evaluate(graph, model, source, target, noise_rate, noise_type, verbose=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    print('data loading...')
    data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_Amazon(path_amazon_data, source)
    data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_Amazon(path_amazon_data, target)
    if noise_type == 'sym':
        data_s_label, Actual_noise = noisify_multiclass_symmetric(data_s_label, noise_rate, random_state=0,
                                                                  nb_classes=2)
    elif noise_type == 'pair':
        data_s_label, Actual_noise = noisify_pairflip(data_s_label, noise_rate, random_state=0, nb_classes=2)
    print('load finished')

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        for t in range(30):
            print('phase:%d' % (t))
            if t < 5:
                forget_rate = 0
            else:
                forget_rate = 0 + min(0.1 * (t - 5) / 5, 0.1)
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
                        [new_data, new_label], int(batch_size / 2))
                    gen_source_only_batch = batch_generator(
                        [data_s_im, data_s_label], batch_size)
                else:
                    gen_source_batch = gen_source_only_batch
                    gen_new_batch = gen_source_only_batch
                    print('No candidate!')

            # Training loop
            for i in range(num_steps):
                lr = FLAGS.learning_rate
                dropout = 0.5 #0.5
                # Training step
                if t == 0:
                    X0, y0 = next(gen_source_only_batch)
                    _, _, batch_loss, w_diff, ploss, p_l1, p_l2, p_acc1, p_acc2 = \
                        sess.run([target_train_op, regular_train_op1, total_loss, weight_diff, total_loss, pred_loss1,
                                  pred_loss2, label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: dropout,
                                            model.KK: kk_c})
                    if verbose and i % 50 == 0:
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
                        forget_rate_t = 0 + min(0.05 * t / 5, 0.05)
                        kk_c_target = int(np.shape(y1)[0] * (1 - forget_rate_t))
                    _, p_acc_t, p_acc_t2 = \
                        sess.run([target_train_op, label_acc_t, label_acc_t2],
                                 feed_dict={model.X: X1, model.y: y1, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout, model.KK: int(kk_c_target)})

                    if verbose and i % 50 == 0:
                        print('loss: %f  w_diff: %f  loss1: %f  loss2: %f  acc1: %f acc2: %f acc_t: %f' % \
                              (batch_loss, w_diff, p_l1, p_l2, p_acc1, p_acc2, p_acc_t))
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
                rate = max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 500)
                new_data, new_label = judge_func_amazon(cand,
                                                        pred1_stack[:rate, :],
                                                        pred2_stack[:rate, :],
                                                        upper=Threshold_confidence,
                                                        num_class=N_CLASS)
            if t != 0:
                cand = data_t_im[perm, :]
                rate = min(max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 500),
                           1500)  # always 20000 was best int(N_source*0.8)
                new_data, new_label = judge_func_amazon(cand,
                                                        pred1_stack[:rate, :],
                                                        pred2_stack[:rate, :],
                                                        upper=Threshold_confidence,
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
            total_target2 = 0
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
                                                       feed_dict={model.X: X1, model.y: y1, model.train: False,
                                                                  model.keep_prob: 1, model.KK: 128})
                total_source += source_acc * len(X0)
                total_target += target_acc * len(X1)
                total_target2 += target_acc2 * len(X1)
                total_acc1 += t_acc1 * len(X1)
                total_acc2 += t_acc2 * len(X1)
                size_t += len(X1)
                size_s += len(X0)
                step += 1
            T_t[t] = total_target / size_t
            S_t1[t] = total_acc1 / size_t
            S_t2[t] = total_acc2 / size_t
            S_s[t] = total_source / size_s
            print('train target', total_target / size_t, total_target2 / size_t, total_acc1 / size_t, total_acc2 / size_t,
                  total_source / size_s)
    return model, total_source / size_s, total_target / size_t, total_acc1 / size_t, total_acc2 / size_t, T_t, S_t1, S_t2, S_s


print('\nTraining Start')
NN = 10
Domains = ['book', 'dvd', 'electronics', 'kitchen']
noise_type = ['sym']
noise_rate = [0.2, 0.45]
Results = np.zeros([16, 4, 2])
all_source_list = np.zeros([16, 4, NN])
all_target_list = np.zeros([16, 4, NN])
T_t_M = np.zeros([16, 4, NN, 30])
S_t1_M = np.zeros([16, 4, NN, 30])
S_t2_M = np.zeros([16, 4, NN, 30])
S_s_M = np.zeros([16, 4, NN, 30])
for D_i in [0, 1, 2, 3]:
    for D_j in [0, 1, 2, 3]:
        if D_i != D_j:
            if True:
                for noise_type_ind in [0]:
                    for noise_rate_ind in [0, 1]:
                        all_source = 0
                        all_target = 0
                        for i in range(NN):
                            print(i, Domains[D_i], Domains[D_j], noise_rate[noise_rate_ind], noise_type[noise_type_ind])
                            model0, source_acc, target_acc, t_acc1, t_acc2, T_t, S_t1, S_t2, S_s = train_and_evaluate(
                                graph, model, Domains[D_i], Domains[D_j], noise_rate[noise_rate_ind],
                                noise_type[noise_type_ind])
                            all_source += source_acc
                            all_target += target_acc
                            T_t_M[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i, :] = T_t
                            S_t1_M[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i, :] = S_t1
                            S_t2_M[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i, :] = S_t2
                            S_s_M[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i, :] = S_s
                            all_source_list[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i] = source_acc
                            all_target_list[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, i] = target_acc
                            print('Source accuracy:', source_acc)
                            print('Target accuracy (Target Classifier):', target_acc)
                            print('Target accuracy (Classifier1):', t_acc1)
                            print('Target accuracy (Classifier2):', t_acc2)
                        Results[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, 0] = all_target / NN
                        Results[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, 1] = all_source / NN
                        print('Source accuracy:', Results[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, 1])
                        print('Target accuracy:', Results[4 * (D_i) + D_j, noise_type_ind * 2 + noise_rate_ind, 0])
# f = open('store_results_Butterfly_Amazon_test.pckl', 'wb')
# pickle.dump([S_s_M, S_t1_M, S_t2_M, T_t_M, all_source_list, all_target_list, Results], f)
# f.close()
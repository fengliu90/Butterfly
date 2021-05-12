import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.examples.tutorials.mnist import input_data
import copy
import hashlib
import errno
from numpy.testing import assert_array_almost_equal


def return_Obj_data(path_data):
    Obj_data_all = loadmat(path_data)
    xx = Obj_data_all['fts'].astype(np.float32)
    yy = Obj_data_all['label']-1
    yy = dense_to_one_hot_amazon(yy, 40)
    return xx, xx, yy, yy

def return_Amazon(path_data, data_name):
    amazon_data_all = loadmat(path_data)
    xx = amazon_data_all['xx'].toarray()
    xxl = xx[0:5000][:]
    offset = amazon_data_all['offset']
    yy = amazon_data_all['yy']
    yy = dense_to_one_hot_amazon(yy,2)
    if data_name == 'book':
        i = 0
        ind1 = int(offset[i])
        ind2 = int(offset[i+1])
        train_feature = np.transpose(xxl[:,ind1:ind1+2000])
        test_feature = np.transpose(xxl[:,ind1+2000:ind2-1])
        train_labels = yy[ind1:2000,:]
        test_labels = yy[ind1+2000:ind2-1,:]
    if data_name == 'dvd':
        i = 1
        ind1 = int(offset[i])
        ind2 = int(offset[i+1])
        train_feature = np.transpose(xxl[:,ind1:ind1+2000])
        test_feature = np.transpose(xxl[:,ind1+2000:ind2-1])
        train_labels = yy[ind1:ind1+2000,:]
        test_labels = yy[ind1+2000:ind2-1,:]
    if data_name == 'electronics':
        i = 2
        ind1 = int(offset[i])
        ind2 = int(offset[i+1])
        train_feature = np.transpose(xxl[:,ind1:ind1+2000])
        test_feature = np.transpose(xxl[:,ind1+2000:ind2-1])
        train_labels = yy[ind1:ind1+2000,:]
        test_labels = yy[ind1+2000:ind2-1,:]
    if data_name == 'kitchen':
        i = 3
        ind1 = int(offset[i])
        ind2 = int(offset[i+1])
        train_feature = np.transpose(xxl[:,ind1:ind1+2000])
        test_feature = np.transpose(xxl[:,ind1+2000:ind2-1])
        train_labels = yy[ind1:ind1+2000,:]
        test_labels = yy[ind1+2000:ind2-1,:]
    return train_feature, test_feature, train_labels, test_labels

def return_svhn(path_train, path_test):
    svhn_train = loadmat(path_train)
    svhn_test = loadmat(path_test)
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 0, 1, 2)
    svhn_train_im = np.reshape(svhn_train_im, (svhn_train_im.shape[0], 32, 32, 3))
    svhn_label = dense_to_one_hot_svhn(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 0, 1, 2)
    svhn_label_test = dense_to_one_hot_svhn(svhn_test['y'])
    svhn_test_im = np.reshape(svhn_test_im, (svhn_test_im.shape[0], 32, 32, 3))

    return svhn_train_im, svhn_test_im, svhn_label, svhn_label_test


def return_mnist(path_train, path_test):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_train = np.reshape(np.load(path_train), (55000, 32, 32, 1))
    mnist_train = np.reshape(mnist_train, (55000, 32, 32, 1))
    mnist_train = mnist_train.astype(np.float32)
    mnist_test = np.reshape(np.load(path_test), (10000, 32, 32, 1)).astype(
        np.float32)
    mnist_test = np.reshape(mnist_test, (10000, 32, 32, 1))
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    return mnist_train, mnist_test, mnist.train.labels, mnist.test.labels


def select_class(labels, data, num_class=10, per_class=10):
    classes = np.argmax(labels, axis=1)
    labeled = []
    train_label = []
    unlabels = []
    for i in range(num_class):
        class_list = np.array(np.where(classes == i))
        class_list = class_list[0]
        class_ind = labels[np.where(classes == i), :]
        rands = np.random.permutation(len(class_list))
        unlabels.append(class_list[rands[per_class:]])
        labeled.append(class_list[rands[:per_class]])
        label_i = np.zeros((per_class, num_class))
        label_i[:, i] = 1
        train_label.append(label_i)
    unlabel_ind = []
    label_ind = []
    for t in unlabels:
        for i in t:
            unlabel_ind.append(i)
    for t in labeled:
        for i in t:
            label_ind.append(i)
    unlabel_data = data[unlabel_ind, :, :, :]
    labeled_data = data[label_ind, :, :, :]
    train_label = np.array(train_label).reshape((num_class * per_class, num_class))
    return np.array(labeled_data), np.array(train_label), unlabel_data


def judge_func(data, pred1, pred2, upper=0.95, num_class=10):
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])
        if ind1 == ind2:
            if max(value1, value2) > upper:
                label_data[0, ind1] = 1
                new_label.append(label_data)
                new_data.append(cand_data)
                new_ind.append(i)
    return np.array(new_data), np.array(new_label)

def judge_func_amazon(data, pred1, pred2, upper, num_class=2):
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    for i in range(num):
        cand_data = data[i, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])
        if ind1 == ind2:
            if max(value1, value2) > upper:
                label_data[0, ind1] = 1
                new_label.append(label_data)
                new_data.append(cand_data)
                new_ind.append(i)
    return np.array(new_data), np.array(new_label)

def judge_func_obj(data, pred1, pred2, upper, num_class=40):
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    for i in range(num):
        cand_data = data[i, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])
        if ind1 == ind2:
            # print(max(value1, value2))
            if max(value1, value2) > upper:
                label_data[0, ind1] = 1
                new_label.append(label_data)
                new_data.append(cand_data)
                new_ind.append(i)
    return np.array(new_data), np.array(new_label)

def weight_variable(shape, stddev=0.1, name=None, train=True):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name:
        return tf.Variable(initial, name=name, trainable=train)
    else:
        return tf.Variable(initial)


def bias_variable(shape, init=0.1, name=None):
    initial = tf.constant(init, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def batch_norm_conv(x, out_channels):
    mean, var = tf.nn.moments(x, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels])
    batch_norm = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)
    return batch_norm


def batch_norm_fc(x, out_channels):
    mean, var = tf.nn.moments(x, axes=[0])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels])
    batch_norm = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)
    return batch_norm


def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p_0 = np.random.permutation(num)
    return [d[p_0] for d in data]


def batch_generator(data, batch_size, shuffle=True, test=False):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if test:
            if batch_count * batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        else:
            if batch_count * batch_size + batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        labels_one_hot[i, t] = 1
    return labels_one_hot

def dense_to_one_hot_amazon(labels_dense, num_classes=2):
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = np.where(labels_dense<0,0,labels_dense)
#    labels_dense = labels_dense + 1
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        labels_one_hot[i, t] = 1
    return labels_one_hot


def dense_to_one_hot_svhn(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
        labels_one_hot[i, t] = 1
    return labels_one_hot

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
#    assert P.shape[0] == P.shape[1]
#    assert np.max(y) < P.shape[0]
#
#    # row stochastic matrix
#    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
#    assert (P >= 0.0).all()

    m = y.shape[0]
    print (m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = np.where(y[idx]==1)
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i[0], :][0], 1)[0]
        new_y[idx] = flipped
    print(new_y)
    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (np.where(y_train_noisy == 1)[1] != np.where(y_train==1)[1]).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print (P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (np.where(y_train_noisy == 1)[1] != np.where(y_train==1)[1]).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print (P)

    return y_train, actual_noise

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


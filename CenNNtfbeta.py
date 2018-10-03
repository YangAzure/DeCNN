import tensorflow as tf
import numpy as np
from numpy.linalg import inv
from sklearn import datasets
from sklearn.model_selection import train_test_split

C = 0.25
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def get_mnist_data():
    from mnist import MNIST
    mndata = MNIST('./python-mnist-master/data')
    data, target = mndata.load_training()
    data = np.asarray(data[:10000])
    target = np.asarray(target[:10000])

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)


with tf.Session() as sess:
    train_X, test_X, train_y, test_y = get_mnist_data()
    #train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 600                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    print train_y
    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))

    # Forward propagation
    h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function

    init = tf.global_variables_initializer()
    sess.run(init)

    hnp = h.eval(feed_dict={X: train_X})
    I_L = np.eye(hnp.shape[1])

    Q = np.matmul(np.transpose(hnp), train_y)
    omega = inv(1/C*I_L + np.matmul(np.transpose(hnp),hnp))
    print omega
    
    beta = np.matmul(omega, Q)
    print beta.shape
    yhat = np.matmul(hnp, beta)  # The \varphi function
    predict = np.argmax(yhat, axis=1)

    print yhat.shape

    print predict
    train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 predict)
    hnp2 = h.eval(feed_dict={X: test_X})
    yhat2 = np.matmul(hnp2, beta)
    predict2 = np.argmax(yhat2, axis=1)
    test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 predict2)
    print("train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (100. * train_accuracy, 100. * test_accuracy))

    sess.close()

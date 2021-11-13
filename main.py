import numpy as np
import math
import operator
from collections import Counter
from random import seed, randrange
import time
import matplotlib.pyplot as plt
import scipy.stats as st
from numba import guvectorize, jit
from joblib import Parallel, delayed
from keras.datasets import mnist


def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    image = np.reshape(image, (28, 28))
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


@jit('float64(float64[:,:], float64[:,:])')
def euclidean_distance(test, train):
    sum = 0.
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            sum += (test[i, j] - train[i, j]) ** 2
    return sum


@jit('float64[:,:,:](float64[:,:])')
def sliding_window(train_sample):
    W = train_sample.shape[0]
    H = train_sample.shape[1]
    shift_trains = np.zeros((9, W+2, H+2))
    shifts = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1)]

    for k in range(len(shifts)):
        dx = shifts[k][0]
        dy = shifts[k][1]
        shift_trains[k, (1 + dx):(W+1 + dx), (1 + dy):(H+1+dy)] = train_sample
    return shift_trains


@jit('float64(float64[:,:], float64[:,:])')
def new_dist_euclidean(test, train):
    W = train.shape[0]
    H = train.shape[1]

    shift_trains = sliding_window(train)

    test_reshape = np.zeros((W+2, H+2))
    test_reshape[1:W+1, 1:H+1] = test

    min_dist = 1e308
    for k in range(shift_trains.shape[0]):
        dist = 0.
        for i in range(shift_trains.shape[1]):
            for j in range(shift_trains.shape[2]):
                dist += (test_reshape[i, j] - shift_trains[k, i, j]) ** 2
        if dist < min_dist:
            min_dist = dist
    return min_dist


@guvectorize(['void(float64[:,:], float64[:,:,:], b1, float64[:])'],
             '(w,h),(n,w,h),()->(n)', nopython=True, target='cpu')
def find_dist(val, train, slide, distances):
    for i in range(train.shape[0]):
        if not slide:
            dist = euclidean_distance(val, train[i, :, :])
        else:
            dist = new_dist_euclidean(val, train[i, :, :])
        distances[i] = dist
    return


def getNeighbors(training, validation, k, i, slide=False):
    if i % 10 == 0:
        print("test sample number %d" %i)
    distances = np.zeros(len(training))
    val_data = validation[1].reshape((28,28))
    train_data = np.array([training[i][1].reshape((28,28)) for i in range(len(training))])
    find_dist(val_data, train_data, slide, distances)
    label_dist = [(training[x][0], distances[x]) for x in range(len(training))]

    label_dist.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(label_dist[x][0])

    vote = Counter(neighbors)
    knn = max(vote.iteritems(), key=operator.itemgetter(1))[0]
    return knn


def cross_validation(training_data, k, folds):
    valid_accuracies = []
    accuracies = []
    training_length = len(training_data)
    for pointer in range(0, folds):  # for the k-fold cross validation

        validation = training_data[pointer * training_length / folds: ((pointer + 1) * training_length / folds)]
        training = np.concatenate((training_data[: (pointer * training_length / folds)] + training_data[((pointer + 1) * training_length / folds):]), axis=0)

        error = 0
        for n in range(0, len(validation)):

            knn = getNeighbors(training, validation[n], k, n)
            if knn != validation[n][0]:
                error += 1

        print("k=%d ,fold=%d ,error=%d" % (k, (pointer + 1), error))
        accuracy = (len(validation) - error) * 100 / (len(validation) * 1.0)
        print("accuracy=%.4f%%" % accuracy)
        valid_accuracies.append(accuracy)

    mean_accuracy = sum(valid_accuracies) / float(len(valid_accuracies))
    
    accuracies.append(mean_accuracy)
    print("average accuracy for cross fold validation %.4f%%" % mean_accuracy)
    print("################################################")
    return valid_accuracies


def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    plt.matshow(df_confusion, cmap='gray_r')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(range(10)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.savefig('confusion_matrix.png')


def confusion_matrix(testing_data, knn):
    
    # confusion matrix
    actual = list(testing_data[x][0] for x in range(len(testing_data)))
    predicted = list(knn[x] for x in range(len(knn)))

    # calculate the confusion matrix:
    cm = np.zeros((len(range(10)), len(range(10))))
    for a, p in zip(actual, predicted):
        cm[a][p] += 1
    print("confusion matrix:")
    print(cm)
    plot_confusion_matrix(cm)
    return cm


def confidence_interval(testing_data, error):
    
    # confidence interval:
    p_hat = (len(testing_data) - error) * 1.0 / len(testing_data)
    sigma = math.sqrt((p_hat * (1-p_hat)) * 1.0 / len(testing_data))
    zscore = st.norm.ppf(1-(1-0.95)/2)
    conf_int_up = p_hat + (zscore * sigma)
    conf_int_down = p_hat - (zscore * sigma)
    print("confidence_interval: [%.4f, %.4f]" % (conf_int_down, conf_int_up))
    return conf_int_down, conf_int_up


def test_function(training_data, testing_data, k, slide=False):
    error = 0
    knn_labels = list()
    knn = Parallel(n_jobs=-1,backend='multiprocessing', batch_size=50, prefer='processes')(
    delayed(getNeighbors)(training_data, testing_data[i], k, i, slide) for i in range(len(testing_data)))

    for n in range(len(testing_data)):
        knn_labels.append((testing_data[n][0], knn[n]))
        if knn[n] != testing_data[n][0]:
            error += 1

    print("error=%d" % error)
    accuracy = (len(testing_data) - error) * 100 / (len(testing_data) * 1.0)

    # confusion matrix
    cm = confusion_matrix(testing_data, knn)
    print("\ntest accuracy=%.4f%%" % accuracy)

    # confidence interval
    conf_int_down, conf_int_up = confidence_interval(testing_data, error)
    return accuracy


def main():
    start_time = time.time()
    seed(1)

    # Download the mnist data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = [[train_X[i], train_y[i]] for i in range(len(train_X))]
    test_X = [[test_X[i], test_y[i]] for i in range(len(test_X))]

    k_inputs = range(1, 11)  # to find the optimal value for k in knn
    validation_accuracies = Parallel(n_jobs=8)(delayed(cross_validation)(train_X, i, folds=10) for i in k_inputs)

    # find the value of k that has the largest accuracy
    k_optimal = np.argmax(np.mean(validation_accuracies, axis=1))
    print("k={} achieved highest accuracy of {} on validation data".format(range(1, 11)[k_optimal], np.mean(validation_accuracies[i])))

    # test
    test_accuracy = test_function(train_X, test_X, k_optimal)
    print('test accuracies: ', test_accuracy)

    slide_accuracy = test_function(train_X, test_X, k_optimal, slide=True)
    print('slide_accuracies: ', slide_accuracy)

    print("\n --- %s seconds ---" % (time.time() - start_time))

main()

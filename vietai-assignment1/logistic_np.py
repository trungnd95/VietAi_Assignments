"""
This file is for two-class vehicle classification
Original Author: Kien Huynh
Editor: Trung Ng
Date: 25/05/2019
VietAI Ha Noi - Course 04 - 2019
Result (equal with result output from sklearn lib) => The formula is correct:
    + Precision(Accuracy) = 72.5% with 1000 epochs, learning_rate = 0.01
    + (*)Precision = 75.3% with 10000 epochs, learning_rate = 0.005
GD with momentum and GD without one are not quite different.
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_vehicle_data
from sklearn.metrics import accuracy_score, recall_score
import pdb


class LogisticClassifier(object):
    def __init__(self, w_shape):
        """__init__
        
        :param w_shape: create w with shape w_shape using normal distribution
        """
        mean = 0
        std = 1
        self.w = np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape)

    def feed_forward(self, x):
        """feed_forward
        This function compute the output of your logistic classification model
        
        :param x: input
        
        :return result: feed forward result (after sigmoid) 
        """
        # [TODO 1.5] - Doing by Trung Ng - 25/05/2019
        #            - State: Done
        # Sigmoid function

        z = np.dot(x, self.w)
        result = 1 / (1 + np.exp(-z))
        return result


    def compute_loss(self, y, y_hat):
        """compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the propabilitis that the given samples belong to class 1
        
        :return loss: a single value
        """
        # [TODO 1.6] - Doing by Trung Ng - 25/05/2019
        #            - State: Done
        # Compute loss value (a single number)
        loss = -np.sum(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat))) / y.shape[0]
        return loss

    def get_grad(self, x, y, y_hat):
        """get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        
        :return w_grad: has the same shape as self.w
        """ 
        # [TODO 1.7] - Doing by Trung Nguyen - 25/05/2019
        #            - State:
        # Compute the gradient of w, it has the same size of w

        w_grad = np.dot(np.transpose(x), (y_hat - y)) / y.shape[0]
        return w_grad


    def update_weight(self, grad, learning_rate):
        """update_weight
        Update w using the computed gradient

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        # [TODO 1.8] - Doing by Trung Ng - 25/05/2019
        #            - State: Done
        # Update w using SGD

        self.w = self.w - learning_rate * grad


    def update_weight_momentum(self, grad, learning_rate, momentum, momentum_rate):
        """update_weight with momentum
        Update w using the algorithm with momnetum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum: the array storing momentum for training w,
                         should have the same shape as w
        :param momentum_rate: float, how much momentum to reuse
                              after each loop (denoted as gamma in the document)
        """
        # [TODO 1.9] - Doing by Trung Ng - Date: 25/05/2019
        #            - State: Doing
        # Update w using SGD with momentum
        try:
            assert(momentum.shape == self.w.shape)
        except AssertionError:
            print()
        momentum = momentum_rate * momentum + learning_rate * grad
        self.w = self.w - momentum

    
def plot_loss(all_loss):
    plt.figure(1)
    plt.clf()
    plt.plot(all_loss)


def normalize_per_pixel(train_x, test_x):
    """normalize_per_pixel
        This function computes train mean and standard deviation on each pixel then applying data scaling on train_x and
        test_x using these computed values
        :param train_x: train images, shape=(num_train, image_height, image_width)
        :param test_x: test images, shape=(num_test, image_height, image_width)
    """

    # [TODO 1.1] - Doing by Trung Ng - 25/05/2019
    #            - State: Done
    # train_mean and train_std should have the shape of (1, image_height, image_width) 

    """
        Initialization
    """
    image_height = train_x.shape[1]
    image_width = train_x.shape[2]
    number_of_train_images = train_x.shape[0]
    train_mean = np.empty((image_height, image_width))
    train_std = np.empty((image_height, image_width))

    """
        Calculate train mean by:
            + Split each image in numpy array shape 2D. 
            + Add it up to compose numpy array with shape = (image_height, image_width)
            + Divide each element of the array just got by number of images in train set
            + Reshape to (1, image_height, image_width)
    """
    train_mean_before_reshape = np.empty((image_height, image_width))
    for each_image in range(number_of_train_images):
        train_mean_before_reshape += train_x[each_image]

    train_mean = np.divide(train_mean_before_reshape, number_of_train_images)\
                   .reshape(1, image_height, image_width)

    """
        Calculate train std by:
            + Splitting each image in numpy array shape 2D. 
            + Subtract element-wise with mean 2D calculated above. Then power 2 to compose 
                array with shape 2D (image_height, image_width)
            + Divide the array with number of images in train set. After that, do square root
            + Reshape to 3D (1, image_height, image_width)
    """
    train_std_before_reshape = np.empty((image_height, image_width))
    for each_image in range(number_of_train_images):
        train_std_before_reshape = (train_x[each_image] - train_mean_before_reshape)**2

    train_std = np.sqrt(np.divide(train_std_before_reshape, number_of_train_images))\
                  .reshape(1, image_height, image_width)

    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std
    return train_x, test_x


def normalize_all_pixel(train_x, test_x):
    """normalize_all_pixel
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x and test_x using these computed values

    :param train_x: train images, shape=(num_train, image_height, image_width)
    :param test_x: test images, shape=(num_test, image_height, image_width)
    """
    # [TODO 1.2] - Doing by Trung Ng - 25/05/2019
    #            - State: Done
    # train_mean and train_std should have the shape of (1, image_height, image_width)

    """
        Initialization
    """
    number_of_train_images = train_x.shape[0]
    image_height = train_x.shape[1]
    image_width = train_x.shape[2]

    """
        Calculate train mean by:
            + Add up all pixel value over entire training dataset
            + Divide each value by product of number of training images, image width
                and image height
            + Reshape to 3D (1, image_height, image_width)
    """
    train_mean = np.divide(np.sum(train_x),
                           (number_of_train_images * image_width * image_height))

    train_std = np.sqrt(np.divide(np.sum(train_x - train_mean),
                                  (number_of_train_images * image_width * image_height)))

    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std
    return train_x, test_x


def reshape2D(tensor):
    """reshape_2D
    Reshape our 3D tensors to 2D.
    A 3D tensor of shape (num_samples, image_height, image_width) must be reshaped
        into (num_samples, image_height*image_width)
    """
    # [TODO 1.3] - Doing by Trung Ng - 25/05/2019
    #            - State: Done
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    return tensor


def add_one(x):
    """add_one
    
    This function add ones as an additional feature for x
    :param x: input data
    """
    # [TODO 1.4] - Doing by Trung Ng - 25/05/2019
    #            - State: Done
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    return x


def test(y_hat, test_y):
    """test
    Compute precision, recall and F1-score based on predicted test values

    :param y_hat: predicted values, output of classifier.feed_forward
    :param test_y: test labels
    """
    
    # [TODO 1.10] - Doing by Trung Ng - Date: 25/05/2019
    #             - State:
    # Compute test scores using test_y and y_hat

    """
        Step 1: Transfer probability matrix result calculated from feed_forward function
        to class label with default threshold is 0.5
    """
    y_hat = (y_hat > 0.5).astype(int)

    """
        Step 2: Filter samples that model predicted correctly.Result is array with true at position 
        matched between y_hat and test_y, false at otherwise. 
        Cause value of True = 1, False = 0, so we will calculate precision value by exactly 
        this binary array. Other assessment values is calculated based on y_hat and test_y 
    """
    matched_binary_array = (y_hat == test_y)
    precision = np.mean(matched_binary_array)

    true_positive_arr = (y_hat == 1) & (test_y == 1)
    recall = np.sum(true_positive_arr) / np.sum(test_y)
    f1 = 2 / ((1/precision) + (1/recall))
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1-score: %.3f" % f1)

    """
        Test my result compare with sklearn built-in lib
    """
    print("Accuracy from sklearn lib: {value}".format(value=accuracy_score(test_y, y_hat)))
    print("Recall from sklearn lib: {recall}".format(recall=recall_score(test_y, y_hat)))
    return precision, recall, f1


def generate_unit_testcase(train_x, train_y):
    train_x = train_x[0:5, :, :]
    train_y = train_y[0:5, :]
    
    testcase = {}
    testcase['output'] = []

    train_x_norm1, _ = normalize_per_pixel(train_x, train_x)
    train_x_norm2, _ = normalize_all_pixel(train_x, train_x)
    train_x = train_x_norm2

    testcase['train_x_norm1'] = train_x_norm1
    testcase['train_x_norm2'] = train_x_norm2

    train_x = reshape2D(train_x)
    testcase['train_x2D'] = train_x

    train_x = add_one(train_x)
    testcase['train_x1'] = train_x
    
    learning_rate = 0.001
    momentum_rate = 0.9

    for i in range(10): 
        test_dict = {}
        classifier = LogisticClassifier((train_x.shape[1], 1))
        test_dict['w'] = classifier.w
        
        y_hat = classifier.feed_forward(train_x)
        loss = classifier.compute_loss(train_y, y_hat)
        grad = classifier.get_grad(train_x, train_y, y_hat)
        classifier.update_weight(grad, 0.001)
        test_dict['w_1'] = classifier.w
        
        momentum = np.ones_like(grad)
        classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)
        test_dict['w_2'] = classifier.w

        test_dict['y_hat'] = y_hat
        test_dict['loss'] = loss
        test_dict['grad'] = grad
         
        testcase['output'].append(test_dict)
    
    np.save('./data/unittest', testcase)


debug = False
if __name__ == "__main__":
    np.random.seed(2018)

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    if debug:
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    num_train = train_x.shape[0]
    num_test = test_x.shape[0]
    #
    # #generate_unit_testcase(train_x.copy(), train_y.copy())
    #
    # Normalize our data: choose one of the two methods before training
    # train_x, test_x = normalize_all_pixel(train_x, test_x)
    train_x, test_x = normalize_per_pixel(train_x, test_x)
    if debug:
        plt.imshow(train_x[0, :, :], cmap='gray')
        plt.show()

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)
    if debug:
        print("Dimension after reshape: (train = {train_shape}, test = {test_shape})".format(
                                                                    train_shape=train_x.shape,
                                                                    test_shape=test_x.shape))

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)
    if debug:
        print("Dimension after add one: (train = {train_shape}, test = {test_shape})".format(
                                                                    train_shape=train_x.shape,
                                                                    test_shape=test_x.shape))
        print(train_x[:, 4096])

    # Create classifier
    num_feature = train_x.shape[1]
    bin_classifier = LogisticClassifier((num_feature, 1))
    momentum = np.zeros_like(bin_classifier.w)

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005
    momentum_rate = 0.95
    epochs_to_draw = 100
    all_loss = []
    plt.ion()
    for e in range(num_epoch):
        print("Training ----> {sequence_of_epoch}".format(sequence_of_epoch=e))
        y_hat = bin_classifier.feed_forward(train_x)
        loss = bin_classifier.compute_loss(train_y, y_hat)
        print("\tLoss value = {}".format(loss))
        grad = bin_classifier.get_grad(train_x, train_y, y_hat)
        # Updating weight: choose either normal SGD or SGD with momentum
        bin_classifier.update_weight(grad, learning_rate)
        # bin_classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)

        all_loss.append(loss)

        if e % epochs_to_draw == epochs_to_draw-1:
            plot_loss(all_loss)
            plt.show()
            plt.pause(0.1)
            print("Epoch %d: loss is %.5f" % (e+1, loss))

    y_hat = bin_classifier.feed_forward(test_x)
    test(y_hat, test_y)

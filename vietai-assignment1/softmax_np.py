"""
This file is for fashion mnist classification

Author: Trung Ng
Date: 27/05/2019
VietAI Ha Noi - Course 04 - 2019
Result:
    Parameters: Epoch 10000, learning_rate=0.005, momentum=0.95
    Training process stopped after 6627 epochs since using early stopping technique(Todo 2.6)
    Diagonal values of confusion matrix = [0.93 0.95 0.59 0.82 0.77 0.89 0.51 0.86 0.93 0.98]
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_mnist_data
from logistic_np import add_one, LogisticClassifier
from sklearn.metrics import confusion_matrix

import pdb

debug=False

class SoftmaxClassifier(LogisticClassifier):
    def __init__(self, w_shape):
        """__init__
        
        :param w_shape: create w with shape w_shape using normal distribution
        """
        super(SoftmaxClassifier, self).__init__(w_shape)


    def softmax(self, x):
        """softmax
        Compute softmax on the second axis of x
    
        :param x: input
        """
        # [TODO 2.3]
        # Compute softmax

        return None


    def feed_forward(self, x):
        """feed_forward
        This function compute the output of your softmax regression model
        
        :param x: input
        """
        # [TODO 2.3] - Doing by Trung Ng - 27/05/2019
        #            - State: Done
        # Compute a feed forward pass


        """
            Step1: Calculate output z: z = w.T * x
                - z.shape = (x.shape[0], self.w.shape[1]) = (number_of_training_samples, output_labels)
                - Each row of z is score of each input sample.
        """
        z = np.dot(x, self.w)

        """
            Step2: Compute softmax values for each sets of scores in z
                - To avoid number overflow, subtract each score value with the max score
                  in each row
                - Do softmax computation: A = np.exp(z-zmax) / np.sum(z). Out put is a matrix with
                shape equals with shape of z = (number_of_training_samples, output_lables)
        """
        e_Z = np.exp(z - np.max(z, axis=1, keepdims=True))
        A = e_Z / e_Z.sum(axis=1).reshape(e_Z.shape[0], 1)
        return A


    def compute_loss(self, y, y_hat):
        """compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the class probabilities of all samples in our data
        """
        # [TODO 2.4] - Doing by Trung Ng - 27/05/2019
        #            - State: Done
        # Compute categorical loss

        """
            Compute loss by categorical cross-entropy
            m = num_of_training_samples
            c = num_of_labels_output
            J(W) = -1/m * np.sum(y*log(y_hat))
        """
        return (-1/ y.shape[0]) * np.sum(y * np.log(y_hat))


    def get_grad(self, x, y, y_hat):
        """get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        """ 
        # [TODO 2.5] - Doing by Trung Ng - 27/05/2019
        #            - State: Done
        # Compute gradient of the loss function with respect to w

        """
            m - x.shape[0] - number_of_training_samples
            Gradient formula: Grad(J/w) = 1/m *  x.T * (y_hat-y)
        """
        return (1/x.shape[0]) * np.dot(x.T, y_hat - y)


def plot_loss(train_loss, val_loss):
    plt.figure(1)
    plt.clf()
    plt.plot(train_loss, color='b')
    plt.plot(val_loss, color='g')


def draw_weight(w):
    label_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    w = w[0:(28*28),:].reshape(28, 28, 10)
    for i in range(10):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(w[:,:,i], interpolation='nearest')
        plt.axis('off')
        ax.set_title(label_names[i])


def normalize(train_x, val_x, test_x):
    """normalize
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x, val_x
    and test_x using these computed values
    Note that in this classification problem, the data is already flatten into a shape of
    (num_samples, image_width*image_height)

    :param train_x: train images, shape=(num_train, image_height*image_width)
    :param val_x: validation images, shape=(num_val, image_height*image_width)
    :param test_x: test images, shape=(num_test, image_height*image_width)
    """
    # [TODO 2.1] - Doing by Trung Ng - 27/05/2019
    #            - State: Done
    # train_mean and train_std should have the shape of (1, 1)
    """
        Initialization
    """
    number_of_train_images = train_x.shape[0]
    image_size = train_x.shape[1]

    """
        Calculate train mean by:
            + Add up all pixel value over entire training dataset
            + Divide each value by product of number of training images, image width
                and image height
    """
    train_mean = np.divide(np.sum(train_x),
                           (number_of_train_images * image_size))

    train_std = np.sqrt(np.divide(np.sum((train_x - train_mean) ** 2),
                                  (number_of_train_images * image_size)))

    train_x = (train_x - train_mean) / train_std
    val_x = (val_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std
    return train_x, val_x, test_x


def create_one_hot(labels, num_k=10):
    """create_one_hot
    This function creates a one-hot (one-of-k) matrix based on the given labels

    :param labels: list of labels, each label is one of 0, 1, 2,... , num_k - 1
    :param num_k: number of classes we want to classify
    """
    # [TODO 2.2] - Doing by Trung Ng - 27/05/2019
    #            - State: Done
    # Create the one-hot label matrix here based on labels
    one_hot_matrix = np.zeros((len(labels), num_k))
    one_hot_matrix[np.arange(len(labels)), labels] = 1
    return one_hot_matrix


def test(y_hat, test_y):
    """test
    Compute the confusion matrix based on labels and predicted values 

    :param classifier: the trained classifier
    :param y_hat: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """
    
    confusion_mat = np.zeros((10,10))

    # [TODO 2.7] - Doing by Trung Ng - 27/05/2019
    #            - State: Doing
    # Compute the confusion matrix here

    """
        Step 1: Convert one-hot matrix to sparse array
    """
    sparse_y_hat = np.array([y_hat[y_hat_label].argmax() for y_hat_label in range(y_hat.shape[0])])
    sparse_test_y = np.array([test_y[test_y_label].argmax() for test_y_label in range(test_y.shape[0])])

    """
        Step 2: Feed up value in confusion matrix
    """
    for actual, predict in zip(sparse_test_y, sparse_y_hat):
        confusion_mat[actual][predict] += 1

    """
        Step 3: Normalization
    """
    confusion_mat = confusion_mat / confusion_mat.sum(axis=1)

    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::11])

    print('-----------------------------------------------')
    cm = confusion_matrix(sparse_test_y, sparse_y_hat)
    print('Confusion matrix calculate by sklearn with checking purpose: ')
    print(cm / cm.sum(axis=1))


if __name__ == "__main__":
    np.random.seed(2018)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]
    if debug:
        print("Shape of data: train_x={train_x}, train_y={train_y}, val_x={val_x}, val_y={val_y}, "
              "test_x={test_x}, test_y={test_y}".format(train_x=train_x.shape, train_y=train_y.shape,
                                                        val_x=val_x.shape, val_y=val_y.shape, test_x=test_x.shape,
                                                        test_y=test_y.shape))

    # generate_unit_testcase(train_x.copy(), train_y.copy())

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)
    if debug:
        print("Shape of labels output after one-hot: {0}, {1}, {2}".format(train_y.shape,
                                                                           val_y.shape, test_y.shape))
    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    if debug:
        plt.imshow(train_x[0].reshape((28, 28)), cmap='gray')
        plt.show()

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    val_x = add_one(val_x)
    test_x = add_one(test_x)
    if debug:
        print("Shape of data after add one: {0}, {1}, {2}".format(train_x.shape,
                                                                  val_x.shape, test_x.shape))

    # Create classifier
    num_feature = train_x.shape[1]
    dec_classifier = SoftmaxClassifier((num_feature, 10))
    momentum = np.zeros_like(dec_classifier.w)

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005
    momentum_rate = 0.95
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()

    for e in range(num_epoch):
        print("Epoch --------> {}".format(e))
        train_y_hat = dec_classifier.feed_forward(train_x)
        val_y_hat = dec_classifier.feed_forward(val_x)

        train_loss = dec_classifier.compute_loss(train_y, train_y_hat)
        val_loss = dec_classifier.compute_loss(val_y, val_y_hat)

        grad = dec_classifier.get_grad(train_x, train_y, train_y_hat)

        # dec_classifier.numerical_check(train_x, train_y, grad)
        # Updating weight: choose either normal SGD or SGD with momentum
        dec_classifier.update_weight(grad, learning_rate)
        # dec_classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)

        # [TODO 2.6] - Doing by Trung Ng - 27/05/2019
        #            - State: Done
        # Propose your own stopping condition here:
        """
            Overall, loss(training_set) will be decreased after each loop.
            To stop update weight (stop training process) - early stopping based on
            loss values of model on training set and validation set are if loss(training_set)
            and loss(validation_set) values are go far from each other after each loop
            (It means that loss (validation set) is increasing )=> Stop
            Specialize:
                - Step 1: Compute difference loss of validation set after each epoch
                - Step 2: In 50 recent epochs, if the later value is large previous one in more than 40 times
                => STOP
        """
        compare_array = np.diff(all_val_loss)
        val_loss_in_recent_50_epochs = compare_array[-50: -1]
        if val_loss_in_recent_50_epochs[val_loss_in_recent_50_epochs > 0].size > 40:
            break

        if e % epochs_to_draw == epochs_to_draw-1:
            plot_loss(all_train_loss, all_val_loss)
            draw_weight(dec_classifier.w)
            plt.show()
            plt.pause(0.1)
            print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss, val_loss))

    y_hat = dec_classifier.feed_forward(test_x)
    test(y_hat, test_y)
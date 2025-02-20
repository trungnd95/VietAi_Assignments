"""
This file is for multiclass fashion-mnist classification using TensorFlow
Author: Kien Huynh
Editor: Trung Ng
Date: 27/05/2019
VietAI Ha Noi - Course 04 - 2019
Result:
    Epochs: 10000, learning_rate=0.01
    Early stopping is not happen. loop all epochs
    Diagonal of confusion matrix: [0.93 0.9  0.59 0.85 0.77 0.87 0.   0.82 0.91 0.94]
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *


if __name__ == "__main__":
    np.random.seed(2018)
    tf.set_random_seed(2018)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)

    """
        Initialization
    """
    num_of_trains = train_x.shape[0]
    num_of_features = train_x.shape[1]
    num_of_labels = train_y.shape[1]

    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    #            - Doing by Trung Ng - 27/05/2019
    #            - State: Done
    x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))

    # [TODO 2.8] Create weights (W) using TF variables
    #           - Doing by Trung Ng - 27/05/2019
    #           - State: Done
    w = tf.Variable(np.zeros((num_of_features, num_of_labels)))

    # [TODO 2.9] Create a feed-forward operator
    #            - Doing by Trung Ng - 27/05/2019
    #            - State: Done
    pred = tf.nn.softmax(tf.matmul(x, tf.cast(w, tf.float32)))

    # [TODO 2.10] Write the cost function
    #           - Doing by Trung Ng - 27/05/2019
    #           - State: Done
    # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer
    #           - Doing by Trung Ng - 27/05/2019
    #           - State: Done
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            # [TODO 2.8] Compute losses and update weights here
            #           - Doing by Trung Ng - 27/05/2019
            #           - State: Done
            train_loss = 0 
            val_loss = 0 
            # Update weights
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            train_loss = sess.run(cost, feed_dict={x: train_x, y: train_y})
            all_train_loss.append(train_loss)

            val_loss = sess.run(cost, feed_dict={x: val_x, y: val_y})
            all_val_loss.append(val_loss)

            # [TODO 2.11] Define your own stopping condition here
            #           - Doing by Trung Ng - 27/05/2019
            #           - State: Done
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
                w_  = sess.run(w)
                draw_weight(w_)
                plt.show()
                plt.pause(0.1)     
                print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss, val_loss))
        
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)

# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf


def x_entr( p_y_given_x_train, y_gt, num_classes, weightPerClass ):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.
    
    #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
    e1 = 1e-6
    log_p_y_given_x_train = tf.log( p_y_given_x_train + e1) #added a tiny so that it does not go to zero and I have problems with nan again...
    
    
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, num_classes, 1, 1, 1])
    weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClass5D
    
    y_one_hot = tf.one_hot( indices=y_gt, depth=num_classes, axis=1, dtype="float32" )

    num_samples = tf.cast( tf.reduce_prod( tf.shape(y_gt) ), "float32")
    
    return - (1./ num_samples) * tf.reduce_sum( weighted_log_p_y_given_x_train * y_one_hot )
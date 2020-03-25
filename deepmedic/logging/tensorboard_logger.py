from __future__ import absolute_import, print_function, division

import tensorflow as tf


class TensorboardLogger(object):

    def __init__(self, log_path, tf_graph):
        #_ = tf.compat.v1.summary.FileWriter(log_path, tf_graph)
        self.logger = tf.summary.create_file_writer(log_path) # tf.compat.v1.summary.FileWriter(log_path, tf_graph)

    def add_summary(self, value, name, step_num):
        #self.logger.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=name, simple_value=value)]),
        #                        global_step=step_num)
        with self.logger.as_default():
            tf.summary.scalar(name, value, step=step_num)
            self.logger.flush()

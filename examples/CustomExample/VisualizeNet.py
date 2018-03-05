#This file is just for visulaizing the 3D U-Net on tensorboard

#code from "https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/F2h33c9LB4U"


import tensorflow as tf


g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('my-model.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='logs/my-model', graph=g)
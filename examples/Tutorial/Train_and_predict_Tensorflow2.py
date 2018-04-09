
from __future__ import print_function

import math
import numpy as np
import random

import gunpowder as gp
from gunpowder import *
from gunpowder.caffe import *
from gunpowder.ext import malis
import tensorflow as tf
import os
import code



import mala
from gunpowder import *
from gunpowder.caffe import *

                ############
                # Training #
                ############

#Declare Arrays
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GT')
loss=gp.ArrayKey('LOSS')
prediction=gp.ArrayKey('prediction')
grad=gp.ArrayKey('gradient')






#define training values
#input_shape = (120, 1200, 1200)
input_shape=(,1200,1200)

raw_tf = tf.placeholder(tf.float32, shape=input_shape)
raw_batched = tf.reshape(raw_tf, (1, 1) + input_shape)

unet = mala.networks.unet(raw_batched, 12, 5, [[1,1,1],[2,4,4],[1,4,4]])



labels_batched = mala.networks.conv_pass(
    unet,
    kernel_size=1,
    num_fmaps=1,
    num_repetitions=1,
    activation='sigmoid')

output_shape_batched = labels_batched.get_shape().as_list()
output_shape = output_shape_batched[2:]
print(output_shape)

labels = tf.reshape(labels_batched, output_shape)
gt_labels = tf.placeholder(tf.float32, shape=output_shape)
loss_weights = tf.placeholder(tf.float32, shape=output_shape)

loss = tf.losses.mean_squared_error(
    gt_labels,
    labels,
    loss_weights)


tf.summary.scalar('loss_total', loss)

opt = tf.train.AdamOptimizer(
    learning_rate=0.5e-6,
    beta1=0.95,
    beta2=0.999,
    epsilon=1e-8)
optimizer = opt.minimize(loss)
merged = tf.summary.merge_all()
tf.train.export_meta_graph(filename='unet.meta')


###############################
# CREMI DATA IS STORED Z,X,Y  #
#           z   x    y        #
# voxelsize=(40 ,4   ,4)      #
# size  =   (125,1250,1250)   #
###############################


#import source
source = gp.Hdf5Source(
        'data.hdf',
        {
            raw: 'volumes/raw',
            gt : 'volumes/labels/neuron_ids'
        },
        array_specs={
            #raw : gp.ArraySpec(voxel_size=Coordinate((4,4,40)),roi=Roi((0,0,0),(1200,1200,120))),
            #gt  : gp.ArraySpec(voxel_size=Coordinate((4,4,40)),roi=Roi((0,0,0),(1200,1200,120)))
            raw : gp.ArraySpec(roi=Roi((0,0,0), (80*40, 4*1200, 4*1200))),
            gt  : gp.ArraySpec(roi=Roi((0,0,0), (80*40, 4*1200, 4*1200)))
        }
    )




#define pipeline
training_pipeline = (
        source+
        gp.RandomProvider()+
        #gp.SimpleAugment() +
        #uses metagraph file 'unet' which contains 3D u-net
        gp.tensorflow.Train('unet',
                            optimizer=optimizer.name,
                            loss=loss.name,
                            inputs={
                                    raw_tf.name : raw,
                                    gt_labels.name : gt
                            },
                            outputs={
                                    labels.name : prediction
                            },
                            gradients={
                                    labels.name : grad
                            }
        )
)



#define request
request = gp.BatchRequest()
#request.add(raw, (120*40, 4*1200, 4*1200))
#request.add(gt,  (120*40, 4*1200, 4*1200))







#Execute train
n=10
with gp.build(training_pipeline) as minibatch_maker:
    for i in range(n):
        minibatch_maker.request_batch(request)























'''
                ##############
                # Prediction #
                ##############

#we want the last itteration
iteration=n





#import the necessary data for the model and define input and output sizes
checkpoint='unet_checkpoint_%i'%iteration
input_size = Coordinate((84,268,268))
output_size = Coordinate((56,56,56))





#define source for prediction
predict_source=Hdf5Source(
                'sample_B_20160501.hdf',
                raw='volumes/raw')


#create pipeline
prediction_pipeline = (
        predict_source +
        Normalize() +
        Predict(checkpoint) +
        Snapshot(
                every=1,
                output_dir=os.path.join('chunks', '%d'%iteration),
                output_filename='chunk.hdf'
        ) +
        PrintProfilingStats() +
                ArraySpec(
                        input_size,
                        output_size
                )+
        Snapshot(
                every=1,
                output_dir=os.path.join('processed', '%d'%iteration),
                output_filename='sample_A_20160501.hdf'
        )
)


# request a "batch" of the size of the whole dataset
with build(prediction_pipeline) as p:
    shape = p.get_spec().roi.get_shape()
    p.request_batch(
            ArraySpec(
                    shape,
                    shape - (input_size-output_size)
            )
    )
'''

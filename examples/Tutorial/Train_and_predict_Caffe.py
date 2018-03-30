
from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *
from gunpowder.caffe import *
from gunpowder.ext import malis


import os

from gunpowder import *
from gunpowder.caffe import *



                ############
                # Training #
                ############

#built in neighborhood function
affinity_neighborhood = malis.mknhood3d()


#initialize solver parameters for caffe
solver_parameters = SolverParameters()
solver_parameters.train_net = 'net.prototxt'
solver_parameters.base_lr = 1e-4
solver_parameters.momentum = 0.95
solver_parameters.momentum2 = 0.999
solver_parameters.delta = 1e-8
solver_parameters.weight_decay = 0.000005
solver_parameters.lr_policy = 'inv'
solver_parameters.gamma = 0.0001
solver_parameters.power = 0.75
solver_parameters.snapshot = 2000
solver_parameters.snapshot_prefix = 'net'
solver_parameters.type = 'Adam'
solver_parameters.resume_from = None
solver_parameters.train_state.add_stage('euclid')



#define
raw = ArrayKey('RAW')
labels = ArrayKey('GT_LABELS')

#create source
data_source=Hdf5Source(
        'FILE.hdf',
        datasets = {
           raw: 'volumes/raw',
           labels: 'volumes/labels',
        }
    )

#define pipeline
training_pipeline = (
        data_source+
        RandomProvider()+
        SimpleAugment(transpose_only_xy=True) +
        Train(solver_parameters, use_gpu=0)
    )


#define request
request = BatchRequest()
request.add(ArrayKey.RAW, (84,268,268))
request.add(ArrayKey.GT_LABELS, (56,56,56))
request.add(ArrayKey.GT_MASK, (56,56,56))

#Execute train
n=10
with build(training_pipeline) as minibatch_maker:
    for i in range(n):
        minibatch_maker.request_batch(request)




                ##############
                # Prediction #
                ##############


#we want the last itteration
iteration=n

#import the necessary data for the model
prototxt = 'net.prototxt'
weights  = 'net_iter_%d.caffemodel'%iteration

input_size = Coordinate((84,268,268))
output_size = Coordinate((56,56,56))

prediction_pipeline = (


        #import
        Hdf5Source(
                'sample_A_20160501.hdf',
                raw_dataset='volumes/raw') +


        Normalize() +
        Pad() +
        IntensityScaleShift(2, -1) +
        Predict(prototxt, weights, use_gpu=0) +
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


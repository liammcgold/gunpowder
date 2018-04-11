
from __future__ import print_function


import numpy as np
import random
#USE  kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')
import gunpowder as gp
from gunpowder import *
from gunpowder.ext import malis
import tensorflow as tf






import mala
from gunpowder import *
from gunpowder.caffe import *

                                                                ############
                                                                # Training #
                                                                ############

#Declare Arrays
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GT')
prediction=gp.ArrayKey('prediction')
grad=gp.ArrayKey('gradient')




#300,000 is too high
#270,750 works (30,95,95)


#define training values
input_shape=(30,95,95)
voxel_size=(40,4,4)
raw_tf = tf.placeholder(tf.float32, shape=input_shape)
raw_batched = tf.reshape(raw_tf, (1, 1) + input_shape)

#unet = mala.networks.unet(raw_batched, 12, 5, [[1,1,1],[2,4,4],[1,4,4]])
unet = mala.networks.unet(raw_batched, 12, 5, [[1,1,1],[1,1,1],[1,1,1]])



labels_batched = mala.networks.conv_pass(
    unet,
    kernel_size=1,
    num_fmaps=1,
    num_repetitions=1,
    activation='relu')

output_shape_batched = labels_batched.get_shape().as_list()
output_shape = output_shape_batched[2:]
print(output_shape)

labels = tf.reshape(labels_batched, output_shape)
gt_labels = tf.placeholder(tf.float32, shape=output_shape)


#loss_weights = tf.placeholder(tf.float32, shape=output_shape)

loss = tf.losses.mean_squared_error(
    gt_labels,
    labels)


tf.summary.scalar('loss_total', loss)

opt = tf.train.AdamOptimizer(
    learning_rate=0.5e-6,
    beta1=0.95,
    beta2=0.999,
    epsilon=1e-8)
optimizer = opt.minimize(loss)
merged = tf.summary.merge_all()
tf.train.export_meta_graph(filename='unet.meta')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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
        }
    )




#define output snapshot
snapshot_request = BatchRequest()
snapshot_request.add(grad,(output_shape[0]*voxel_size[0],output_shape[1]*voxel_size[1],output_shape[2]*voxel_size[2]))
snapshot_request.add(prediction,(output_shape[0]*voxel_size[0],output_shape[1]*voxel_size[1],output_shape[2]*voxel_size[2]))






#define pipeline
training_pipeline = (
        source+
        gp.RandomLocation()+
        #gp.SimpleAugment() +
        #uses metagraph file 'unet' which contains 3D u-net
        gp.tensorflow.Train('unet',
                            optimizer=optimizer.name,
                            loss=loss.name,
                            save_every=10,
                            inputs={
                                    raw_tf.name : raw,
                                    gt_labels.name : gt
                                    #loss_weights.name : loss_w
                            },
                            outputs={
                                    labels.name : prediction
                            },
                            gradients={
                                    labels.name : grad
                            }
        )+
        Snapshot({
                    raw: 'volumes/raw',
                    gt: 'volumes/gt',
                    grad: 'volumes/gradient',
                    prediction: 'volumes/prediction',
        }, output_filename='batch_{id}.hdf', additional_request=snapshot_request)
)

tf.reset_default_graph()

#define request
request = gp.BatchRequest()
request.add(raw,(input_shape[0]*voxel_size[0],input_shape[1]*voxel_size[1],input_shape[2]*voxel_size[2]))
request.add(gt,(output_shape[0]*voxel_size[0],output_shape[1]*voxel_size[1],output_shape[2]*voxel_size[2]))






gp.set_verbose()

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

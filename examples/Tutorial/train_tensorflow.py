from __future__ import print_function
import gunpowder as gp
from gunpowder import *
from gunpowder.ext import malis
import tensorflow as tf
import mala



############
# Training #
############

# Declare Arrays
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GT')
mask= gp.ArrayKey('mask')
prediction = gp.ArrayKey('prediction')
grad = gp.ArrayKey('gradient')

# define training values
input_shape = (40, 300, 300)
voxel_size = (40, 4, 4)


#define network parameters these will be used to define the feed dict for the network
raw_tf = tf.placeholder(tf.float32, shape=input_shape)
raw_batched = tf.reshape(raw_tf, (1, 1) + input_shape)

unet = mala.networks.unet(raw_batched, 3, 3, [[1, 1, 1], [1, 1, 1], [3, 3, 3]])

#since we want binary predictions we will be using 1 feature map per
output = mala.networks.conv_pass(
    unet,
    kernel_size=1,
    num_fmaps=1,
    num_repetitions=1,
    activation='sigmoid')

#get the correct output size to compare to gt for loss
output_shape_batched = output.get_shape().as_list()
output_shape = output_shape_batched[2:] # strip the batch dimension

#this creates the tensor that holds the predicted binary outputs and the tensor that will be fed the real gt masks
binary = tf.reshape(output, output_shape)
gt_binaries = tf.placeholder(tf.float32, shape=output_shape)

#deine loss to optimize
loss = tf.losses.mean_squared_error(
    gt_binaries,
    binary)

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




# import source
source = gp.Hdf5Source(
    'data_with_mask.hdf',
    {
        raw: 'volumes/raw',
        gt: 'volumes/labels',
        mask: 'volumes/masks'
    }
)



# define output snapshot
snapshot_request = BatchRequest()
snapshot_request.add(grad, (
    output_shape[0] * voxel_size[0], output_shape[1] * voxel_size[1], output_shape[2] * voxel_size[2]))
snapshot_request.add(prediction, (
    output_shape[0] * voxel_size[0], output_shape[1] * voxel_size[1], output_shape[2] * voxel_size[2]))





# define pipeline
training_pipeline = (
        source +
        gp.RandomLocation() +
        # gp.SimpleAugment() +
        gp.PreCache(cache_size=300, num_workers=80) +
        # uses metagraph file 'unet' which contains 3D u-net
        gp.tensorflow.Train('unet',
                            optimizer=optimizer.name,
                            loss=loss.name,
                            save_every=100,
                            checkpoint_dir="./checkpoints/",
                            inputs={
                                raw_tf.name: raw,
                                gt_binaries.name: mask,
                            },
                            outputs={
                                binary.name: prediction
                            },
                            gradients={
                                unet.name: grad
                            }
                            ) +

        Snapshot({
            raw: 'volumes/raw',
            gt: 'volumes/gt',
            grad: 'volumes/gradient',
            prediction: 'volumes/prediction',
        }, output_filename='batch_{id}.hdf', additional_request=snapshot_request, every=100)
)



tf.reset_default_graph()




# define request
request = gp.BatchRequest()
request.add(raw, (input_shape[0] * voxel_size[0], input_shape[1] * voxel_size[1], input_shape[2] * voxel_size[2]))
request.add(gt, (output_shape[0] * voxel_size[0], output_shape[1] * voxel_size[1], output_shape[2] * voxel_size[2]))
request.add(mask, (output_shape[0] * voxel_size[0], output_shape[1] * voxel_size[1], output_shape[2] * voxel_size[2]))


# Execute train
n = 100
with gp.build(training_pipeline) as minibatch_maker:
    for i in range(n):
        minibatch_maker.request_batch(request)

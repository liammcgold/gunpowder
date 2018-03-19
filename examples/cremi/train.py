from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *
from gunpowder.caffe import *
from gunpowder.ext import malis



# this does the actual training
def train():


#define many values needed for training

    #creates some sort of offset, according to documentation "nhood is just the offset vector that the edge corresponds to"
    affinity_neighborhood = malis.mknhood3d()
    #Initializes Cafee solver parameters object with atributes filled below, probably should not mess with these without
    #good reason
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

    #batch request states what is to be contained in the batch that is requested
    request = BatchRequest()
    request.add(ArrayKeys.RAW, (84,268,268))
    request.add(ArrayKeys.GT_LABELS, (56,56,56))
    request.add(ArrayKeys.GT_MASK, (56,56,56))
    request.add(ArrayKeys.GT_IGNORE, (56,56,56))
    request.add(ArrayKeys.GT_AFFINITIES, (56,56,56))

    #import data, normalize it, pick random volumes within it
    data_sources = tuple(
        Hdf5Source(
            'sample_'+s+'_padded_20160501.aligned.filled.cropped.hdf',
            datasets = {
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                ArrayKeys.GT_MASK: 'volumes/labels/mask',
            }
        ) +
        Normalize() +
        RandomLocation()
        #not sure what this line below does, somehow I think it defines how the random location does its sampling
        for s in ['A', 'B', 'C']
    )


    #atrifcat source, not sure whats being imported but after its inmported its randomly sampled volumetrically, normalized,
    #randomly augmented for intesity, elasticity and randomly mirrored and transposed

    artifact_source = (
        Hdf5Source(
            'sample_ABC_padded_20160501.defects.hdf',
            datasets = {
                ArrayKeys.RAW: 'defect_sections/raw',
                ArrayKeys.ALPHA_MASK: 'defect_sections/mask',
            }
        ) +
        RandomLocation(min_masked=0.05, mask_array_key=ArrayKeys.ALPHA_MASK) +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        SimpleAugment(transpose_only_xy=True)
    )





    #Creates snapshots to be saved that contain LOSS_GRADIENT, this will be pulled from upstream provider
    snapshot_request = BatchRequest()
    snapshot_request.add_array_request(ArrayKeys.LOSS_GRADIENT, (56,56,56))



    # creates "directed acyclic graph" DAG with each term being a source or a batch provider, the first temp is the source
    # "data_soruces" and the remaining terms are batch_providers, when they get the request they send it up stream to the
    #  source and eventually return it
    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ExcludeLabels([8094], ignore_mask_erode=12) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], prob_slip=0.05,prob_shift=0.05,max_misalign=25) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.1) +
        ZeroOutConstSections() +
        IntensityScaleShift(2,-1) +
        BalanceAffinityLabels() +
        PreCache(
            cache_size=10,
            num_workers=5) +
        Train(solver_parameters, use_gpu=0) +
        Snapshot(every=10, output_filename='batch_{id}.hdf', additional_request=snapshot_request)
    )

    n = 10
    print("Training for", n, "iterations")


    #batch request is a training itteration, build takes the function created for batch provider tree and initializes things
    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(request)

    print("Finished")

if __name__ == "__main__":
    train()


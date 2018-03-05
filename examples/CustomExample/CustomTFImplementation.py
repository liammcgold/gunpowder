from __future__ import print_function

#converted to tensorflow implemetation



import math
import numpy as np
import random

from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.ext import malis
from gunpowder.contrib import ZeroOutConstSections


# this does the actual training
def train():



    # define many values needed for training

    # creates some sort of offset, according to documentation "nhood is just the offset vector that the edge corresponds to"
    # Creates matrix of values in neighborhood
    # 3 numnbers with (x,y,x, and a 4th number that represents which connection you want between it)
    # this simplifies the data structure from 6 numbers to 4 because an ofsett of 001 may be represented by a single number
    # affinity neighborhood simply initializes this data structure making it easier to access the afinity graph.
    affinity_neighborhood = malis.mknhood3d()


    #find this name for use
    meta_graph_filename="null"
    #Optimizer for TF
    optimizer="ADAM"
    #


    # batch request states what is to be contained in the batch that is requested
    request = BatchRequest()
    request.add_array_request(ArrayKeys.RAW, (84, 268, 268))
    request.add_array_request(ArrayKeys.GT_LABELS, (56, 56, 56))
    request.add_array_request(ArrayKeys.GT_MASK, (56, 56, 56))
    request.add_array_request(ArrayKeys.GT_IGNORE, (56, 56, 56))
    request.add_array_request(ArrayKeys.GT_AFFINITIES, (56, 56, 56))

    # import data, normalize it, pick random volumes within it
    data_sources = tuple(
        Hdf5Source(
            'sample_' + s + '_padded_20160501.aligned.filled.cropped.hdf',
            datasets={
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                ArrayKeys.GT_MASK: 'volumes/labels/mask',
            }
        ) +
        Normalize() +
        RandomLocation()
        # not sure what this line below does, somehow I think it defines how the random location does its sampling
        for s in ['A', 'B', 'C']
    )


    # Creates snapshots to be saved that contain LOSS_GRADIENT, and current graph state this will be pulled from upstream provider
    snapshot_request = BatchRequest()
    snapshot_request.add_array_request(ArrayKeys.LOSS_GRADIENT, (56, 56, 56))

    # creates "directed acyclic graph" DAG with each term being a source or a batch provider, the first term is the source
    # "data_soruces" and the remaining terms are batch_providers, when they get the request they send it up stream to the
    #  source and eventually return it
    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ExcludeLabels([8094], ignore_mask_erode=12) +
        ElasticAugment([4, 40, 40], [0, 2, 2], [0, math.pi / 2.0], prob_slip=0.05, prob_shift=0.05, max_misalign=25) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ZeroOutConstSections() +
        IntensityScaleShift(2, -1) +
        # changed from "BalanceAffinityLabels" because that function does not exist
        BalanceLabels() +
        PreCache(
            cache_size=10,
            num_workers=5) +

        Train(meta_graph_filename=meta_graph_filename,optimizer=optimizer) +
        Snapshot(every=10, output_filename='batch_{id}.hdf', additional_request=snapshot_request)
    )

    n = 10
    print("Training for", n, "iterations")

    # batch request is a training itteration, build takes the function created for batch provider tree and initializes things
    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()

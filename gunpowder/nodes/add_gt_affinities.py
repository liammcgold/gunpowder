import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.array import Array

logger = logging.getLogger(__name__)

class AddGtAffinities(BatchFilter):
    '''Add an array with affinities for a given label array and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood(list of offsets): List of offsets for the 
            affinities to consider for each voxel.

        gt_labels(:class:``ArrayKey``): The array to read the labels from.

        gt_affinities(:class:``ArrayKey``): The array to generate containing
            the affinities.

        gt_labels_mask(:class:``ArrayKey``, optional): The array to use as a
            mask for ``gt_labels``. Affinities connecting at least one masked
            out label will be masked out in ``gt_affinities_mask``. If not
            given, ``gt_affinities_mask`` will contain ones everywhere (if
            requested).

        gt_unlabelled(:class:``ArrayKey``, optional): A binary array to
            indicate unlabelled areas with 0. Affinities from labelled to
            unlabelled voxels are set to 0, affinities between unlabelled voxels
            are masked out (they will not be used for training).

        gt_affinities_mask(:class:``ArrayKey``, optional): The array to
            generate containing the affinitiy mask, as derived from parameter
            ``gt_labels_mask``.
    '''

    def __init__(
            self,
            affinity_neighborhood,
            gt_labels,
            gt_affinities,
            gt_labels_mask=None,
            gt_unlabelled=None,
            gt_affinities_mask=None):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.gt_labels = gt_labels
        self.gt_unlabelled = gt_unlabelled
        self.gt_labels_mask = gt_labels_mask
        self.gt_affinities = gt_affinities
        self.gt_affinities_mask = gt_affinities_mask

    def setup(self):

        assert self.gt_labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddGtAffinities"%self.gt_labels)

        voxel_size = self.spec[self.gt_labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.gt_labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = np.float32

        self.provides(self.gt_affinities, spec)
        if self.gt_affinities_mask:
            self.provides(self.gt_affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.gt_labels_mask:
            assert (
                request[self.gt_labels].roi ==
                request[self.gt_labels_mask].roi),(
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same."%(
                    request[self.gt_labels].roi,
                    request[self.gt_labels_mask].roi))

        if self.gt_unlabelled:
            assert (
                request[self.gt_labels].roi ==
                request[self.gt_unlabelled].roi),(
                "requested GT label roi %s and GT unlabelled mask roi %s are not "
                "the same."%(
                    request[self.gt_labels].roi,
                    request[self.gt_unlabelled].roi))

        gt_labels_roi = request[self.gt_labels].roi
        logger.debug("downstream %s request: "%self.gt_labels + str(gt_labels_roi))

        # grow labels ROI to accomodate padding
        gt_labels_roi = gt_labels_roi.grow(-self.padding_neg, self.padding_pos)
        request[self.gt_labels].roi = gt_labels_roi

        # same for label mask
        if self.gt_labels_mask:
            request[self.gt_labels_mask].roi = gt_labels_roi.copy()
        # and unlabelled mask
        if self.gt_unlabelled:
            request[self.gt_unlabelled].roi = gt_labels_roi.copy()

        logger.debug("upstream %s request: "%self.gt_labels + str(gt_labels_roi))

    def process(self, batch, request):

        gt_labels_roi = request[self.gt_labels].roi

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = malis.seg_to_affgraph(
                batch.arrays[self.gt_labels].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)


        # crop affinities to original label ROI
        offset = gt_labels_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = gt_labels_roi.shift(shift)
        crop_roi /= self.spec[self.gt_labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        gt_affinities = gt_affinities[(slice(None),)+crop]

        spec = self.spec[self.gt_affinities].copy()
        spec.roi = gt_labels_roi
        batch.arrays[self.gt_affinities] = Array(gt_affinities, spec)

        if self.gt_affinities_mask and self.gt_affinities_mask in request:

            if self.gt_labels_mask:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                gt_affinities_mask = malis.seg_to_affgraph(
                    batch.arrays[self.gt_labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood)
                gt_affinities_mask = gt_affinities_mask[(slice(None),)+crop]

            else:

                gt_affinities_mask = np.ones_like(gt_affinities)

            if self.gt_unlabelled:

                # 1 for all affinities between unlabelled voxels
                unlabelled = (1 - batch.arrays[self.gt_unlabelled].data)
                unlabelled_mask = malis.seg_to_affgraph(
                    unlabelled.astype(np.int32),
                    self.affinity_neighborhood)
                unlabelled_mask = unlabelled_mask[(slice(None),)+crop]

                # 0 for all affinities between unlabelled voxels
                unlabelled_mask = (1 - unlabelled_mask)

                # combine with mask
                gt_affinities_mask = gt_affinities_mask*unlabelled_mask

            gt_affinities_mask = gt_affinities_mask.astype(np.float32)
            batch.arrays[self.gt_affinities_mask] = Array(gt_affinities_mask, spec)

        else:

            if self.gt_labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        batch.arrays[self.gt_labels] = batch.arrays[self.gt_labels].crop(gt_labels_roi)

        # same for label mask
        if self.gt_labels_mask:
            batch.arrays[self.gt_labels_mask] = batch.arrays[self.gt_labels_mask].crop(gt_labels_roi)
        # and unlabelled mask
        if self.gt_unlabelled:
            batch.arrays[self.gt_unlabelled] = batch.arrays[self.gt_unlabelled].crop(gt_labels_roi)

        batch.affinity_neighborhood = self.affinity_neighborhood

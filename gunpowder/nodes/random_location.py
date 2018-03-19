import copy
import logging
from random import random, randint, choice

import numpy as np
from skimage.transform import integral_image, integrate
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


#Randomly samples volumetrically
class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream
    provider.

    The random location is chosen such that the batch request roi lies entirely
    inside the provider's roi.

    If `min_masked` and `mask` are set, only batches are returned that have at
    least the given ratio of masked-in voxels. This is in general faster than
    using the ``Reject`` node, at the expense of storing an integral array of
    the complete mask.

    If 'ensure_nonempty' is set to a :class:``PointsKey``, only batches are
    returned that have at least one point of this point collection within the
    requested ROI.

    Args:

        min_masked(float, optional): If non-zero, require that the random
            sample contains at least that ratio of masked-in voxels.

        mask(:class:``ArrayKey``): The array to use for mask checks.

        ensure_nonempty(:class:``PointsKey``, optional): Ensures that when
            finding a random location, a request for ``ensure_nonempty`` will
            contain at least one point. This does only work if all upstream
            nodes are deterministic (e.g., there is no
            :class:``RandomProvider`` upstream).
    '''

    def __init__(self, min_masked=0, mask=None, ensure_nonempty=None):

        self.min_masked = min_masked
        self.mask = mask
        self.mask_spec = None
        self.ensure_nonempty = ensure_nonempty


    def setup(self):

        upstream = self.get_upstream_provider()
        self.upstream_spec = upstream.spec

        if self.mask and self.min_masked > 0:

            assert self.mask in self.upstream_spec, (
                    "Upstream provider does not have %s"%self.mask)
            self.mask_spec = self.upstream_spec.array_specs[self.mask]

            logger.info("requesting complete mask...")

            mask_request = BatchRequest({self.mask: self.mask_spec})
            mask_batch = upstream.request_batch(mask_request)

            logger.info("allocating mask integral array...")

            mask_data = mask_batch.arrays[self.mask].data
            mask_integral_dtype = np.uint64
            logger.debug("mask size is " + str(mask_data.size))
            if mask_data.size < 2**32:
                mask_integral_dtype = np.uint32
            if mask_data.size < 2**16:
                mask_integral_dtype = np.uint16
            logger.debug("chose %s as integral array dtype"%mask_integral_dtype)

            self.mask_integral = np.array(mask_data>0, dtype=mask_integral_dtype)
            self.mask_integral = integral_image(self.mask_integral)

        if self.ensure_nonempty:

            assert self.ensure_nonempty in self.upstream_spec, (
                "Upstream provider does not have %s"%self.ensure_nonempty)
            points_spec = self.upstream_spec.points_specs[self.ensure_nonempty]

            logger.info("requesting all %s points...", self.ensure_nonempty)

            points_request = BatchRequest({self.ensure_nonempty: points_spec})
            points_batch = upstream.request_batch(points_request)

            self.points = points_batch.points[self.ensure_nonempty]

            logger.info("retrieved %d points", len(self.points.data))

        # clear bounding boxes of all provided arrays and points -- 
        # RandomLocation does not have limits (offsets are ignored)
        for identifier, spec in self.spec.items():
            spec.roi.set_shape(None)
            self.updates(identifier, spec)

    def prepare(self, request):

        logger.debug("request: %s", request.array_specs)
        logger.debug("my spec: %s", self.spec)

        shift_roi = self.__get_possible_shifts(request)

        if request.array_specs.keys():

            lcm_voxel_size = self.spec.get_lcm_voxel_size(
                    request.array_specs.keys())
            lcm_shift_roi = shift_roi/lcm_voxel_size

            logger.debug(
                "restricting random locations to multiples of voxel size %s",
                lcm_voxel_size)

        else:

            lcm_voxel_size = None
            lcm_shift_roi = shift_roi

        random_shift = self.__select_random_shift(
            request,
            lcm_shift_roi,
            lcm_voxel_size)

        # shift request ROIs
        self.random_shift = random_shift
        for specs_type in [request.array_specs, request.points_specs]:
            for (key, spec) in specs_type.items():
                roi = spec.roi.shift(random_shift)
                logger.debug("new %s ROI: %s"%(key, roi))
                specs_type[key].roi = roi

    def process(self, batch, request):

        # reset ROIs to request
        for (array_key, spec) in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi
        for (points_key, spec) in request.points_specs.items():
            batch.points[points_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for points_key in request.points_specs.keys():
            for point_id, point in batch.points[points_key].data.items():
                batch.points[points_key].data[point_id].location -= self.random_shift

    def __get_possible_shifts(self, request):

        total_shift_roi = None

        for identifier, spec in request.items():

            request_roi = spec.roi
            provided_roi = self.upstream_spec[identifier].roi

            shift_roi = provided_roi.shift(-request_roi.get_begin()).grow((0,0,0),-request_roi.get_shape())

            if total_shift_roi is None:
                total_shift_roi = shift_roi
            else:
                total_shift_roi = total_shift_roi.intersect(shift_roi)

        logger.debug("valid shifts for request in " + str(total_shift_roi))

        assert not total_shift_roi.unbounded(), (
            "Can not pick a random location, intersection of upstream ROIs is "
            "unbounded.")
        assert total_shift_roi.size() > 0, (
            "Can not satisfy batch request, no location covers all requested "
            "ROIs.")

        return total_shift_roi

    def __select_random_shift(self, request, lcm_shift_roi, lcm_voxel_size):

        while True:

            if self.ensure_nonempty:
                random_shift = self.__select_random_location_with_points(
                    request,
                    lcm_shift_roi,
                    lcm_voxel_size)
            else:
                random_shift = self.__select_random_location(
                    lcm_shift_roi,
                    lcm_voxel_size)

            logger.debug("random shift: " + str(random_shift))

            if self.__is_min_masked(random_shift, request):
                return random_shift
            else:
                logger.debug("reject random shift, min_masked not exceeded")

    def __is_min_masked(self, random_shift, request):

            if not self.mask or self.min_masked == 0:
                return True

            # get randomly chosen mask ROI
            request_mask_roi = request.array_specs[self.mask].roi
            request_mask_roi = request_mask_roi.shift(random_shift)

            # get coordinates inside mask array
            mask_voxel_size = self.spec[self.mask].voxel_size
            request_mask_roi_in_array = request_mask_roi/mask_voxel_size
            request_mask_roi_in_array -= self.mask_spec.roi.get_offset()/mask_voxel_size

            # get number of masked-in voxels
            num_masked_in = integrate(
                self.mask_integral,
                [request_mask_roi_in_array.get_begin()],
                [request_mask_roi_in_array.get_end()-(1,)*self.mask_integral.ndim]
            )[0]

            mask_ratio = float(num_masked_in)/request_mask_roi_in_array.size()
            logger.debug("mask ratio is %f", mask_ratio)

            return mask_ratio >= self.min_masked

    def __select_random_location_with_points(
            self,
            request,
            lcm_shift_roi,
            lcm_voxel_size):

        while True:

            # pick a random point
            point_id = choice(self.points.data.keys())
            point = self.points.data[point_id]

            # get all possible shifts of the request ROI that would contain
            # this point
            request_points_roi = request[self.ensure_nonempty].roi
            request_points_shape = request_points_roi.get_shape()
            point_shift_roi = Roi(
                point.location - request_points_shape,
                request_points_shape)
            point_shift_roi = point_shift_roi.shift(
                -request_points_roi.get_offset())

            # align with lcm_voxel_size
            if lcm_voxel_size is not None:
                lcm_point_shift_roi = point_shift_roi/lcm_voxel_size
            else:
                lcm_point_shift_roi = point_shift_roi

            # intersect with total shift ROI
            if not lcm_point_shift_roi.intersects(lcm_shift_roi):
                logger.debug(
                    "reject random shift, random point %s shift ROI %s does "
                    "not intersect total shift ROI %s", point.location,
                    lcm_point_shift_roi, lcm_shift_roi)
                continue
            lcm_point_shift_roi = lcm_point_shift_roi.intersect(lcm_shift_roi)

            random_shift = self.__select_random_location(
                lcm_point_shift_roi,
                lcm_voxel_size)

            # count all points inside the shifted ROI
            points_request = BatchRequest()
            points_request[self.ensure_nonempty] = PointsSpec(
                    roi=request_points_roi.shift(random_shift))
            points_batch = self.get_upstream_provider().request_batch(points_request)

            point_ids = points_batch.points[self.ensure_nonempty].data.keys()
            assert point_id in point_ids, (
                "Requested batch to contain point %s, but got points "
                "%s"%(point_id, point_ids))
            num_points = len(point_ids)

            # accept this shift with p=1/num_points
            #
            # This is to compensate the bias introduced by close-by points.
            accept = random() <= 1.0/num_points
            if accept:
                return random_shift

    def __select_random_location(self, lcm_shift_roi, lcm_voxel_size):

        # select a random point inside ROI
        random_shift = Coordinate(
                randint(int(begin), int(end-1))
                for begin, end in zip(lcm_shift_roi.get_begin(), lcm_shift_roi.get_end()))

        if lcm_voxel_size is not None:
            random_shift *= lcm_voxel_size

        return random_shift

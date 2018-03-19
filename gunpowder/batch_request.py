import copy
from .points import PointsKey
from .points_spec import PointsSpec
from .provider_spec import ProviderSpec
from .roi import Roi
from .array import ArrayKey
from .array_spec import ArraySpec

class BatchRequest(ProviderSpec):
    '''A collection of (possibly partial) :class:`ArraySpec` and
    :class:`PointsSpec` forming a request.

    For usage, see the documentation of :class:`ProviderSpec`.
    '''

    def add(self, identifier, shape, voxel_size=None):
        '''Convenience method to add an array or point spec by providing only
        the shape of a ROI (in world units).

        A ROI with zero-offset will be generated. If more than one request is
        added, the ROIs with smaller shapes will be shifted to be centered in
        the largest one.


        #ROI is region of interest, a rectangular region
        def add_volume_request(self, volume_type, shape):
        Args:
            identifier: A :class:`ArrayKey` or `PointsKey` instance to refer to the output.


            shape: A tuple containing the shape of the desired roi

            voxel_size: A tuple contening the voxel sizes for each corresponding dimension
        '''

        if isinstance(identifier, ArrayKey):
            spec = ArraySpec()
        elif isinstance(identifier, PointsKey):
            spec = PointsSpec()
        else:
            raise RuntimeError("Only ArrayKey or PointsKey can be added.")

        spec.roi = Roi((0,)*len(shape), shape)

        if voxel_size is not None:
            spec.voxel_size = voxel_size

        self[identifier] = spec
        self.__center_rois()

    def copy(self):
        '''Create a copy of this request.'''
        return copy.deepcopy(self)

    def __center_rois(self):
        '''Ensure that all ROIs are centered around the same location.'''

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.get_center()

        for specs_type in [self.array_specs, self.points_specs]:
            for identifier in specs_type:
                roi = specs_type[identifier].roi
                specs_type[identifier].roi = roi.shift(center - roi.get_center())

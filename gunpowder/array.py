from .freezable import Freezable
from copy import deepcopy
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ArrayType(Freezable):
    '''Describes general properties of an array type.

    Args:

        identifier (string):
            A human readable identifier for this array type. Will be used as a
            static attribute in :class:`ArrayTypes`. Should be upper case (like
            ``RAW``, ``GT_LABELS``).
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class ArrayTypes:
    '''An expandable collection of array types, which initially contains:

        =================================  ====================================================
        identifier                         purpose
        =================================  ====================================================
        ``RAW``                            Raw intensity arrays.
        ``ALPHA_MASK``                     Alpha mask for blending
                                           raw arrays
                                           (used in :class:`DefectAugment`).
        ``GT_LABELS``                      Ground-truth object IDs.
        ``GT_MASK``                        Binary mask (1-use, 0-don't use) on ground-truth. No
                                           assumptions about masked out area (i.e., end of
                                           ground-truth).
        ``GT_IGNORE``                      Binary mask (1-use, 0-don't use) on ground-truth.
                                           Assumes that transition between 0 and 1 lies on an
                                           object boundary.
        ``GT_AFFINITIES``                  Ground-truth affinities.
        ``GT_AFFINITIES_MASK``             Binary mask (1-use, 0-don't use) on ground-truth. No
                                           assumptions about masked out area (i.e., end of
                                           ground-truth).
        ``PRED_AFFINITIES``                Predicted affinities.
        ``LOSS_SCALE``                     Used for element-wise multiplication with loss for
                                           training.
        ``LOSS_GRADIENT``                  Gradient of the training loss.
        ``GT_BM_PRESYN``                   Ground truth of binary map for presynaptic locations
        ``GT_BM_PRESYN``                   Ground truth of binary map for postsynaptic locations
        ``GT_MASK_EXCLUSIVEZONE_PRESYN``   ExculsiveZone binary mask (1-use,
                                           0-don't use) around presyn locations
        ``GT_MASK_EXCLUSIVEZONE_POSTSYN``  ExculsiveZone binary mask (1-use,
                                           0-don't use) around postsyn locations
        ``PRED_BM_PRESYN``                 Predicted presynaptic locations
        ``PRED_BM_POSTSYN``                Predicted postsynaptic locations
        =================================  ====================================================

    New array types can be added with :func:`register_array_type`.
    '''
    pass

def register_array_type(identifier):
    '''Register a new array type.

    For example, the following call::

            register_array_type('IDENTIFIER')

    will create a new array type available as ``ArrayTypes.IDENTIFIER``.
    ``ArrayTypes.IDENTIFIER`` can then be used in dictionaries, as it is done
    in :class:`BatchRequest` and :class:`ProviderSpec`, for example.
    '''
    array_type = ArrayType(identifier)
    logger.debug("Registering array type " + str(array_type))
    setattr(ArrayTypes, array_type.identifier, array_type)

register_array_type('RAW')
register_array_type('ALPHA_MASK')
register_array_type('GT_LABELS')
register_array_type('GT_AFFINITIES')
register_array_type('GT_AFFINITIES_MASK')
register_array_type('GT_MASK')
register_array_type('GT_IGNORE')
register_array_type('PRED_AFFINITIES')
register_array_type('LOSS_SCALE')
register_array_type('LOSS_GRADIENT')
register_array_type('MALIS_COMP_LABEL')

register_array_type('GT_BM_PRESYN')
register_array_type('GT_BM_POSTSYN')
register_array_type('GT_MASK_EXCLUSIVEZONE_PRESYN')
register_array_type('GT_MASK_EXCLUSIVEZONE_POSTSYN')
register_array_type('PRED_BM_PRESYN')
register_array_type('PRED_BM_POSTSYN')
register_array_type('LOSS_GRADIENT_PRESYN')
register_array_type('LOSS_GRADIENT_POSTSYN')

register_array_type('LOSS_SCALE_BM_PRESYN')
register_array_type('LOSS_SCALE_BM_POSTSYN')


class Array(Freezable):
    '''A numpy array with a specification describing the data.

    Args:

        data (array-like): The data to be stored in the array. Will be
            converted to an numpy array, if necessary.

        spec (:class:`ArraySpec`, optional): A spec describing the data.
    '''

    def __init__(self, data, spec=None, attrs=None):

        self.spec = deepcopy(spec)
        self.data = np.asarray(data)
        self.attrs = attrs

        if attrs is None:
            self.attrs = {}

        if spec is not None:
            for d in range(len(spec.voxel_size)):
                assert spec.voxel_size[d]*data.shape[-spec.roi.dims()+d] == spec.roi.get_shape()[d], \
                        "ROI %s does not align with voxel size %s * data shape %s"%(spec.roi, spec.voxel_size, data.shape)

        self.freeze()

    def crop(self, roi, copy=True):
        '''Create a cropped copy of this Array.

        Args:

            roi(:class:``Roi``): ROI in world units to crop to.

            copy(bool): Make a copy of the data (default).
        '''

        assert self.spec.roi.contains(roi), "Requested crop ROI (%s) doesn't fit in array (%s)"\
        %(roi, self.spec.roi)

        voxel_size = self.spec.voxel_size
        data_roi = (roi - self.spec.roi.get_offset())/voxel_size
        slices = data_roi.get_bounding_box()

        while len(slices) < len(self.data.shape):
            slices = (slice(None),) + slices

        data = self.data[slices]
        if copy:
            data = np.array(data)

        spec = deepcopy(self.spec)
        attrs = deepcopy(self.attrs)
        spec.roi = deepcopy(roi)
        return Array(data, spec, attrs)
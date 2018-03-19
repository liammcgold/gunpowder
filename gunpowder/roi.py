import copy
from .coordinate import Coordinate
from .freezable import Freezable
import numbers

class Roi(Freezable):
    '''A rectangular region of interest, defined by an offset and a shape.

    Args:

        offset (array-like of int, optional): The starting point (inclusive) of
            the ROI. Can be `None` (default) if the ROI only characterizes a
            shape.

        shape (array-like): The shape of the ROI. Entries can be `None` to
            undicate unbounded dimensions.
    '''

    def __init__(self, offset=None, shape=None):

        self.__offset = None
        self.__shape = None
        self.freeze()

        self.set_shape(shape)
        if offset is not None:
            self.set_offset(offset)

    def set_offset(self, offset):

        self.__offset = Coordinate(offset)
        self.__consolidate_offset()

    def set_shape(self, shape):
        '''Set the shape of this ROI.

        Args:

            shape (tuple or None): The new shape. Entries can be `None` to
                indicate unboundedness. If `None` is passed instead of a tuple,
                all dimensions are set to `None`, if the number of dimensions
                can be inferred from an existing offset or previous shape.
        '''

        if shape is None:

            if self.__shape is not None:

                dims = self.__shape.dims()

            else:

                assert self.__offset is not None, (
                    "Can not infer dimension of ROI (there is no offset or "
                    "previous shape). Call set_shape with a tuple.")

                dims = self.__offset.dims()

            self.__shape = Coordinate((None,)*dims)

        else:

            self.__shape = Coordinate(shape)

        self.__consolidate_offset()

    def __consolidate_offset(self):
        '''Ensure that offsets for unbound dimensions are None.'''

        if self.__offset is not None:

            assert self.__offset.dims() == self.__shape.dims(), (
                "offset dimension %d != shape dimension %d"%(
                    self.__offset.dims(),
                    self.__shape.dims()))

            self.__offset = Coordinate((
                o
                if s is not None else None
                for o, s in zip(self.__offset, self.__shape)))

    def get_offset(self):
        return self.__offset

    def get_begin(self):
        '''Smallest coordinate inside ROI.'''
        return self.__offset

    def get_end(self):
        '''Smallest coordinate which is component-wise larger than any inside ROI.'''
        if not self.__shape:
            return self.__offset

        return self.__offset + self.__shape

    def get_shape(self):
        return self.__shape

    def get_center(self):

        return self.__offset + self.__shape/2

    def get_bounding_box(self):

        if self.__offset is None:
            return None

        return tuple(
                slice(
                    int(self.__offset[d])
                    if self.__shape[d] is not None
                    else None,
                    int(self.__offset[d] + self.__shape[d])
                    if self.__shape[d] is not None
                    else None)
                for d in range(self.dims())
        )

    def dims(self):

        if self.__shape is None:
            return 0
        return self.__shape.dims()

    def size(self):

        if self.unbounded():
            return None

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def empty(self):

        return self.size() == 0

    def unbounded(self):

        return None in self.__shape

    def contains(self, other):

        if isinstance(other, Roi):

            if other.empty():
                return True

            return (
                self.contains(other.get_begin())
                and
                self.contains(other.get_end() - (1,)*other.dims()))

        elif isinstance(other, Coordinate):

            return all([
                (b is None or p is not None and p >= b)
                and
                (e is None or p is not None and p < e)
                for p, b, e in zip(other, self.get_begin(), self.get_end() )
            ])

        else:

            raise RuntimeError("contains() can only be applied to Roi and Coordinate")

    def intersects(self, other):

        assert self.dims() == other.dims()

        if self.empty() or other.empty():
            return False

        # separated if at least one dimension is separated
        separated = any([
            # a dimension is separated if:
            # none of the shapes is unbounded
            (None not in [b1, b2, e1, e2])
            and
            (
                # either b1 starts after e2
                (b1 >= e2)
                or
                # or b2 starts after e1
                (b2 >= e1)
            )
            for b1, b2, e1, e2 in zip(
                self.get_begin(),
                other.get_begin(),
                self.get_end(),
                other.get_end())
        ])

        return not separated

    def intersect(self, other):

        if not self.intersects(other):
            return Roi(shape=(0,)*self.dims()) # empty ROI

        begin = Coordinate((
            max(b1, b2) # max(x, None) is x, so this does the right thing
            for b1, b2 in zip(self.get_begin(), other.get_begin())
        ))
        end = Coordinate((
            min(e1, e2) # min(x, None) is min, but we want x
            if e1 is not None and e2 is not None
            else max(e1, e2) # so we just take the other value or None if both
                             # are None
            for e1, e2 in zip(self.get_end(), other.get_end())
        ))

        return Roi(begin, end - begin)

    def union(self, other):

        begin = Coordinate((
            min(b1, b2) # min(x, None) is None, so this does the right thing
            for b1, b2 in zip(self.get_begin(), other.get_begin())
        ))
        end = Coordinate((
            max(e1, e2) # max(x, None) is x, but we want None
            if e1 is not None and e2 is not None
            else None
            for e1, e2 in zip(self.get_end(), other.get_end())
        ))

        return Roi(begin, end - begin)

    def shift(self, by):

        return Roi(self.__offset + by, self.__shape)

    def grow(self, amount_neg, amount_pos):
        '''Grow a ROI by the given amounts in each direction:

        amount_neg: Coordinate or None

            Amount (per dimension) to grow into the negative direction.

        amount_pos: Coordinate or None

            Amount (per dimension) to grow into the positive direction.
        '''

        if amount_neg is None:
            amount_neg = Coordinate((0,)*self.dims())
        if amount_pos is None:
            amount_pos = Coordinate((0,)*self.dims())

        assert len(amount_neg) == self.dims()
        assert len(amount_pos) == self.dims()

        offset = self.__offset - amount_neg
        shape = self.__shape + amount_neg + amount_pos

        return Roi(offset, shape)

    def copy(self):
        '''Create a copy of this ROI.'''
        return copy.deepcopy(self)

    def __add__(self, other):

        assert isinstance(other, tuple), "can only add Coordinate or tuples to Roi"
        return self.shift(other)

    def __sub__(self, other):

        assert isinstance(other, Coordinate), "can only subtract Coordinate from Roi"
        return self.shift(-other)

    def __mul__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only multiply with a number or tuple of numbers"
        return Roi(self.__offset*other, self.__shape*other)

    def __div__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset/other, self.__shape/other)

    def __truediv__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset/other, self.__shape/other)

    def __floordiv__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset//other, self.__shape//other)

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        if self.empty():
            return "[empty ROI]"
        return str(self.get_begin()) + "--" + str(self.get_end()) + " [" + "x".join(str(a) for a in self.__shape) + "]"

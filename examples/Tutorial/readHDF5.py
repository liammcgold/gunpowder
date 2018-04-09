import h5py
import code


a = h5py.File('data.hdf')
raw = a['volumes']['raw']


code.interact(local=locals())






from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *
from gunpowder.caffe import *
from gunpowder.ext import malis


affinity_neighborhood = malis.mknhood3d()

print(affinity_neighborhood)
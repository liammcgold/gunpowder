
from __future__ import print_function


import numpy as np
import random
#USE  kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')
import gunpowder as gp
from gunpowder import *
from gunpowder.ext import malis
import tensorflow as tf




                                                                ##############
                                                                # Prediction #
                                                                ##############

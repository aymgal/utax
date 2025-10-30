__author__ = 'aymgal'

import unittest
import os
import numpy as np
import numpy.testing as npt

import jax
jax.config.update("jax_enable_x64", True)

import utax
from utax.interpolation import *


class TestBilinearInterpolator(unittest.TestCase):

    def setup(self):
        utax_path = os.path.dirname(utax.__path__[0])
        data_path = os.path.join(utax_path, 'test', 'data')

        # load some test images obtained using from pysparse (sparse2d) from PySAP
        self.image = np.load(os.path.join(data_path, 'galaxy_image.npy'))
        # reduce size for faster computations
        nx, ny = 20, 20
        image = image[30:30+ny, 30:30+nx]
        
        # create a coordinate grid
        self.pix_scl = 0.08
        self.x_coord = np.arange(-3., +3., nx) * self.pix_scl
        self.x_coord = np.arange(-3., +3., ny) * self.pix_scl
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)
        

    # TODO: add tests

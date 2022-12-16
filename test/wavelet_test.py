__author__ = 'aymgal'


import pytest
import os
import numpy as np
import numpy.testing as npt
# from scipy import signal, ndimage
# import pysap

import utax
from utax.wavelet import *



class TestWaveletTransform(object):


    def setup(self):
        # load some test images
        utax_path = os.path.dirname(utax.__path__[0])
        data_path = os.path.join(utax_path, 'test', 'data')
        self.image = np.load(os.path.join(data_path, 'galaxy_image.npy'))
        self.coeffs_st_gen1 = np.load(os.path.join(data_path, 'galaxy_starlet_coeffs_gen1_pysparse.npy'))
        self.coeffs_st_gen2 = np.load(os.path.join(data_path, 'galaxy_starlet_coeffs_gen2_pysparse.npy'))
        self.n_scales = self.coeffs_st_gen1.shape[0]-1

    def test_decomposition(self):
        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=False)
        coeffs = starlet.decompose(self.image)
        npt.assert_almost_equal(coeffs, self.coeffs_st_gen1, decimal=6)

        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=True)
        coeffs = starlet.decompose(self.image)
        npt.assert_almost_equal(coeffs, self.coeffs_st_gen2, decimal=6)

    def test_reconstruction(self):
        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=False)
        image = starlet.reconstruct(self.coeffs_st_gen1)
        npt.assert_almost_equal(image, self.image, decimal=6)

        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=True)
        image = starlet.reconstruct(self.coeffs_st_gen2)
        npt.assert_almost_equal(image, self.image, decimal=6)

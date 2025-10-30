__author__ = 'aymgal'


import unittest
import os
import numpy as np
import numpy.testing as npt
from test.convolution_test import gaussian_kernel

from jax import config
config.update("jax_enable_x64", True)

import utax
from utax.wavelet import *


class TestStarletTransform(unittest.TestCase):

    def setup_method(self):
        utax_path = os.path.dirname(utax.__path__[0])
        data_path = os.path.join(utax_path, 'test', 'data')

        # load some test images obtained using from pysparse (sparse2d) from PySAP
        self.image = np.load(os.path.join(data_path, 'galaxy_image.npy'))
        self.vec1d = gaussian_kernel(odd=True, sigma=30., truncate=5., ndim=1)
        self.coeffs_st_gen1 = np.load(os.path.join(data_path, 'galaxy_starlet_coeffs_gen1_pysparse.npy'))
        self.coeffs_st_gen2 = np.load(os.path.join(data_path, 'galaxy_starlet_coeffs_gen2_pysparse.npy'))
        self.n_scales = self.coeffs_st_gen1.shape[0]-1

    def test_decomposition(self):
        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=False)
        coeffs = starlet.decompose(self.image)
        npt.assert_almost_equal(coeffs, self.coeffs_st_gen1, decimal=6)

        # 2nd generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=True)
        coeffs = starlet.decompose(self.image)
        npt.assert_almost_equal(coeffs, self.coeffs_st_gen2, decimal=6)

    def test_decomposition_reconstrcution1D(self):
        # 1st generation starlet, 1 Dimension
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=False, dim=1)
        coeffs = starlet.decompose(self.vec1d)
        reconstruction = starlet.reconstruct(coeffs)
        npt.assert_almost_equal(self.vec1d, reconstruction, decimal=6)

    def test_reconstruction(self):
        # 1st generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=False)
        image = starlet.reconstruct(self.coeffs_st_gen1)
        npt.assert_almost_equal(image, self.image, decimal=6)

        # 2nd generation starlet
        starlet = WaveletTransform(self.n_scales, wavelet_type='starlet', second_gen=True)
        image = starlet.reconstruct(self.coeffs_st_gen2)
        npt.assert_almost_equal(image, self.image, decimal=6)

    def test_scale_norms(self):
        starlet = WaveletTransform(self.n_scales)
        norms = starlet.scale_norms
        assert norms.size == self.n_scales + 1
        # check that values are decreasing (a Dirac impulse has more power in high frequencies)
        assert np.all(norms[:-1] > norms[1:])


class TestBLWTransform(unittest.TestCase):

    def setUp(self):
        utax_path = os.path.dirname(utax.__path__[0])
        data_path = os.path.join(utax_path, 'test', 'data')

        # load some test images obtained using from pysparse (sparse2d) from PySAP
        self.image = np.load(os.path.join(data_path, 'galaxy_image.npy'))
        self.coeffs_bl1 = np.load(os.path.join(data_path, 'galaxy_battle-lemarie-1_coeffs_gen1_pysparse.npy'))
        self.n_scales = self.coeffs_bl1.shape[0]-1

    def test_decomposition(self):
        starlet = WaveletTransform(self.n_scales, wavelet_type='battle-lemarie-1', second_gen=False)
        coeffs = starlet.decompose(self.image)
        npt.assert_almost_equal(coeffs, self.coeffs_bl1, decimal=6)

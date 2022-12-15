__author__ = 'aymgal'

import pytest
import numpy as np
import numpy.testing as npt
from scipy import signal

from utax.convolution import *


def gaussian_kernel(odd=True, sigma=1., truncate=5., ndim=1):
    if sigma <= 0:
        return np.ones(1) if ndim == 1 else np.ones((1, 1))

    radius = int(np.ceil(2 * truncate * sigma)) // 2
    npix = radius * 2.
    if odd is True:
        npix += 1
    else:
        radius -= 0.5

    # Compute the kernel
    if ndim == 1:
        x = np.arange(npix)
        kernel = np.exp(-(x-radius)**2 / sigma**2)
    else:
        x, y = np.meshgrid(np.arange(npix), np.arange(npix))  # pixel coordinates
        kernel = np.exp(-((x-radius)**2+(y-radius)**2) / sigma**2)
    return kernel / kernel.sum()


def test_convolve_separable_dilated():
    # test the utax function against scipy's method convolve2d

    # odd kernels
    kernel_1d = gaussian_kernel(sigma=0.4, odd=True, ndim=1)
    kernel_2d = gaussian_kernel(sigma=0.4, odd=True, ndim=2)
    npt.assert_almost_equal(np.outer(kernel_1d, kernel_1d), kernel_2d, decimal=10)  # sanity check
    np.random.seed(36)
    image = np.random.randn(10, 10)
    image_conv = np.array(convolve_separable_dilated(image, kernel_1d, boundary='wrap'))
    image_conv_ref = signal.convolve2d(image, kernel_2d, mode='same', boundary='wrap')
    npt.assert_almost_equal(image_conv, image_conv_ref, decimal=5)

    # TODO: fix the even kernel case
    # kernel_1d = gaussian_kernel(sigma=0.4, odd=False, ndim=1)
    # kernel_2d = gaussian_kernel(sigma=0.4, odd=False, ndim=2)
    # npt.assert_almost_equal(np.outer(kernel_1d, kernel_1d), kernel_2d, decimal=10)  # sanity check

    # np.random.seed(36)
    # image = np.random.randn(10, 10)
    # image_conv = np.array(convolve_separable_dilated(image, kernel_1d, boundary='wrap'))
    # image_conv_ref = scipy.signal.convolve2d(image, kernel_2d, mode='same', boundary='wrap')
    # print(image_conv.shape, image_conv_ref.shape)
    # npt.assert_almost_equal(image_conv, image_conv_ref, decimal=5)


# class TestGaussianFilter(object):

    # def test_gaussian_kernel():
    #     ref = gaussian_kernel()

# class TestConvolution(object):

#     def setup(self):
#         self.image = np.random.randn(10, 10)

#     def test_convolution_operator(self):
#         # using scipy function
#         kernel = gaussian_kernel()
#         result_ref = scipy.signal.convolve2d(kernel, self.image, mode='same')
#         result_utax = 

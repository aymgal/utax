__author__ = 'aymgal'

import pytest
import numpy as np
import numpy.testing as npt
from scipy import signal, ndimage

from jax.config import config
config.update("jax_enable_x64", True)  # makes a difference when comparing to scipy's routines!!

from utax.convolution import *


def kernel_from_size(nxk, nyk):
    ckx, cky = 0.1, 0.1  # off-centered kernel
    sigkx, sigky = 0.2, 0.1 # different sigma along both axes
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, nyk), np.linspace(-0.5, 0.5, nxk))
    kernel = np.exp(-(xx-ckx)**2/sigkx**2 - (yy-cky)**2/sigky**2)
    return kernel / kernel.sum()


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
        kernel = np.exp(-(x-radius)**2 / (2*sigma**2))
    else:
        x, y = np.meshgrid(np.arange(npix), np.arange(npix))  # pixel coordinates
        kernel = np.exp(-((x-radius)**2+(y-radius)**2) / (2*sigma**2))
    return kernel / kernel.sum()


def test_convolve_separable_dilated():
    # test the utax function against scipy's method convolve2d
    np.random.seed(36)
    image = np.random.randn(10, 10)

    # odd kernels
    kernel_1d = gaussian_kernel(sigma=0.4, odd=True, ndim=1)
    kernel_2d = gaussian_kernel(sigma=0.4, odd=True, ndim=2)
    npt.assert_almost_equal(np.outer(kernel_1d, kernel_1d), kernel_2d, decimal=10)  # sanity check
    image_conv = np.array(convolve_separable_dilated(image, kernel_1d, boundary='wrap'))
    image_conv_ref = signal.convolve2d(image, kernel_2d, mode='same', boundary='wrap')
    npt.assert_almost_equal(image_conv, image_conv_ref, decimal=5)

    # TODO: fix the even kernel case
    # kernel_1d = gaussian_kernel(sigma=0.4, odd=False, ndim=1)
    # kernel_2d = gaussian_kernel(sigma=0.4, odd=False, ndim=2)
    # npt.assert_almost_equal(np.outer(kernel_1d, kernel_1d), kernel_2d, decimal=10)  # sanity check
    # image_conv = np.array(convolve_separable_dilated(image, kernel_1d, boundary='wrap'))
    # image_conv_ref = signal.convolve2d(image, kernel_2d, mode='same', boundary='wrap')
    # print(image_conv.shape, image_conv_ref.shape)
    # npt.assert_almost_equal(image_conv, image_conv_ref, decimal=5)


def test_gaussian_filter():
    sigma = 0.3
    truncate = 5.
    gaussian_filter = GaussianFilter(sigma, truncate=truncate, mode='wrap')

    # first test the gaussian kernel itself
    kernel = gaussian_filter.gaussian_kernel(sigma, truncate)
    kernel_ref = gaussian_kernel(odd=True, sigma=sigma, truncate=truncate, ndim=1)
    npt.assert_almost_equal(kernel, kernel_ref, decimal=7)

    # then test the result of the convolution
    np.random.seed(36)
    image = np.random.randn(10, 10)
    image_filt = gaussian_filter(image)
    image_filt_ref = ndimage.gaussian_filter(image, sigma, truncate=truncate, mode='wrap')
    npt.assert_almost_equal(image_filt, image_filt_ref, decimal=6)

    # test with dumb sigma value
    gaussian_filter_id = GaussianFilter(-1., truncate=truncate, mode='wrap')
    image_filt = gaussian_filter_id(image)
    npt.assert_almost_equal(image_filt, image, decimal=7)


def test_blurring_operator():
    np.random.seed(36)

    # odd image + odd kernel smaller than image
    nx, ny = 9, 9
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(5, 5)
    conv_matrix = build_convolution_matrix(kernel, image.shape)
    npt.assert_almost_equal(conv_matrix.dot(image.flatten()).reshape(image.shape), 
                            signal.convolve2d(image, kernel, mode='same'),
                            decimal=10)

    # even image + odd kernel smaller than image
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(5, 5)
    conv_matrix = build_convolution_matrix(kernel, image.shape)
    npt.assert_almost_equal(conv_matrix.dot(image.flatten()).reshape(image.shape), 
                            signal.convolve2d(image, kernel, mode='same'),
                            decimal=10)


def test_blurring_operator_class():
    np.random.seed(36)

    # odd image + odd kernel smaller than image
    nx, ny = 9, 9
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(5, 5)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # even image + odd kernel smaller than image
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(5, 5)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # even image + even kernel smaller than image
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(4, 4)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # odd image + even kernel smaller than image
    nx, ny = 9, 9
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(4, 4)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # even image + odd kernel larger than image
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(13, 13)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # even image + odd kernel larger than image
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(13, 13)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='full'), 
                            signal.convolve2d(image, kernel, mode='full'), 
                            decimal=10)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # non-square image + non-square kernel
    nx, ny = 10, 11
    image = np.random.randn(nx, ny)
    kernel = kernel_from_size(6, 5)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(bop.convolve(image, out_padding='same'), 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)

    # Dirac kernel (identity operation)
    nx, ny = 10, 10
    image = np.random.randn(ny, nx)
    kernel = kernel_from_size(1, 1)
    bop = BlurringOperator(nx, ny, kernel)
    npt.assert_almost_equal(image, 
                            signal.convolve2d(image, kernel, mode='same'), 
                            decimal=10)



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

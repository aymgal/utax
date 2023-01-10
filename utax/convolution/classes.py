from functools import partial
import numpy as np
from scipy.linalg import toeplitz
import jax.numpy as jnp
from jax.scipy import stats as jstats
from jax import jit

from utax.convolution import convolve_separable_dilated


class GaussianFilter(object):
    """JAX-friendly Gaussian filter."""
    def __init__(self, sigma, truncate=4.0, mode='edge'):
        """Convolve an image by a gaussian filter.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian kernel.
        truncate : float, optional
            Truncate the filter at this many standard deviations.
            Default is 4.0.

        Note
        ----
        Reproduces `scipy.ndimage.gaussian_filter` with high accuracy.

        """
        if sigma <= 0:
            self.kernel = None
        else:
            self.kernel = self.gaussian_kernel(sigma, truncate)
        self.mode = mode

    def gaussian_kernel(self, sigma, truncate):
        # Determine the kernel pixel size (rounded up to an odd int)
        self.radius = int(jnp.ceil(2 * truncate * sigma)) // 2
        npix = self.radius * 2 + 1  # always at least 1

        # Return the identity if sigma is not a positive number
        if sigma <= 0:
            return jnp.ones(1)

        # Compute the kernel
        x = jnp.ravel(jnp.indices((npix,)))  # pixel coordinates
        kernel = jstats.norm.pdf((x-self.radius) / sigma)
        kernel /= kernel.sum()

        return kernel

    @partial(jit, static_argnums=(0,))
    def __call__(self, image):
        """Jit-compiled convolution an image by a gaussian filter.

        Parameters
        ----------
        image : array_like
            Image to filter.
        """
        # Convolve
        # pad_mode = ['constant', 'edge'][mode == 'nearest']
        # image_padded = jnp.pad(image, pad_width=radius, mode=pad_mode)
        if self.kernel is None:
            return image
        return convolve_separable_dilated(image, self.kernel, boundary=self.mode)


class BlurringOperator(object):

    def __init__(self, nx, ny, kernel):
        self.target_shape = nx, ny
        self.kernel = kernel
        conv_matrix, self.temp_shape = self.toeplitz_matrix(self.target_shape, self.kernel)
        self.conv_matrix = jnp.array(conv_matrix)
        # get the indices for cropping the output of the full convolution
        nxk, nyk = self.kernel.shape
        if nxk % 2 == 0:
            self.i1, self.i2 = nxk//2-1, -nxk//2
        else:
            self.i1, self.i2 = nxk//2, -nxk//2+1
        if nyk % 2 == 0:
            self.j1, self.j2 = nyk//2-1, -nyk//2
        else:
            self.j1, self.j2 = nyk//2, -nyk//2+1

    @partial(jit, static_argnums=(0, 2))
    def convolve(self, image, out_padding='same'):
        image_conv = self.v2m(self.conv_matrix.dot(self.m2v(image)), self.temp_shape)
        if out_padding == 'same':
            return image_conv[self.i1:self.i2, self.j1:self.j2]
        elif out_padding == 'full':
            return image_conv
        else:
            raise ValueError(f"padding model '{out_padding}' is not supported.")

    # @partial(jit, static_argnums=(0, 2))
    # def convolve_transpose(self, image, in_padding='same'):
    #     if in_padding == 'full':
    #         image_padded = image
    #     elif in_padding == 'same':
    #         image_padded = jnp.pad(image, ((self.i1, -self.i2), (self.j1, -self.j2)), 
    #                                'constant', constant_values=0)
    #     else:
    #         raise ValueError(f"padding model '{in_padding}' is not supported.")
    #     image_conv_t = self.v2m(self.conv_matrix.T.dot(self.m2v(image_padded)), self.target_shape)
    #     return image_conv_t

    @staticmethod
    def toeplitz_matrix(input_shape, kernel, verbose=False):
        """
        Performs 2D convolution between input I and filter F by converting the F to a toeplitz matrix and multiply it
          with vectorizes version of I
          By : AliSaaalehi@gmail.com
          
        Arg:
        input shape of the image to be convolved (I) -- 2D numpy matrix
        convolution kernel (F) -- numpy 2D matrix
        verbose -- if True, all intermediate resutls will be printed after each step of the algorithms
        
        Returns: 
        output -- 2D numpy matrix, result of convolving I with F
        """
        # number of columns and rows of the input 
        I_row_num, I_col_num = input_shape 

        # number of columns and rows of the filter
        F_row_num, F_col_num = kernel.shape

        #  calculate the output dimensions
        output_row_num = I_row_num + F_row_num - 1
        output_col_num = I_col_num + F_col_num - 1
        if verbose: print('output dimension:', output_row_num, output_col_num)

        # zero pad the filter
        F_zero_padded = np.pad(kernel, ((output_row_num - F_row_num, 0),
                                        (0, output_col_num - F_col_num)), 
                               'constant', constant_values=0)
        if verbose: print('F_zero_padded: ', F_zero_padded)

        # use each row of the zero-padded F to creat a toeplitz matrix. 
        #  Number of columns in this matrices are same as numbe of columns of input signal
        toeplitz_list = []
        for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
            c = F_zero_padded[i, :] # i th row of the F 
            r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                                # the result is wrong
            toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
            toeplitz_list.append(toeplitz_m)
            if verbose: print('F '+ str(i)+'\n', toeplitz_m)

            # doubly blocked toeplitz indices: 
        #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
        c = range(1, F_zero_padded.shape[0]+1)
        r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
        doubly_indices = toeplitz(c, r)
        if verbose: print('doubly indices \n', doubly_indices)

        ## creat doubly blocked matrix with zero values
        toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
        h = toeplitz_shape[0]*doubly_indices.shape[0]
        w = toeplitz_shape[1]*doubly_indices.shape[1]
        doubly_blocked_shape = [h, w]
        doubly_blocked = np.zeros(doubly_blocked_shape)

        # tile toeplitz matrices for each row in the doubly blocked matrix
        b_h, b_w = toeplitz_shape # hight and withs of each block
        for i in range(doubly_indices.shape[0]):
            for j in range(doubly_indices.shape[1]):
                start_i = i * b_h
                start_j = j * b_w
                end_i = start_i + b_h
                end_j = start_j + b_w
                doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

        if verbose: print('doubly_blocked: ', doubly_blocked)
        
        out_shape = (output_row_num, output_col_num)
        
        return doubly_blocked, out_shape

    @staticmethod
    def m2v(mat):
        return jnp.flipud(mat).flatten(order='C')

    @staticmethod
    def v2m(vec, output_shape):
        return jnp.flipud(vec.reshape(output_shape, order='C'))

from functools import partial
import jax.numpy as jnp
from jax.scipy import stats
from jax import jit
from jax.lax import conv_general_dilated, conv_dimension_numbers


@partial(jit, static_argnums=(2,))
def convolveSeparableDilated(image2D, kernel1D, dilation=1):
    """
     
     Convolves an image contained in image2D with the 1D kernel kernel1D.
     The operation is basically the following:
         blured2D = image2D * (kernel1D ∧ kernel1D )
     where ∧ is a wedge product, here a tensor product. 
     
     

    Parameters
    ----------
    image2D : 2D array
        imaged to be convolved with the kernel.
    kernel1D : 1D array
        kernel to convolve the image with..
    dilation : TYPE, optional
        makes the spacial extent of the kernel bigger. The default is 1.

    Returns
    -------
    2D array
        image convoluted by the kernel.

    """
    
    # padding
    b = int(kernel1D.size // 2) * dilation
    padded = jnp.pad(image2D, ((b, b), (b, b)), mode='edge')
    # Fred D.: THIS PADDING IS DANGEROUS AS IT WILL OVERFLOW MEMORY VERY QUICKLY
    # I LEAVE IT AS ORIGINALLY IMPLEMENTED AS I DO NOT WANT TO CHANGE THE
    # OUTPUT OF THE WAVELET TRANSFORM (this could have impact on the science)
    
    
    # specify the row and column operations for the jax convolution routine:
    image = jnp.expand_dims(padded, (2,))
    # shape (Nx, Ny, 1) -- (N, W, C)
    # we treat the Nx as the batch number!! (because it is a 1D convolution 
    # over the rows)
    kernel = jnp.expand_dims(kernel1D, (0,2,))
    # here we have kernel shape ~(I,W,O)
    # so: 
    # (Nbatch, Width, Channel) * (Inputdim, Widthkernel, Outputdim) 
    #                                            -> (Nbatch, Width, Channel)
    # where Nbatch is our number of rows.
    dimension_numbers = ('NWC', 'IWO', 'NWC')
    dn = conv_dimension_numbers(image.shape, 
                                kernel.shape, 
                                dimension_numbers)
    # with these conv_general_dilated knows how to handle the different
    # axes:
    rowblur = conv_general_dilated(image, kernel,
                                   window_strides=(1,),
                                   padding='VALID',
                                   rhs_dilation=(dilation,),
                                   dimension_numbers=dn)
    
    # now we do the same for the columns, hence this time we have
    # (Height, Nbatch, Channel) * (Inputdim, Widthkernel, Outputdim) 
    #                                            -> (Height, Nbatch, Channel)
    # where Nbatch is our number of columns.
    dimension_numbers = ('HNC', 'IHO', 'HNC')
    dn = conv_dimension_numbers(image.shape, 
                                kernel.shape, 
                                dimension_numbers)
    
    rowcolblur = conv_general_dilated(rowblur, kernel,
                                      window_strides=(1,),
                                      padding='VALID',
                                      rhs_dilation=(dilation,),
                                      dimension_numbers=dn)
    
    return rowcolblur[:,:,0]




class GaussianFilter(object):
    """JAX-friendly Gaussian filter."""
    def __init__(self, sigma, truncate=4.0):
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
        self.kernel = self.gaussian_kernel(sigma, truncate)

    def gaussian_kernel(self, sigma, truncate):
        # Determine the kernel pixel size (rounded up to an odd int)
        self.radius = int(jnp.ceil(2 * truncate * sigma)) // 2
        npix = self.radius * 2 + 1  # always at least 1

        # Return the identity if sigma is not a positive number
        if sigma <= 0:
            return jnp.array([[1.]])

        # Compute the kernel
        x = jnp.ravel(jnp.indices((npix,)))  # pixel coordinates
        kernel = stats.norm.pdf((x-self.radius) / sigma)
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
        return convolveSeparableDilated(image, self.kernel)

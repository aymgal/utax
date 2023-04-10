from functools import partial
import numpy as np
from scipy import sparse
import jax.numpy as jnp
from jax.scipy import stats as jstats
from jax import jit
from jax.lax import conv_general_dilated, conv_dimension_numbers


@partial(jit, static_argnums=(2, 3))
def convolve_separable_dilated(image2D, kernel1D, dilation=1, boundary='edge'):
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
    padded = jnp.pad(image2D, ((b, b), (b, b)), mode=boundary)
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

@partial(jit, static_argnums=(2, 3))
def convolve_dilated1D(signal1D, kernel1D, dilation=1, boundary='edge'):
    """

     Convolves a vector contained in signal1D with the 1D kernel.
     The operation is basically the following:
         blured1D = signal1D * kernel1D

    Parameters
    ----------
    signal1D : 1D array
        vector to be convolved with the kernel.
    kernel1D : 1D array
        kernel to convolve signal1D with..
    dilation : TYPE, optional
        makes the spacial extent of the kernel bigger. The default is 1.

    Returns
    -------
    1D array convoluted by the kernel.
    """

    # padding
    b = int(kernel1D.size // 2) * dilation
    padded = jnp.pad(signal1D, ((b, b)), mode=boundary)

    shape = kernel1D.shape
    strides = tuple(1 for s in shape)
    rowblur = conv_general_dilated(padded[None, None], kernel1D[None, None],
                                   window_strides=strides,
                                   padding='VALID',
                                   rhs_dilation=(dilation,),
                                   )

    return rowblur[0, 0]



def build_convolution_matrix(psf_kernel_2d, image_shape):
    """
    Build a sparse matrix to convolve an image via matrix-vector product.
    Ported from C++ code in VKL from Vernardos & Koopmans 2022.

    Note: only works with square kernel with odd number of pixels on the side, 
    lower than the number of pixels on the side of the image to be convolved.

    Authors: @gvernard, @aymgal
    """
    Ni, Nj = image_shape
    Ncropx, Ncropy = psf_kernel_2d.shape

    def setCroppedLimitsEven(k, Ncrop, Nimg, Nquad):
        if k < (Nquad - 1):
            Npre   = k
            Npost  = Nquad
            offset = Nquad - k
        elif k > (Nimg - Nquad - 1):
            Npre   = Nquad
            Npost  = Nimg - k
            offset = 0
        else:
            Npre   = Nquad
            Npost  = Nquad
            offset = 0
        return Npre, Npost, offset

    def setCroppedLimitsOdd(k, Ncrop, Nimg, Nquad):
        if k < (Nquad - 1):
            Npre   = k
            Npost  = Nquad
            offset = Nquad - 1 - k
        elif k > (Nimg - Nquad - 1):
            Npre   = Nquad - 1
            Npost  = Nimg - k
            offset = 0
        else:
            Npre   = Nquad-1
            Npost  = Nquad
            offset = 0
        return Npre, Npost, offset

    # get the correct method to offset the PSF kernel from the above
    if Ncropx % 2 == 0:
        # Warning: this might be broken in certain cases
        func_limits_x = setCroppedLimitsEven
        Nquadx = Ncropx//2
    else:
        func_limits_x = setCroppedLimitsOdd
        Nquadx = int(np.ceil(Ncropx/2.))
    if Ncropx % 2 == 0:
        # Warning: this might be broken in certain cases
        func_limits_y = setCroppedLimitsEven
        Nquady = Ncropy//2
    else:
        func_limits_y = setCroppedLimitsOdd
        Nquady = int(np.ceil(Ncropy/2.))

    # create the blurring matrix in a sparse form
    blur = psf_kernel_2d.flatten()
    sparse_B_rows, sparse_B_cols = [], []
    sparse_B_values  = []
    for i in range(Ni):  # loop over image rows
        for j in range(Nj):  # loop over image columns

            Nleft, Nright, crop_offsetx = func_limits_x(j, Ncropx, Nj, Nquadx)
            Ntop, Nbottom, crop_offsety = func_limits_y(i, Ncropy, Ni, Nquady)
            
            crop_offset = crop_offsety*Ncropx + crop_offsetx

            for ii in range(i-Ntop, i+Nbottom):  # loop over PSF rows
                ic = ii - i + Ntop

                for jj in range(j-Nleft, j+Nright):  # loop over PSF columns
                    jc = jj - j + Nleft;

                    val = blur[crop_offset + ic*Ncropx + jc]

                    # save entries 
                    # (note: rows and cols were inverted from the VKL code)
                    sparse_B_rows.append(ii*Nj + jj)
                    sparse_B_cols.append(i*Nj + j)
                    sparse_B_values.append(val)

    # populate the sparse matrix
    blurring_matrix = sparse.csr_matrix((sparse_B_values, (sparse_B_rows, sparse_B_cols)), 
                                        shape=(Ni**2, Nj**2))
    return blurring_matrix

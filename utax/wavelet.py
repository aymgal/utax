from functools import patial
import jax.numpy as jnp
from jax import jit

from utax.convolution import convolveSeparableDilated


class WaveletTransform(object):
    """
    Class that handles wavelet transform using JAX, using the 'a trous' algorithm

    Parameters
    ----------
    nscales : number of scales in the decomposition
    self._type : supported types are 'starlet', 'battle-lemarie-1', 'battle-lemarie-3'

    """
    def __init__(self, nscales, wavelet_type='starlet', second_gen=False):
        self._n_scales = nscales
        self._second_gen = second_gen
        if wavelet_type == 'starlet':
            self._h = jnp.array([1, 4, 6, 4, 1]) / 16.
            self._fac = 2 # basically len(self._h)//2 
        elif wavelet_type == 'battle-lemarie-1':
            self._h = jnp.array([-0.000122686, -0.000224296, 0.000511636, 
                        0.000923371, -0.002201945, -0.003883261, 0.009990599, 
                        0.016974805, -0.051945337, -0.06910102, 0.39729643, 
                        0.817645956, 0.39729643, -0.06910102, -0.051945337, 
                        0.016974805, 0.009990599, -0.003883261, -0.002201945,
                        0.000923371, 0.000511636, -0.000224296, -0.000122686])
            self._h /= 1.4140825479999999  # sum of coefficients above
            self._fac = 11
        elif wavelet_type == 'battle-lemarie-3':
            self._h = jnp.array([0.000146098, -0.000232304, -0.000285414, 
                           0.000462093, 0.000559952, -0.000927187, -0.001103748, 
                           0.00188212, 0.002186714, -0.003882426, -0.00435384, 
                           0.008201477, 0.008685294, -0.017982291, -0.017176331, 
                           0.042068328, 0.032080869, -0.110036987, -0.050201753, 
                           0.433923147, 0.766130398, 0.433923147, -0.050201753, 
                          -0.110036987, 0.032080869, 0.042068328, -0.017176331, 
                          -0.017982291, 0.008685294, 0.008201477, -0.00435384, 
                          -0.003882426, 0.002186714, 0.00188212, -0.001103748, 
                          -0.000927187, 0.000559952, 0.000462093, -0.000285414, 
                          -0.000232304, 0.000146098])
            self._h /= 1.4141580200000003  # sum of coefficients above
            self._fac = 20
        else:
            raise ValueError(f"'{wavelet_type}' starlet transform is not supported")
            
        if self._second_gen:
            self.decompose = self._decompose_2nd_gen 
            self.reconstruct = self._reconstruct_2nd_gen
        else:
            self.decompose = self._decompose_1st_gen
            self.reconstruct = self._reconstruct_1st_gen

    @property
    def scale_norms(self):
        if not hasattr(self, '_norms'):
            npix_dirac = 2**(self._n_scales + 2)
            dirac = jnp.diag((jnp.arange(npix_dirac) == int(npix_dirac / 2)).astype(float))
            wt_dirac = self.decompose(dirac)
            self._norms = jnp.sqrt(jnp.sum(wt_dirac**2, axis=(1, 2,)))
        return self._norms

    
    @partial(jit, static_argnums=(0,))
    def _decompose_1st_gen(self, image):
        """Decompose an image into the chosen wavelet basis"""
        # Validate input
        assert self._n_scales >= 0, "nscales must be a non-negative integer"
        if self._n_scales == 0:
            return image

        # Preparations
        image = jnp.copy(image)
        kernel = self._h.copy()

        # Compute the first scale:
        c1 = convolveSeparableDilated(image, kernel)
        # Wavelet coefficients:
        w0 = (image - c1)  
        result = jnp.expand_dims(w0, 0)
        cj = c1
        
        # Compute the remaining scales
        # at each scale, the kernel becomes larger ( a trou ) using the
        # dilation argument in the jax wrapper for convolution.
        for step in range(1, self._n_scales):
            cj1 = convolveSeparableDilated(cj, kernel, dilation=self._fac**step)
            # wavelet coefficients
            wj = (cj - cj1)
            result = jnp.concatenate((result, jnp.expand_dims(wj, 0)), axis=0)
            cj = cj1
        
        # Append final coarse scale
        result = jnp.concatenate((result, jnp.expand_dims(cj, axis=0)), axis=0)
        return result
    
    @partial(jit, static_argnums=(0,))
    def _decompose_2nd_gen(self, image):
        """Decompose an image into the chosen wavelet basis"""
        # Validate input
        assert self._n_scales >= 0, "nscales must be a non-negative integer"
        if self._n_scales == 0:
            return image

        # Preparations
        image = jnp.copy(image)
        kernel = self._h.copy()

        # Compute the first scale:
        c1 = convolveSeparableDilated(image, kernel)
        c1p = convolveSeparableDilated(c1, kernel)
        # Wavelet coefficients:
        w0 = (image - c1p)  
        result = jnp.expand_dims(w0, 0)
        cj = c1
        
        # Compute the remaining scales
        # at each scale, the kernel becomes larger ( a trou ) using the
        # dilation argument in the jax wrapper for convolution.
        for step in range(1, self._n_scales):
            cj1  = convolveSeparableDilated(cj, kernel, dilation=self._fac**step)
            cj1p = convolveSeparableDilated(cj1, kernel, dilation=self._fac**step)
            # wavelet coefficients
            wj = (cj - cj1p)
            result = jnp.concatenate((result, jnp.expand_dims(wj, 0)), axis=0)
            cj = cj1
        
        # Append final coarse scale
        result = jnp.concatenate((result, jnp.expand_dims(cj, axis=0)), axis=0)
        return result


    @partial(jit, static_argnums=(0,))
    def _reconstruct_1st_gen(self, coeffs):
        return jnp.sum(coeffs, axis=0)
    
    
    @partial(jit, static_argnums=(0,))
    def _reconstruct_2nd_gen(self, coeffs):
        # Validate input
        assert coeffs.shape[0] == self._n_scales+1, \
               "Wavelet coefficients are not consistent with number of scales"
        if self._n_scales == 0:
            return coeffs[0, :, :]
        
        kernel = self._h
        
        # Start with the last scale 'J-1'
        cJ = coeffs[self._n_scales, :, :]
        cJp = convolveSeparableDilated(cJ, kernel, 
                                       dilation=self._fac**(self._n_scales-1))
        

        wJ = coeffs[self._n_scales-1, :, :]
        cj = cJp + wJ
        
        # Compute the remaining scales
        for ii in range(self._n_scales-2, -1, -1):
            cj1 = cj
            cj1p = convolveSeparableDilated(cj1, kernel, dilation=self._fac**ii)
            wj1 = coeffs[ii, :, :]
            cj = cj1p + wj1

        result = cj
        return result
import numpy as np
import jax.numpy as jnp
from jax import lax, vmap  #, jit



class BilinearInterpolator(object):
    """Bilinear interpolation of a 2D field.

    Functionality is modelled after scipy.interpolate.RectBivariateSpline
    when `kx` and `ky` are both equal to 1. Results match the scipy version when
    interpolated values lie within the x and y domain (boundaries included).
    Returned values can be significantly different outside the natural domain,
    as the scipy version does not extrapolate. Evaluation of this jax version
    is MUCH SLOWER as well.

    """
    def __init__(self, x, y, z, allow_extrapolation=True):
        self.z = jnp.array(z)

        # Sort x if not increasing
        x = jnp.array(x)
        x_sorted = jnp.sort(x)
        flip_x = ~jnp.all(jnp.diff(x) >= 0)

        def x_keep_fn(_):
            return x, self.z

        def x_sort_fn(_):
            return x_sorted, jnp.flip(self.z, axis=0)

        self.x, self.z = lax.cond(flip_x, x_sort_fn, x_keep_fn, operand=None)

        # Sort y if not increasing
        y = jnp.array(y)
        y_sorted = jnp.sort(y)
        flip_y = ~jnp.all(jnp.diff(y) >= 0)

        def y_keep_fn(_):
            return y, self.z

        def y_sort_fn(_):
            return y_sorted, jnp.flip(self.z, axis=1)

        self.y, self.z = lax.cond(flip_y, y_sort_fn, y_keep_fn, operand=None)
        self._extrapol_bool = allow_extrapolation

    def __call__(self, x, y, dx=0, dy=0):
        """Vectorized evaluation of the interpolation or its derivatives.

        Parameters
        ----------
        x, y : array_like
            Position(s) at which to evaluate the interpolation.
        dx, dy : int, either 0 or 1
            If 1, return the first partial derivative of the interpolation
            with respect to that coordinate. Only one of (dx, dy) should be
            nonzero at a time.

        """
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)

        error_msg_type = "dx and dy must be integers"
        error_msg_value = "dx and dy must only be either 0 or 1"
        assert isinstance(dx, int) and isinstance(dy, int), error_msg_type
        assert dx in (0, 1) and dy in (0, 1), error_msg_value
        if dx == 1: dy = 0

        return vmap(self._evaluate, in_axes=(0, 0, None, None))(x, y, dx, dy)

    # @partial(jit, static_argnums=(0,))
    def _compute_coeffs(self, x, y):
        # Find the pixel that the point (x, y) falls in
        # x_ind = jnp.digitize(x, self.x_padded) - 1
        # y_ind = jnp.digitize(y, self.y_padded) - 1
        x_ind = jnp.searchsorted(self.x, x, side='right') - 1
        x_ind = jnp.clip(x_ind, a_min=0, a_max=(len(self.x) - 2))
        y_ind = jnp.searchsorted(self.y, y, side='right') - 1
        y_ind = jnp.clip(y_ind, a_min=0, a_max=(len(self.y) - 2))

        # Determine the coordinates and dimensions of this pixel
        x1 = self.x[x_ind]
        x2 = self.x[x_ind + 1]
        y1 = self.y[y_ind]
        y2 = self.y[y_ind + 1]
        area = (x2 - x1) * (y2 - y1)

        # Compute function values at the four corners
        # Edge padding is implicitly constant
        v11 = self.z[x_ind, y_ind]
        v12 = self.z[x_ind, y_ind + 1]
        v21 = self.z[x_ind + 1, y_ind]
        v22 = self.z[x_ind + 1, y_ind + 1]

        # Compute the coefficients
        a0_ = v11 * x2 * y2 - v12 * x2 * y1 - v21 * x1 * y2 + v22 * x1 * y1
        a1_ = -v11 * y2 + v12 * y1 + v21 * y2 - v22 * y1
        a2_ = -v11 * x2 + v12 * x2 + v21 * x1 - v22 * x1
        a3_ = v11 - v12 - v21 + v22

        return a0_ / area, a1_ / area, a2_ / area, a3_ / area

    def _evaluate(self, x, y, dx=0, dy=0):
        """Single-point evaluation of the interpolation."""
        a0, a1, a2, a3 = self._compute_coeffs(x, y)
        if (dx, dy) == (0, 0):
            result = a0 + a1 * x + a2 * y + a3 * x * y
        elif (dx, dy) == (1, 0):
            result = a1 + a3 * y
        else:
            result = a2 + a3 * x
        # if extrapolation is not allowed, then we mask out values outside the original bounding box
        result = lax.cond(self._extrapol_bool, 
                          lambda _: result, 
                          lambda _: result * (x >= self.x[0]) * (x <= self.x[-1]) * (y >= self.y[0]) * (y <= self.y[-1]), 
                          operand=None)
        return result


class BicubicInterpolator(object):
    """Bicubic interpolation of a 2D field.

    Functionality is modelled after scipy.interpolate.RectBivariateSpline
    when `kx` and `ky` are both equal to 3.

    """
    def __init__(self, x, y, z, zx=None, zy=None, zxy=None, allow_extrapolation=True):
        self.z = jnp.array(z)
        if np.all(np.diff(x) >= 0):  # check if sorted in increasing order
            self.x = jnp.array(x)
        else:
            self.x = jnp.array(np.sort(x))
            self.z = jnp.flip(self.z, axis=1)
        if np.all(np.diff(y) >= 0):  # check if sorted in increasing order
            self.y = jnp.array(y)
        else:
            self.y = jnp.array(np.sort(y))
            self.z = jnp.flip(self.z, axis=0)

        # Assume uniform coordinate spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Compute approximate partial derivatives if not provided
        if zx is None:
            self.zx = jnp.gradient(z, axis=0) / self.dx
        else:
            self.zx = zy
        if zy is None:
            self.zy = jnp.gradient(z, axis=1) / self.dy
        else:
            self.zy = zx
        if zxy is None:
            self.zxy = jnp.gradient(self.zx, axis=1) / self.dy
        else:
            self.zxy = zxy

        # Prepare coefficients for function evaluations
        self._A = jnp.array([[1., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [-3., 3., -2., -1.],
                            [2., -2., 1., 1.]])
        self._B = jnp.array([[1., 0., -3., 2.],
                            [0., 0., 3., -2.],
                            [0., 1., -2., 1.],
                            [0., 0., -1., 1.]])
        row0 = [self.z[:-1,:-1], self.z[:-1,1:], self.dy * self.zy[:-1,:-1], self.dy * self.zy[:-1,1:]]
        row1 = [self.z[1:,:-1], self.z[1:,1:], self.dy * self.zy[1:,:-1], self.dy * self.zy[1:,1:]]
        row2 = self.dx * jnp.array([self.zx[:-1,:-1], self.zx[:-1,1:],
                                   self.dy * self.zxy[:-1,:-1], self.dy * self.zxy[:-1,1:]])
        row3 = self.dx * jnp.array([self.zx[1:,:-1], self.zx[1:,1:],
                                   self.dy * self.zxy[1:,:-1], self.dy * self.zxy[1:,1:]])
        self._m = jnp.array([row0, row1, row2, row3])

        self._m = jnp.transpose(self._m, axes=(2, 3, 0, 1))

        self._extrapol_bool = allow_extrapolation

    def __call__(self, x, y, dx=0, dy=0):
        """Vectorized evaluation of the interpolation or its derivatives.

        Parameters
        ----------
        x, y : array_like
            Position(s) at which to evaluate the interpolation.
        dx, dy : int, either 0, 1, or 2
            Return the nth partial derivative of the interpolation
            with respect to the specified coordinate. Only one of (dx, dy)
            should be nonzero at a time.

        """
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        if x.ndim == 1:
            vmap_call = vmap(self._evaluate, in_axes=(0, 0, None, None))
        elif x.ndim == 2:
            vmap_call = vmap(vmap(self._evaluate, in_axes=(0, 0, None, None)),
                             in_axes=(0, 0, None, None))
        return vmap_call(x, y, dx, dy)

    def _evaluate(self, x, y, dx=0, dy=0):
        """Evaluate the interpolation at a single point."""
        # Determine which pixel (i, j) the point (x, y) falls in
        i = jnp.maximum(0, jnp.searchsorted(self.x, x) - 1)
        j = jnp.maximum(0, jnp.searchsorted(self.y, y) - 1)

        # Rescale coordinates into (0, 1)
        u = (x - self.x[i]) / self.dx
        v = (y - self.y[j]) / self.dy

        # Compute interpolation coefficients
        a = jnp.dot(self._A, jnp.dot(self._m[i, j], self._B))

        if dx == 0:
            uu = jnp.asarray([1., u, u**2, u**3])
        if dx == 1:
            uu = jnp.asarray([0., 1., 2. * u, 3. * u**2]) / self.dx
        if dx == 2:
            uu = jnp.asarray([0., 0., 2., 6. * u]) / self.dx**2
        if dy == 0:
            vv = jnp.asarray([1., v, v**2, v**3])
        if dy == 1:
            vv = jnp.asarray([0., 1., 2. * v, 3. * v**2]) / self.dy
        if dy == 2:
            vv = jnp.asarray([0., 0., 2., 6. * v]) / self.dy**2
        result = jnp.dot(uu, jnp.dot(a, vv))

        # if extrapolation is not allowed, then we mask out values outside the original bounding box
        result = lax.cond(self._extrapol_bool, 
                          lambda _: result, 
                          lambda _: result * (x >= self.x[0]) * (x <= self.x[-1]) * (y >= self.y[0]) * (y <= self.y[-1]), 
                          operand=None)
        return result
        
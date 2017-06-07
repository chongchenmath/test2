# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
Example of shape-based image reconstruction with linearized deformations.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from builtins import super
from odl.operator.operator import Operator
from odl.deform import LinDeformFixedTempl
from odl.phantom import shepp_logan, disc_phantom
from odl.solvers import CallbackShow, CallbackPrintIteration
import odl
import numpy as np
import time
import matplotlib.pyplot as plt
standard_library.install_aliases()


def K(x, y, sigma):
    """Define the K matrix as symmetric gaussian"""
    return np.exp(-((x[0] - y[0])**2 + (x[1] - y[1])**2) / sigma**2) * np.eye(2)


def v(x, grid, alphas, sigma):
    """Calculate the translation at point x"""
    alpha1, alpha2 = alphas  # unpack translations per direction
    result = np.zeros_like(x)
    for i, (point, a1, a2) in enumerate(zip(grid.points(), alpha1, alpha2)):
        result += K(x, point, sigma).dot([a1, a2]).squeeze()
    return result


class LinearDeformation(Operator):
    """A linear deformation given by:
        ``g(x) = f(x + v(x))``
    Where ``f(x)`` is the input template and ``v(x)`` is the translation at
    point ``x``. ``v(x)`` is computed using gaussian kernels with midpoints at
    ``grid``.
    """
    def __init__(self, fspace, vspace, grid, sigma):
        self.grid = grid
        self.sigma = sigma
        super().__init__(odl.ProductSpace(fspace, vspace), fspace, False)

    def _call(self, x):
        # Unpack input
        f, alphas = x
#        extension = f.space.extension(f.ntuple)  # this syntax is improved in pull #276

        # Array of output values
        out_values = np.zeros(f.size)

        for i, point in enumerate(self.range.points()):
            # Calculate deformation in each point
            point += v(point, self.grid, alphas, self.sigma)

            if point in f.space.domain:
                # Use extension operator of f
                out_values[i] = f.interpolation(point)
            else:
                # Zero boundary condition
                out_values[i] = 0

        return out_values


class Functional(Operator):

    """Quick hack for a functional class."""

    def __init__(self, domain, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluated
        linear : `bool`, optional
            If `True`, assume that the functional is linear
        """
        super().__init__(domain=domain, range=domain.field, linear=linear)

    def gradient(self, x, out=None):
        """Evaluate the gradient of the functional.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : domain element
            Result of the gradient calcuation. If ``out`` was given,
            the returned object is a reference to it.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an operator, this corresponds to
        operator composition:

            ``op1 * op2 <==> (x --> op1(op2(x))``

        If ``other`` is a scalar, this corresponds to right
        multiplication of scalars with operators:

            ``op * scalar <==> (x --> op(scalar * x))``

        If ``other`` is a vector, this corresponds to right
        multiplication of vectors with operators:

            ``op * vector <==> (x --> op(vector * x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : {`Operator`, `LinearSpaceVector`, scalar}
            `Operator`:
                The `Operator.domain` of ``other`` must match this
                operator's `Operator.range`.

            `LinearSpaceVector`:
                ``other`` must be an element of this operator's
                `Operator.domain`.

            scalar:
                The `Operator.domain` of this operator must be a
                `LinearSpace` and ``other`` must be an
                element of the ``field`` of this operator's
                `Operator.domain`.

        Returns
        -------
        mul : `Functional`
            Multiplication result

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.

        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            raise NotImplementedError
            if self.is_linear:
                return OperatorLeftScalarMult(self, other)
            else:
                return OperatorRightScalarMult(self, other)
        elif isinstance(other, LinearSpaceVector) and other in self.domain:
            raise NotImplementedError
            return OperatorRightVectorMult(self, other.copy())
        else:
            return NotImplemented


class DisplacementOperator(Operator):

    """Operator mapping parameters to an inverse displacement for
    the domain of the target.

    This operator computes for the momenta::

        alpha --> D(alpha)

    where

        D(alpha)(y) = v(y).

    The operator vector field ``v`` depends on the deformation parameters
    ``alpha_j`` as follows::

        v(y) = sum_j (K(y, y_j) * alpha_j)

    Here, ``K`` is the RKHS kernel matrix and each ``alpha_j`` is an
    element of ``R^n``, can be seen as the momenta alpha at control point y_j.
    """

    def __init__(self, par_space, control_points, discr_space, ft_kernel):
        """Initialize a new instance.

        Parameters
        ----------
        par_space : `ProductSpace` or `Rn`
            Space of the parameters. For one-dimensional deformations,
            `Rn` can be used. Otherwise, a `ProductSpace` with ``n``
            components is expected.
        control_points : `TensorGrid` or `array-like`
            The points ``x_j`` controlling the deformation. They can
            be given either as a tensor grid or as a point array. In
            the latter case, its shape must be ``(N, n)``, where
            ``n`` is the dimension of the template space, and ``N``
            the number of ``alpha_j``, i.e. the size of (each
            component of) ``par_space``.
        discr_space : `DiscreteSpace`
            Space of the image grid of the template.
        kernel : `callable`
            Function to determine the kernel at the control points ``K(y_j)``
            The function must accept a real variable and return a real number.
        """
        # TODO: use kernel operator instead of function & matrix
#        if isinstance(par_space, odl.Fn):
#            # Make a product space with one component
#            par_space = odl.ProductSpace(par_space, 1)
#        elif isinstance(par_space, odl.ProductSpace):
#            pass
#        else:
#            raise TypeError('expected Rn or ProductSpace as par_space, got '
#                            '{!r}.'.format(par_space))

        if par_space.size != discr_space.ndim:
            raise ValueError('dimensions of product space and image grid space'
                             ' do not match ({} != {})'
                             ''.format(par_space.size, discr_space.ndim))

        # The operator maps from the parameter space to an inverse
        # displacement for the domain of the target.

        self.discr_space = discr_space
        self.range_space = odl.ProductSpace(self.discr_space,
                                            self.discr_space.ndim)

        super().__init__(par_space, self.range_space, linear=True)

        self.ft_kernel = ft_kernel

        if not isinstance(control_points, odl.RectGrid):
            self._control_pts = np.asarray(control_points)
            if self._control_pts.shape != (self.num_contr_pts, self.ndim):
                raise ValueError(
                    'expected control point array of shape {}, got {}.'
                    ''.format((self.num_contr_pts, self.ndim),
                              self.control_points.shape))
        else:
            self._control_pts = control_points

        # TODO: check that number of control points is the same as alphas

    @property
    def ndim(self):
        """Number of dimensions of the deformation."""
        return self.domain.ndim

    @property
    def contr_pts_is_grid(self):
        """`True` if the control points are given as a grid."""
        return isinstance(self.control_points, odl.TensorGrid)

    @property
    def control_points(self):
        """The control points for the deformations."""
        return self._control_pts

    @property
    def num_contr_pts(self):
        """Number of control points for the deformations."""
        if self.contr_pts_is_grid:
            return self.control_points.size
        else:
            return len(self.control_points)

    @property
    def image_grid(self):
        """Spatial sampling grid of the image space."""
        return self.discr_space.grid

    def displacement_ft(self, alphas):
        """Calculate the inverse translation at point y by n-D FFT.

        inverse displacement: v(y)

        Parameters
        ----------
        alphas: `ProductSpaceElement`
            Deformation parameters for control points. It has ``n``
            components, each of which has size ``N``. Here, ``n`` is
            the number of dimensions and ``N`` the number of control
            points. Note that here N = M.
        kernel: Gaussian kernel fuction
        """
        ft_momenta = vectorial_ft_fit_op(alphas)
        ft_displacement = self.ft_kernel * ft_momenta
        return vectorial_ft_fit_op.inverse(ft_displacement)
        # scaling
#        return (vectorial_ft_op_inverse(ft_displacement) /
#                self.discr_space.cell_volume * 2.0 * np.pi)

    def _call(self, alphas):
        """Implementation of ``self(alphas, out)``.

        Parameters
        ----------
        alphas: `ProductSpaceElement`
            Deformation parameters for control points. It has ``n``
            components, each of which has size ``N``. Here, ``n`` is
            the number of dimensions and ``N`` the number of control
            points.

        out : `DiscreteLpElement`
            Element where the result is stored
        """

        return self.displacement_ft(alphas)

    def derivative(self, alphas):
        """Frechet derivative of this operator in ``alphas``.

        Parameters
        ----------
        alphas: `ProductSpaceElement`
            Deformation parameters for control points. It has ``n``
            components, each of which has size ``N``. Here, ``n`` is
            the number of dimensions and ``N`` the number of control
            points.

        Returns
        -------
        deriv : `Operator`
            The derivative of this operator, evaluated at ``alphas``
        """
        deriv_op = DisplacementDerivative(
            alphas, self.control_points, self.discr_space, self.ft_kernel)

        return deriv_op


class DisplacementDerivative(DisplacementOperator):

    """Frechet derivative of the displacement operator at alphas."""

    def __init__(self, alphas, control_points, discr_space, ft_kernel):
        """Initialize a new instance.

        Parameters
        ----------
        alphas : `ProductSpaceElement`
            Displacement parameters in which the derivative is evaluated
        control_points : `TensorGrid` or `array-like`
            The points ``x_j`` controlling the deformation. They can
            be given either as a tensor grid or as a point array. In
            the latter case, its shape must be ``(N, n)``, where
            ``n`` is the dimension of the template space, and ``N``
            the number of ``alpha_j``, i.e. the size of (each
            component of) ``par_space``.
        discr_space : `DiscreteSpace`
            Space of the image grid of the template.
        kernel : `callable`
            Function to determine the kernel at the control points ``K(y_j)``
            The function must accept a real variable and return a real number.
        """

        super().__init__(alphas.space, control_points, discr_space, ft_kernel)
        self.discr_space = discr_space
        self.range_space = odl.ProductSpace(self.discr_space,
                                            self.discr_space.ndim)
        Operator.__init__(self, alphas.space, self.range_space, linear=True)
        self.alphas = alphas

    def _call(self, betas):
        """Implementation of ``self(betas)``.

        Parameters
        ----------
        betas: `ProductSpaceElement`
            Deformation parameters for control points. It has ``n``
            components, each of which has size ``N``. Here, ``n`` is
            the number of dimensions and ``N`` the number of control
            points. It should be in the same space as alpha.
        """

        return self.displacement_ft(betas)

    @property
    def adjoint(self):
        """Adjoint of the displacement derivative."""
        adj_op = DisplacementDerivativeAdjoint(
            self.alphas, self.control_points, self.discr_space, self.ft_kernel)

        return adj_op


class DisplacementDerivativeAdjoint(DisplacementDerivative):

    """Adjoint of the Displacement operator derivative.
    """
    def __init__(self, alphas, control_points, discr_space, ft_kernel):
        """Initialize a new instance.

        Parameters
        ----------
        alphas : `ProductSpaceElement`
            Displacement parameters in which the derivative is evaluated
        control_points : `TensorGrid` or `array-like`
            The points ``x_j`` controlling the deformation. They can
            be given either as a tensor grid or as a point array. In
            the latter case, its shape must be ``(N, n)``, where
            ``n`` is the dimension of the template space, and ``N``
            the number of ``alpha_j``, i.e. the size of (each
            component of) ``par_space``.
        discr_space : `DiscreteSpace`
            Space of the image grid of the template.
        kernel : `callable`
            Function to determine the kernel at the control points ``K(y_j)``
            The function must accept a real variable and return a real number.
        """

        super().__init__(alphas, control_points, discr_space, ft_kernel)

        # Switch domain and range
        self.discr_space = discr_space
        self.range_space = odl.ProductSpace(self.discr_space,
                                            self.discr_space.ndim)
        Operator.__init__(self, self.range_space, alphas.space, linear=True)
#        self._domain, self._range = self._range, self._domain

    def _call(self, grad_func):
        """Implement ``self(func)```.

        Parameters
        ----------
        func : `DiscreteLpElement`
            Element of the image's gradient space.
        """
        # If control grid is not the image grid, the following result for
        # the ajoint is not right. Because the kernel matrix in fitting
        # term is not symmetric.
        return self.displacement_ft(grad_func)


class ShapeRegularizationFunctional(Operator):

    """Regularization functional for linear shape deformations.

    The shape regularization functional is given as

        S(alpha) = 1/2 * ||v(alpha)||^2 = 1/2 * alpha^T K alpha,

    where ``||.||`` is the norm in a reproducing kernel Hilbert space
    given by parameters ``alpha``. ``K`` is the kernel matrix.
    """
    # TODO: let user specify K

    def __init__(self, par_space, ft_kernel):
        """Initialize a new instance.

        Parameters
        ----------
        par_space : `ProductSpace`
            Parameter space of the deformations, i.e. the space of the
            ``alpha_k`` parametrizing the deformations
        kernel_op : `numpy.ndarray` or `Operator`
            The kernel matrix defining the norm. If an operator is
            given, it is called to evaluate the matrix-vector product
            of the kernel matrix with a given ``alpha``.
        """
        super().__init__(par_space, odl.RealNumbers(), linear=False)
#        if isinstance(kernel_op, Operator):
#            self._kernel_op = kernel_op
#        else:
#            self._kernel_op = odl.MatVecOperator(kernel_op)
        self.par_space = par_space
        self.ft_kernel = ft_kernel

    def _call(self, alphas):
        """Return ``self(alphas)``."""
        # TODO: how to improve

        # Compute the shape energy by fft
        ft_momenta = vectorial_ft_shape_op(alphas)
        stack = vectorial_ft_shape_op.inverse(self.ft_kernel * ft_momenta)
        return sum(s.inner(s.space.element(
                       np.asarray(a).reshape(-1, order=self.domain[0].order)))
                   for s, a in zip(stack, alphas)) / 2

#        # Compute the shape energy by matrix times vector
#        stack = [self._kernel_op(
#                     np.asarray(a).reshape(-1, order=self.domain[0].order))
#                 for a in alphas]
#        return sum(s.inner(s.space.element(
#                       np.asarray(a).reshape(-1, order=self.domain[0].order)))
#                   for s, a in zip(stack, alphas)) / 2

    def _gradient(self, alphas):
        """Return the gradient at ``alphas``.

        The gradient of the functional is given by

            grad(S)(alpha) = K alpha
        """
#        return self.domain.element([self._kernel_op(np.asarray(a).reshape(-1))
#                                    for a in alphas])
        pass

    def _gradient_ft(self, alphas):
        """Return the gradient at ``alphas``.

        The gradient of the functional is given by

            grad(S)(alpha) = K alpha.

        This is used for the n-D case: control grid = image grid.
        """
        ft_momenta = vectorial_ft_shape_op(alphas)
        return vectorial_ft_shape_op.inverse(self.ft_kernel * ft_momenta)
#        return (vectorial_ft_op_inverse(ft_displacement) /
#                self.par_space[0].cell_volume * 2.0 * np.pi)


class L2DataMatchingFunctional(Functional):

    """Basic data matching functional using the L2 norm.

    This functional computes::

        1/2 * ||f - g||_2^2

    for a given element ``g``.
    """

    def __init__(self, space, data):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` with exponent 2.0
            Space where the data is matched
        data : `DiscreteLp` element-like
            Data which is to be matched
        """
        if not (isinstance(space, odl.DiscreteLp) and space.exponent == 2.0):
            raise ValueError('not an L2 space.')
        super().__init__(space, linear=False)
        self.data = self.domain.element(data)

    def _call(self, x):
        """Return ``self(x)``."""
        return self.domain.dist(x, self.data)

    def gradient(self, x):
        """Return the gradient in the point ``x``."""
        return x - self.data

    def derivative(self, x):
        """Return the derivative in ``x``."""
        return self.gradient(x).T


# Kernel function for any dimensional
def gauss_kernel(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2))


# Kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


def gaussian_kernel_matrix(grid, sigma):
    """Return the kernel matrix for Gaussian kernel.

    The Gaussian kernel matrix ``K`` in ``n`` dimensions is defined as::

        k_ij = exp(- |x_i - x_j|^2 / (2 * sigma^2))

    where ``x_i, x_j`` runs through all grid points. The final matrix
    has size ``N x N``, where ``N`` is the total number of grid points.

    Parameters
    ----------
    grid : `RegularGrid`
        Grid where the control points are defined
    sigma : `float`
        Width of the Gaussian kernel
    """
    point_arrs = grid.points().T
    matrices = [parr[:, None] - parr[None, :] for parr in point_arrs]
    for mat in matrices:
        mat *= mat

    sq_sum = np.sqrt(np.sum(mat for mat in matrices))
    kernel_matrix = np.exp(-sq_sum / (2 * sigma ** 2))
    return kernel_matrix


def SNR(signal, noise, impl='general'):
    """Compute the signal-to-noise ratio.
    This compute::
        impl='general'
            SNR = s_power / n_power
        impl='dB'
            SNR = 10 * log10 (
                s_power / n_power)
    Parameters
    ----------
    signal : projection
    noise : white noise
    impl : implementation method
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal)/signal.size
        ave2 = np.sum(noise)/noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            snr = s_power/n_power
        else:
            snr = 10.0 * np.log10(s_power/n_power)

        return snr

    else:
        return float('inf')


def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padding_op = odl.ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = odl.trafos.FourierTransform(
        padding_op.range, halfcomplex=False, shift=shifts)

    return ft_op * padding_op


def shape_kernel_ft(kernel):
    """Compute the n-D Fourier transform of the discrete kernel ``K``.

    Calculate the n-D Fourier transform of the discrete kernel ``K`` on the
    control grid points {y_i} to its reciprocal points {xi_i}.
    """

    # Create the array of kernel values on the grid points
    discretized_kernel = vspace.element(
        [cptsspace.element(kernel) for _ in range(cptsspace.ndim)])
    return vectorial_ft_shape_op(discretized_kernel)


def fitting_kernel_ft(kernel):
    """Compute the n-D Fourier transform of the discrete kernel ``K``.

    Calculate the n-D Fourier transform of the discrete kernel ``K`` on the
    image grid points {y_i} to its reciprocal points {xi_i}.

    """
    kspace = odl.ProductSpace(discr_space, discr_space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [discr_space.element(kernel) for _ in range(discr_space.ndim)])
    return vectorial_ft_fit_op(discretized_kernel)


def shepp_logan_ellipse_2d_template():
    """Return ellipse parameters for a 2d Shepp-Logan phantom.

    This assumes that the ellipses are contained in the square
    [-1, -1]x[-1, -1].
    """
    # 5/A, 7/A + noise
    return [[2.00, .6000, .6000, 0.0000, 0.1200, 0],
            [-.98, .5624, .5640, 0.0000, -.0184 + 0.12, 0],
            [-.02, .1100, .1100, 0.2600, 0.1500, -18],
            [-.02, .1300, .1300, -.2500, 0.1900, 18],
            [0.01, .1650, .1650, 0.0000, 0.3000, 0],
            [0.01, .0300, .0300, 0.0000, 0.1600, 0],
            [0.01, .0300, .0300, -.1500, 0.1000, 0],
            [0.01, .0360, .0230, -.0770, -.0750, 0],
            [0.01, .0230, .0230, 0.0000, -.0760, 0],
            [0.01, .0230, .0360, 0.0600, -.0750, 0]] 


def modified_shepp_logan_ellipses(ellipses):
    """Modify ellipses to give the modified Shepp-Logan phantom.

    Works for both 2d and 3d.
    """
    intensities = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    assert len(ellipses) == len(intensities)

    for ellipse, intensity in zip(ellipses, intensities):
        ellipse[0] = intensity


def shepp_logan_ellipses(ndim, modified=False):
    """Ellipses for the standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    ndim : {2, 3}
        Dimension of the space the ellipses should be in.
    modified : bool, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    ellipse_phantom : Function for creating arbitrary ellipse phantoms
    shepp_logan : Create a phantom with these ellipses

    References
    ----------
    .. _Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    if ndim == 2:
        ellipses = shepp_logan_ellipse_2d_template()
    else:
        raise ValueError('dimension not 2, no phantom available')

    if modified:
        modified_shepp_logan_ellipses(ellipses)

    return ellipses


def shepp_logan_2d(space, modified=False):
    """Standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
    modified : `bool`, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    shepp_logan_ellipses : Get the parameters that define this phantom
    ellipse_phantom : Function for creating arbitrary ellipse phantoms

    References
    ----------
    .. Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    ellipses = shepp_logan_ellipses(space.ndim, modified)

    return odl.phantom.geometric.ellipse_phantom(space, ellipses)


def donut(discr, smooth=True, taper=20.0):
    """Return a 'donut' phantom.

    This phantom is used in [Okt2015]_ for shape-based reconstruction.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 2:
        if smooth:
            return _donut_2d_smooth(discr, taper)
        else:
            return _donut_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _donut_2d_smooth(discr, taper):
    """Return a 2d smooth 'donut' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle_1(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.26, 0.26]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_circle_2(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.115, 0.115]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle_1) - discr.element(blurred_circle_2)
    return out.ufuncs.minimum(1, out=out)


def _donut_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'donut' phantom."""

    def circle_1(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def circle_2(x):
        """Characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.1, 0.1]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle_1) - discr.element(circle_2)
    return out.ufuncs.minimum(1, out=out)


# Create seed for random generator
# np.random.seed(0)


# Create 2-D discretization reconstruction space
# The size of the domain should be proportional to the given images
# 128 for shepp-logan
discr_space = odl.uniform_discr([-16, -16], [16, 16], [101, 101],
                                dtype='float32', interp='linear')

# Create 2-D discretization space for control points
cptsspace = odl.uniform_discr([-16, -16], [16, 16], [101, 101],
                              dtype='float32', interp='linear')

# Create discretization space for vector field
vspace = odl.ProductSpace(cptsspace, cptsspace.ndim)

# Create the ground truth as the Shepp-Logan phantom
#ground_truth = shepp_logan(discr_space, modified=True)
ground_truth = donut(discr_space, smooth=True, taper=50)

#template = shepp_logan_2d(discr_space, modified=True)
template = disc_phantom(discr_space, smooth=True, taper=50)

## Create the template for Shepp-Logan phantom
#deform_field_space = discr_space.vector_field_space
#disp_func = [
#    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
#    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
## Discretization of the space
#spc = odl.uniform_discr([0, 0], [1, 1], [128, 128])
#
## deformation space
## n: number of gridpoints for deformation, usually smaller than m
#pspace = odl.ProductSpace(odl.uniform_discr([0, 0], [1, 1], [5, 5]), 2)
#
## Deformation operator
#deformation = LinearDeformation(spc, pspace, pspace[0].grid, sigma=0.2)
#
## Create input function
#f = shepp_logan(spc, True)
#
## Create deformation field
#values = np.zeros([2, 5, 5])
#values[0, :, :5//2] = 0.02  # movement in "x" direction
#values[1, 5//2, :] = 0.01   # movement in "y" direction
#def_coeff = pspace.element(values)
## Show input
#f.show(title='f')
#def_coeff.show(title='deformation')
#
## Calculate deformed function
#result = deformation([f, def_coeff])
#result.show(title='result')
#template_deformed_1 = discr_space.element(result)
#
#disp_field = deform_field_space.element(disp_func)
### Create the template from the deformed ground truth
#template = discr_space.element(geometric_deform(
#    template_deformed_1, disp_field))

# Show ground truth and template
ground_truth.show('ground truth')
template.show('template')

# Give the number of directions
num_angles = 4

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, 3.*np.pi/4., num_angles,
                                        nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
# 181 for 128 shepp-logan
detector_partition = odl.uniform_partition(-24, 24, 151)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create forward projection operator by X-ray transform
xray_trafo_op = odl.tomo.RayTransform(discr_space, geometry, impl='astra_cuda')

# Create projection data by given setting
proj_data = xray_trafo_op(ground_truth)

# Create white Gaussian noise
noise = 0.1 * odl.phantom.white_noise(xray_trafo_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio
snr = SNR(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Do the backprojection reconstruction
backproj = xray_trafo_op.adjoint(noise_proj_data)

# FFT setting for regularization shape term, 1 means 100% padding
# FFT setting for data matching term, 1 means 100% padding
padded_size = 2 * cptsspace.shape[0]
padded_ft_shape_op = padded_ft_op(cptsspace, padded_size)
vectorial_ft_shape_op = odl.DiagonalOperator(
    *([padded_ft_shape_op] * cptsspace.ndim))

# FFT setting for data matching term, 1 means 100% padding
padded_ft_fit_op = padded_ft_op(discr_space, padded_size)
vectorial_ft_fit_op = odl.DiagonalOperator(
    *([padded_ft_fit_op] * discr_space.ndim))

# Initialize deformation vector field
momenta = vspace.zero()

# Fix the sigma parameter in the kernel
sigma = 4.0

# Compute Fourier trasform of the kernel function in data matching term
ft_kernel_fitting = fitting_kernel_ft(kernel)

# Compute Fourier trasform of the kernel function in shape regularization term
ft_kernel_shape = shape_kernel_ft(kernel)

# Create displacement operator
displacement_op = DisplacementOperator(vspace, cptsspace.grid,
                                       discr_space, ft_kernel_fitting)

# Compute the displacement at momenta
displ = displacement_op(momenta)

# Create linearized deformation operator
linear_deform_op = LinDeformFixedTempl(template)

# Compute the deformed template
deformed_template = linear_deform_op(displ)

# Create X-ray transform operator
proj_deformed_template = xray_trafo_op(deformed_template)

# Create L2 data matching (fitting) term
l2_data_fit_func = L2DataMatchingFunctional(xray_trafo_op.range,
                                            noise_proj_data)

# Composition of the L2 fitting term with three operators
# data_fitting_term = l2_data_fit_func * xray_trafo_op * linear_deform_op * displacement_op

# Compute the kernel matrix for the method without Fourier transform
# If the dimension is too large, it could cause MemoryError
# kernelmatrix = gaussian_kernel_matrix(cptsspace.grid, sigma)

# Compute the gradient of shape regularization term
shape_func = ShapeRegularizationFunctional(vspace, ft_kernel_shape)

# Shape regularization parameter, should be nonnegtive
lambda_shape = 0.0000001

# Step size for the gradient descent method
eta = 0.002

# Maximum iteration number
niter = 1000

callback = CallbackShow('iterates', display_step=5) & CallbackPrintIteration()

# Test time, set starting time
start = time.clock()

# Iterations for updating alphas
for i in range(niter):

    # Compute the gradient for the shape term by Fourier transform
    grad_shape_func = shape_func._gradient_ft(momenta)

    displ = displacement_op(momenta)
    deformed_template = linear_deform_op(displ)
    proj_deformed_template = xray_trafo_op(deformed_template)
    temp1 = l2_data_fit_func.gradient(proj_deformed_template)
    temp2 = linear_deform_op.derivative(displ).adjoint(xray_trafo_op.adjoint(temp1))
    grad_data_fitting_term = displacement_op.derivative(momenta).adjoint(temp2)

#    # Compute the gradient for data fitting term
#    grad_data_fitting_term = data_fitting_term.gradient(momenta)

    # Update momenta
    momenta -= eta * (
        lambda_shape * grad_shape_func + grad_data_fitting_term)

    # Show the middle reconstrcted results
    if (i+1) % 100 == 0:
        print(i+1)

    if callback is not None:
        displ = displacement_op(momenta)
        deformed_template = linear_deform_op(displ)
        callback(deformed_template)

# Test time, set end time
end = time.clock()

# Output the computational time
print(end - start)

# Compute the projections of the reconstructed image
displ = displacement_op(momenta)
deformed_template = linear_deform_op(displ)
rec_proj_data = xray_trafo_op(deformed_template)

# Plot the results of interest
plt.figure(1, figsize=(20, 10))
plt.clf()

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()), plt.axis('off')
plt.colorbar()
plt.title('Template')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(deformed_template), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()), plt.axis('off')
plt.colorbar()
plt.title('Reconstructed result')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()), plt.axis('off')
plt.colorbar()
plt.title('Ground truth')

plt.subplot(2, 3, 4)
plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(noise_proj_data)[0], 'r')
plt.axis([0, 181, -3, 10])
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(np.asarray(proj_data)[2], 'b', np.asarray(noise_proj_data)[2], 'r')
plt.axis([0, 181, -3, 10])
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(np.asarray(proj_data)[4], 'b', np.asarray(noise_proj_data)[4], 'r')
plt.axis([0, 181, -3, 10])
plt.grid(True)


## TV reconstruction by Chambolle-Pock algorithm
## Initialize gradient operator
#gradient = odl.Gradient(discr_space, method='forward')
## Column vector of two operators
#op = odl.BroadcastOperator(xray_trafo_op, gradient)
## Create the proximal operator for unconstrained primal variable
#proximal_primal = odl.solvers.proximal_const_func(op.domain)
## Create proximal operators for the dual variable
## l2-data matching
#prox_convconj_l2 = odl.solvers.proximal_cconj_l2_squared(xray_trafo_op.range,
#                                                         g=noise_proj_data)
## Isotropic TV-regularization i.e. the l1-norm
#prox_convconj_l1 = odl.solvers.proximal_cconj_l1(gradient.range, lam=0.6,
#                                                 isotropic=True)
## Combine proximal operators, order must correspond to the operator K
#proximal_dual = odl.solvers.combine_proximals(prox_convconj_l2,
#                                              prox_convconj_l1)
## --- Select solver parameters and solve using Chambolle-Pock --- #
## Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
#op_norm = 1.1 * odl.power_method_opnorm(op)
#
#niter = 1000  # Number of iterations
#tau = 1.0 / op_norm  # Step size for the primal variable
#sigma = 1.0 / op_norm  # Step size for the dual variable
#gamma = 0.2
#
## Optionally pass callback to the solver to display intermediate results
#callback = (odl.solvers.CallbackPrintIteration() &
#            odl.solvers.CallbackShow())
#
## Choose a starting point
#x = op.domain.zero()
#
## Run the algorithm
#odl.solvers.chambolle_pock_solver(
#    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
#    proximal_dual=proximal_dual, niter=niter, callback=callback,
#    gamma=gamma)
#
#plt.imshow(np.rot90(x), cmap='bone',
#           vmin=0., vmax=1.), plt.axis('off')
##plt.colorbar()
#plt.title('Reconstructed result')

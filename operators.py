"""
Operators and functions for shape-based image reconstruction using linearized deformations.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from builtins import super
from odl.operator.operator import Operator
from odl.discr import RectGrid, ResizingOperator
from odl.trafos import FourierTransform 
from odl.space import ProductSpace
from odl.set import RealNumbers
from odl.operator import DiagonalOperator
import numpy as np

standard_library.install_aliases()


class DisplacementOperator(Operator):
    """Operator mapping parameters to an inverse displacement.

    This operator computes for the momenta::

        alpha --> D(alpha)

    where

        D(alpha) = v.

    The vector field ``v`` depends on the deformation parameters
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
        ft_kernel : `callable`
            Function to determine the FT of kernel at the control points ``K(y_j)``
            The function must accept a real variable and return a real number.
        """

        if par_space.size != discr_space.ndim:
            raise ValueError('dimensions of product space and image grid space'
                             ' do not match ({} != {})'
                             ''.format(par_space.size, discr_space.ndim))

        self.discr_space = discr_space
        self.range_space = ProductSpace(self.discr_space,
                                        self.discr_space.ndim)

        super().__init__(par_space, self.range_space, linear=True)

        self.ft_kernel = ft_kernel

        if not isinstance(control_points, RectGrid):
            self._control_pts = np.asarray(control_points)
            if self._control_pts.shape != (self.num_contr_pts, self.ndim):
                raise ValueError(
                    'expected control point array of shape {}, got {}.'
                    ''.format((self.num_contr_pts, self.ndim),
                              self.control_points.shape))
        else:
            self._control_pts = control_points

    @property
    def ndim(self):
        """Number of dimensions of the deformation."""
        return self.domain.ndim

    @property
    def contr_pts_is_grid(self):
        """`True` if the control points are given as a grid."""
        return isinstance(self.control_points, RectGrid)

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
        temp_op = vectorial_ft_shape_op(alphas.space[0])
        ft_momenta = temp_op(alphas)
        ft_displacement = self.ft_kernel * ft_momenta
        return temp_op.inverse(ft_displacement)
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
        self.range_space = ProductSpace(self.discr_space,
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
    """Adjoint of the Displacement operator derivative."""

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
        self.range_space = ProductSpace(self.discr_space,
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
        super().__init__(par_space, RealNumbers(), linear=False)
#        if isinstance(kernel_op, Operator):
#            self._kernel_op = kernel_op
#        else:
#            self._kernel_op = odl.MatVecOperator(kernel_op)
        self.par_space = par_space
        self.ft_kernel = ft_kernel

    def _call(self, alphas):
        """Return ``self(alphas)``."""

        # Compute the shape energy by fft
        ft_momenta = vectorial_ft_shape_op(alphas)
        stack = vectorial_ft_shape_op.inverse(self.ft_kernel * ft_momenta)
        return sum(s.inner(s.space.element(
                       np.asarray(a).reshape(-1, order=self.domain[0].order)))
                   for s, a in zip(stack, alphas)) / 2

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
        temp_op = vectorial_ft_shape_op(alphas.space[0])
        ft_momenta = temp_op(alphas)
        return temp_op.inverse(self.ft_kernel * ft_momenta)
#        return (vectorial_ft_op_inverse(ft_displacement) /
#                self.par_space[0].cell_volume * 2.0 * np.pi)


def snr(signal, noise, impl='general'):
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


def _padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padding_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padding_op.range, halfcomplex=False, shift=shifts)

    return ft_op * padding_op


def vectorial_ft_shape_op(cptsspace):

    padded_size = 2 * cptsspace.shape[0]
    padded_ft_shape_op = _padded_ft_op(cptsspace, padded_size)
    return DiagonalOperator(*([padded_ft_shape_op] * cptsspace.ndim))

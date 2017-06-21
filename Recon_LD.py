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
from odl.deform import LinDeformFixedTempl
from odl.solvers import CallbackShow, CallbackPrintIteration
import odl
import numpy as np
import time
import matplotlib.pyplot as plt
from Linearized_deformation import (SNR, donut, padded_ft_op,
                                    DisplacementOperator,
                                    L2DataMatchingFunctional,
                                    ShapeRegularizationFunctional)
standard_library.install_aliases()


# Kernel function for any dimensional
def gauss_kernel(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2))


# Kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


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
# ground_truth = odl.phantom.shepp_logan(discr_space, modified=True)
ground_truth = donut(discr_space, smooth=True, taper=50)

#template = shepp_logan_2d(discr_space, modified=True)
template = odl.phantom.disc_phantom(discr_space, smooth=True, taper=50)

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

"""
Example of shape-based image reconstruction using linearized deformations.
"""

import odl
import time
import matplotlib.pyplot as plt
from operators import (vectorial_ft_shape_op, snr, DisplacementOperator,
                       ShapeRegularizationFunctional)
from phantoms import donut
import numpy as np


# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


# Compute the FT of the vectorized kernel function in space
def kernel_ft(space, kernel):
    """Compute the n-D Fourier transform of the discrete kernel ``K``.

    Calculate the n-D Fourier transform of the discrete kernel ``K`` on the
    control grid points {y_i} to its reciprocal points {xi_i}.
    """
    # Create the array of kernel values on the control grid points
    pspace = space.tangent_bundle
    discretized_kernel = pspace.element(
        [space.element(kernel) for _ in range(space.ndim)])

    return vectorial_ft_op(discretized_kernel)


# Give kernel function in matrix form
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


# Create 2-D discretization reconstruction space
# The size of the domain should be proportional to the given images
# 128 for shepp-logan
discr_space = odl.uniform_discr([-16, -16], [16, 16], [101, 101],
                                dtype='float32', interp='linear')

# Create 2-D discretization space for control points
cptsspace = odl.uniform_discr([-16, -16], [16, 16], [101, 101],
                              dtype='float32', interp='linear')

# Create discretization space for vector field
vspace = cptsspace.tangent_bundle

# Create the ground truth as the Shepp-Logan phantom
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
xray_trafo_op = odl.tomo.RayTransform(discr_space, geometry,
                                      impl='astra_cuda')

# Create projection data by given setting
proj_data = xray_trafo_op(ground_truth)

# Create white Gaussian noise
noise = 0.1 * odl.phantom.white_noise(xray_trafo_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Do the backprojection reconstruction
backproj = xray_trafo_op.adjoint(noise_proj_data)

# FFT setting for regularization shape term, 1 means 100% padding
vectorial_ft_op = vectorial_ft_shape_op(cptsspace)

# Initialize deformation vector field
momenta = vspace.zero()

# Fix the sigma parameter in the kernel
sigma = 4.0

# Compute Fourier trasform of the kernel function in shape regularization term
ft_kernel_shape = kernel_ft(cptsspace, kernel)

# Create displacement operator
displacement_op = DisplacementOperator(vspace, cptsspace.grid,
                                       discr_space, ft_kernel_shape)

# Compute the displacement at momenta
displ = displacement_op(momenta)

# Create linearized deformation operator
linear_deform_op = odl.deform.LinDeformFixedTempl(template)

# Compute the deformed template
deformed_template = linear_deform_op(displ)

# Create X-ray transform operator
proj_deformed_template = xray_trafo_op(deformed_template)

# Create L2 data matching (fitting) term
l2_data_fit_func = odl.solvers.L2NormSquared(xray_trafo_op.range).translated(noise_proj_data)

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
niter = 100

callback = odl.solvers.CallbackShow('iterates', display_step=5) & \
    odl.solvers.CallbackPrintIteration()

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
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max()), plt.axis('off')
plt.colorbar()
plt.title('Template')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(deformed_template), cmap='bone',
           vmin=np.asarray(deformed_template).min(),
           vmax=np.asarray(deformed_template).max()), plt.axis('off')
plt.colorbar()
plt.title('Reconstructed result')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()), plt.axis('off')
plt.colorbar()
plt.title('Ground truth')

plt.subplot(2, 3, 4)
plt.plot(np.asarray(proj_data)[0], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[0], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -3, 10])
plt.grid(True, linestyle='--')

plt.subplot(2, 3, 5)
plt.plot(np.asarray(proj_data)[1], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[1], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -3, 10])
plt.grid(True, linestyle='--')

plt.subplot(2, 3, 6)
plt.plot(np.asarray(proj_data)[2], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[2], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -3, 10])
plt.grid(True, linestyle='--')

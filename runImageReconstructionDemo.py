#  runImageReconstructionDemo.py
#
# This script will create phaseless measurements from a test image, and
# then recover the image using phase retrieval methods.  We now describe
# the details of the simple recovery problem that this script implements.
#
#                         Recovery Problem
# This script loads a test image, and converts it to grayscale.
# Measurements of the image are then obtained by applying a linear operator
# to the image, and computing the magnitude (i.e., removing the phase) of
# the linear measurements.
#
#                       Measurement Operator
# Measurement are obtained using a linear operator, called 'A', that
# obtains masked Fourier measurements from an image.  Measurements are
# created by multiplying the image (coordinate-wise) by a 'mask,' and then
# computing the Fourier transform of the product.  There are 8 masks,
# each of which is an array of binary (+1/-1) variables. The output of
# the linear measurement operator contains the Fourier modes produced by
# all 8 masks.  The measurement operator, 'A', is defined as a separate
# function near the end of the file.  The adjoint/transpose of the
# measurement operator is also defined, and is called 'At'.
#
#                         Data structures
# PhasePack assumes that unknowns take the form of vectors (rather than 2d
# images), and so we will represent our unknowns and measurements as a
# 1D vector rather than 2D images.
#
#                      The Recovery Algorithm
# The image is recovered by calling the method 'solvePhaseRetrieval', and
# handing the measurement operator and linear measurements in as arguments.
# A struct containing options is also handed to 'solvePhaseRetrieval'.
# The entries in this struct specify which recovery algorithm is used.
#
# For more details, see the Phasepack user guide.
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017


from numpy.linalg import norm
from imageio import imread
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

from phasepack.util import Options, ConvMatrix
from phasepack.solvers import solvePhaseRetrieval

#########################################################################
# Measurement Operator Definition
#########################################################################
# Create a measurement operator that maps a vector of pixels into Fourier
# measurements using the random binary masks defined above.
def Afunc(pixels, masks):
    # The reconstruction method stores iterates as vectors, so this
    # function needs to accept a vector as input.  Let's convert the vector
    # back to a 2D image.
    num_fourier_masks, numrows, numcols = masks.shape
    im = pixels.reshape(numrows, numcols)
    # Allocate space for all the measurements
    measurements = np.zeros((num_fourier_masks, numrows, numcols), dtype='complex128')
    # Loop over each mask, and obtain the Fourier measurements
    for m in range(num_fourier_masks):
        this_mask = masks[m, ...]
        measurements[m, ...] = np.fft.fft2(im*this_mask)
    # Convert results into vector format
    measurements = measurements.reshape(-1, 1)
    return measurements

# The adjoint/transpose of the measurement operator

def Atfunc(measurements, masks):
    # The reconstruction method stores measurements as vectors, so we need
    # to accept a vector input, and convert it back into a 3D array of
    # Fourier measurements.
    num_fourier_masks, numrows, numcols = masks.shape
    measurements = measurements.reshape(num_fourier_masks, numrows, numcols)
     # Allocate space for the returned value
    im = np.zeros((numrows, numcols))
    for m in range(num_fourier_masks):
        this_mask = masks[m, ...]
        this_measurements = measurements[m, ...]
        im = im + this_mask*np.fft.ifft2(this_measurements)*numrows*numcols
    # Vectorize the results before handing them back to the reconstruction
    # method
    return im.reshape(-1, 1)

# Specify the target image and number of measurements/masks
image = imread('data/logo.jpg')      # Load the image from the 'data' folder.
image = color.rgb2gray(image) # convert image to grayscale
num_fourier_masks = 8               # Select the number of Fourier masks

# Create 'num_fourier_masks' random binary masks. Store them in a 3d array.
numrows, numcols = image.shape # Record image dimensions
random_vars = rand(num_fourier_masks, numrows, numcols) # Start with random variables
masks = (random_vars<.5)*2 - 1  # Convert random variables into binary (+1/-1) variables
mv = lambda pixels: Afunc(pixels, masks)
rmv= lambda measurements: Atfunc(measurements, masks)

# Compute phaseless measurements
# Note, the measurement operator 'A', and it's adjoint 'At', are defined
# below as separate functions
x = image.reshape(-1, 1)   # Convert the signal/image into a vector so PhasePack can handle it
# b = abs(A(x)) Use the measurement operator 'A', defined below, to obtain phaseless measurements.
b = np.abs(Afunc(x, masks))
A = ConvMatrix(mv=mv, rmv=rmv, shape=(numrows*numcols*num_fourier_masks, numrows*numcols))

# Run the Phase retrieval Algorithm
# Set options for PhasePack - this is where we choose the recovery algorithm.
opts = Options(algorithm = 'Fienup',          # Use the truncated Wirtinger flow method to solve the retrieval
                                              # problem.  Try changing this to 'Fienup'.
               initMethod = 'optimal',        # Use a spectral method with optimized data pre-processing
                                              # to generate an initial starting point for the solver.
               tol = 1E-3,                    # The tolerance - make this smaller for more accurate
                                              # solutions, or larger for faster runtimes.
               verbose = 2)                   # Print out lots of information as the solver runs
                                              # (set this to 1 or 0 for less output)
print('Running %s algorithm\n' % opts.algorithm)
# Call the solver using the measurement operator 'A', its adjoint 'At', the
# measurements 'b', the length of the signal to be recovered, and the
# options.  Note, this method can accept either function handles or
# matrices as measurement operators.   Here, we use function handles
# because we rely on the FFT to do things fast.
x, outs, opts = solvePhaseRetrieval(A=A, b0=b, n=x.size, opts=opts)

# Convert the vector output back into a 2D image
recovered_image = x.reshape(numrows, numcols)

# Phase retrieval can only recover images up to a phase ambiguity.
# Let's apply a phase rotation to align the recovered image with the
# original so it looks nice when we display it.
rotation = (recovered_image.conjugate().T@image)/np.abs(recovered_image.conjugate().T@image)
recovered_image = np.real(rotation*recovered_image)

# Print some useful info to the console
print('Image recovery required %d iterations (%f secs)\n' % (outs.iterationCount, outs.solveTimes[-1]))
# Plot results
fig, axes = plt.subplots(1, 3)
# Plot the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
# Plot the recovered image
axes[1].imshow(np.real(recovered_image))
axes[1].set_title('Recovered Image')
plt.show()

"""
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
"""
# import cProfile
from numpy.linalg import norm
from imageio import imread
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import time
from phasepack.util import Options, ConvMatrix
from phasepack.solvers import solvePhaseRetrieval
import scipy
#########################################################################
# Measurement Operator Definition
#########################################################################
# Create a measurement operator that maps a vector of pixels into Fourier
# measurements using the random binary masks defined above.

def Afunc(pixels, masks):
    """ The fourier mask matrix operator
    As the reconstruction method stores iterates as vectors, this
    function needs to accept a vector as input.
    """
    nmasks, numrows, numcols = masks.shape
    im = pixels.reshape(numrows, numcols)
#    measurements = np.array([np.fft.fft2(im*m).reshape(numrows*numcols,)
#                             for m in masks]).reshape(-1,1)
    measurements = np.fft.fft2(masks*im)
    return measurements.reshape(nmasks*numrows*numcols, 1)

# The adjoint/transpose of the measurement operator

def Atfunc(measurements, masks):
    # The reconstruction method stores measurements as vectors, so we need
    # to accept a vector input, and convert it back into a 3D array of
    # Fourier measurements.
    nmasks, numrows, numcols = masks.shape
    measurements = measurements.reshape(nmasks, numrows, numcols)
     # Allocate space for the returned value
    #im = np.zeros((numrows, numcols))
#    im = np.array([np.fft.ifft2(measurements[m, ...])*masks[m, ...]*
#                               numrows*numcols for m in range(nmasks)])
#    im_out = sum(im).reshape(-1, 1)
#    print(im_out.shape)
    im = np.fft.ifft2(measurements)*masks*numrows*numcols
    im_out = np.sum(im,axis=0).reshape(-1, 1)
#    print(im_out.shape)
    return im_out

# Specify the target image and number of measurements/masks
image = imread('data/logo.jpg')      # Load the image from the 'data' folder.
image = color.rgb2gray(image) # convert image to grayscale
num_fourier_masks = 16              # Select the number of Fourier masks

# Create 'num_fourier_masks' random binary masks. Store them in a 3d array.
numrows, numcols = image.shape # Record image dimensions
random_vars = rand(num_fourier_masks, numrows, numcols) # Start with random
                                                        # variables
masks = (random_vars<.5)*2 - 1  # Convert random variables into binary (+1/-1)
                                # variables
mv = lambda pixels: Afunc(pixels, masks)
rmv= lambda measurements: Atfunc(measurements, masks)
# Nete, the meanurement operator 'A', and it's adjoint 'At', are defined
# below as separate functions
x = image.reshape(-1, 1)   # Convert the signal/image into a vector so PhasePac
                           #k can handle it
# b = abs(A(x)) Use the measurement operator 'A', defined below, to obtain
# phaseless measurements.
b = np.abs(Afunc(x, masks))
A = ConvMatrix(mv=mv, rmv=rmv, shape=(numrows*numcols*num_fourier_masks,
                                     numrows*numcols))
# Run the Phase retrieval Algorithm
# Set options for PhasePack - this is where we choose the recovery algorithm.
opts = Options(algorithm = 'twf',      # Use the truncated Wirtinger flow
                                       # method to solve the retrieval
                                       # problem. Try changing to 'Fienup'.
               initMethod = 'optimal', # Use a spectral method with optimized

                                       # data pre-processing to generate an
                                       # initial starting point for the solver.
               tol = 1E-2,             # The tolerance - make this smaller for
                                       # more accurate solutions, or larger
                                       # for faster runtimes.
               verbose = 2)            # Print out lots of information as the
                                       # solver runs (set this to 1 or 0 for
                                       # less output)
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
rotation = (recovered_image.conjugate().T@image)/\
            np.abs(recovered_image.conjugate().T@image)
print(rotation.shape)
recovered_image = np.real(recovered_image)



# Print some useful info to the console
print('Image recovery required %d iterations (%f secs)\n'
      % (outs.iterationCount, outs.solveTimes[-1]))
# Print some useful info to the console
print('Image recovery required %d iterations (%f secs)\n' %
      (outs.iterationCount, outs.solveTimes[-1]))
# Plot results
fig, axes = plt.subplots(1, 3)
# Plot the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
# Plot the recovered image
axes[1].imshow(np.real(recovered_image))
axes[1].set_title('Recovered Image')
plt.show()

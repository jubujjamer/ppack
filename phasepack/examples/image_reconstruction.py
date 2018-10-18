"""
This script will create phaseless measurements from a test image, and
then recover the image using phase retrieval methods.  We now describe
the details of the simple recovery problem that this script implements.

                        Recovery Problem
This script loads a test image, and converts it to grayscale.
Measurements of the image are then obtained by applying a linear operator
to the image, and computing the magnitude (i.e., removing the phase) of
the linear measurements.

                      Measurement Operator
Measurement are obtained using a linear operator, called 'A', that
obtains masked Fourier measurements from an image.  Measurements are
created by multiplying the image (coordinate-wise) by a 'mask,' and then
computing the Fourier transform of the product.  There are 8 masks,
each of which is an array of binary (+1/-1) variables. The output of
the linear measurement operator contains the Fourier modes produced by
all 8 masks.  The measurement operator, 'A', is defined as a separate
function near the end of the file.  The adjoint/transpose of the
measurement operator is also defined, and is called 'At'.

                        Data structures
PhasePack assumes that unknowns take the form of vectors (rather than 2d
images), and so we will represent our unknowns and measurements as a
1D vector rather than 2D images.

                     The Recovery Algorithm
The image is recovered by calling the method 'solve_phase_retrieval', and
handing the measurement operator and linear measurements in as arguments.
A struct containing options is also handed to 'solve_phase_retrieval'.
The entries in this struct specify which recovery algorithm is used.

For more details, see the Phasepack user guide.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""
# import cProfile
from imageio import imread
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import time
import scipy

from phasepack.containers import Options
from phasepack.matops import ConvolutionMatrix, FourierOperator
from phasepack.retrieval import Retrieval

## Build a test problem
# Specify the target image and number of measurements/masks
image = imread('phasepack/data/logo.jpg')  # Load the image from the 'data' folder.
image = color.rgb2gray(image)    # Convert image to grayscale.
num_fourier_masks = 16           # Select the number of Fourier masks.
numrows, numcols = image.shape # Image dimensions
# In this example we create masks consisting of random PSF's
random_vars = rand(num_fourier_masks, numrows, numcols)
masks = (random_vars<.5)*2 - 1  # Convert into binary (+1/-1) variables
# Create the selected number of random binary masks, and build transformation
# operators using the methods from the FourierOperator class.
fo = FourierOperator(masks)
mv = fo.mv # Operator vector
rmv = fo.rmv # Transposed operator
x = image.reshape(-1, 1)   # Convert the signal/image
# Use the measurement operator mv, to obtain phaseless measurements.
b = np.abs(mv(x))
A = ConvolutionMatrix(mv=mv, rmv=rmv, shape=(numrows*numcols*num_fourier_masks,
                                     numrows*numcols))

## Run the Phase retrieval Algorithm
# Set options for PhasePack - this is where we choose the recovery algorithm.
opts = Options(algorithm = 'twf', init_method = 'optimal', tol = 1E-3,
               verbose = 2)
# Create an instance of the phase retrieval class, which manages initializers
# and selection of solvers acording to the options provided.
retrieval = Retrieval(A, b, opts)
# Call the solver using the measurement operator all the information provided
# by the ConvolutionMatrix 'A'.
x, outs, opts = retrieval.solve_phase_retrieval()
# Convert the vector output back into a 2D image
recovered_image = x.reshape(numrows, numcols)
# Phase retrieval can only recover images up to a phase ambiguity.
# Let's apply a phase rotation to align the recovered image with the original
# so it looks nice when we display it.
rotation = (recovered_image.conjugate().T@image)/\
            np.abs(recovered_image.conjugate().T@image)
recovered_image = np.real(recovered_image)

## Print and plot results
print('Image recovery required %d iterations (%f secs)\n' %
      (outs.iteration_count, outs.solve_times[-1]))
fig, axes = plt.subplots(1, 3)
# Plot the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
# Plot the recovered image
axes[1].imshow(np.real(recovered_image))
axes[1].set_title('Recovered Image')
plt.show()

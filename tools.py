# -*- coding: utf-8 -*-
__author__ = 'Domen Gorjup'

'''
Various tools, used by the pyDIC application.
'''

import glob
import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.interpolate import RectBivariateSpline
import collections
import matplotlib.pyplot as plt
import dic

def allfiles(path, filetype='.tif'):
    '''
    Get all fles of filetype in specified folder.

    :param path: path to folder
    :param filetype: file type extension
    :return: list of all files of specified type in specified folder
    '''
    query = path+'/*'+filetype
    return glob.glob(query)

def get_sequence(path, filetype='.tif', nmax=None, dstack=False):
    '''
    Get a sequence of image files as 3D numpy array.

    :param path: path to folder
    :param filetype: file type extension
    :param nmax: maximum number of images in sequence
    ;param dstack: True to return 3D (x,y,t) image sequence array
    :return: 3D array of image sequence
    '''
    files = allfiles(path, filetype)
    if not files:
        return []

    # Only display 100 or less images:
    if nmax and len(files) > nmax:
        for_display = files[::len(files)//nmax]
    else:
        for_display = files
    imarrays = [scipy.ndimage.imread(file) for file in for_display]
    if dstack:
        imarrays = np.dstack(imarrays).transpose((2, 1, 0))

    return imarrays

def get_info(path):
    '''
    Get info from .cih file in path, return it as dict.

    :param path: Path to folder with .cih file and image data.
    :return: info_dict: .cih file contents as dict.
    '''
    wanted_info = ['Date',
                   'Camera Type',
                   'Record Rate(fps)',
                   'Shutter Speed(s)',
                   'Total Frame',
                   'Image Width',
                   'Image Height',
                   'File Format',
                   'EffectiveBit Depth',
                   'Comment Text']

    query = path+'/*.cih'
    cih_file = glob.glob(query)[0]

    info_dict = collections.OrderedDict([])

    with open (cih_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip().split(' : ')
            if line[0] in wanted_info:
                key, value = line[0], line[1]
                info_dict[key] = bytes(value, "utf-8").decode("unicode_escape") # Evaluate escape characters

    return info_dict


def get_integer_translation(path, roi_reference, roi_size, initial_only=False, n_im=0, progressBar=None):
    '''
    Quickly get integer precision rigid body translation data from image, using FFT cross correlation.
    Only valid for object with minimal rotations and deformation.

    :param path: Path to directory containing .tif image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param initial_only: If True, only the initial guess is returned.
    :param n_im: Number of images extracted from original sequence. If 0, whole sequence is used.
    :return: trans: Array of integer precision translation data, extracted fro mimage sequence, shaped [y, x].
    '''

    path_list = allfiles(path)

    if n_im:
        inc = len(path_list)//n_im + 1
    else:
        inc = 1

    path_list = path_list[::inc]
    image = scipy.ndimage.imread(path_list[0])
    roi_image = _get_roi_image(image, roi_reference, roi_size)
    initial_guess = dic.get_initial_guess(image, roi_image)[0]
    if initial_only:
        return initial_guess

    trans = np.array([[0, 0]], dtype=int)
    for i in range(len(path_list)-1):
        image = scipy.ndimage.imread(path_list[i+1])
        new_trans = (dic.get_initial_guess(image, roi_image)[0] - initial_guess).astype(int)
        trans = np.vstack((trans, new_trans+trans[i]))
        roi_image = _get_roi_image(image, roi_reference, roi_size)

        if progressBar:
            progressBar.setValue((i+1)/len(path_list)*100)

    return trans, inc


def get_rigid_movement(path, roi_reference, roi_size, progressBar=None, tol=1e-5, maxiter=1000, int_order=1, increment=1):
    '''
    Get rigid body movement (translation and rotation) data from image, using the Newton-Gauss optimization method with
    a Zero Normalized Cross Correlation based DIC algorithm.

    :param path: Path to directory containing .tif image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param tol: Convergence condition (maximum parameter iteration vector norm).
    :param int_order: Bivariate spline interpolation order.
    :param increment: Only read every n-th image from sequence.
    :return: results: Numpy array, containing extracted data, shaped as [y, x, phi] arrays at given images.
    :return: iters: Array, number of iterations required to reach converegence for each image pair.
    '''
    path_list = allfiles(path)[::increment]                 # List of images in folder

    # Precomputable stuff:
    F = scipy.ndimage.imread(path_list[0])                  # Initial image of the sequence is only used once.
    G = scipy.ndimage.imread(path_list[1])                  # First target image, used for the initial guess.
    ROI = _get_roi_image(F, roi_reference, roi_size)        # First ROI image, used for the initial guess.
    ROI_st = (np.mean(ROI), np.std(ROI))                    # Mean and standard deviation of ROI gray values.
    in_guess = dic.get_initial_guess(G, ROI)[0]             # Cross-correlation initial guess is only used once.
    jac = dic.jacobian_rigid(roi_size[0], roi_size[1])      # The Jacobian is constant throughout the computation.
    grad = dic.get_gradient(ROI)                            # Compute gradient images of ROI.
    sd_im = dic.sd_images(grad, jac)                        # Steepest descent images for the current ROI.
    H = dic.hessian(sd_im, n_param=3)                       # Hessian matrix for the current ROI.
    inv_H = np.linalg.inv(H)                                # Inverse of Hessian.

    # Loop through all the images in given folder:
    for i in range(2, len(path_list)):                      # First two images were loaded already.

        h, w = G.shape                                      # Get the shape of the current image for interpolation.
        spl = RectBivariateSpline(x=np.arange(h),           # Calculate cubic bivariate spline interpolation of G.
                                  y=np.arange(w),
                                  z=G,
                                  kx=int_order,
                                  ky=int_order,
                                  s=0)

        err = 1.                                            # Initialize the convergence condition.
        niter = 0                                           # Initialize optimization loop iteration counter.
        # Optimization loop:
        while err > tol and niter < maxiter:
            if not 'p' in locals():                             # If this is the first iteration:
                p = np.array([in_guess[0], in_guess[1], 0.])    # Set initial parameters to in_guess.
                warped_ROI = _get_roi_image(G, in_guess, roi_size)  # Since in_guess are integer, extract new ROI directly.
                warp = dic.rigid_transform_matrix(p)            # Get the affine transformation matrix form initial p.
            else:                                               # Else, use last computed parameters, interpolate new ROI.
                xi, yi = dic.coordinate_warp(warp, roi_size)    # Get warped coordinates to new ROI, using last optimal warp.
                warped_ROI = dic.interpolate_warp(xi, yi,       # Compute new ROI image by interpolating the reference image
                                                  target=G,
                                                  output_shape=roi_size,
                                                  spl=spl,
                                                  order=int_order)
            error_im = dic.get_error_image(ROI, ROI_st, warped_ROI)  # Compute error image, according to the ZNSSD criterion.
            b = dic.get_sd_error_vector(sd_im, error_im, 3)     # Compute the right-side vector in the optimization system.
            dp = np.dot(inv_H, b)                               # Compute the optimal transform parameters increment.
            err = np.linalg.norm(dp)                            # Incremental parameter norm = convergence criterion.
            dp_warp = dic.rigid_transform_matrix(dp)            # Construct the increment transformation matrix.
            inverse_increment_warp = np.linalg.inv(dp_warp)     # Construct the inverse of increment warp materix.
            warp = np.dot(warp, inverse_increment_warp)         # The updated iteration of warp matrix.
            p = dic.param_from_rt_matrix(warp)                  # The updated iteration of transformation parameters.
            niter += 1                                          # Update the optimization loop iteration counter.

        if i == 2:                                                  # If this is the first optimization iteration:
            p_ref = p                                               # Initialize a reference for all following calculations.
            results = np.array([[0, 0, 0]], dtype=np.float64)       # Initialize the results array.
            iters = np.array([], dtype=int)                         # Initialize array of iteration counters.
        else:                                                       # If optimization had already been initialized:
            p_relative = p - p_ref                                  # Calculate the relative translation and rotation.
            results = np.vstack((results, p_relative))              # Update the results array.
            iters = np.append(iters, niter)                         # Append current iteration number to list.

        if progressBar:
            progressBar.setValue(i / len(path_list) * 100)  # Update the progress bar.

        G = scipy.ndimage.imread(path_list[i])              # Load the next target image.

    if np.max(iters) >= maxiter:                            # If maximum number of iterations was reached:
        iters = np.append(iters, np.argmax(iters))          # Append index of first iteration number maximum.
        iters = np.append(iters, 0)                         # Append 0 to the iters list (warning signal!)

    print('Max niter: ', np.max(iters), ' (mean, std: ',np.mean(iters), np.std(iters),')')
    return results, iters, increment


def _get_roi_image(target, roi_reference, roi_size):
    '''
    Get 2D ROI array from target image, ROI position and size.

    :param target: Target iamge.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :return: ROI image (2D numpy array).
    '''
    ul = roi_reference   # Upper left vertex of ROI
    roi_image = target[ul[0]:ul[0]+roi_size[0], ul[1]:ul[1]+roi_size[1]]
    return roi_image

def plot_data(data):
    '''
    Temporary function for DIC results visualization, using matplotlib.

    :param tyx: Array containing DIC result data.
    :return:
    '''

    # # LaTeX text backend
    # import matplotlib
    # rc = matplotlib.rc
    # font = {'family': 'serif',
    #         'serif': 'CM',
    #         'size': 16}
    # rc('font', **font)
    # rc('text', usetex=True)
    # rc('text.latex', unicode=True)

    t = data[:, 0]
    ty = data[:, 1]
    tx = data[:, 2]
    n = len(t)

    if len(data[0]) == 3:
        plt.figure()
        plt.plot(t, ty, label=r'$y$')
        plt.plot(t, tx, label=r'$x$')
        plt.legend()
        plt.grid()
        plt.show()

    elif len(data[0]) == 4:

        phi = data[:, 3]
        f, ax = plt.subplots(3, 1)
        ax[0].plot(t, ty-np.mean(ty), label=r'$y$')
        ax[0].plot(t, tx-np.mean(tx), label=r'$x$')
        ax[0].set_ylabel(r'$x, y$ [piksel]')
        ax[0].set_xlabel(r'$t$ [s]')
        ax[0].legend()
        ax[0].grid()
        ax[1].plot(t, phi-np.mean(phi), label=r'$\alpha$')
        ax[1].grid()
        ax[1].set_ylabel(r'$\alpha$ [rad]')
        ax[1].set_xlabel(r'$t$ [s]')
        ax[2].plot(np.fft.rfftfreq(10*n, t[1]-t[0]), np.real(np.fft.rfft(ty-np.mean(ty), 10*n)))
        ax[2].grid()
        ax[2].set_xlabel(r'$f$ [Hz]')
        ax[2].set_ylabel(r'$Re(c_n)$ [/]')
        ax[2].set_xlim([0, 200])
        plt.tight_layout()
        plt.show()


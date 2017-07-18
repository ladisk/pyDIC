# -*- coding: utf-8 -*-
__author__ = 'Domen Gorjup'

'''
Various tools, used by the pyDIC application.
'''

import os
import sys
import glob
import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.interpolate import RectBivariateSpline
import collections
import matplotlib.pyplot as plt
import dic
import warnings

# Disable warnings printout
warnings.filterwarnings("ignore")


def allfiles(path, filetype='.tif'):
    '''
    Get all fles of filetype in specified folder.

    :param path: path to folder
    :param filetype: file type extension
    :return: list of all files of specified type in specified folder
    '''
    query = path+'/*'+filetype
    return glob.glob(query)


def get_sequence(mraw_path, file_shape, nmax=None, offset=0):
    '''
    Get a sequence of image files as 3D numpy array.

    :param mraw_path: path to .mraw file containing image data
    :param file_shape: tuple, (ntotal, height, width) of images in .mraw file
    :param nmax: maximum number of images in sequence
    :param offset: First image to be read
    :return: 3D array of image sequence
    '''
    ntotal, h, w = file_shape
    byte_size = 2*h*w                   # Number of bytes for one image 
    byte_offset = offset * byte_size    # Offset to first byte to be read

    # If only a single image was requested:
    if nmax and nmax == 1:
        with open(mraw_path, 'rb') as mraw:
            imarray = np.memmap(mraw, dtype=np.uint16, offset=byte_offset, mode='r', shape=(h, w))
    # Only display nmax or less images:
    elif nmax and ntotal > nmax:
        image_step = ntotal//nmax
        with open(mraw_path, 'rb') as mraw:
            memmap = np.memmap(mraw, dtype=np.uint16, offset=byte_offset, mode='r', shape=(ntotal-offset, h, w))
            imarray = memmap[::image_step, :, :]
    # If there are less than nmax images:
    else:
        with open(mraw_path, 'rb') as mraw:
            imarray = np.memmap(mraw, dtype=np.uint16, offset=byte_offset, mode='r', shape=(ntotal-offset, h, w))

    return imarray


def _tiff_to_temporary_file(dir_path):
    '''
    Saves all .tif files inside direcotry into a .npy file to read as memmap.

    :param dir_path: Path to direcotry containing .tif image files.
    :return out_file: Path to output .npy file.
    return cih_file: Path to generated .cih file
    '''
    dir_path = dir_path.replace('\\', '/')

    im_path = glob.glob(dir_path + '/*.tif') + glob.glob(dir_path + '/*.tiff') 
    out_file = os.path.join(dir_path, '_images.npy')

    if not os.path.isfile(out_file):
        with open(out_file, 'wb') as file:
            for image_file in im_path:
                image = scipy.ndimage.imread(image_file).astype(np.uint16)
                image.tofile(file)
        file_shape = (len(im_path), image.shape[0], image.shape[1])

    if len(glob.glob(dir_path + '/*.cih')) == 0:
        wanted_info = ['Date', 'Camera Type', 'Record Rate(fps)', 'Shutter Speed(s)', 'Total Frame', 
        'Image Width', 'Image Height', 'File Format', 'EffectiveBit Depth', 'Color Bit', 'Comment Text']
        image_info = {'Date': '/',
                    'Camera Type': '/',
                    'Record Rate(fps)': '{:d}'.format(1),
                    'Shutter Speed(s)': '{:.6f}'.format(1),
                    'Total Frame': '{:d}'.format(int(len(im_path))),
                    'Image Width': '{:d}'.format(image.shape[1]),
                    'Image Height': '{:d}'.format(image.shape[0]),
                    'File Format' : 'tif',
                    'EffectiveBit Depth': '12',
                    'Comment Text': 'Reading from .tiff.<br>Modify measurement info in<br>created .cih file if necessary.',
                    'Color Bit': '16'}

        cih_file = os.path.join(dir_path, 'slika.cih')
        with open(cih_file, 'w') as file:
            for key in wanted_info:
                file.write('{:s} : {:s}\n'.format(key, image_info[key]))
    else:
        cih_file = glob.glob(dir_path + '/*.cih')[0]
    
    return out_file, cih_file


def get_info(path):
    '''
    Get info from .cih file in path, return it as dict.

    :param path: Path to .cih file.
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
                   'Comment Text',
                   'Color Bit']

    info_dict = collections.OrderedDict([])

    with open(path, 'r') as file:
        for line in file:
            line = line.rstrip().split(' : ')
            if line[0] in wanted_info:                
                key, value = line[0], line[1]#[:20]
                info_dict[key] = bytes(value, "utf-8").decode("unicode_escape") # Evaluate escape characters

    return info_dict


def get_integer_translation(mraw_path, roi_reference, roi_size, file_shape, initial_only=False, n_im=0, progressBar=None):
    '''
    Quickly get integer precision rigid body translation data from image, using FFT cross correlation.
    Only valid for object with minimal rotations and deformation.

    :param mraw_path: Path to .mraw file containing image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param file_shape: Tuple, (ntotal, height, width) of images in .mraw file.
    :param initial_only: If True, only the initial guess is returned.
    :param n_im: Number of images to be extracted from original sequence. If 0, whole sequence is used.
    :return: trans: Array of integer precision translation data, extracted fro mimage sequence, shaped [y, x].
    '''
    ntotal, h, w = file_shape
    with open(mraw_path, 'rb') as mraw:
        memmap = np.memmap(mraw, dtype=np.uint16, mode='r', shape=file_shape)
    if n_im:
        inc = ntotal//n_im + 1
    else:
        inc = 1

    imarray = memmap[::inc, :, :]
    image = imarray[0]
    roi_image = _get_roi_image(image, roi_reference, roi_size)
    initial_guess = dic.get_initial_guess(image, roi_image)[0]
    if initial_only:
        return initial_guess

    trans = np.array([[0, 0]], dtype=int)
    for i in range(imarray.shape[0]):
        image = imarray[i]
        new_trans = (dic.get_initial_guess(image, roi_image)[0] - initial_guess).astype(int)
        trans = np.vstack((trans, new_trans+trans[i]))
        roi_image = _get_roi_image(image, roi_reference, roi_size)

        if progressBar:
            progressBar.setValue(i/ntotal*100)

    return trans, inc


def get_simple_translation(mraw_path, roi_reference, roi_size, file_shape, progressBar=None, increment=1, debug=False):
    '''
    Get translation data from image, using a Zero Normalized Cross Correlation based Lucas-Kanade algorithm..

    :param mraw_path: Path to .mraw file containing image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param file_shape: Tuple, (ntotal, height, width) of images in .mraw file.
    :param increment: Only read every n-th image from sequence.
    :return: results: Numpy array, containing extracted data, shaped as [y, x, phi] arrays at given images.
    :return: iters: Array, number of iterations required to reach converegence for each image pair.
    :param: debug: If True, display debug output.
    '''
    with open(mraw_path, 'rb') as mraw:
        memmap = np.memmap(mraw, dtype=np.uint16, mode='r', shape=file_shape)             # Map to all images in file
    
    memmap = memmap[::increment]                            # Apply sequence increment
    N_inc = len(memmap)
    errors = {}                                             # Initialize warnings dictionary

    # Precomputable stuff:
    roi_reference = np.asarray(roi_reference)
    F = _get_roi_image(memmap[0], roi_reference, roi_size)  # First ROI image, used for the initial guess.

    Fx, Fy = dic.get_gradient(F)
    Fx2 = np.sum(Fx**2)
    Fy2 = np.sum(Fy**2)
    FxFy = np.sum(Fx * Fy)
    FxF = np.sum(Fx * F)
    FyF = np.sum(Fy * F)

    mean_F = np.mean(F)
    Fi = F - mean_F
    denominator = np.sum(Fi**2)

    results = np.array([[0, 0]], dtype=np.float64)          # Initialize the results array.
    p_ref = roi_reference                                   # Initialize a reference for all following calculations.

    # Loop through all the images in .mraw file:
    for i in range(1, len(memmap)):                         # First image was loaded already.
        d_int = np.round(results[-1])                       # Last calculated integer translation.
        G = _get_roi_image(memmap[i], p_ref + d_int, roi_size) # Current image at integer location.
        mean_G = np.mean(G)
        Gi = G - mean_G
        
        # Optimization step:
        numerator = np.sum(Fi * Gi)
        a_opt = numerator / denominator
        b_opt = mean_G - mean_F * a_opt

        Gb = G - b_opt
        A = np.array([[Fx2, FxFy],
                      [FxFy, Fx2]]) * a_opt
        b = np.array([-a_opt*FxF + np.sum(Fx*Gb), 
                      -a_opt*FyF + np.sum(Fy*Gb)])
        d = np.linalg.solve(A, b) # dx, dy

        results = np.vstack((results, d_int+d[::-1])) # y, x
        
        if progressBar:
            progressBar.setValue(i / N_inc * 100)  # Update the progress bar.
        else:
            stdout_print_progress(i - 2, N_inc)  # Print progress info to stdout.

    memmap._mmap.close()                                    # Close the loaded memmap
    return results, increment


def get_rigid_movement(mraw_path, roi_reference, roi_size, file_shape, progressBar=None, tol=1e-6, maxiter=100, int_order=1, increment=1, crop=False, debug=False):
    '''
    Get rigid body movement (translation and rotation) data from image, using the Newton-Gauss optimization method with
    a Zero Normalized Cross Correlation based DIC algorithm.

    :param mraw_path: Path to .mraw file containing image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param file_shape: Tuple, (ntotal, height, width) of images in .mraw file.
    :param tol: Convergence condition (maximum parameter iteration vector norm).
    :param int_order: Bivariate spline interpolation order.
    :param increment: Only read every n-th image from sequence.
    :return: results: Numpy array, containing extracted data, shaped as [y, x, phi] arrays at given images.
    :return: iters: Array, number of iterations required to reach converegence for each image pair.
    :param: crop: Border size to crop loaded images (if 0, do not crop).
    :param: debug: If True, display debug output.
    '''
    with open(mraw_path, 'rb') as mraw:
        memmap = np.memmap(mraw, dtype=np.uint16, mode='r', shape=file_shape)             # Map to all images in file
    
    memmap = memmap[::increment]                            # Apply sequence increment
    N_inc = len(memmap)
    errors = {}                                             # Initialize warnings dictionary

    # Precomputable stuff:
    F = memmap[0]                                           # Initial image of the sequence is only used once.
    roi_reference = np.asarray(roi_reference)
    ROI = _get_roi_image(F, roi_reference, roi_size)        # First ROI image, used for the initial guess.
    ROI_st = (np.mean(ROI), np.std(ROI))                    # Mean and standard deviation of ROI gray values.
    if crop:
        crop_slice = _crop_with_border_slice(roi_reference, roi_size, crop)
        F = F[crop_slice]                                   # Crop the initial image.
    in_guess = dic.get_initial_guess(F, ROI)[0]             # Cross-correlation initial guess is only used once.
    jac = dic.jacobian_rigid(roi_size[0], roi_size[1])      # The Jacobian is constant throughout the computation.
    grad = dic.get_gradient(ROI)                            # Compute gradient images of ROI.
    sd_im = dic.sd_images(grad, jac)                        # Steepest descent images for the current ROI.
    H = dic.hessian(sd_im, n_param=3)                       # Hessian matrix for the current ROI.
    inv_H = np.linalg.inv(H)                                # Inverse of Hessian.

    results = np.array([[0, 0, 0]], dtype=np.float64)       # Initialize the results array.
    iters = np.array([], dtype=int)                         # Initialize array of iteration counters.
    p_shift = np.zeros(3, dtype=np.float64)                 # Initialize cropped image shift
    p_ref = np.array([in_guess[0], in_guess[1], 0.])        # Initialize a reference for all following calculations.

    # Loop through all the images in .mraw file:
    for i in range(1, len(memmap)):                         # First image was loaded already.
        if crop:
            roi_translation = (results[-1, :2]).astype(int)         # Last calculated integer displacement
            new_roi_reference = roi_reference + roi_translation     # Shift cropped section new position
            crop_slice = _crop_with_border_slice(new_roi_reference, roi_size, crop) # Calculate crop indices for new image section
            
            if _is_in_image(crop_slice, file_shape): # If still inside image frame
                G = memmap[i][crop_slice]                                   # Load the next target image.
            else:
                roi_translation = np.zeros(2, dtype=int)            # If crop indices outside image frame: don't crop
                G = memmap[i]

        else:
            roi_translation = np.zeros(2, dtype=int)
            G = memmap[i]

        h, w = G.shape                                      # Get the shape of the current image for interpolation.
        spl = RectBivariateSpline(x=np.arange(h),           # Calculate the bivariate spline interpolation of G.
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
            try:                                                # Singular warp matrix error handling.
                inverse_increment_warp = np.linalg.inv(dp_warp)     # Construct the inverse of increment warp materix.
                warp = np.dot(warp, inverse_increment_warp)         # The updated iteration of warp matrix.
                p = dic.param_from_rt_matrix(warp)                  # The updated iteration of transformation parameters.
            except Exception as e:
                errors[i] = {'image':G, 'ROI':warped_ROI, 'message': e, 'warp_matrix': dp_warp}
            niter += 1                                          # Update the optimization loop iteration counter.

        p_shift = np.array([roi_translation[0], roi_translation[1], 0.]) # ROI shift if cropping
        p_relative = p_shift + p - p_ref                        # Calculate the relative translation and rotation.
        results = np.vstack((results, p_relative))              # Update the results array.
        iters = np.append(iters, niter)                         # Append current iteration number to list.

        if progressBar:
            progressBar.setValue(i / N_inc * 100)  # Update the progress bar.
        else:
            stdout_print_progress(i - 2, N_inc)  # Print progress info to stdout.

        if debug:   # DEBUG
            if len(errors) != 0:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(G, cmap='gray', interpolation='nearest')
                ax[1].imshow(ROI, cmap='gray', interpolation='nearest')
                ax[2].imshow(warped_ROI, cmap='gray', interpolation='nearest')
                #plt.show()

    if np.max(iters) >= maxiter:                            # If maximum number of iterations was reached:
        iters = np.append(iters, np.argmax(iters))          # Append index of first iteration number maximum.
        iters = np.append(iters, 0)                         # Append 0 to the iters list (warning signal!)
    
    print('Max niter: ', np.max(iters), ' (mean, std: ', np.mean(iters), np.std(iters),')')
    memmap._mmap.close()                                    # Close the loaded memmap
    return results, errors, iters, increment


def get_affine_deformations(mraw_path, roi_reference, roi_size, file_shape, progressBar=None, tol=1e-5, maxiter=100, int_order=3, increment=1, crop=False, debug=False):
    '''
    Get defformations of te Region of Interest, using the Newton-Gauss optimization method with
    a Zero Normalized Cross Correlation based DIC algorithm.

    :param mraw_path: Path to .mraw file containing image data.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :param file_shape: Tuple, (ntotal, height, width) of images in .mraw file.
    :param tol: Convergence condition (maximum parameter iteration vector norm).
    :param int_order: Bivariate spline interpolation order.
    :param increment: Only read every n-th image from sequence.
    :param: crop: Border size to crop loaded images (if 0, do not crop).
    :param: debug: If True, display debug output.
    :return: results: Numpy array, containing extracted data, shaped as [p1, p2, p3, p4, p5, p6] arrays for all images.
    :return: iters: Array, number of iterations required to reach converegence for each image pair.
    '''
    with open(mraw_path, 'rb') as mraw:
        memmap = np.memmap(mraw, dtype=np.uint16, mode='r', shape=file_shape)             # Map to all images in file

    memmap = memmap[::increment]                            # Apply sequence increment
    N_inc = len(memmap)
    errors = {}                                             # Initialize warnings dictionary

    # Precomputable stuff:
    F = memmap[0]                                           # Initial image of the sequence is only used once.
    ROI = _get_roi_image(F, roi_reference, roi_size)        # First ROI image, used for the initial guess.
    ROI_st = (np.mean(ROI), np.std(ROI))                    # Mean and standard deviation of ROI gray values.
    if crop:
        crop_slice = _crop_with_border_slice(roi_reference, roi_size, crop)
        F = F[crop_slice]                                   # Crop the initial image.
    in_guess = dic.get_initial_guess(F, ROI)[0]             # Cross-correlation initial guess is only used once.
    jac = dic.jacobian_affine(roi_size[0], roi_size[1])     # The Jacobian is constant throughout the computation.         ##
    grad = dic.get_gradient(ROI, prefilter_gauss=True)      # Compute gradient images of ROI.
    sd_im = dic.sd_images(grad, jac)                        # Steepest descent images for the current ROI.
    H = dic.hessian(sd_im, n_param=6)                       # Hessian matrix for the current ROI.                          ##
    inv_H = np.linalg.inv(H)                                # Inverse of Hessian.

    results = np.array([np.zeros(6, dtype=np.float64)])     # Initialize the results array.                                ##
    iters = np.array([], dtype=int)                         # Initialize array of iteration counters.
    p_shift = np.zeros(6, dtype=np.float64)                 # Initialize cropped image shift                               ##                                                                                   
    p_ref = np.zeros(6, dtype=np.float64)                   # Initialize a reference for all following calculations.
    p_ref[2], p_ref[5] = in_guess[1], in_guess[0]           # ...

    # Loop through all the images in .mraw file:
    for i in range(1, len(memmap)):                         # First image was loaded already.
        if crop:
            this_p = results[-1]
            roi_translation = np.array([this_p[5], this_p[2]]).astype(int)  # Last calculated integer displacement
            new_roi_reference = roi_reference + roi_translation             # Shift cropped section new position
            crop_slice = _crop_with_border_slice(new_roi_reference, roi_size, crop) # Calculate crop indices for new image section
            
            if _is_in_image(crop_slice, file_shape): # If still inside image frame
                G = memmap[i][crop_slice]                                   # Load the next target image.
            else:
                roi_translation = np.zeros(2, dtype=int)            # If crop indices outside image frame: don't crop
                G = memmap[i]

        else:
            roi_translation = np.zeros(2, dtype=int)
            G = memmap[i]

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
                p = np.zeros(6, dtype=np.float64)               # Initialize optimization parameters vector                          ##
                p[2], p[5] = in_guess[1], in_guess[0]           # Set initial parameters to in_guess.                                ##
                warped_ROI = _get_roi_image(G, in_guess, roi_size)  # Since in_guess are integer, extract new ROI directly.          
                warp = dic.affine_transform_matrix(p)           # Get the affine transformation matrix form initial p.               ##
            else:                                               # Else, use last computed parameters, interpolate new ROI.
                xi, yi = dic.coordinate_warp(warp, roi_size)    # Get warped coordinates to new ROI, using last optimal warp.
                warped_ROI = dic.interpolate_warp(xi, yi,       # Compute new ROI image by interpolating the reference image
                                                  target=G,
                                                  output_shape=roi_size,
                                                  spl=spl,
                                                  order=int_order)
            error_im = dic.get_error_image(ROI, ROI_st, warped_ROI)  # Compute error image, according to the ZNSSD criterion.
            b = dic.get_sd_error_vector(sd_im, error_im, 6)     # Compute the right-side vector in the optimization system.           ##
            dp = np.dot(inv_H, b)                               # Compute the optimal transform parameters increment.
            err = np.linalg.norm(dp)                            # Incremental parameter norm = convergence criterion.
            dp_warp = dic.affine_transform_matrix(dp)           # Construct the increment transformation matrix.                      ##
            try:                                                # Singular warp matrix error handling.
                inverse_increment_warp = np.linalg.inv(dp_warp)     # Construct the inverse of increment warp materix.
                warp = np.dot(warp, inverse_increment_warp)         # The updated iteration of warp matrix.
                p = dic.param_from_affine_matrix(warp)                  # The updated iteration of transformation parameters.         ##
            except Exception as e:
                errors[i] = {'image':G, 'ROI':warped_ROI, 'message': e, 'warp_matrix': dp_warp}
            niter += 1                                          # Update the optimization loop iteration counter.
        
        p_shift = np.zeros(6, dtype=np.float64)
        p_shift[2], p_shift[5] = roi_translation[1], roi_translation[0]
        p_relative = p_shift + p - p_ref                        # Calculate the relative translation and rotation.
        results = np.vstack((results, p_relative))              # Update the results array.
        iters = np.append(iters, niter)                         # Append current iteration number to list.

        if progressBar:
            progressBar.setValue(i / N_inc * 100)  # Update the progress bar.
        else:
            stdout_print_progress(i-2, N_inc)      # Print progress info to stdout.
        
        if debug:   # DEBUG
            if len(errors) != 0:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(G, cmap='gray', interpolation='nearest')
                ax[1].imshow(ROI, cmap='gray', interpolation='nearest')
                ax[2].imshow(warped_ROI, cmap='gray', interpolation='nearest')
                #plt.show()

    if np.max(iters) >= maxiter:                            # If maximum number of iterations was reached:
        iters = np.append(iters, np.argmax(iters))          # Append index of first iteration number maximum.
        iters = np.append(iters, 0)                         # Append 0 to the iters list (warning signal!)

    print('Max niter: ', np.max(iters), ' (mean, std: ', np.mean(iters), np.std(iters), ')')
    memmap._mmap.close()                                    # Close the loaded memmap
    return results, errors, iters, increment


def get_GL_strains(p):
    '''
    Get Green-Lagrangian strains from displacement gradients in p.

    :param p: The 6 affine warp parameters.
        p = [du/dx, du/dy, u, dv/dx, dv/dy, v]
    :return: Array of Green-Lagrangian strain values
        E = [E_xx, E_xy, E_yy]
    '''
    E_xx = 1/2*(2*p[0] + p[0]**2 + p[3]**2)
    E_xy = 1/2*(p[1] + p[3] + p[0]*p[1] + p[3]*p[4])
    E_yy = 1/2*(2*p[4] + p[1]**2 + p[4]**2)
    return np.array([E_xx, E_xy, E_yy], dtype=np.float64)


def _get_roi_image(target, roi_reference, roi_size):
    '''
    Get 2D ROI array from target image, ROI position and size.

    :param target: Target iamge.
    :param roi_reference: Upper left coordinate point of ROI, (y, x).
    :param roi_size: ROI size, (h, w) [px].
    :return: ROI image (2D numpy array).
    '''
    ul = np.array(roi_reference).astype(int)   # Upper left vertex of ROI
    m, n = target.shape
    ul = np.clip(np.array(ul), 0, [m-roi_size[0]-1, n-roi_size[1]-1])
    roi_image = target[ul[0]:ul[0]+roi_size[0], ul[1]:ul[1]+roi_size[1]]
    return roi_image


def _crop_with_border_slice(reference, size,  border=10):
    '''
    Returns a slice that crops an image according to given reference (upper left) point, section size and border size.

    :param: reference: The upper-left coordiante of the desired image section, (y, x).
    :param: size: Size of desired cropped image (y, x).
    :param border: Border size.
    :return: crop_slice: tuple of (size[0]+border[0], size[1]+border[1]) to use for slicing.
    '''
    y, x = np.array(reference).astype(int)
    m, n = np.array(size).astype(int)
    by, bx = border, border
    yslice = slice(y - by, y + m + by)
    xslice = slice(x - bx, x + n + bx)
    crop_slice = (yslice, xslice)
    return crop_slice


def _is_in_image(crop_slice, file_shape):
    '''
    Finds out if crop slice indices are still inside image boundaries.

    :param crop_slice: Slice to be use in image crop.
    :param file_shape: Shape of input image.
    :return b: True if slice still inside image boundaries.
    '''
    b = (crop_slice[0].stop < file_shape[1] and 
        crop_slice [1].stop < file_shape[2] and 
        crop_slice[0].start > 0 and
         crop_slice[1].start > 0)
    return b


def get_roi_center(roi_reference, roi_size):
    '''
    Returns the coordiantes of ROI center.

    :param roi_reference: (y, x) coordinates of ROI upper-left vertex.
    :param roi_size: (y. x) size of ROI.
    :return roi_center: (y, x) coordinates of ROI center pixel.
    '''
    return (roi_reference[0] + roi_size[0] // 2, roi_reference[1] + roi_size[1] // 2)


def stdout_print_progress(current, all):
    '''
    Prints current analysis progress, relative to full length of image sequence.

    :param current: Current image being analyzed.
    :param all: Full length of image sequence.
    :return:
    '''
    if current == 0:
        sys.stdout.write('Current image: {:d} (of {:d})'.format(current, all))
    else:
        sys.stdout.write('\r\r')
        sys.stdout.flush()
        sys.stdout.write('Current image: {:d} (of {:d})'.format(current, all))
    if current+2 == all:
        print()


def plot_data(data, unit):
    '''
    Temporary function for DIC results visualization, using matplotlib.

    :param tyx: Array containing DIC result data.
    :param unit: y-axis label unit.
    :return:
    '''

    zp_factor = 10
    lw = 1
    highpass = False
    latex = False

    if highpass:
        fc = 1/350  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        b = 1/700  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1  # Make sure that N is odd.
        n = np.arange(N)

        # Compute a low-pass filter.
        h = np.sinc(2 * fc * (n - (N - 1) / 2.))
        w = np.blackman(N)
        h = h * w
        h = h / np.sum(h)

        # Create a high-pass filter from the low-pass filter through spectral inversion.
        h = -h
        h[(N - 1) / 2] += 1
        print(len(h))

    if latex:
        # LaTeX text backend
        import matplotlib
        rc = matplotlib.rc
        font = {'family': 'serif',
                'serif': 'CM',
                'size': 16}
        rc('font', **font)
        rc('text', usetex=True)
        rc('text.latex', unicode=True)

    t = data[:, 0]
    n = len(t)

    if len(data[0]) == 3:
        ty = data[:, 1]
        tx = data[:, 2]
        plt.figure()
        plt.plot(t, ty-ty[0], label=r'$y$', lw=lw)
        plt.plot(t, tx-tx[0], label=r'$x$', lw=lw)
        plt.legend()
        plt.grid()
        plt.show()

    elif len(data[0]) == 4:
        ty = data[:, 1]
        tx = data[:, 2]
        phi = data[:, 3]
        f, ax = plt.subplots(2, 1)
        ax[0].plot(t, ty-ty[0], label=r'$y$', lw=lw)
        ax[0].plot(t, tx-tx[0], label=r'$x$', lw=lw)
        ax[0].set_ylabel(r'$x, y$ [{:s}]'.format(unit))
        ax[0].set_xlabel(r'$t$ [s]')
        ax[0].legend()
        ax[0].grid()
        ax[1].plot(t, phi-phi[0], label=r'$\alpha$', lw=lw)
        ax[1].grid()
        ax[1].set_ylabel(r'$\alpha$ [rad]')
        ax[1].set_xlabel(r'$t$ [s]')
        # ax[2].plot(np.fft.rfftfreq(zp_factor*n, t[1]-t[0]), np.real(np.fft.rfft(ty-np.mean(ty), zp_factor*n))/n*2, lw=lw)
        # ax[2].grid()
        # ax[2].set_xlabel(r'$f$ [Hz]')
        # ax[2].set_ylabel(r'$Re(c_n)$ [/]')
        # ax[2].set_xlim([0, 200])
        plt.tight_layout()
        plt.show()

    elif len(data[0]) == 7:
        #p = [du/dx, du/dy, u, dv/dx, dv/dy, v]
        u_x = data[:, 1]
        u_y = data[:, 2]
        u = data[:, 3]
        v_x = data[:, 4]
        v_y = data[:, 5]
        v = data[:, 6]

        if highpass:
            u_x = np.convolve(u_x, h, mode='same')
            u_y = np.convolve(u_y, h, mode='same')
            v_x = np.convolve(v_x, h, mode='same')
            v_y = np.convolve(v_y, h, mode='same')

        f, ax = plt.subplots(3, 1)

        ax[0].plot(t, u - np.mean(u), label=r'$u$', lw=lw)
        ax[0].plot(t, v - np.mean(v), label=r'$v$', lw=lw)
        ax[0].set_ylabel(r'$x, y$ [{:s}]'.format(unit))
        ax[0].set_xlabel(r'$t$ [s]')
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(t, u_x-np.mean(u_x), label=r'$du/dx$', lw=lw)
        ax[1].plot(t, u_y-np.mean(u_y), label=r'$du/dy$', lw=lw)
        ax[1].plot(t, v_x-np.mean(v_x), label=r'$dv/dx$', lw=lw)
        ax[1].plot(t, v_y-np.mean(v_y), label=r'$dv/dy$', lw=lw)
        ax[1].set_xlabel(r'$t$ [s]')
        ax[1].legend(loc=(0,1.01), ncol=4)
        ax[1].grid()

        # for k, v in {'u': u, 'v': v}.items():
        #     ax[2].plot(np.fft.rfftfreq(10*n, t[1]-t[0]), np.real(np.fft.rfft(v-np.mean(v), 10*n)), label=k)
        # ax[2].grid()
        # ax[2].set_xlabel(r'$f$ [Hz]')
        # ax[2].set_ylabel(r'$Re(c_n)$ [/]')
        # ax[2].set_xlim([0, 200])
        # ax[2].legend()

        for k, v in collections.OrderedDict(((r'$du/dx$', u_x), (r'$du/dy$', u_y), (r'$dv/dx$', v_x), (r'$dv/dy$', v_y))).items():
            ax[2].plot(np.fft.rfftfreq(zp_factor*n, t[1]-t[0]), 2/n*np.real(np.fft.rfft(v-np.mean(v), zp_factor*n)), label=k, lw=lw)
        ax[2].grid()
        ax[2].set_xlabel(r'$f$ [Hz]')
        ax[2].set_ylabel(r'$Re(c_n)$ [/]')
        ax[2].set_xlim([0, 15000])
        ax[2].legend(loc=(0,1.01), ncol=4)

        plt.tight_layout()
        plt.show()

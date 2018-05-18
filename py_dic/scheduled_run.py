__author__ = 'Domen Gorjup'
__package__ = 'py_dic'

import os
import numpy as np
import time
import datetime
import pickle
import logging
from . import dic_tools as tools

# Nastavitve:
conv_tol = 1e-9
max_iter = 30
int_order = 3
sequence_increment = 1
roi_mode = 'center'
crop_px = 3 # samo za majhne pomike okoli ravnovesne lege!
results_file_name = 'DIC_analysis_'
save_path = None
settings_log = True
deformable = True
debug = True

roi_positions = [(50, 50)]

# Poti do .mraw datotek:
paths = [['D:\delo-kamera\python_performance\dataset\_images.npy', roi_positions, (35, 35)]]

this_folder = os.path.dirname(os.path.abspath(__file__))

if debug:
    settings_log = True
    logger_level = logging.DEBUG
else:
    logger_level = logging.INFO

# Logging:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
formatter = logging.Formatter('%(asctime)s %(levelname)s from %(name)s: %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
# Set logger levels:
my_logger_keys = ['dic_tools', '__main__', 'main']
logger_dict = logging.Logger.manager.loggerDict
for lkey, l in logger_dict.items():
    if lkey in my_logger_keys:
        for h in l.handlers:
            h.setLevel(logger_level)
            

# Funkcije:
def pickledump(save_path, data, stamp=''):
    '''
    Dumps data into a .pkl file.

    :param save_path: The folder into which the results are saved.
    :param data: DIC results array, with columns of (t,y,x) shape.
    :param stamp: Optional time stamp to add to file name.
    :return:
    '''
    os.makedirs(save_path, exist_ok=True)  # If the directory does not exist, create it.
    if len(data[0]) == 3:
        dict_keys = ['t', 'v', 'u']
    elif len(data[0]) == 4:
        dict_keys = ['t', 'v', 'u', 'phi']
    elif len(data[0]) == 7:
        dict_keys = ['t', 'u_x', 'u_y', 'u', 'v_x', 'v_y', 'v']
    else:
        raise ValueError('Unrecognized result format.')

    dict_data = dict(zip(dict_keys, [data[:, _] for _ in range(len(data[0]))]))
    pickle.dump(dict_data, open(save_path + results_file_name + stamp + '.pkl', 'wb'))


# Izvedi analize:
logger.info('Analysis started at {:}'.format(datetime.datetime.now().strftime(("%d-%m-%y, %H:%M:%S."))))
logger.debug('DEBUG mode.')

mraw_counter = 0
errors = {}
for path, multiple_roi_position, roi_size in paths:
    cih_path = path.split('.')[0] + '.cih'

    # POLJUBNO POIMENOVANJE DATOTEKE Z REZULTATI:
    #results_file_name = path.split('\\')[-1].split('.')[0] + '_'

    if os.path.splitext(path)[-1] != '.mraw':
        try:
            logger.warning('Input file not .mraw! Attemptiong to read .tiff...')
            path, cih_path = tools._tiff_to_temporary_file(os.path.dirname(path))
        except Exception as e:
            logger.error(e)
            break
    # Za različne .mraw datoteke
    mraw_counter += 1
    position_counter = 0

    for roi_reference in multiple_roi_position:
        # Za različne položaje ROI
        position_counter += 1

        # Če so podana središča ROI
        if roi_mode == 'center':
            roi_center = roi_reference
            roi_reference = [roi_center[0] - roi_size[0]//2, roi_center[1] - roi_size[1]//2]
        # Če so podani levi zgornji robovi ROI:
        else:
            roi_center = tools.get_roi_center(roi_reference, roi_size)
        
        logger.debug('ROI position no. {:d}: reference at {} px, center at {} px'.format(position_counter, roi_reference, roi_center))
        
        if save_path is None:
            save_folder = 'DIC_results\\' + '\\center_at_{:d}_{:d}\\'.format(roi_center[0], roi_center[1]) 
            save_folder = os.path.join(this_folder, save_folder)
            #print(this_folder, save_folder)
        else:
            save_folder = save_path
        
        info_dict = tools.get_info(cih_path)
        fps = float(info_dict['Record Rate(fps)']) / sequence_increment
        N = int(info_dict['Total Frame'])
        h = int(info_dict['Image Height'])
        w = int(info_dict['Image Width'])
        timestamp = datetime.datetime.now().strftime(("%H:%M:%S"))

        print()
        logger.info('Sequence {:^3d}, ROI position {:^3d}: '.format(mraw_counter, position_counter))
        logger.info('{:}: Analyzing sequence in <{:}>\n\t({:d} frames, ROI center at ({:d}, {:d}))'.format(timestamp, path, int(info_dict['Total Frame']), roi_center[0], roi_center[1]))
        beginning_time = time.time()
        

        if not deformable:
            logger.debug('Model: RIGID')
            try:
                result = tools.get_rigid_movement(path,
                                                roi_reference,
                                                roi_size,
                                                (N, h, w),
                                                tol=conv_tol,
                                                maxiter=max_iter,
                                                int_order=int_order,
                                                increment=sequence_increment,
                                                crop=crop_px,
                                                debug=False)
            except Exception as e:
                logger.error('An error occurred, stopping analysis. Error message: \n"{}"'.format(e))
        else:
            logger.debug('Model: DEFORMABLE')
            try:
                result = tools.get_affine_deformations(path,
                                                roi_reference,
                                                roi_size,
                                                (N, h, w),
                                                tol=conv_tol,
                                                maxiter=max_iter,
                                                int_order=int_order,
                                                increment=sequence_increment,
                                                crop=crop_px,
                                                debug=False)
            except Exception as e:
                logger.error('An error occurred, stopping analysis. Error message: \n"{}"'.format(e))

        time_taken = (time.time() - beginning_time)
        timestamp = datetime.datetime.now().strftime(("%H:%M:%S"))
        logger.info('Analyzed {:d} frames (time taken:\t{:d} min {:.2f} s).'.format(len(result[0]), int(time_taken // 60), time_taken % 60))

        n_iters = result[-2]
        inc = result[-1]
        kin = result[0]
        errors = result[1]
        t = np.reshape(np.arange(len(kin)) * (inc / fps), (len(kin), 1))

        # Save the results:
        tkin_data = np.hstack((t, kin))
        timestamp = datetime.datetime.now().strftime('%d-%m-%H-%M-%S')

        pickledump(save_folder, data=tkin_data, stamp=timestamp)
        logger.info('Results saved to {:}.'.format(save_folder))
        
        if settings_log:
            with open(save_folder+'/settings_log.txt', 'w') as file:
                file.write('mraw_file =  {:s}\n'.format(path))
                file.write('time_taken = {:d} min {:.2f} s.\n'.format(int(time_taken // 60), time_taken % 60))
                file.write('conv_tol = {:.3e}\n'.format(conv_tol))
                file.write('max_iter = {:d}\n'.format(max_iter))
                file.write('int_order = {:d}\n'.format(int_order))
                file.write('sequence_increment = {:d}\n'.format(sequence_increment))
                file.write('roi_size = ({:d}, {:d})\n'.format(roi_size[0], roi_size[1]))
                file.write('roi_center = ({:d}, {:d})\n'.format(roi_center[0], roi_center[1]))
                if debug:
                    if len(errors.keys()) != 0:
                        file.write('-'*50)
                        file.write('\n\t{:d} errors occurred:\n'.format(len(errors.keys())))
                        file.write('\tWrap matrices saved to {:s}.\n'.format(save_folder+'/warp_matrices.pkl'))
                        keys = sorted(errors.keys())
                        for key in keys:
                            file.write('\tImage {:d}: {}\n'.format(key, errors[key]['message']))
                        matrices = [{key: {'M':errors[key]['warp_matrix'], 'G':errors[key]['image'], 'ROI': errors[key]['ROI']}} for key in keys]
                        pickle.dump(matrices, open(save_folder + '/warp_matrices.pkl', 'wb'))

print()
logger.info('Analysis ended at {:}.'.format(datetime.datetime.now().strftime(("%d-%m-%y, %H:%M:%S"))))
if len(errors.keys()) != 0 and debug:
    logger.warning('{:d} errors encountered during analysis, first at image {:d}. See log for more info.'.format(len(errors.keys()), min(errors.keys())))

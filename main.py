# -*- coding: utf-8 -*-
__author__ = 'Domen Gorjup, Janko Slavič, Miha Boltežar'

"""
A basic GUI for the pyDIC application.
"""

import os
import sys
import time
import csv
import glob
import pickle
import datetime
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import tools
import numpy as np
import scipy.ndimage
import configparser
import warnings

from matplotlib import animation
import matplotlib.pyplot as plt

# Disable warnings printout
warnings.filterwarnings("ignore")

class GlavnoOkno(QtGui.QWidget):

    def __init__(self):
        super(GlavnoOkno, self).__init__()

        self.initUI()

    def initUI(self):

        # Other parameterss
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.results_file_name = 'DIC_analysis_'
        self.mode = 'rigid'     # The default analysis mode.
        self.all_modes = ['rigid', 'deformations', 'integer']
        self.all_mode_names = ['Rigid', 'Deformable', 'Integer translation']
        self.mode_descriptions = {'rigid': 'Rigid body displacement (translation, rotation).',
                                  'deformations': 'Translation and deformation.',
                                  'integer': 'Integer translation - displacements of 1 px and above only!'}

        #Layout Elements:
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(5)
        BottomFrame = QtGui.QGroupBox()
        infoFrame = QtGui.QGroupBox()
        infobox = QtGui.QVBoxLayout()
        calibFrame = QtGui.QGroupBox()
        calibbox = QtGui.QHBoxLayout()
        imageFrame = QtGui.QGroupBox()
        imagebox = QtGui.QVBoxLayout()
        imctrlHbox = QtGui.QHBoxLayout()

        # Widgets:
        self.openFolder = QtGui.QPushButton('Choose a .cih or .tif file', self)
        self.openFolder.clicked.connect(self.open_file)
        self.openFolder.setToolTip('Choose the .cih or .tif file to be read (if .tif, a sequence of every .tif image in directory is loaded).')

        self.naprejButton = QtGui.QPushButton('Next', self)
        self.naprejButton.clicked.connect(self.select_image)
        self.naprejButton.setEnabled(False)
        self.naprejButton.setToolTip('Confirm the choice of image sequence for analysis.')
        self.nazajButton = QtGui.QPushButton('Back', self)
        self.nazajButton.clicked.connect(self.to_beginning)
        self.nazajButton.setEnabled(False)
        self.nazajButton.setToolTip('Back to image sequence seelction.')

        self.pathlabel = QtGui.QLabel()
        self.pathlabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.pathlabel.setMaximumSize(infoFrame.frameGeometry().width(), 200)
        self.pathlabel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.pathlabel.setWordWrap(True)
        self.pathlabel.setTextFormat(1)
        self.info = QtGui.QLabel()
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.info.setMaximumSize(infoFrame.frameGeometry().width(), 1000)
        self.info.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.info.setWordWrap(True)
        self.info.setTextFormat(1)

        self.calibButton = QtGui.QPushButton('Calibrate', self)
        self.calibButton.clicked.connect(self.calibrate)
        self.calibButton.setEnabled(False)
        self.calibButton.setToolTip('Unit calibration.')
        self.resetbButton = QtGui.QPushButton('Reset', self)
        self.resetbButton.clicked.connect(self.default_calibration)
        self.resetbButton.setEnabled(False)
        self.resetbButton.setToolTip('Reset unit calibration.')
        self.caliblabel = QtGui.QLabel()
        self.caliblabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Info section layout
        infobox.addWidget(self.pathlabel)
        infobox.addWidget(self.info)
        infobox.addStretch(1)
        infoFrame.setLayout(infobox)
        infoFrame.setTitle('Info:')
        infoFrame.setToolTip('Basic image sequence info, from the .cih file.')

        # Calibration tools section layout
        calibbox.addWidget(self.calibButton)
        calibbox.addWidget(self.resetbButton)
        calibbox.addWidget(self.caliblabel)
        calibbox.addStretch(1)
        calibFrame.setLayout(calibbox)
        calibFrame.setTitle("Unit calibration:")

        # Customized ImageView GUI for sequence preview:
        self.imw = pg.ImageView()
        self.imw.ui.roiBtn.hide()
        self.imw.ui.menuBtn.hide()
        self.imw.roi.sigRegionChanged.disconnect(self.imw.roiChanged)   # disconnect original ROI

        # Image view control, ROI info
        self.autoscaleButton = QtGui.QPushButton('Center')
        self.autoscaleButton.setEnabled(False)
        self.autoscaleButton.clicked.connect(self.imw.autoRange)
        self.roiLabel = QtGui.QLabel()

        # Image widget layout
        imctrlHbox.addWidget(self.autoscaleButton)
        imctrlHbox.addWidget(self.roiLabel)
        imctrlHbox.addStretch(1)
        imagebox.addWidget(self.imw)
        imagebox.addLayout(imctrlHbox)
        imageFrame.setLayout(imagebox)

        # DIC controls and settings:
        mode_label = QtGui.QLabel('Mode:')
        self.mode_selection = QtGui.QComboBox(self)
        self.mode_selection.setEnabled(False)
        self.mode_selection.addItems(self.all_mode_names)
        self.mode_selection.currentIndexChanged.connect(self.set_mode)
        self.mode_description = QtGui.QLabel(self.mode_descriptions[self.mode])
        # Simulation controls:
        self.beginButton = QtGui.QPushButton('Run analysis')
        self.beginButton.setEnabled(False)
        self.beginButton.clicked.connect(self.begin_DIC)
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.hide()

        # DIC Setting frame:
        dic_settings_frame = QtGui.QGroupBox()
        dic_settings_vbox = QtGui.QVBoxLayout()
        dic_box = QtGui.QVBoxLayout()
        dic_settings_row1 = QtGui.QHBoxLayout()
        dic_settings_row2 = QtGui.QHBoxLayout()
        dic_settings_row1.addWidget(mode_label)
        dic_settings_row1.addWidget(self.mode_selection)
        dic_settings_row1.addWidget(self.mode_description)
        dic_settings_row1.addStretch(1)
        dic_settings_vbox.addLayout(dic_settings_row1)
        dic_settings_frame.setLayout(dic_settings_vbox)
        # Control options:
        dic_control_box = QtGui.QHBoxLayout()
        dic_control_box.addWidget(self.beginButton)
        dic_control_box.addStretch(1)
        dic_control_box.addWidget(self.progressBar)
        dic_box.addWidget(dic_settings_frame)
        dic_box.addLayout(dic_control_box)
        BottomFrame.setLayout(dic_box)
        BottomFrame.setTitle('DIC:')

        self.grid.addWidget(self.openFolder, 0, 0, 1, 2)
        self.grid.addWidget(self.nazajButton, 1,0,1,1)
        self.grid.addWidget(self.naprejButton, 1,1,1,1)
        self.grid.addWidget(infoFrame, 2, 0, 8, 2)
        self.grid.addWidget(calibFrame, 9, 0, 1, 2)
        self.grid.addWidget(imageFrame, 0, 2, 10, 6)
        self.grid.addWidget(BottomFrame, 11, 0, 1,8)

        self.setLayout(self.grid)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('pyDIC')
        self.show()


    def open_file(self):
        '''
        Opens the selected folder and checks for image data. If found, displays a preview.
        '''
        self.load_settings(configfile=(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings.ini')))
        self.naprejButton.setEnabled(False)
        self.imw.setImage(np.zeros((100,100)))
        selected_path = QtGui.QFileDialog.getOpenFileName(self, 'Select path to .cih or .tif file.', self.dir_path, filter=("cih (*.cih);; tiff (*.tif)"))
        self.dir_path = os.path.dirname(selected_path)
        self.save_path = self.dir_path + '/DIC_results/'
        file_extension = selected_path.split('.')[-1].lower()

        def _process_tif_images(dir_path):
            tif_warning = QtGui.QMessageBox.warning(self, 'Warning!',
                                        '".tif" file chosen. The image sequence must first be loaded and ' +
                                        'converted. This may take a few minutes. Do you wish to proceed?',
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if tif_warning == QtGui.QMessageBox.Yes:
                self.info.setText('Processing .tif files.<br>Please wait.')
                self.mraw_path, self.cih_path = tools._tiff_to_temporary_file(self.dir_path)
                self.info_dict = tools.get_info(self.cih_path)
        
        def _incorrect_filetype_warning():
            type_warning = QtGui.QMessageBox.warning(self, 'Warning!',
                                        'Invalid file type selected. pyDIC currently only supports .mraw and .tif. ' +
                                        'Please select a valid file')

        error = 0       # Initialize possible file type error variable.
        if not selected_path:
            error = 1

        if file_extension == 'cih':
            self.cih_path = selected_path
            self.info_dict = tools.get_info(self.cih_path)
            self.image_type = self.info_dict['File Format'].lower()

            if self.image_type == 'mraw':
                self.mraw_path = self.cih_path[:-4] + '.mraw'
            elif self.image_type in ['tif', 'tiff']:
                _process_tif_images(self.dir_path)
            else:
                _incorrect_filetype_warning()
                error = 1
                
        elif file_extension in ['tif', 'tiff']:
            self.image_type = 'tif'
            _process_tif_images(self.dir_path)         
        
        if not error:
            self.show_sequence(self.mraw_path, nmax=100)


    def show_sequence(self, filepath, nmax=100):
        '''
        Displays a preview of found image data for DIC purposes.

        :param filepath: Path to .mraw file, containing image data.
        :param nmax: Maximum number of images in sequence preview.
        '''
        dirpath = os.path.dirname(filepath)
        # Extract basic info from .cih:
        self.fps = float(self.info_dict['Record Rate(fps)'])
        self.h = int(self.info_dict['Image Height'])
        self.w = int(self.info_dict['Image Width'])
        self.N_images = int(self.info_dict['Total Frame'])
        
        if int(self.info_dict['Color Bit']) != 16:
            if self.info_dict['File Format'].lower() in ['tif', 'tiff']:    # 8-bit .tiff images are supported.
                if self.debug:
                    print('Warning, {:d} bit images!'.format(int(self.info_dict['Color Bit'])))
            else:                                                           # 8-bit .mraw files are not upported!
                self.pathlabel.setText('Invalid image type, pyDIC only works with 16-bit images!')
                self.imw.setImage(np.zeros((100,100)))
                self.naprejButton.setEnabled(False)
                self.autoscaleButton.setEnabled(False)
                self.info.setText('')
                return          
        
        pathtext = '<b>Selected directory:</b><br>' + dirpath
        pathtext += '<br>'+'(Preview, {:} images)'.format(nmax)
        infolabel = ''
        for key, value in self.info_dict.items():
            infolabel += '<b>{}</b> : {}<br>'.format(key, value.replace('\n', '<br>'))
        # Get a preview image sequence (t, W, H)!:
        data = tools.get_sequence(filepath, file_shape=(self.N_images, self.h, self.w), nmax=nmax).transpose(0,2,1) 
        self.pathlabel.setText(pathtext)
        self.info.setText(infolabel)
        self.imw.setImage(data)
        self.naprejButton.setEnabled(True)
        self.autoscaleButton.setEnabled(True)


    def select_image(self):
        '''
        Select current image sequence to begin DIC setup.
        '''
        self.openFolder.setEnabled(False)
        self.naprejButton.setEnabled(False)
        self.nazajButton.setEnabled(True)
        self.calibButton.setEnabled(True)
        self.beginButton.setEnabled(True)
        self.mode_selection.setEnabled(True)
        pathtext = '<b>Selected directory:</b><br>' + self.dir_path
        pathtext += '<br>(initial image - specify the region of interest)'
        self.pathlabel.setText(pathtext)
        self.default_calibration()
        self.initialImage = tools.get_sequence(self.mraw_path, file_shape=(self.N_images, self.h, self.w), nmax=1).T
        self.imw.setImage(self.initialImage)
        self.imw.ui.histogram.hide()

        # Add ROI widget to the view:
        image_shape = self.initialImage.shape
        roi_pos = (image_shape[0]//2 - self.min_roi_size[0], image_shape[1]//2 - self.min_roi_size[1])
        ch_roi_pos = tools.get_roi_center(roi_pos, self.initial_roi_size)
        roi_bound = QtCore.QRectF(0, 0, image_shape[0], image_shape[1])
        self.imw.roi = pg.ROI(roi_pos, self.initial_roi_size, maxBounds=roi_bound, scaleSnap=True, translateSnap=True)
        self.imw.CHroi = DotROI(ch_roi_pos, size=1, movable=False)
        self.imw.view.addItem(self.imw.roi)
        self.imw.view.addItem(self.imw.CHroi)
        self.imw.roi.sigRegionChanged.connect(self.imw.roiChanged)
        self.imw.roi.sigRegionChanged.connect(self.moving_roi)
        self.imw.roi.sigRegionChangeFinished.connect(self.update_roi_position)
        self.imw.roi.addScaleHandle(pos=(1, 1), center=(0.5, 0.5))
        self.imw.roi.setPen(pg.mkPen(color=pg.mkColor('00FF6F'), width=2))
        self.imw.CHroi.setPen(pg.mkPen(color=pg.mkColor('FF0000'), width=2))


    def load_settings(self, configfile='settings.ini'):
        '''
        Load settings from .ini file.
        ;param configfile: Path to .ini configuration file.
        '''
        Config = configparser.ConfigParser()
        Config.read(configfile)

        self.conv_tol = float(Config.get('DIC', 'convergence_tolerance'))
        self.max_iter = int(Config.get('DIC', 'max_iterations'))
        self.int_order = int(Config.get('DIC', 'interpolation_order'))
        self.crop_px = int(Config.get('DIC', 'crop_pixels'))
        self.sequence_increment = int(Config.get('DIC', 'sequence_increment'))
        self.min_roi_size = tuple(int((vel)) for vel in Config.get('GUI', 'min_roi_size').split(', '))
        self.initial_roi_size = tuple(int((vel)) for vel in Config.get('GUI', 'initial_roi_size').split(', '))
        self.debug = bool(Config.get('DIC', 'debug'))


    def to_beginning(self):
        '''
        Returns to initial screen, displaying currently set image sequence.
        '''
        self.openFolder.setEnabled(True)
        self.naprejButton.setEnabled(False)
        self.nazajButton.setEnabled(False)
        self.calibButton.setEnabled(False)
        self.beginButton.setEnabled(False)
        self.mode_selection.setEnabled(False)
        self.imw.setImage(np.zeros((100,100)))
        self.initialImage._mmap.close()             # Close displayed image memmap
        self.imw.ui.histogram.show()
        self.imw.roi.sigRegionChanged.disconnect(self.imw.roiChanged)
        self.imw.CHroi.hide()
        self.roiLabel.setText('')
        self.default_calibration()
        self.progressBar.hide()
        self.progressBar.reset()


    def update_roi_position(self):
        '''
        Updates the size of ROI upon changing its size to be of the <2N+1> form.
        Extracts ROI size and position in the original, nonscaled image, dispays info on screen.
        '''
        w, h = self.imw.roi.size()
        if w < self.min_roi_size[1]: w = self.min_roi_size[1]
        if h < self.min_roi_size[0]: h = self.min_roi_size[0]
        # ROI dimensions must be odd:
        m = w//2
        n = h//2
        self.imw.roi.setSize((2*m+1, 2*n+1), finish=False)
        # ROI vertices on the reference image:
        points = self.imw.roi.getArrayRegion(self.initialImage, self.imw.getImageItem(), returnMappedCoords=True)[1]
        # All x, y  values in the ROI region:
        xlist = points[0][:, 0]
        ylist = points[1][0, :]
        # ROI reference, center and size:
        self.roi_reference = (ylist[0], xlist[0])
        self.roi_size = (len(ylist), len(xlist))
        self.roi_center = tools.get_roi_center(self.roi_reference, self.roi_size)
        #self.imw.CHroi.show()
        self.imw.CHroi.setPos(self.roi_center[::-1])
        roi_info = 'ROI (y,x):\tcenter: {:}\t size: {:}\t[px]'.format(self.roi_center, self.roi_size)
        self.roiLabel.setText(roi_info)

    
    def moving_roi(self):
        '''
        Updates the ROI position label and center mark while the ROI is being moved.
        '''
        # ROI vertices on the reference image:
        points = self.imw.roi.getArrayRegion(self.initialImage, self.imw.getImageItem(), returnMappedCoords=True)[1]
        # All x, y  values in the ROI region:
        xlist = points[0][:, 0]
        ylist = points[1][0, :]
        # ROI reference, center and size:
        self.roi_reference = (ylist[0], xlist[0])
        self.roi_size = (len(ylist), len(xlist))
        self.roi_center = tools.get_roi_center(self.roi_reference, self.roi_size)
        self.imw.CHroi.setPos(self.roi_center[::-1])
        roi_info = 'ROI (y,x):\tcenter: {:}\t size: {:}\t[px]'.format(self.roi_center, self.roi_size)
        self.roiLabel.setText(roi_info)


    def set_mode(self, mode):
        '''
        Sets the current DIC mode to the user selected value.
        :param mode: Mode index, emitted by the combo box.
        '''
        self.mode = self.all_modes[mode]
        self.mode_description.setText(self.mode_descriptions[self.mode])


    def calibrate(self):
        '''
        Calibration of units, based on current ROI frame size.
        '''
        width_mm, ok = QtGui.QInputDialog.getDouble(self, 'Unit calibration', 
            'Specify ROI width in mm:')
        
        if ok:
            self.mm_px = width_mm / self.roi_size[1]
            self.unit = 'mm'
            self.caliblabel.setText('1 mm = {:^7.2f} px'.format(1/self.mm_px))
            self.resetbButton.setEnabled(True)


    def default_calibration(self):
        '''
        Reset unit calibration.
        '''
        self.mm_px = 1
        self.unit='pixel'
        self.resetbButton.setEnabled(False)
        self.caliblabel.setText('1 mm = {:^7.2f} px'.format(1/self.mm_px))


    def begin_DIC(self):
        '''
        Begins DIC analysis, using the selected parameters.
        '''
        # Test initial guess:
        reply = self.test_init()
        if reply == QtGui.QMessageBox.No:
            return
        # If initial guess test is positive, begin analysis:
        self.progressBar.show()
        # Begin timing:
        beginning_time = time.time()

        if self.mode == 'integer':
            result = tools.get_integer_translation(self.mraw_path,
                                                   self.roi_reference,
                                                   self.roi_size,
                                                   (self.N_images, self.h, self.w),
                                                   progressBar=self.progressBar,
                                                   n_im=10)
            inc = result[-1]  # Image sequence selection increment.
            # Unit calibration:
            if self.mm_px != 1:
                result[0][:, 0] = result[0][:, 0] * self.mm_px
                result[0][:, 1] = result[0][:, 1] * self.mm_px

        elif self.mode == 'rigid':
            if self.debug: 
                print('Model: RIGID')
                print('Interpolating cropped ROI ({:d} px border).'.format(self.crop_px))
            try:
                result = tools.get_rigid_movement(self.mraw_path,
                                                  self.roi_reference,
                                                  self.roi_size,
                                                  (self.N_images, self.h, self.w),
                                                  progressBar=self.progressBar,
                                                  tol=self.conv_tol,
                                                  maxiter=self.max_iter,
                                                  int_order=self.int_order,
                                                  increment=self.sequence_increment,
                                                  crop=self.crop_px)
            except ValueError:
                if self.debug:
                    print('An error occurred attemptiong to use cropped ROI, continuing without cropping.')
                result = tools.get_rigid_movement(self.mraw_path,
                                                  self.roi_reference,
                                                  self.roi_size,
                                                  (self.N_images, self.h, self.w),
                                                  progressBar=self.progressBar,
                                                  tol=self.conv_tol,
                                                  maxiter=self.max_iter,
                                                  int_order=self.int_order,
                                                  increment=self.sequence_increment,
                                                  crop=False)

            n_iters = result[-2]
            inc = result[-1]
            errors = result[1]

            # Unit calibration:
            if self.mm_px != 1:
                result[0][:, 0] = result[0][:, 0] * self.mm_px
                result[0][:, 1] = result[0][:, 1] * self.mm_px

        elif self.mode == 'deformations':
            if self.debug: 
                print('Model: DEFORMABLE')
                print('Interpolating cropped ROI ({:d} px border).'.format(self.crop_px))
            try:
                result = tools.get_affine_deformations(self.mraw_path,
                                                       self.roi_reference,
                                                       self.roi_size,
                                                       (self.N_images, self.h, self.w),
                                                       progressBar=self.progressBar,
                                                       tol=self.conv_tol,
                                                       maxiter=self.max_iter,
                                                       int_order=self.int_order,
                                                       increment=self.sequence_increment,
                                                       crop=self.crop_px)
            except ValueError:
                if self.debug:
                    print('An error occurred attemptiong to use cropped ROI, continuing without cropping.')
                result = tools.get_affine_deformations(self.mraw_path,
                                                       self.roi_reference,
                                                       self.roi_size,
                                                       (self.N_images, self.h, self.w),
                                                       progressBar=self.progressBar,
                                                       tol=self.conv_tol,
                                                       maxiter=self.max_iter,
                                                       int_order=self.int_order,
                                                       increment=self.sequence_increment,
                                                       crop=False)
            n_iters = result[-2]
            inc = result[-1]
            errors = result[1]

            # Unit calibration:
            if self.mm_px != 1:
                result[0][:, 2] = result[0][:, 2] * self.mm_px
                result[0][:, 5] = result[0][:, 5] * self.mm_px
        
        # Hide ROI center marker
        self.imw.CHroi.hide()

        # If maximum number of iterations was reached:
        if n_iters[-1] == 0:
            if self.debug:
                print('\nMaximum iterations reached. Iteration numbers by image:\n{:}\n'.format(n_iters)) # Print optimization loop iteration numbers.

            niter_warning = QtGui.QMessageBox.warning(self, 'Waring!',
                                    'Maximum iterations reached in the optimization process ' +
                                    '(image {:}).\n(Iterations: mean: {:0.3f}, std: {:0.3f})\n'.format(n_iters[-2] + 1,
                                                                                                    np.mean(n_iters[:-2]),
                                                                                                    np.std(n_iters[:-2])) +
                                    'If this occurred early in the analyis process, the selected ' +
                                    'region of interest might be inappropriate.\n' +
                                    'Try moving the ROI or increasing its size.\n\n' +
                                    'Do yo wish to prooceed to analysis resuts anyway?',
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if niter_warning == QtGui.QMessageBox.No:
                self.to_beginning()
                return
        
        # If warnings errors were raised:
        if len(errors.keys()) != 0:
            if self.debug:
                print('\nErrors ({:d}) occurred during analysis. See log for more info.'.format(len(errors.keys())))
                matrices = [{key: item['warp_matrix']} for key, item in errors.items()]
                pickle.dump(matrices, open(self.save_path + '/warp_matrices.pkl', 'wb'))
            error_warning = QtGui.QMessageBox.warning(self, 'Waring!',
                                    'Errors occurred during the analysis ' +
                                    '(first at image {:d}).\n(Total: {:d} errors\n'.format(min(errors.keys()), len(errors.keys())) +
                                    'Iterations: mean: {:0.3f}, std: {:0.3f})\n'.format(n_iters[-2] + 1,
                                                                                                    np.mean(n_iters[:-2]),
                                                                                                    np.std(n_iters[:-2])) +
                                    'If this occurred early in the analyis process, the selected ' +
                                    'region of interest might be inappropriate.\n' +
                                    'Try moving the ROI or increasing its size.\n\n' +
                                    'Do yo wish to prooceed to analysis resuts anyway?',
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if error_warning == QtGui.QMessageBox.No:
                self.to_beginning()
                return

        time_taken = time.time() - beginning_time
        self.kin = result[0]
        self.t = np.reshape(np.arange(len(self.kin)) * (inc / self.fps), (len(self.kin), 1))

        # Save the results:
        tkin_data = np.hstack((self.t, self.kin))
        timestamp = datetime.datetime.now().strftime('%d-%m-%H-%M-%S')
        self.save_csv(data=tkin_data, stamp=timestamp)
        self.pickledump(data=tkin_data, stamp=timestamp)

        # Show black image - to close loaded memmap:
        self.imw.setImage(np.zeros((100,100)))

        # End-of-analysis pop-up message:
        end_reply = QtGui.QMessageBox.question(self, 'Analysis eneded!',
                                                '{:} images proessed (in {:0.1f} s).\n'.format(len(self.kin), time_taken) +
                                                'Results saved to:\n{} ({}).\n\n'.format(self.save_path.replace('\\', '/'), timestamp) +
                                                'Do yo wish to prooceed to analysis resuts?',
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)

        # Result visualization:
        if end_reply == QtGui.QMessageBox.Yes:
            tools.plot_data(tkin_data, self.unit)
                
        self.to_beginning()

        # Delete temporary file:
        head, tail = os.path.split(self.mraw_path)                                        
        if self.image_type in ['tif', 'tiff'] and tail == '_images.npy':
            delete_temp_reply = QtGui.QMessageBox.question(self, 'Delete temporary files',
                                                'A temporary file has been created from .tif images ' + 
                                                '({:s}). Do you wish to remove it? '.format(self.mraw_path) + 
                                                '(Select "No", if you plan to analyse the same image sequence again.)',
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
            
            if delete_temp_reply == QtGui.QMessageBox.Yes:
                if self.debug:      
                    print('Deleting temporary .npy file.')          
                os.remove(self.mraw_path)



    def test_init(self):
        '''
        Tests if ROI selection is appropriate (a good initial guess can be found using cross correlation template
        matching).

        :return: (bool) True if a good guess has been found, False otherwise.
        '''
        correct = np.array(self.roi_reference)
        guess = tools.get_integer_translation(self.mraw_path, self.roi_reference, self.roi_size,
                                                 (self.N_images, self.h, self.w), initial_only=True)
        if not np.allclose(correct, guess):
            reply = QtGui.QMessageBox.question(self, 'ROI invalid!',
                                               'Initial guess could not be found eith selected ROI. ' +
                                               'Try moving the ROI or increasing its size.\n\n' +
                                               'Do you wish to proceed anyway?', QtGui.QMessageBox.Yes |
                                               QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            return reply
        else:
            return True


    def save_csv(self, data, stamp=''):
        '''
        Saves data array to a .csv file.

        :param data: DIC results array, with columns of (t,y,x) shape.
        :param stamp: Optional time stamp to add to file name.
        :return:
        '''
        os.makedirs(self.save_path, exist_ok=True)  # If the directory does not exist, create it.
        with open(self.save_path + self.results_file_name + stamp + '.csv', 'w', newline='') as csvfile:
            if len(data[0]) == 3:
                fieldnames = ['t', 'v', 'u']
            elif len(data[0]) == 4:
                fieldnames = ['t', 'v', 'u', 'phi']
            elif len(data[0]) == 7:
                fieldnames = ['t', 'u_x', 'u_y', 'u', 'v_x', 'v_y', 'v']
            else:
                raise ValueError('Unrecognized result format.')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for line in data:
                writer.writerow(dict(zip(fieldnames, line)))


    def pickledump(self, data, stamp=''):
        '''
        Dumps data into a .pkl file.

        :param data: DIC results array, with columns of (t,y,x) shape.
        :param stamp: Optional time stamp to add to file name.
        :return:
        '''
        os.makedirs(self.save_path, exist_ok=True)  # If the directory does not exist, create it.
        if len(data[0]) == 3:
            dict_keys = ['t', 'v', 'u']
        elif len(data[0]) == 4:
            dict_keys = ['t', 'v', 'u', 'phi']
        elif len(data[0]) == 7:
            dict_keys = ['t', 'u_x', 'u_y', 'u', 'v_x', 'v_y', 'v']
        else:
            raise ValueError('Unrecognized result format.')

        dict_data = dict(zip(dict_keys, [data[:, _] for _ in range(len(data[0]))]))
        pickle.dump(dict_data, open(self.save_path + self.results_file_name + stamp + '.pkl', 'wb'))


class DotROI(pg.ROI):
    """
    Elliptical ROI subclass withhout handles.
    
    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """
    def __init__(self, pos, size=(1,1), **args):
        pg.ROI.__init__(self, pos, size, **args)

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        
        p.scale(r.width(), r.height())## workaround for GL bug
        r = QtCore.QRectF(r.x()/r.width(), r.y()/r.height(), 1,1)
        
        p.drawEllipse(r)


def main():
    app = QtGui.QApplication(sys.argv)
    ex = GlavnoOkno()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
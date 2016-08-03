# -*- coding: utf-8 -*-
__author__ = 'Domen Gorjup'

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

# Disable warnings printout
warnings.filterwarnings("ignore")

class GlavnoOkno(QtGui.QWidget):

    def __init__(self):
        super(GlavnoOkno, self).__init__()

        self.initUI()

    def initUI(self):

        # Other parameterss
        self.results_file_name = 'DIC_analysis_'
        self.mode = 'rigid'     # The default analysis mode.
        self.all_modes = ['rigid', 'integer']
        self.all_mode_names = ['Kinematika togih teles', 'Celoštevilski pomiki']
        self.mode_descriptions = {'rigid': 'Pomiki togih teles (translacija in rotacija).',
                                  'integer': 'Iskanje celoštevilskoh pomikov - zazna le pomike amplitude 1px ali več!'}

        #Layout Elements:
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(5)
        BottomFrame = QtGui.QGroupBox()
        infoFrame = QtGui.QGroupBox()
        infobox = QtGui.QVBoxLayout()
        imageFrame = QtGui.QGroupBox()
        imagebox = QtGui.QVBoxLayout()
        imctrlHbox = QtGui.QHBoxLayout()

        # Widgets:
        self.openFolder = QtGui.QPushButton('Odpri mapo', self)
        self.openFolder.clicked.connect(self.open_folder)
        self.openFolder.setToolTip('Izberi mapo s slikovnimi datotekami\n(v .tif formatu).')

        self.naprejButton = QtGui.QPushButton('Naprej', self)
        self.naprejButton.clicked.connect(self.select_image)
        self.naprejButton.setEnabled(False)
        self.naprejButton.setToolTip('Izberi trenutno sekvenco slik in nadaljuj z analizo.')
        self.nazajButton = QtGui.QPushButton('Nazaj', self)
        self.nazajButton.clicked.connect(self.to_beginning)
        self.nazajButton.setEnabled(False)
        self.nazajButton.setToolTip('Nazaj na izbiro mape.')

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

        # Info section layout
        infobox.addWidget(self.pathlabel)
        infobox.addWidget(self.info)
        infobox.addStretch(1)
        infoFrame.setLayout(infobox)
        infoFrame.setTitle('Informacije:')
        infoFrame.setToolTip('Osnovne informacije o naloženih datotekah.\nPo možnosti iz .cih datoteke.')

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
        mode_label = QtGui.QLabel('Način:')
        self.mode_selection = QtGui.QComboBox(self)
        self.mode_selection.setEnabled(False)
        self.mode_selection.addItems(self.all_mode_names)
        self.mode_selection.currentIndexChanged.connect(self.set_mode)
        self.mode_description = QtGui.QLabel(self.mode_descriptions[self.mode])
        # Simulation controls:
        self.beginButton = QtGui.QPushButton('Zančni analizo')
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
        self.grid.addWidget(imageFrame, 0, 2, 10, 6)
        self.grid.addWidget(BottomFrame, 11,0, 1,8)

        self.setLayout(self.grid)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('DIC')
        self.show()


    def open_folder(self):
        '''
        Opens the selected folder and checks for image data. If found, displays a preview.
        '''
        self.naprejButton.setEnabled(False)
        self.path = QtGui.QFileDialog.getExistingDirectory(self, 'Izberi mapo s .tif daotekami.', '../')
        self.save_path = self.path + '/DIC_results/'
        self.show_sequence(self.path, nmax=100)


    def show_sequence(self, filepath, nmax=100):
        '''
        Displays a preview of found image data for DIC purposes.

        :param filepath: Path to folder, containing image data.
        :param nmax: Maximum number of images in sequence preview.
        '''
        data = tools.get_sequence(filepath, nmax=nmax, dstack=True)
        if len(data) < 2:
            self.pathlabel.setText('V izbrani mapi ni ustreznih slikovnih datotek.')
            self.imw.setImage(np.zeros((100,100)))
            self.naprejButton.setEnabled(False)
            self.autoscaleButton.setEnabled(False)
            self.info.setText('')
        else:
            pathtext = '<b>Izbrana mapa:</b><br>'+filepath
            pathtext += '<br>'+'(predogled, {:} slik)'.format(nmax)
            infolabel = ''
            if  glob.glob(filepath+'/*.cih'):
                self.info_dict = tools.get_info(filepath)
                self.fps = float(self.info_dict['Record Rate(fps)'])
                for key, value in self.info_dict.items():
                    infolabel += '<b>{}</b> : {}<br>'.format(key, value.replace('\n', '<br>'))
            else:
                infolabel = 'V izbrani mapi ni ustrezne datoteke s podatki o meritvi (.cih)! <br><br>'
                infolabel += 'Prkazani podatki bodo nepopolni.'
                self.fps = 1.

            self.pathlabel.setText(pathtext)
            self.info.setText(infolabel)
            self.imw.setImage(data)
            self.naprejButton.setEnabled(True)
            self.autoscaleButton.setEnabled(True)


    def select_image(self):
        '''
        Select current image sequence to begin DIC setup.
        '''
        self.load_settings()
        self.openFolder.setEnabled(False)
        self.naprejButton.setEnabled(False)
        self.nazajButton.setEnabled(True)
        self.beginButton.setEnabled(True)
        self.mode_selection.setEnabled(True)
        pathtext = '<b>Izbrana mapa:</b><br>'+self.path
        pathtext += '<br>(začetna slika - izbira področja zanimanja)'
        self.pathlabel.setText(pathtext)

        self.initialImage = scipy.ndimage.imread(tools.allfiles(self.path)[0]).transpose()
        self.imw.setImage(self.initialImage)
        self.imw.ui.histogram.hide()

        # Add ROI widget to the view:
        image_shape = self.initialImage.shape
        roi_pos = (image_shape[0]//2 - self.min_roi_size[0], image_shape[1]//2 - self.min_roi_size[1])
        roi_bound = QtCore.QRectF(0, 0, image_shape[0], image_shape[1])
        self.imw.roi = pg.ROI(roi_pos, self.initial_roi_size, maxBounds=roi_bound, scaleSnap=True, translateSnap=True)
        self.imw.view.addItem(self.imw.roi)
        self.imw.roi.sigRegionChanged.connect(self.imw.roiChanged)
        self.imw.roi.sigRegionChangeFinished.connect(self.update_roi_position)
        self.imw.roi.addScaleHandle(pos=(1, 1), center=(0.5, 0.5))
        self.imw.roi.setPen(pg.mkPen(color=pg.mkColor('00FF6F'), width=2))


    def load_settings(self, configfile='nastavitve.ini'):
        '''
        Load settings from .ini file.
        ;param configfile: Path to .ini configuration file.
        '''
        Config = configparser.ConfigParser()
        Config.read(configfile)

        self.conv_tol = float(Config.get('DIC', 'convergence_tolerance'))
        self.max_iter = int(Config.get('DIC', 'max_iterations'))
        self.int_order = int(Config.get('DIC', 'interpolation_order'))
        self.sequence_increment = int(Config.get('DIC', 'sequence_increment'))
        self.min_roi_size = tuple(int((vel)) for vel in Config.get('GUI', 'min_roi_size').split(', '))
        self.initial_roi_size = tuple(int((vel)) for vel in Config.get('GUI', 'initial_roi_size').split(', '))


    def to_beginning(self):
        '''
        Returns to initial screen, displaying currently set image sequence.
        '''
        self.openFolder.setEnabled(True)
        self.naprejButton.setEnabled(False)
        self.nazajButton.setEnabled(False)
        self.beginButton.setEnabled(False)
        self.mode_selection.setEnabled(False)
        self.show_sequence(self.path, nmax=100)
        self.imw.ui.histogram.show()
        self.imw.roi.sigRegionChanged.disconnect(self.imw.roiChanged)
        self.roiLabel.setText('')
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
        self.roi_center = ((ylist[0]+ylist[-1])//2, (xlist[0]+xlist[-1])//2)
        self.roi_size = (len(ylist), len(xlist))
        roi_info = 'ROI (y,x):\tizhodišče: {:}\t velikost: {:}\t[px]'.format(self.roi_reference, self.roi_size)
        self.roiLabel.setText(roi_info)


    def set_mode(self, mode):
        '''
        Sets the current DIC mode to the user selected value.
        :param mode: Mode index, emitted by the combo box.
        '''
        self.mode = self.all_modes[mode]
        self.mode_description.setText(self.mode_descriptions[self.mode])


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
            result = tools.get_integer_translation(self.path,
                                                   self.roi_reference,
                                                   self.roi_size,
                                                   progressBar=self.progressBar,
                                                   n_im=10)
            inc = result[-1]  # Image sequence selection increment.

        else:
            result = tools.get_rigid_movement(self.path,
                                              self.roi_reference,
                                              self.roi_size,
                                              progressBar=self.progressBar,
                                              tol=self.conv_tol,
                                              maxiter=self.max_iter,
                                              int_order=self.int_order,
                                              increment=self.sequence_increment)

            n_iters = result[1]
            inc = result[-1]

            # If maximum number of iterations was reached:
            if n_iters[-1] == 0:
                niter_warning = QtGui.QMessageBox.warning(self, 'Opozorilo!',
                                        'Pri postopku optimizacije je bilo doseženo maksimalno število iteracij ' +
                                        '(slika {:}).\n(Število iteracij: mean: {:0.3f}, std: {:0.3f})\n'.format(n_iters[-2] + 1,
                                                                                                        np.mean(n_iters),
                                                                                                        np.std(n_iters)) +
                                        'Če je do tega prišlo kmalu po začetku sekvence slik, je to verjetno polsedica ' +
                                        'neustrezno izbranega območja zanimanja.\n' +
                                        'Predlagamo, da poskusite povečati ali premakniti območje zanimanja.\n\n' +
                                        'Si želite vseeno ogledati in shraniti rezultate?',
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
                if niter_warning == QtGui.QMessageBox.No:
                    self.to_beginning()
                    return

        time_taken = time.time() - beginning_time
        self.yx = result[0]
        self.t = np.reshape(np.arange(len(self.yx)) * (inc / self.fps), (len(self.yx), 1))

        # Save the results:
        tyx_data = np.hstack((self.t, self.yx))
        timestamp = datetime.datetime.now().strftime('%d-%m-%H-%M-%S')
        self.save_csv(data=tyx_data, stamp=timestamp)
        self.pickledump(data=tyx_data, stamp=timestamp)

        # End-of-analysis pop-up message:
        end_reply = QtGui.QMessageBox.question(self, 'Analiza končana!',
                                                'Obdelanih {:} slik ({:0.1f} s).\n'.format(len(self.yx)+1, time_taken) +
                                                'Rezultati shranjeni v:\n{} ({}).\n\n'.format(self.save_path.replace('\\', '/'), timestamp) +
                                                'Si želite ogledati rezultate?',
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        # Result visualization:
        if end_reply == QtGui.QMessageBox.Yes:
            tools.plot_data(tyx_data)

        self.to_beginning()


    def test_init(self):
        '''
        Tests if ROI selection is appropriate (a good initial guess can be found using cross correlation template
        matching).

        :return: (bool) True if a good guess has been found, False otherwise.
        '''
        correct = np.array(self.roi_reference)
        guess = tools.get_integer_translation(self.path, self.roi_reference, self.roi_size, initial_only=True)
        if not np.allclose(correct, guess):
            reply = QtGui.QMessageBox.question(self, 'Napaka izbire območja zanimanja (ROI)!',
                                               'Pri izbranem območju zanimanja ustrezen začetni približek ni bil najden. ' +
                                               'Poskusite povečati ali premakniti območje zanimanja.\n\n' +
                                               'Želite vseeno nadaljevati?', QtGui.QMessageBox.Yes |
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
        with open(self.save_path + self.results_file_name + stamp + '.csv', 'w') as csvfile:
            if len(data[0]) == 3:
                fieldnames = ['t', 'y', 'x']
            else:
                fieldnames = ['t', 'y', 'x', 'phi']
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
            dict_keys = ['t', 'y', 'x']
        else:
            dict_keys = ['t', 'y', 'x', 'phi']
        dict_data = dict(zip(dict_keys, [data[:, _] for _ in range(len(data[0]))]))
        pickle.dump(dict_data, open(self.save_path + self.results_file_name + stamp + '.pkl', 'wb'))


def main():
    app = QtGui.QApplication(sys.argv)
    ex = GlavnoOkno()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
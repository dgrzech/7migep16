import os
import pydicom
import numpy as np
import scipy, scipy.stats, scipy.ndimage.filters
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import filters
from skimage.restoration import (denoise_wavelet, estimate_sigma)

import sklearn.mixture as mixture
from skimage import measure
from mpl_toolkits.mplot3d import axes3d, Axes3D

from ConnectedComponents import *

from Visualiser import *


class PlantComponents3D(ConnectedComponents3D):
    def __init__(self, binary):
        super(PlantComponents3D, self).__init__(binary)
    def normalise_per_component(self, data):
        new_data = np.copy(data)
        new_data = new_data.astype(float)
        for label in self.labels:
          mask = (self.binary_labelling==label)
          norm_val = np.sum(  data[mask]   )
          #print(norm_val)
          tmp = data[self.binary_labelling==label]
          tmp = tmp.astype(float)
          tmp = (np.divide( tmp , float(norm_val)) )*100.
          #print tmp
          new_data[self.binary_labelling==label] = np.copy(tmp)
        #print('Testing')
        #print np.max(new_data)
        #print np.min(new_data)
        return new_data

class Data:
  def __init__(self, dataset_day=1, scan_number=1):
    dataset_day -= 1
    scan_number -= 1

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = dir_path + '/../config/config.py'
    exec(compile(open(config_file, "rb").read(), config_file, 'exec'))

    self.config = config;
    self.dataset = config['datasets'][dataset_day]
    self.scan = config['scans'][scan_number];

    self.data_dir = dir_path + '/../../' + config['data_path'] + self.dataset + '/' + self.scan + '/'

    self.data_spect = self.data_dir + 'SPECT-processed'
    self.data_ct = self.data_dir + 'CT-processed'

    self.normalised = False


  def import_spect(self):
    spect_files = os.listdir(self.data_spect)


    spect_file = self.data_spect + '/' + spect_files[0];
    ds = pydicom.dcmread(spect_file)
    print(ds.dir()) # PixelSpacing, PixelData, PixelRepresentation, pixel_array
    data = ds.pixel_array
    self.data = data

    self.ConstPixelDims = np.shape(data)
    self.ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
    self.x = np.arange(0., (self.ConstPixelDims[0])*self.ConstPixelSpacing[0], self.ConstPixelSpacing[0])
    self.y = np.arange(0., (self.ConstPixelDims[1])*self.ConstPixelSpacing[1], self.ConstPixelSpacing[1])
    self.z = np.arange(0., (self.ConstPixelDims[2])*self.ConstPixelSpacing[2], self.ConstPixelSpacing[2])

  def import_ct(self):
    ct_files = os.listdir(self.data_ct)

    ct_file = self.data_ct + '/' + ct_files[0];
    ds = pydicom.dcmread(ct_file)
    print(ds.dir()) # PixelSpacing, PixelData, PixelRepresentation, pixel_array
    data = ds.pixel_array
    print np.shape(data)
    self.data = data

    self.ConstPixelDims = np.shape(data)
    self.ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
    self.x = np.arange(0., (self.ConstPixelDims[0])*self.ConstPixelSpacing[0], self.ConstPixelSpacing[0])
    self.y = np.arange(0., (self.ConstPixelDims[1])*self.ConstPixelSpacing[1], self.ConstPixelSpacing[1])


  def histogram_analysis(self):
    pass
    '''
    reshapedData = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )
    mask = reshapedData>400;

    fig = plt.figure()
    reshapedData2 = reshapedData[mask];
    mask = reshapedData>0;
    plt.hist(reshapedData2)

    fig = plt.figure()
    reshapedData2 = reshapedData[~mask];
    plt.hist(reshapedData2)
    #plt.show()

    fig = plt.figure()
    mask = reshapedData>0;
    reshapedData2 = reshapedData[mask];
    plt.hist(np.log(reshapedData2), 100)
    '''

  def restore_original_data(self):
    tmp = np.copy(self.data)
    self.data = np.copy(self.original_data)
    self.saved_calculation = tmp
  def restore_calculated_data(self):
    self.data = np.copy(self.saved_calculation)


  def noise_removal(self, n_gmm_clusters=5, n_largest_components=2):
    self.original_data = np.copy(self.data)
    print np.shape(self.data)
    print(self.data.dtype)

    reshapedData = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )


    sampling = 0.10
    num_pts = len(reshapedData.flatten())
    X_subset = np.random.choice(reshapedData.flatten(),int(num_pts*sampling)).reshape(-1, 1)
    n_clusters = n_gmm_clusters

    gmm = mixture.GaussianMixture(n_components=n_clusters)
    gmm.fit(X_subset)
    y = gmm.predict(reshapedData.flatten().reshape(-1, 1))

    set_values = []
    set_means = []
    for i in np.arange(n_clusters):
      set_values.append(   np.ma.masked_array(reshapedData, mask=( (y!=i) ) )    )
      set_means.append(set_values[i].mean())

    max_set_idx = set_means.index(max(set_means))
    max_set = set_values[max_set_idx]
    max_set_mean = set_means[max_set_idx]

    max_set.set_fill_value(0.)
    reshapedData = max_set
    self.segment = (y==max_set_idx)

    reshapedData.set_fill_value(0.)
    self.data = np.copy(np.reshape(reshapedData, np.shape(self.data)));
    self.segment = np.copy(np.reshape(self.segment, np.shape(self.data)));


    self.comp = PlantComponents3D(self.segment)
    #self.comp = ConnectedComponents3D(self.segment)
    self.data_normalised = self.comp.normalise_per_component(self.data)
    self.main_comp = self.comp.n_largest_components(n=n_largest_components)

    self.data[self.main_comp==False]=0.
    self.data_normalised[self.main_comp==False]=0.

    self.normalised = True

    print(self.data.dtype)
    print(self.data_normalised.dtype)
    # For component visualisation
    #self.data = np.copy(self.main_comp)

import os
import pydicom
import numpy as np
import scipy, scipy.stats
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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

    self.import_spect()

  def import_spect(self):
    spect_files = os.listdir(self.data_spect);
    spect_file = self.data_spect + '/' + spect_files[0];
    ds = pydicom.dcmread(spect_file)
    print(ds.dir()) # PixelSpacing, PixelData, PixelRepresentation, pixel_array
    #print(ds.pixel_array)
    data = ds.pixel_array
    self.data = data
    #print np.shape(np.reshape(data, [5548800, 1]))
    #print scipy.stats.mode(np.reshape(data, [5548800, 1]))
    #ModeResult(mode=array([[0]], dtype=uint16), count=array([[2069750]]))

    self.ConstPixelDims = np.shape(data)
    self.ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
    self.x = np.arange(0., (ConstPixelDims[0])*ConstPixelSpacing[0], ConstPixelSpacing[0])
    self.y = np.arange(0., (ConstPixelDims[1])*ConstPixelSpacing[1], ConstPixelSpacing[1])
    self.z = np.arange(0., (ConstPixelDims[2])*ConstPixelSpacing[2], ConstPixelSpacing[2])

    #X,Y = np.meshgrid(x,y)

    print np.shape(x);
    print np.shape(y);

    print np.shape(data[:,:,2]);

class Visualiser:
  def __init__(self, Data): # Data class above is a prototype of Data
    self.Data = Data;

    fig, ax = plt.subplots()
    self.fig = fig
    #plt.axes().set_aspect('equal', 'datalim')
    #ax.set_cmap(plt.gray())
    #this_plot = ax.pcolormesh(x,y, np.transpose(data[:,:,60]))
    self.this_plot = ax.imshow(np.transpose(data[:,:,40]))

    ax_z = fig.add_axes([0.2, 0.95, 0.65, 0.03])
    self.slider_z = Slider(ax_z, 'Z-Slice', 0, data.shape[2]-1, valinit=40, valfmt='%i')
    self.slider_z.on_changed(self.update)

    plt.show()
  def update(self, val):
      i = int(self.slider_z.val)
      im = self.spect_data[:,:,i]
      self.this_plot_spect.set_data(np.transpose(im))
      self.fig.canvas.draw_idle()


    

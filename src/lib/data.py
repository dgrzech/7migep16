import os
import pydicom
import numpy as np
import scipy, scipy.stats, scipy.ndimage.filters
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import filters
from skimage.restoration import (denoise_wavelet, estimate_sigma)
  


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
    self.x = np.arange(0., (self.ConstPixelDims[0])*self.ConstPixelSpacing[0], self.ConstPixelSpacing[0])
    self.y = np.arange(0., (self.ConstPixelDims[1])*self.ConstPixelSpacing[1], self.ConstPixelSpacing[1])
    self.z = np.arange(0., (self.ConstPixelDims[2])*self.ConstPixelSpacing[2], self.ConstPixelSpacing[2])

    #X,Y = np.meshgrid(x,y)

    print np.shape(self.x);
    print np.shape(self.y);

    print np.shape(data[:,:,2]);

  def noise_removal(self):
    ''' 
    # ignore zeros
    data_tmp = np.reshape(self.data, [np.product(np.shape(self.data)), 1])
    data_tmp = np.ma.masked_equal(data_tmp,0)
    data_tmp = data_tmp.compressed()
    #data_tmp.mean()
    #mode = float(scipy.stats.mode(data_tmp))
    #print('Mode: ', mode)
    mode = 1.
    self.data = scipy.ndimage.filters.gaussian_filter(self.data, sigma=(mode, mode, mode));
    '''

    #self.data = self.data[ (self.data <= 10.) ]
    #self.data = np.ma.masked_array(self.data, mask = (  (self.data < 10)  ) )
    #self.data = self.data.filled(0.);

    '''
    crow = len(self.x)/2
    ccol = len(self.y)/2
    cslice = len(self.z)/2
    print(crow, ccol, cslice)
    f = np.fft.fftn(self.data)
    fshift = np.fft.fftshift(f)
    original = np.copy(fshift)
    x_window = crow-40
    y_window = ccol-20
    z_window = cslice-20
    fshift[crow-x_window:crow+x_window, ccol-y_window:ccol+y_window, cslice-z_window:cslice+z_window] = 0
    f_ishift= np.fft.ifftshift(original - fshift)
    self.data = np.abs(np.fft.ifftn(f_ishift))
    #print(self.data)
    '''
    '''f = np.fft.fftn(self.data)
    #k=[ [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.] ]
    k_size = 5
    k = np.zeros(shape=(k_size,k_size,k_size), dtype=float)
    k_mid = np.int(np.ceil(float(k_size)/2.)-1.)
    k[k_mid,k_mid,k_mid] = 1./9.
    print k
    f_conv = np.ones(shape=np.shape(f), dtype=complex)
    f_conv_real = scipy.ndimage.filters.convolve(np.real(f), k, f_conv.real, 'constant')
    f_conv_imag = scipy.ndimage.filters.convolve(np.imag(f), k, f_conv.imag, 'constant')

    mode = 0.5
    f_conv_real = scipy.ndimage.filters.gaussian_filter(np.real(f), sigma=(mode, mode, mode), output=f_conv.real)
    f_conv_imag = scipy.ndimage.filters.gaussian_filter(np.imag(f), sigma=(mode, mode, mode), output=f_conv.imag)
    
    #f_conv = f_conv_real
    #f_conv.imag = f_conv_imag
    self.data = np.abs(np.fft.ifftn(f_conv))
    #camera = data.camera()
    #val = filters.threshold_otsu(self.data)
    #print val
    #mask = self.data > val
    #self.data = np.ma.masked_array(self.data, mask = (  (mask==True)  ) )
    #self.data = self.data.filled(0.);


    #self.data = denoise_wavelet(self.data, multichannel=False, convert2ycbcr=False,mode='soft')
    self.data = scipy.ndimage.filters.median_filter(self.data, size=(1,1,1))
    #self.data = cv2.GaussianBlur(self.data,(3,3),0)
    #ret3,th3 = cv2.threshold(self.data,0,255,0)
    '''
    #self.data = np.ma.masked_array(self.data, mask = (  (self.data<=0)  ) )
    #self.data = self.data.filled(0.)
    #th3 = cv2.adaptiveThreshold(self.data,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    fig = plt.figure()
    reshapedData = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )
    plt.hist(reshapedData)
    #plt.show()

class Visualiser:
  def __init__(self, Data): # Data class above is a prototype of Data
    self.Data = Data;

    fig, ax = plt.subplots()
    self.fig = fig
    #plt.axes().set_aspect('equal', 'datalim')
    #ax.set_cmap(plt.gray())
    #this_plot = ax.pcolormesh(x,y, np.transpose(data[:,:,60]))
    self.this_plot = ax.imshow(np.transpose(Data.data[:,:,40]))

    ax_z = fig.add_axes([0.2, 0.95, 0.65, 0.03])
    self.slider_z = Slider(ax_z, 'Z-Slice', 0, Data.data.shape[2]-1, valinit=40, valfmt='%i')
    self.slider_z.on_changed(self.update)

    plt.show()
  def update(self, val):
      i = int(self.slider_z.val)
      im = self.Data.data[:,:,i]
      self.this_plot.set_data(np.transpose(im))
      self.fig.canvas.draw_idle()


    

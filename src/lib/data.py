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

    #print np.shape(self.x);
    #print np.shape(self.y);

    #print np.shape(data[:,:,2]);

  def histogram_analysis(self):

    reshapedData = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )
    mask = reshapedData>400;

    fig = plt.figure()
    reshapedData2 = reshapedData[mask];
    plt.hist(reshapedData2)

    fig = plt.figure()
    reshapedData2 = reshapedData[~mask];
    plt.hist(reshapedData2)
    #plt.show()

    fig = plt.figure()
    mask = reshapedData>0;
    reshapedData2 = reshapedData[mask];
    plt.hist(np.log(reshapedData2), 100)

  def restore_original_data(self):
    tmp = np.copy(self.data)
    self.data = np.copy(self.original_data)
    self.saved_calculation = tmp
  def restore_calculated_data(self):
    self.data = np.copy(self.saved_calculation)


  def noise_removal(self):
    self.original_data = np.copy(self.data)
    print np.shape(self.data)
    print(self.data.dtype)

    reshapedData = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )


    sampling = 0.10
    num_pts = len(reshapedData.flatten())
    X_subset = np.random.choice(reshapedData.flatten(),int(num_pts*sampling)).reshape(-1, 1)

    n_clusters = 5

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

    
    self.comp = ConnectedComponents(self.segment)
    self.main_comp = self.comp.largest_connected_component()

    '''
    # Remove for component vis.
    reshaped_data = np.reshape(self.data, [np.product(np.shape(self.data)), 1] )
    reshaped_comp = np.reshape(self.main_comp, [np.product(np.shape(self.main_comp)), 1] )

    reshaped_data = np.ma.masked_array(reshaped_data, mask=( (reshaped_comp!=True) ) ) 
    reshaped_data.set_fill_value(0.)

    self.data = np.copy(np.reshape(reshaped_data, np.shape(self.data))   )
    '''
    self.data[self.main_comp==False]=0.
    
    print(self.data.dtype)
    # For component visualisation    
    #self.data = np.copy(self.main_comp)


class ConnectedComponents:

  def __init__(self, binary):
    self.labels = []
    self.binary_labelling = np.zeros(np.shape(binary))
    
    label_numbers_for_unification = []

    dims =np.shape(binary)

    current_label = self.pick_label()   
    
    # Time to search

    for i in np.arange(dims[0], dtype=int):
      print i
      for j in np.arange(dims[1], dtype=int):
        for k in np.arange(dims[2], dtype=int):
          #print(i, j, k)
          if binary[i][j][k] == 1:
            # Normal Case
            #cells_to_check = [binary[i-1][j][k], binary[i-1][j-1][k], binary[i][j-1][k], binary[i][j][k-1], binary[i-1][j][k-1], binary[i-1][j-1][k-1], binary[i][j-1][k-1] ]
            cells_to_check = [ [i-1,  j, k], [i-1, j-1, k], [i,j-1,k], [i,j,k-1], [i-1,j,k-1], [i-1,j-1,k-1], [i,j-1,k-1] ]

            if((i==0 )    &   ( j!=0 )    &   ( k!=0)):
              cells_to_check = [ [i,j-1,k], [i,j,k-1], [i,j-1,k-1] ]
            if((i!=0 )    &   ( j==0 )    &   ( k!=0)):
              cells_to_check = [ [i-1,  j, k],  [i,j-1,k], [i,j,k-1], [i-1,j,k-1], ]
            if((i!=0 )    &   ( j!=0 )    &   ( k==0)):
              cells_to_check = [ [i-1,  j, k], [i-1, j-1, k], [i,j-1,k] ]
            if((i==0 )    &   ( j==0 )    &   ( k!=0)):
              cells_to_check = [ [i,j,k-1]]   
            if((i!=0 )    &   ( j==0 )    &   ( k==0)):
              cells_to_check = [ [i-1,  j, k] ]  
            if((i==0 )    &   ( j!=0 )    &   ( k==0)):
              cells_to_check = [ [i,j-1,k] ]    
            
            
            labelled = False
            labelled_with = []

            if((i==0) & (j==0) & (k==0)):
              pass
            else:
              for cell in cells_to_check:
                this_cell_label = self.binary_labelling[cell[0]][cell[1]][cell[2]]
                if (this_cell_label > 0):
                  #print(this_cell_label, self.binary_labelling[i][j][k], this_cell_label<self.binary_labelling[i][j][k], labelled==True )
                  if ( (labelled==False) | ( (labelled==True) & (this_cell_label<self.binary_labelling[i][j][k])) ) :
                    self.binary_labelling[i][j][k] = this_cell_label
                  labelled = True
                  labelled_with.append(this_cell_label)
                if(len(labelled_with)>1):
                  label_numbers_for_unification.append(labelled_with)

            if labelled==False:
              self.binary_labelling[i][j][k] = self.pick_label()

    old_label_list = np.copy(self.labels)
    # Labels all found. Now it is time to union
    for ii in np.arange(len(label_numbers_for_unification)):
      i = (len(label_numbers_for_unification)-1)-ii
      this_label_set = label_numbers_for_unification[i]
      smallest_label = np.min(this_label_set)
      for this_label in this_label_set:
        if this_label != smallest_label:
          self.binary_labelling[self.binary_labelling==this_label]=smallest_label
          del(self.labels[self.labels.index(this_label)])
          #############################################
          # Why are the below lines of code neccessary?
          for jj in np.arange(len(label_numbers_for_unification)):
            for jj_thislabel_no in np.arange(len(label_numbers_for_unification[jj])):
              if(label_numbers_for_unification[jj][jj_thislabel_no]==this_label):
                label_numbers_for_unification[jj][jj_thislabel_no]=smallest_label


  def pick_label(self):
    i=1
    no_found = False
    while no_found == False:
      if i in self.labels:
        i +=1
      else:
        no_found=True
    self.labels.append(i)
    return i

  def largest_connected_component(self, num_connected=2):
    biggest_val = 0.
    biggest_component_label = 0.

    values = []
    component_labels = []



    for label in self.labels:
      mask = (self.binary_labelling==label)
      val = np.sum(mask)
      values.append(val)
      component_labels.append(label)

    sorted_values, sorted_component_labels = zip(*[(x, y) for x, y in sorted(zip(values, component_labels))])
    print(sorted_values)
    print(sorted_component_labels)

    connected_component = np.zeros(np.shape(self.binary_labelling))
    num_components = len(sorted_values)
    for i in np.arange(num_connected):
      this_label = sorted_component_labels[(num_components-1) - i]
      connected_component = (  (connected_component==True) | (self.binary_labelling==this_label) )

    return connected_component 
  def largest_connected_component_old(self):
    biggest_val = 0.
    biggest_component_label = 0.



    for label in self.labels:
      mask = (self.binary_labelling==label)
      val = np.sum(mask)
      if val>biggest_val:
        biggest_val = np.copy(val)
        biggest_component_label = np.copy(label)

    
    print(biggest_component_label)
    print(biggest_val)
    connected_component = (self.binary_labelling==biggest_component_label)
    return connected_component 



class Visualiser:
  def __init__(self, Data): # Data class above is a prototype of Data
    self.Data = Data;

    
    tmp = np.ma.masked_array(Data.data, mask=(Data.data==0.))

    themin = 0.
    themax = 1.

    themin=0.
    themax=4000.

    print(themin, themax)

    fig, ax = plt.subplots()
    self.fig = fig
    #plt.axes().set_aspect('equal', 'datalim')
    #ax.set_cmap(plt.gray())
    #this_plot = ax.pcolormesh(x,y, np.transpose(data[:,:,60]))
    self.this_plot = ax.imshow(np.transpose(Data.data[:,:,40]), clim=[themin,themax])

    ax_z = fig.add_axes([0.2, 0.95, 0.65, 0.03])
    self.slider_z = Slider(ax_z, 'Z-Slice', 0, Data.data.shape[2]-1, valinit=40, valfmt='%i')
    self.slider_z.on_changed(self.update)



    verts, faces = measure.marching_cubes_classic(self.Data.data, 0, spacing=(0.1, 0.1, 0.1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap='Spectral', lw=1)
    plt.show()

  def update(self, val):
      i = int(self.slider_z.val)
      im = self.Data.data[:,:,i]
      self.this_plot.set_data(np.transpose(im))
      self.fig.canvas.draw_idle()


    

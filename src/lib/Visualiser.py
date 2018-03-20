import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from skimage import measure
from mpl_toolkits.mplot3d import axes3d, Axes3D


class Visualiser:
  def __init__(self, Data, plot3D=False, plotNorm=False, norm_fac=20.): # Data class above is a prototype of Data
    self.Data = Data;
    self.this_data = self.Data.data;

    themin=0.
    themax=4000.
    if(plotNorm==True):
        if(self.Data.normalised == True):
            themin = 0.
            themax = np.mean(self.Data.normalised)/norm_fac
            self.this_data = self.Data.data_normalised
            print("Data is normalised per component")
        else:
            print("Data wasn't normalised so plotting unnormalised data")


    tmp = np.ma.masked_array(Data.data, mask=(Data.data==0.))




    print(themin, themax)

    fig, ax = plt.subplots()
    self.fig = fig
    #plt.axes().set_aspect('equal', 'datalim')
    #ax.set_cmap(plt.gray())
    #this_plot = ax.pcolormesh(x,y, np.transpose(data[:,:,60]))
    self.this_plot = ax.imshow(np.transpose(self.this_data[:,:,40]), clim=[themin,themax])

    ax_z = fig.add_axes([0.2, 0.95, 0.65, 0.03])
    self.slider_z = Slider(ax_z, 'Z-Slice', 0, self.this_data.shape[2]-1, valinit=40, valfmt='%i')
    self.slider_z.on_changed(self.update)


    if plot3D==True:
      verts, faces = measure.marching_cubes_classic(self.this_data, 0, spacing=(0.1, 0.1, 0.1))

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                      cmap='Spectral', lw=1)
    plt.show()

  def update(self, val):
      i = int(self.slider_z.val)
      im = self.this_data[:,:,i]
      self.this_plot.set_data(np.transpose(im))
      self.fig.canvas.draw_idle()

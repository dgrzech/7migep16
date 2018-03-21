import numpy as np
import copy

import sklearn.mixture as mixture
from src.lib import data as DataManager



class PlantSegmentation3D:
    def __init__(self, Spect, n_segmentations):
        self.Spect = Spect
        euc_distro = Spect.main_comp
        dims = np.shape(euc_distro)
        num = np.sum(euc_distro)
        coord_list = np.zeros([num, 3], dtype=int)
        count = 0
        for i in np.arange(dims[0]):
            for j in np.arange(dims[1]):
                for k in np.arange(dims[2]):
                    if(euc_distro[i][j][k] != 0):
                        coord_list[count] = [i, j, k]
                        count = count + 1
        self.coord_list = coord_list
        self.coord_list_num = count

        self.n_segmentations = n_segmentations

        reshapedData = coord_list
        sampling = 0.10
        num_pts = len(reshapedData)
        X_subset = np.random.permutation(reshapedData)
        X_subset = X_subset[0:int(num_pts*sampling)]
        n_clusters = n_segmentations

        gmm = mixture.GaussianMixture(n_components=n_clusters)
        gmm.fit(X_subset)
        y = gmm.predict(reshapedData)


        self.n_segmentations = n_segmentations
        self.segmentation = y
    def build_segmentation_data(self):
        #self.data = np.ones(np.shape(self.Spect.data))*np.nan
        dim = [self.n_segmentations]
        dim1 = list(np.shape(self.Spect.data))
        dim.extend(dim1)
        #dim.append(self.n_segmentations)
        self.data_segmented_norm = np.zeros(dim)
        self.data_segmented_nonorm = np.zeros(dim)
        self.data_segmented_bin = np.zeros(dim1)
        print('Started building the segmented parts')
        for i in np.arange(self.coord_list_num):
            this_label = self.segmentation[i]
            this_coordinate = self.coord_list[i]
            this_activity_norm = self.Spect.data_normalised[this_coordinate[0]][this_coordinate[1]][this_coordinate[2]]
            this_activity_nonorm = self.Spect.data[this_coordinate[0]][this_coordinate[1]][this_coordinate[2]]
            self.data_segmented_norm[this_label][this_coordinate[0]][this_coordinate[1]][this_coordinate[2]] = this_activity_norm
            self.data_segmented_nonorm[this_label][this_coordinate[0]][this_coordinate[1]][this_coordinate[2]] = this_activity_nonorm
            self.data_segmented_bin[this_coordinate[0]][this_coordinate[1]][this_coordinate[2]] = np.copy(this_label)+1;
        print('End building the segmented parts')


    def segmentation_activity(self):
        activity_counts = np.zeros([self.n_segmentations])
        for i in np.arange(self.coord_list_num):
            this_label = self.segmentation[i]
            this_coordinate = self.coord_list[i]
            this_activity = self.Spect.data_normalised[this_coordinate[0]][this_coordinate[1]][this_coordinate[2]]
            activity_counts[this_label] = activity_counts[this_label] + this_activity
        for i in np.arange(self.n_segmentations):
            print('Segmentation ' + str(i) + ': ' + str(activity_counts[i]))
    def visualise_segmentations(self):
        Visualisers = []
        for i in np.arange(self.n_segmentations):
            ThisSpect = copy.deepcopy(self.Spect)
            ThisSpect.data = self.data_segmented_nonorm[i]
            ThisSpect.data_normalised = self.data_segmented_norm[i]
            Visualisers.append(DataManager.Visualiser(ThisSpect, plot3D=True, plotNorm=True, show_plot=False))

        ThisSpect = copy.deepcopy(self.Spect)
        ThisSpect.data = self.data_segmented_bin
        ThisSpect.data_normalised = self.data_segmented_bin
        Visualisers.append(DataManager.Visualiser(ThisSpect, plot3D=True, plotNorm=True, show_plot=False, binary_cmap=True))


        Visualisers[0].show_plot()

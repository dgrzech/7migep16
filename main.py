from src.lib import data as DataManager
from src.lib import Segmentation

Spect = DataManager.Data(dataset_day=1, scan_number=1)
Spect.import_spect()
Spect.histogram_analysis()
Spect.noise_removal(n_gmm_clusters=5, n_largest_components=2)

Spect.restore_original_data() # Plot the data with no noise removal
Visualiser = DataManager.Visualiser(Spect, show_plot=False)
Spect.restore_calculated_data() # Restore the data with noise removal and segmentation of connected activity





#Visualiser = DataManager.Visualiser(Spect, plot3D=False, plotNorm=True)


#n_components = input("How many components are present in the plant?")
n_components = 9
Seg = Segmentation.PlantSegmentation3D(Spect, n_components)
Seg.segmentation_activity()
Seg.build_segmentation_data()
Seg.visualise_segmentations()

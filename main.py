from src.lib import data as DataManager
from src.lib import Segmentation

auto=False

if auto==False:
    dataset_day = input("Which dataset day do you want to look at? (1) = 7March, (2) = 12March: ")
    scan_number = input("Which scan do you want to look at? (1) = 9(L)7(R),  (2) = 3(R)4(L), (3) = 1(L)2(R): ")
else:
    dataset_day=2
    scan_number = 3

Spect = DataManager.Data(dataset_day=dataset_day, scan_number=scan_number)
Spect.import_spect()
Spect.histogram_analysis()
Visualiser = DataManager.Visualiser(Spect, show_plot=True, plot3D=False)
if auto==False:
    n_components = input("How many components are present in the plant: ")
else:
    n_components = 2
Spect.noise_removal(n_gmm_clusters=5, n_largest_components=n_components)

Spect.restore_original_data() # Plot the data with no noise removal
#Visualiser = DataManager.Visualiser(Spect, show_plot=True)
Spect.restore_calculated_data() # Restore the data with noise removal and segmentation of connected activity





Visualiser = DataManager.Visualiser(Spect, plot3D=False, plotNorm=False, show_plot=False)
Visualiser = DataManager.Visualiser(Spect, plot3D=False, plotNorm=True, show_plot=True)

if auto==False:
    n_segmentations = input("How many segmentations are present in the plant: ")
else:
    n_segmentations = 9
Seg = Segmentation.PlantSegmentation3D(Spect, n_segmentations)
Seg.segmentation_activity()
Seg.build_segmentation_data()
Visualiser = DataManager.Visualiser(Spect, plot3D=False, plotNorm=False, show_plot=False)
Visualiser = DataManager.Visualiser(Spect, plot3D=False, plotNorm=True, show_plot=False)
Seg.visualise_segmentations()

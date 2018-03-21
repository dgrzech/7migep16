from src.lib import data


Spect = data.Data(dataset_day=1, scan_number=2)
Spect.import_spect()
Spect.histogram_analysis()
Spect.noise_removal()

Spect.restore_original_data()
Visualiser = data.Visualiser(Spect)


Spect.restore_calculated_data()
Visualiser = data.Visualiser(Spect)


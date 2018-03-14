from src.lib import data


Spect = data.Data()
Spect.import_spect()
Spect.noise_removal()

Visualiser = data.Visualiser(Spect);


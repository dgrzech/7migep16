import os
import pydicom

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
    

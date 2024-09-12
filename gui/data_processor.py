import os
import pandas as pd
import numpy as np
from ieeg.auth import Session
from pyedflib import highlevel
from montages import generate_bipolar_montages, car_montage

class DataProcessor:
    def __init__(self, username, password, file):
        self.username = username
        self.password = password
        self.file = file
        self.current_dir = os.getcwd()
        self.result_file_path = os.path.join(self.current_dir, "result_dataframe.csv")
        self.result_dataframe = self.load_or_create_dataframe()
        self.time_in_s = 0
        self.bipolar_montage = None
        self.car_montage = None

    def load_or_create_dataframe(self):
        if os.path.exists(self.result_file_path):
            return pd.read_csv(self.result_file_path)
        else:
            columns = ["sleep_state_0", "prediction_0", "start_0", "end_0"]
            num_rows = len(self.file)
            data = np.full((num_rows, len(columns)), -1)
            return pd.DataFrame(data, columns=columns)

    def parse_calculate(self):
        raw_data = None
        if self.username is not None:
            with Session(self.username, self.password) as session:
                dataset = session.open_dataset(self.file.iloc[0].dataset_name)
                channels = list(range(len(dataset.ch_labels)))
                raw_data = dataset.get_data(self.file.iloc[0].dataset_start_time, self.file.iloc[0].duration, channels)
                raw_data = pd.DataFrame(raw_data, columns=dataset.ch_labels)
                session.close_dataset(self.file.iloc[0].dataset_name)
        else:
            cur_edf_filepath = os.path.join(self.file, os.listdir(self.file)[0])
            raw_data, channel_metadata, scan_metadata = highlevel.read_edf(cur_edf_filepath)
            channels = highlevel.read_edf_header(cur_edf_filepath)['channels']
            clean_channel_map = []
            for ichannel in channels:
                regex_match = re.match(r"(\D+)(\d+)", ichannel)
                lead = regex_match.group(1).replace("EEG", "").strip()
                contact = int(regex_match.group(2))
                clean_channel_map.append(f"{lead}{contact:02d}")
            raw_data = pd.DataFrame(raw_data)
            raw_data = raw_data.T
            raw_data.columns = clean_channel_map

        sns.set_theme(style="dark")

        downsample_factor = 10
        downsampled_data = raw_data.iloc[::downsample_factor, :]
        num_rows = downsampled_data.shape[0]
        self.time_in_s = np.arange(num_rows) * (self.file.iloc[0].duration / num_rows) / 1000000

        self.bipolar_montage = generate_bipolar_montages(downsampled_data)
        which_chs = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', "Pz", 'T3', 'T4', 'T5', 'T6']
        self.car_montage = car_montage(downsampled_data, which_chs)
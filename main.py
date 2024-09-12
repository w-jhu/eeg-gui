import argparse
import os
import pandas as pd
from gui.gui_init import EEGGraphGUI
from utils import is_valid_csv_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', required=True, help='username')
    parser.add_argument('-p', '--password', help='password')
    parser.add_argument('data_file', help='name of local CSV file/folder directory containing dataset information')
    parser.add_argument('--delay', help='delay time of input (default = 0.5 s)')
    args = parser.parse_args()

    username = args.user
    password = args.password
    delay = float(args.delay) if args.delay else 0.5
    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, args.data_file)

    if is_valid_csv_file(data_file_path):
        file = pd.read_csv(data_file_path)
    else:
        file = data_file_path

    EEGGraphGUI(username, password, file, delay)
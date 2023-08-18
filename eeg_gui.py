# -*- coding: utf-8 -*-

import argparse
import os
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from ieeg.auth import Session
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyedflib import highlevel


class EEGGraphGUI(tk.Tk):
    def __init__(self, username, password, file, delay):
        tk.Tk.__init__(self)
        
        # Store arguments as instance variables
        self.username = username
        self.password = password
        self.file = file
        self.delay = delay
        self.current_dir = os.getcwd()
        self.result_file_path = os.path.join(self.current_dir, "result_dataframe.csv")
        
        # Initialize GUI components
        self.initialize_gui()
        
        self.mainloop()
        plt.close("all")

    def initialize_gui(self):
        self.initialize_instance_variables()
        self.create_or_load_result_dataframe()
        self.create_main_figure()
        self.create_buttons_frame()
        self.bind_keyboard_shortcuts()
        self.load_initial_graph()
        
    def initialize_instance_variables(self):
        self.data_index = 0
        self.current_dataset_info = None
        self.dataset_name = None
        self.dataset_start_time = None
        self.duration = None
        self.current_montage_str = "bipolar"
        self.gain = 1.0
        self.current_montage = None
        self.bipolar_montage = None
        self.car_montage = None
        self.time_in_s = 0
        self.font_size = 16
        self.timeframe_selection_started = False
        self.start_time = None
        self.end_time = None
        self.has_seizure = None
        self.sleep_state = None
        self.timer_id = None
        self.folder_file_list = None
        self.is_csv_format = self.username is not None
        self.delay = delay
        # Handle different input formats
        if(self.is_csv_format):
            self.current_dataset_info = self.file.iloc[self.data_index]
            self.dataset_name = self.current_dataset_info.dataset_name
            self.dataset_start_time = self.current_dataset_info.dataset_start_time
            self.duration = self.current_dataset_info.duration
        else:
            self.folder_file_list = os.listdir(self.file)

    def create_or_load_result_dataframe(self):
        if os.path.exists(self.result_file_path):
            self.result_dataframe = pd.read_csv(self.result_file_path)
        else:
            columns = ["sleep_state_0", "prediction_0", "start_0", "end_0"]
            num_rows = len(self.file)
            data = np.full((num_rows, len(columns)), -1)
            self.result_dataframe = pd.DataFrame(data, columns=columns)
    
    def create_main_figure(self):
        self.figure = plt.figure(figsize=(20, 12))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_buttons_frame(self):
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, pady=10)
        self.seizure_label = tk.Label(self.buttons_frame, text="Sleep State: ", font=("Arial", 16))
        self.seizure_label.pack(side=tk.LEFT, padx=10)
        
        # Create sleep state buttons
        self.awake_button = tk.Button(self.buttons_frame, text="Awake", command=self.on_awake_button_click, font=("Arial", 16))
        self.n1_button = tk.Button(self.buttons_frame, text="N1", command=self.on_n1_button_click, font=("Arial", 16))
        self.n2_button = tk.Button(self.buttons_frame, text="N2", command=self.on_n2_button_click, font=("Arial", 16))
        self.n3_button = tk.Button(self.buttons_frame, text="N3", command=self.on_n3_button_click, font=("Arial", 16))
        self.rem_button = tk.Button(self.buttons_frame, text="REM", command=self.on_rem_button_click, font=("Arial", 16))
        self.undet_button = tk.Button(self.buttons_frame, text="Undetermined", command=self.on_undet_button_click, font=("Arial", 16))
        
        self.awake_button.pack(side=tk.LEFT)
        self.n1_button.pack(side=tk.LEFT)
        self.n2_button.pack(side=tk.LEFT)
        self.n3_button.pack(side=tk.LEFT)
        self.rem_button.pack(side=tk.LEFT)
        self.undet_button.pack(side=tk.LEFT)
        
        # Create "Yes" and "No" buttons
        self.yes_seizure = tk.Button(self.buttons_frame, text="Yes", command=self.on_yes_seizure_click, font=("Arial", 16))
        self.no_seizure = tk.Button(self.buttons_frame, text="No", command=self.on_no_seizure_click, font=("Arial", 16))
        self.yes_save = tk.Button(self.buttons_frame, text = "Yes", command=self.on_yes_save_click, font=("Arial", 16))
        self.no_save = tk.Button(self.buttons_frame, text="No", command=self.on_no_save_click, font=("Arial", 16))
        
        # Create help button
        question_mark_unicode = "\u2753"
        self.question_mark_label = tk.Label(self.buttons_frame, text=question_mark_unicode, font=("Arial", 20), cursor="hand2")
        self.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)
        self.question_mark_label.bind("<Button-1>", self.show_popup_window)
        
    def bind_keyboard_shortcuts(self):
        self.bind("<Left>", lambda event: self.switch_to_bipolar())
        self.bind("<Right>", lambda event: self.switch_to_car())
        self.bind("<Up>", self.increase_gain)
        self.bind("<Down>", self.decrease_gain)
        self.bind("r", lambda event: self.reset_GUI())
        self.bind("<Configure>", self.on_configure)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def load_initial_graph(self):
        self.parse_calculate()
        self.plot_graph()
        
    def show_popup_window(self, event):
        popup_window = tk.Toplevel(self)
        popup_window.title("Help")
        popup_window.geometry("800x250")  # Customize the size of the pop-up window
        
        # Unicode arrow characters for visualizations
        left_arrow_unicode = "\u2190"
        right_arrow_unicode = "\u2192"
        up_arrow_unicode = "\u2191"
        down_arrow_unicode = "\u2193"
        
        info_text = f"{left_arrow_unicode} {right_arrow_unicode}: Switch montages\n" \
                    f"{up_arrow_unicode} {down_arrow_unicode}: Increase/decrease gain\n" \
                        f"R: Reset (annotations, gain, montage)\n" \
                            f"Data saves to result_dataframe.csv"

        font_size = 30 
        font_style = "Arial"
        info_label = tk.Label(popup_window, text=info_text, font=(font_style, font_size))
        info_label.pack(pady=20)
        
        # Calculate the centering offsets relative to the GUI window
        x_offset = (self.winfo_x() + self.winfo_width() // 2) - (popup_window.winfo_reqwidth() // 2)
        y_offset = (self.winfo_y() + self.winfo_height() // 2) - (popup_window.winfo_reqheight() // 2)
        
        # Center the pop-up window on top of the main window
        popup_window.geometry(f"+{x_offset}+{y_offset}")
        
        # Make the pop-up window transient to the main window (it will stay on top of the main window)
        popup_window.transient(self)
    
    def save_result_dataframe(self, file_path):
        self.result_dataframe.to_csv(file_path, index=False)
    
    def on_configure(self, event):
        # If there is a previous timer, cancel it
        if self.timer_id:
            self.after_cancel(self.timer_id)
        
        # Set a new timer to call replot_graph after 1000 milliseconds (1 second)
        self.timer_id = self.after(500, lambda: self.replot_graph(event))
        
    def switch_to_bipolar(self):
        if self.current_montage_str != "bipolar":
            self.current_montage_str = "bipolar"
            self.plot_graph()

    def switch_to_car(self):
        if self.current_montage_str != "car":
            self.current_montage_str = "car"
            self.plot_graph()

    def increase_gain(self, event):
        self.gain *= 1.25
        self.plot_graph()

    def decrease_gain(self, event):
        self.gain /= 1.25
        self.plot_graph()
    
    def switch_to_seizure_buttons(self):
        time.sleep(delay)
        
        self.question_mark_label.forget()
        
        # Add buttons
        self.yes_seizure.pack(side=tk.LEFT)
        self.no_seizure.pack(side=tk.LEFT)
        
        # Remove buttons
        sleep_state_buttons = [self.awake_button, self.n1_button, self.n2_button, self.n3_button, self.rem_button, self.undet_button]

        for button in sleep_state_buttons:
            button.pack_forget()
        
        # Set new text
        self.seizure_label.configure(text = "Seizure Present: ")
        
        self.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)
        
    def switch_to_save_buttons(self):
        time.sleep(delay)
        
        self.question_mark_label.forget()
        
        # Add buttons
        self.yes_save.pack(side=tk.LEFT)
        self.no_save.pack(side=tk.LEFT)
        
        # Remove buttons
        self.yes_seizure.pack_forget()
        self.no_seizure.pack_forget()
        
        # Set new text
        self.seizure_label.configure(text = "Save: ")
        
        self.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)
        
        
    def on_awake_button_click(self):
        print("Awake selected")
        self.sleep_state = 0
        self.switch_to_seizure_buttons()
    
    def on_n1_button_click(self):
        print("N1 selected")
        self.sleep_state = 1
        self.switch_to_seizure_buttons()
    
    def on_n2_button_click(self):
        print("N2 selected")
        self.sleep_state = 2
        self.switch_to_seizure_buttons()
        
    def on_n3_button_click(self):
        print("N3 selected")
        self.sleep_state = 3
        self.switch_to_seizure_buttons()
        
    def on_rem_button_click(self):
        print("REM selected")
        self.sleep_state = 4
        self.switch_to_seizure_buttons()
        
    def on_undet_button_click(self):
        print("Undetermined selected")
        self.sleep_state = 5
        self.switch_to_seizure_buttons()
        
    def on_yes_seizure_click(self):
        print("Yes seizure clicked")
        time.sleep(delay)
        
        # Set prediction result
        self.has_seizure = True
        
        # Remove buttons and label
        self.yes_seizure.pack_forget()
        self.no_seizure.pack_forget()

        # Add new text
        self.seizure_label.configure(text="Select seizure timeframe (start and end)")
        self.seizure_label.pack(side=tk.LEFT, padx=10)

        # Set the new state to indicate that timeframe selection has started
        self.timeframe_selection_started = True
        
    def on_no_seizure_click(self):
        print("No seizure clicked")      
        
        # Set prediction result
        self.has_seizure = False
        
        self.switch_to_save_buttons()
        
    def on_yes_save_click(self):
        print("Yes save clicked")
        
        # Creates new columns if needed
        default_values = self.result_dataframe.iloc[self.data_index].eq(-1).tolist()
        if True not in default_values:
            num_existing_columns = len(self.result_dataframe.columns)
            index_for_new_columns = num_existing_columns // 4
            new_columns = [f"sleep_state_{index_for_new_columns}",
               f"prediction_{index_for_new_columns}",
               f"start_{index_for_new_columns}",
               f"end_{index_for_new_columns}"]
            self.result_dataframe[new_columns] = pd.DataFrame([[-1, -1, -1, -1]], index=self.result_dataframe.index)
            default_values = self.result_dataframe.iloc[self.data_index].eq(-1).tolist()
        
        # Inserts user input
        starting_position = default_values.index(True)
        self.result_dataframe.iloc[self.data_index, starting_position:starting_position + 4] = self.sleep_state, int(self.has_seizure), self.start_time, self.end_time
        
        print(self.result_dataframe)
        
        self.load_next_graph()
        
    def on_no_save_click(self):
        print("No save clicked")
        self.reset_GUI()
        
    def on_canvas_click(self, event):
        if self.timeframe_selection_started:
            if event.xdata is not None:  # Check if the click event occurred within the plot area
                # Get the x-coordinate of the mouse click on the plot
                x_coord = event.xdata

                # If the start time is not set, record the start time
                if self.start_time is None:
                    self.start_time = x_coord
                    self.plot_graph()
                elif self.end_time is None:
                    # Record the end time and reset the selection process
                    self.end_time = x_coord
                    if(self.end_time > self.start_time):
                        self.plot_graph()
                        print("End time recorded: ", self.end_time)
                        self.switch_to_save_buttons()
                    else:
                        self.start_time = None
                        self.end_time = None
                        self.plot_graph()
                        print("End time must be after start")
                        self.seizure_label.configure(text="Select seizure timeframe (start and end) - End time must be AFTER start")

    def reset_GUI(self):
        # Resets instance variables
        self.timeframe_selection_started = False
        self.start_time = None
        self.end_time = None
        self.has_seizure = None
        self.sleep_state = None
        self.gain = 1.0
        self.switch_to_bipolar()
        # Reset the buttons and labels
        self.question_mark_label.forget()
        self.yes_seizure.pack_forget()
        self.no_seizure.pack_forget()
        self.yes_save.pack_forget()
        self.no_save.pack_forget()
        self.awake_button.pack(side=tk.LEFT)
        self.n1_button.pack(side=tk.LEFT)
        self.n2_button.pack(side=tk.LEFT)
        self.n3_button.pack(side=tk.LEFT)
        self.rem_button.pack(side=tk.LEFT)
        self.undet_button.pack(side=tk.LEFT)
        self.seizure_label.configure(text="Sleep State: ")
        self.seizure_label.pack(side=tk.LEFT, padx=10)
        self.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)
        
        self.plot_graph()
        
    def replot_graph(self, event):
        self.font_size = self.plot_graph()
        self.resize_menu()
    
    def resize_menu(self):
        self.awake_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.n1_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.n2_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.n3_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.rem_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.undet_button.configure(font=("Arial", int(self.font_size * 1.5)))
        self.yes_seizure.configure(font=("Arial", int(self.font_size * 1.5)))
        self.no_seizure.configure(font=("Arial", int(self.font_size * 1.5)))
        self.yes_save.configure(font=("Arial", int(self.font_size * 1.5)))
        self.no_save.configure(font=("Arial", int(self.font_size * 1.5)))
        self.seizure_label.configure(font=("Arial", int(self.font_size * 1.5)))
        self.question_mark_label.configure(font=("Arial", int(self.font_size * 1.75)))
        
    def load_next_graph(self):
        result_file_path = os.path.join(current_dir, "result_dataframe.csv")
        self.save_result_dataframe(result_file_path)
        self.data_index += 1
        
        # Handles different input formats
        if self.is_csv_format:
            source_data = self.file
        else:
            source_data = self.folder_file_list
    
        if self.data_index < len(source_data):
            if self.is_csv_format:
                self.current_dataset_info = source_data.iloc[self.data_index]
                self.dataset_name = self.current_dataset_info.dataset_name
                self.dataset_start_time = self.current_dataset_info.dataset_start_time
                self.duration = self.current_dataset_info.duration       
            self.parse_calculate()
            self.reset_GUI()
        else:
            self.yes_save.pack_forget()
            self.no_save.pack_forget()
            self.seizure_label.configure(text="All data analyzed. Window will automatically close shortly.")
            self.after(5000, self.destroy)
        
    def parse_calculate(self):
        # Extracts raw data depending on input format
        raw_data = None
        # IEEG database method
        if(self.is_csv_format):
            with Session(self.username, self.password) as session:
                dataset = session.open_dataset(self.dataset_name)
                print(dataset)
                channels = list(range(len(dataset.ch_labels)))
                print(dataset.ch_labels)
                raw_data = dataset.get_data(self.dataset_start_time, self.duration, channels)
                raw_data = pd.DataFrame(raw_data, columns=dataset.ch_labels)
                print(raw_data)
                session.close_dataset(self.dataset_name)
        # Local EDF file method
        else:
            cur_edf_filepath = self.folder_file_list[self.data_index]
            cur_edf_filepath = os.path.join(self.file, cur_edf_filepath)
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
            print(raw_data)
            
        sns.set_theme(style="dark")
        
        # Downsamples data
        downsample_factor = 10
        downsampled_data = raw_data.iloc[::downsample_factor, :]
        num_rows = downsampled_data.shape[0]
        self.time_in_s = np.arange(num_rows) * (self.duration / num_rows) / 1000000

        self.bipolar_montage = generate_bipolar_montages(downsampled_data)
        which_chs = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', "Pz", 'T3', 'T4', 'T5', 'T6']
        self.car_montage = car_montage(downsampled_data, which_chs)
            
        self.current_montage = self.bipolar_montage
    
    def plot_graph(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if (self.current_montage_str == "bipolar"):
            self.current_montage = self.bipolar_montage
        else:
            self.current_montage = self.car_montage
                
        offset = 0
        self.ax.set_yticklabels([])
            
        # Dynamically adjust font size based on the figure size
        figure_width, figure_height = self.figure.get_size_inches()
        scaling_factor = min(figure_width / 20, figure_height / 12)
        font_size = int(20 * scaling_factor)
            
        for column in self.current_montage.columns:
            line = sns.lineplot(
                x=self.time_in_s,
                y=(self.current_montage[column] - self.current_montage[column].iloc[0] + offset),
                ax=self.ax
            )
            line.annotate(column, xy=(self.time_in_s[0], self.current_montage[column].iloc[0] - self.current_montage[column].iloc[0] + offset),
                xytext=(-10, 0), textcoords='offset points', ha='right', fontsize=font_size)
            offset -= 1000 / self.gain
                
        if self.start_time is not None:
            self.ax.axvline(x=self.start_time, color='black', linestyle='--', linewidth=2)
                
        if self.end_time is not None:
            self.ax.axvline(x=self.end_time, color='black', linestyle='--', linewidth=2)

        if self.current_montage_str == "bipolar":
            title = 'AP Bipolar\n'
        if self.current_montage_str == "car":
            title = 'Common Average Reference\n'
        self.ax.set_title(title, fontsize=int(font_size) * 1.33)
        self.ax.set_xlabel('Time (s)', fontsize=font_size)
        self.ax.tick_params(axis='x', labelsize=font_size)
        self.ax.set_ylabel('')
        self.ax.set_ylim(-1000 * (self.current_montage.shape[1] + 1) / self.gain, 1000 / self.gain)

        self.canvas.draw()
            
        return font_size

def generate_bipolar_montages(dataframe):
    # Define bipolar pairs
    bi_pairs = [('Fp1', 'F7'),
                ('F7', 'T3'),
                ('T3', 'T5'),
                ('T5', 'O1'),
                ('Fp2', 'F8'),
                ('F8', 'T4'),
                ('T4', 'T6'),
                ('T6', 'O2'),
                ('Fp1', 'F3'),
                ('F3', 'C3'),
                ('C3', 'P3'),
                ('P3', 'O1'),
                ('Fp2', 'F4'),
                ('F4', 'C4'),
                ('C4', 'P4'),
                ('P4', 'O2'),
                ('Fz', 'Cz')]

    bi_values = []
    bi_labels = []

    for pair in bi_pairs:
        first, second = pair

        if first in dataframe.columns and second in dataframe.columns:
            if dataframe[first].isnull().any() or dataframe[second].isnull().any():
                bi_values.append(None)
            else:
                montage_values = dataframe[first] - dataframe[second]
                bi_values.append(montage_values)    
            bi_labels.append(f"{first}-{second}")

    bi_values = pd.DataFrame(bi_values).T
    bi_labels = pd.Series(bi_labels)
    bi_values.rename(columns=dict(zip(bi_values.columns, bi_labels)), inplace=True)

    return bi_values


def car_montage(df, which_chs):
    car_labels = []
    car_values = df.copy()
    column_names = car_values.columns.tolist()

    # Filter out columns with NaN values
    valid_columns = [col for col in df.columns if df[col].notna().all()]
    invalid_columns = [col for col in column_names if col not in valid_columns]
    
    valid_chs = list(set(which_chs).intersection(valid_columns))
    valid_chs_indices = [df.columns.get_loc(channel) for channel in valid_chs]
    
    average = np.nanmean(car_values.iloc[:, valid_chs_indices], axis=1, keepdims=True)

    car_values -= average

    for invalid_col in invalid_columns:
        car_values[invalid_col] = None
        
    for label in car_values.columns:
        car_labels.append(label + '-CAR')

    car_values.rename(columns=dict(zip(car_values.columns, car_labels)), inplace=True)
    return car_values

def is_valid_csv_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', required=True, help='username')
    parser.add_argument('-p', '--password', help='password')
    parser.add_argument('data_file', help='name of local CSV file/folder directory containing dataset information')
    parser.add_argument('--delay', help='delay time of input (default = 0.5 s)')
    args = parser.parse_args()
        
    username = args.user
    password = args.password
    if (args.delay is not None):
        delay = args.delay
    else:
        delay = 0.5
    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, args.data_file)
    if is_valid_csv_file(data_file_path):
        file = pd.read_csv(data_file_path)
    else:
        file = data_file_path 
    EEGGraphGUI(username, password, file, delay)
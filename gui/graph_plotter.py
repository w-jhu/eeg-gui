import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GraphPlotter:
    def __init__(self, gui):
        self.gui = gui
        self.figure = plt.figure(figsize=(20, 12))
        self.current_montage_str = "bipolar"
        self.gain = 1.0
        self.current_montage = None
        self.timer_id = None

    def plot_graph(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if self.current_montage_str == "bipolar":
            self.current_montage = self.gui.data_processor.bipolar_montage
        else:
            self.current_montage = self.gui.data_processor.car_montage

        offset = 0
        self.ax.set_yticklabels([])

        figure_width, figure_height = self.figure.get_size_inches()
        scaling_factor = min(figure_width / 20, figure_height / 12)
        font_size = int(20 * scaling_factor)

        for column in self.current_montage.columns:
            line = sns.lineplot(
                x=self.gui.data_processor.time_in_s,
                y=(self.current_montage[column] - self.current_montage[column].iloc[0] + offset),
                ax=self.ax
            )
            line.annotate(column, xy=(self.gui.data_processor.time_in_s[0], self.current_montage[column].iloc[0] - self.current_montage[column].iloc[0] + offset),
                          xytext=(-10, 0), textcoords='offset points', ha='right', fontsize=font_size)
            offset -= 1000 / self.gain

        if self.gui.start_time is not None:
            self.ax.axvline(x=self.gui.start_time, color='black', linestyle='--', linewidth=2)

        if self.gui.end_time is not None:
            self.ax.axvline(x=self.gui.end_time, color='black', linestyle='--', linewidth=2)

        title = 'AP Bipolar\n' if self.current_montage_str == "bipolar" else 'Common Average Reference\n'
        self.ax.set_title(title, fontsize=int(font_size) * 1.33)
        self.ax.set_xlabel('Time (s)', fontsize=font_size)
        self.ax.tick_params(axis='x', labelsize=font_size)
        self.ax.set_ylabel('')
        self.ax.set_ylim(-1000 * (self.current_montage.shape[1] + 1) / self.gain, 1000 / self.gain)

        self.gui.canvas.draw()

        return font_size

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

    def replot_graph(self):
        self.font_size = self.plot_graph()
        self.gui.button_handlers.resize_menu()

    def close_plots(self):
        plt.close("all")
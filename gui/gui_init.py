import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from button_handlers import ButtonHandlers
from graph_plotter import GraphPlotter
from data_processor import DataProcessor

class EEGGraphGUI(tk.Tk):
    def __init__(self, username, password, file, delay):
        super().__init__()
        self.data_processor = DataProcessor(username, password, file)
        self.graph_plotter = GraphPlotter(self)
        self.button_handlers = ButtonHandlers(self)

        self.delay = delay
        self.result_file_path = self.data_processor.result_file_path

        self.initialize_gui()
        self.mainloop()
        self.graph_plotter.close_plots()

    def initialize_gui(self):
        self.setup_figure()
        self.setup_buttons()
        self.bind_shortcuts()
        self.load_initial_graph()

    def setup_figure(self):
        self.figure = self.graph_plotter.figure
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def setup_buttons(self):
        self.button_handlers.setup_buttons()

    def bind_shortcuts(self):
        self.bind("<Left>", lambda event: self.graph_plotter.switch_to_bipolar())
        self.bind("<Right>", lambda event: self.graph_plotter.switch_to_car())
        self.bind("<Up>", self.graph_plotter.increase_gain)
        self.bind("<Down>", self.graph_plotter.decrease_gain)
        self.bind("r", lambda event: self.reset_GUI())
        self.bind("<Configure>", self.on_configure)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def load_initial_graph(self):
        self.data_processor.parse_calculate()
        self.graph_plotter.plot_graph()

    def on_configure(self, event):
        if self.graph_plotter.timer_id:
            self.after_cancel(self.graph_plotter.timer_id)
        self.graph_plotter.timer_id = self.after(500, lambda: self.replot_graph(event))

    def on_canvas_click(self, event):
        self.button_handlers.handle_canvas_click(event)

    def reset_GUI(self):
        self.button_handlers.reset_GUI()

    def replot_graph(self, event):
        self.graph_plotter.replot_graph()
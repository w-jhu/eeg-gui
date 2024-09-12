import tkinter as tk
import time

class ButtonHandlers:
    def __init__(self, gui):
        self.gui = gui

    def setup_buttons(self):
        self.gui.buttons_frame = tk.Frame(self.gui)
        self.gui.buttons_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, pady=10)
        self.gui.seizure_label = tk.Label(self.gui.buttons_frame, text="Sleep State: ", font=("Arial", 16))
        self.gui.seizure_label.pack(side=tk.LEFT, padx=10)

        self.create_sleep_buttons()
        self.create_seizure_buttons()
        self.create_help_button()

    def create_sleep_buttons(self):
        self.gui.awake_button = tk.Button(self.gui.buttons_frame, text="Awake", command=self.gui.on_awake_button_click, font=("Arial", 16))
        self.gui.n1_button = tk.Button(self.gui.buttons_frame, text="N1", command=self.gui.on_n1_button_click, font=("Arial", 16))
        self.gui.n2_button = tk.Button(self.gui.buttons_frame, text="N2", command=self.gui.on_n2_button_click, font=("Arial", 16))
        self.gui.n3_button = tk.Button(self.gui.buttons_frame, text="N3", command=self.gui.on_n3_button_click, font=("Arial", 16))
        self.gui.rem_button = tk.Button(self.gui.buttons_frame, text="REM", command=self.gui.on_rem_button_click, font=("Arial", 16))
        self.gui.undet_button = tk.Button(self.gui.buttons_frame, text="Undetermined", command=self.gui.on_undet_button_click, font=("Arial", 16))

        self.gui.awake_button.pack(side=tk.LEFT)
        self.gui.n1_button.pack(side=tk.LEFT)
        self.gui.n2_button.pack(side=tk.LEFT)
        self.gui.n3_button.pack(side=tk.LEFT)
        self.gui.rem_button.pack(side=tk.LEFT)
        self.gui.undet_button.pack(side=tk.LEFT)

    def create_seizure_buttons(self):
        self.gui.yes_seizure = tk.Button(self.gui.buttons_frame, text="Yes", command=self.gui.on_yes_seizure_click, font=("Arial", 16))
        self.gui.no_seizure = tk.Button(self.gui.buttons_frame, text="No", command=self.gui.on_no_seizure_click, font=("Arial", 16))
        self.gui.yes_save = tk.Button(self.gui.buttons_frame, text="Yes", command=self.gui.on_yes_save_click, font=("Arial", 16))
        self.gui.no_save = tk.Button(self.gui.buttons_frame, text="No", command=self.gui.on_no_save_click, font=("Arial", 16))

    def create_help_button(self):
        question_mark_unicode = "\u2753"
        self.gui.question_mark_label = tk.Label(self.gui.buttons_frame, text=question_mark_unicode, font=("Arial", 20), cursor="hand2")
        self.gui.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)
        self.gui.question_mark_label.bind("<Button-1>", self.gui.show_popup_window)

    def handle_canvas_click(self, event):
        if self.gui.timeframe_selection_started:
            if event.xdata is not None:
                x_coord = event.xdata
                if self.gui.start_time is None:
                    self.gui.start_time = x_coord
                    self.gui.graph_plotter.plot_graph()
                elif self.gui.end_time is None:
                    self.gui.end_time = x_coord
                    if self.gui.end_time > self.gui.start_time:
                        self.gui.graph_plotter.plot_graph()
                        self.switch_to_save_buttons()
                    else:
                        self.gui.start_time = None
                        self.gui.end_time = None
                        self.gui.graph_plotter.plot_graph()
                        self.gui.seizure_label.configure(text="Select seizure timeframe (start and end) - End time must be AFTER start")

    def reset_GUI(self):
        self.gui.timeframe_selection_started = False
        self.gui.start_time = None
        self.gui.end_time = None
        self.gui.has_seizure = None
        self.gui.sleep_state = None
        self.gui.gain = 1.0
        self.gui.graph_plotter.switch_to_bipolar()

        self.gui.question_mark_label.forget()
        self.gui.yes_seizure.pack_forget()
        self.gui.no_seizure.pack_forget()
        self.gui.yes_save.pack_forget()
        self.gui.no_save.pack_forget()
        self.gui.awake_button.pack(side=tk.LEFT)
        self.gui.n1_button.pack(side=tk.LEFT)
        self.gui.n2_button.pack(side=tk.LEFT)
        self.gui.n3_button.pack(side=tk.LEFT)
        self.gui.rem_button.pack(side=tk.LEFT)
        self.gui.undet_button.pack(side=tk.LEFT)
        self.gui.seizure_label.configure(text="Sleep State: ")
        self.gui.seizure_label.pack(side=tk.LEFT, padx=10)
        self.gui.question_mark_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)

        self.gui.graph_plotter.plot_graph()
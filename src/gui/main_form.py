
import tkinter as tk
from tkinter import ttk
from sys import platform

from globals import log_message
from .tab_data_import import TabDataImport
from .tab_training import TabTraining

class Mainform:

    def __init__(self):
        
        self.window = tk.Tk()        
        self.note_book = ttk.Notebook(self.window)

        self.tabImport = TabDataImport(self.note_book)
        self.tabTraining = TabTraining(self.note_book)
        self.tabImport.training_tab = self.tabTraining

        self.note_book.add(self.tabImport, text = 'Import Datasets')
        self.note_book.add(self.tabTraining, text = 'Training Models')
        self.note_book.pack(expand = 1, fill = 'both')
        self.note_book.tab(1, state = 'disabled')

        
    def start(self):
        self.window.title('Activity Recognition Modelling')

        # Checking the OS platform
        if platform == "darwin": # Mac OS
            self.window.geometry('1230x700')
            self.window.geometry('+{}+{}'.format(30, 30))
        else: # Windows or others
            self.window.geometry('935x700')
            self.window.geometry('+{}+{}'.format(210, 30))
        
        self.window.resizable(0, 0)
        # self.window.deiconify()
        self.window.mainloop()
        log_message('Thank you for your patience!')
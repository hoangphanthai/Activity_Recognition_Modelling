import  sys
from os import path
import globals
from gui.main_form import Mainform
from data import ini_file

if __name__ == '__main__':
    
    # Initializing the global variables
    globals.init()
    
    # Getting global settings from app.ini file
    ini_file.initialize_general_settings()
    
    mainUI = Mainform()
    mainUI.start()

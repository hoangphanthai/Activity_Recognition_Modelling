import os
import datetime
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import globals
from globals import log_message
from data import data_importer, ini_file
from data.data_resampling import get_original_sampling_rate


class TabDataImport(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        
        self.build_controls()
        self.fill_init_data_into_controls()
        self.rdoDataSourceSelectDeselect()              
        self.training_tab = None

    def build_controls(self):
        self.rdoDataSource = tk.IntVar()
        self.txtDBini_text = tk.StringVar()
        self.txtHost_text = tk.StringVar()
        self.txtPort_text = tk.StringVar()
        self.txtDB_text = tk.StringVar()
        self.txtUser_text = tk.StringVar()
        self.txtPwd_text = tk.StringVar()
        self.txtTrainTable_text = tk.StringVar()       
        self.selectMonitoringActivityOnOff = tk.IntVar()
        self.txtMonitoringTable_Text = tk.StringVar()       
        self.txtResultTable_Text = tk.StringVar()
      
        self.txtTrainCSVFile_Text = tk.StringVar()
        self.selectMonitoringCSVActivityOnOff = tk.IntVar() 
        self.txtMonitoringCSVFile_Text = tk.StringVar() 
        self.txtResultCSVFile_Text =  tk.StringVar()
        
        self.lbl_frame_select_data_source = tk.LabelFrame(self, text = 'Select data source',font = ('Sans', '10', 'bold'))
        self.lbl_frame_select_data_source.grid (row = 0, column = 0, padx = 20, pady = 10, sticky = tk.W)

        self.radSourceDB = tk.Radiobutton(self.lbl_frame_select_data_source, text = 'Database', command = self.rdoDataSourceSelectDeselect, variable = self.rdoDataSource, value = 1)
        self.radSourceDB.grid(row = 0, column = 0)
        self.radSourceDB.configure(font = ('Sans', '10'))
        self.radSourceCSV = tk.Radiobutton(self.lbl_frame_select_data_source, text = 'CSV files', command = self.rdoDataSourceSelectDeselect, variable = self.rdoDataSource, value = 0)
        self.radSourceCSV.grid(row = 0, column = 1)
        self.radSourceCSV.configure(font = ('Sans', '10'))
       
        self.lbl_frame_database_group = tk.LabelFrame(self, padx = 5, pady = 5, text='Database',font = ('Sans', '10', 'bold'))
        self.lbl_frame_database_group.grid (row = 1, column = 0, padx = 20, pady = 5)

        self.lblIni = tk.Label(self.lbl_frame_database_group, text = 'Database credentials file path')
        self.lblIni.grid(row = 0, column = 0, padx = 5, pady = 1, sticky=tk.W)

        self.txtDBini = tk.Entry(self.lbl_frame_database_group, width = 80, textvariable = self.txtDBini_text)
        self.txtDBini.grid(row = 0, column = 1, padx = 5, pady = 2)
        self.btnIni = tk.Button(self.lbl_frame_database_group, text = 'Select another INI file', command = self.btnIniSelect_clicked)
        self.btnIni.grid(row = 0, column = 2, padx = 5, pady = 1, sticky = tk.W)

        self.lblHost = tk.Label(self.lbl_frame_database_group, text = 'Host')
        self.lblHost.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtHost = tk.Entry(self.lbl_frame_database_group, width = 40, textvariable = self.txtHost_text)
        self.txtHost.grid(row = 1, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblPort = tk.Label(self.lbl_frame_database_group, text = 'Port')
        self.lblPort.grid(row = 2, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtPort = tk.Entry(self.lbl_frame_database_group, width = 40, textvariable = self.txtPort_text)
        self.txtPort.grid(row = 2, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblDB = tk.Label(self.lbl_frame_database_group, text = 'Database')
        self.lblDB.grid(row = 3, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtDB = tk.Entry(self.lbl_frame_database_group, width = 40, textvariable = self.txtDB_text)
        self.txtDB.grid(row = 3, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblUser = tk.Label(self.lbl_frame_database_group, text = 'User')
        self.lblUser.grid(row = 4, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtUser = tk.Entry(self.lbl_frame_database_group, width = 40, textvariable = self.txtUser_text)
        self.txtUser.grid(row = 4, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblPwd = tk.Label(self.lbl_frame_database_group, text = 'Password')
        self.lblPwd.grid(row = 5, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtPwd = tk.Entry(self.lbl_frame_database_group, width = 40, show = '*', textvariable = self.txtPwd_text)
        self.txtPwd.grid(row = 5, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblTable = tk.Label(self.lbl_frame_database_group, text = 'Train_Valid_Test data table')
        self.lblTable.grid(row = 6, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtTable = tk.Entry(self.lbl_frame_database_group, width = 80, textvariable = self.txtTrainTable_text)
        self.txtTable.grid(row = 6, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.btnMonitoringOnOff = tk.Checkbutton(self.lbl_frame_database_group, command = self.monitoringDBSelectDeselect, text = 'Statistics_Monitoring data table',
                                        variable = self.selectMonitoringActivityOnOff, onvalue = 1, offvalue = 0, height = 1)
        self.btnMonitoringOnOff.grid(row = 7, column = 0, padx = 5, pady = 2, sticky = tk.W)
        
        self.txtMonitoringTable = tk.Entry(self.lbl_frame_database_group, width = 80, textvariable = self.txtMonitoringTable_Text)
        self.txtMonitoringTable.grid(row = 7, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.lblResultTable = tk.Label(self.lbl_frame_database_group, text = 'Table to store result')
        self.lblResultTable.grid(row = 8, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtResultTable = tk.Entry(self.lbl_frame_database_group, width = 80, textvariable = self.txtResultTable_Text)
        self.txtResultTable.grid(row = 8, column = 1, padx = 5, pady = 2, sticky = tk.W)

        # CSV data source section
        self.lbl_frame_csv_group = tk.LabelFrame(self, padx = 5, pady = 5, text='CSV files',font = ('Sans', '10', 'bold'))
        self.lbl_frame_csv_group.grid(row = 2, column = 0, padx = 20, pady = 5, sticky = tk.W)

        self.lblTrainCSVFile = tk.Label(self.lbl_frame_csv_group, text = 'Train_Valid_Test data file')
        self.lblTrainCSVFile.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = tk.W)
        self.txtTrainCSVFile = tk.Entry(self.lbl_frame_csv_group, width = 80, textvariable = self.txtTrainCSVFile_Text)
        self.txtTrainCSVFile.grid(row = 0, column = 1, padx = 5, pady = 2, sticky = tk.W)
        
        self.btnIniTrainingCSV = tk.Button(self.lbl_frame_csv_group, text = 'Select another CSV file', command = self.btnIniSelectTraining_clicked)
        self.btnIniTrainingCSV.grid(row = 0, column = 2, padx = 5, pady = 1, sticky = tk.W)


        self.btnMonitoringCSVOnOff = tk.Checkbutton(self.lbl_frame_csv_group, command = self.monitoringCSVSelectDeselect, text = 'Statistics_Monitoring data file',
                                        variable = self.selectMonitoringCSVActivityOnOff, onvalue = 1, offvalue = 0, height = 1)
        self.btnMonitoringCSVOnOff.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = tk.W)

        self.txtMonitoringCSVFile = tk.Entry(self.lbl_frame_csv_group, width = 80, textvariable = self.txtMonitoringCSVFile_Text)
        self.txtMonitoringCSVFile.grid(row = 1, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.btnIniMonitoringCSV = tk.Button(self.lbl_frame_csv_group, text = 'Select another CSV file', command = self.btnIniSelectMonitoring_clicked)
        self.btnIniMonitoringCSV.grid(row = 1, column = 2, padx = 5, pady = 1, sticky = tk.W)

        self.lblResultCSVFile = tk.Label(self.lbl_frame_csv_group, text = 'Experiment result will be stored in ')
        self.lblResultCSVFile.grid(row = 2, column = 0, padx = 5, pady = 1, sticky = tk.W)

        self.txtResultCSV = tk.Entry(self.lbl_frame_csv_group, width = 80, textvariable = self.txtResultCSVFile_Text)
        self.txtResultCSV.grid(row = 2, column = 1, padx = 5, pady = 1, sticky = tk.W)

        self.lblNotification = tk.Label(self, text = '')
        self.lblNotification.grid(row = 3, column = 0, padx = 18, pady = 1, sticky = tk.W)
        
        self.btnImport = tk.Button(self,  height = 2, text = 'Connect \\ Import Data', bg = 'gold', command = self.connect_import_clicked)
        self.btnImport.grid(row = 4, column = 0, padx = 20, pady = 5, sticky = tk.E)
       
    def fill_init_data_into_controls(self):
        
        # Fill the previous section data
        db_credentials_file_path, params_db_select, params_db, params_csv = ini_file.get_import_data_controls_init_params()
        if len(params_db) < 9:
            messagebox.showinfo('Alert', 'Postgresql credentials from {} are missing \n Please input your own db credentials'.format(os.path.join(globals.dir_path, globals.APP_INI_FILE)))           
        else:
            
            # Database section
            self.txtDBini_text.set(db_credentials_file_path)
            self.txtHost_text.set(params_db[0][1])
            self.txtPort_text.set(params_db[1][1])
            self.txtDB_text.set(params_db[2][1])
            self.txtTrainTable_text.set(params_db[3][1])
            self.txtMonitoringTable_Text.set(params_db[4][1])
            
            if params_db[5][1] ==  '1':
                self.btnMonitoringOnOff.select()
                self.txtMonitoringTable.configure(state = 'normal')
                globals.monitoring_mode = True
            else:
                self.btnMonitoringOnOff.deselect()
                self.txtMonitoringTable.configure(state = 'disabled')
                globals.monitoring_mode = False
            
            self.txtResultTable_Text.set(params_db[6][1])
            self.txtUser_text.set(params_db[7][1])
            self.txtPwd_text.set(params_db[8][1])


            # CSV section
            # Check the existing of the input files otherwise set the ones from ini template, for better user experience
            txtTrainCSVFile = params_csv[0][1]
            if not os.path.isfile(txtTrainCSVFile):
                txtTrainCSVFile = os.path.join(globals.root_dir, 'datasets', 'sensor_data_lilith_35_hours.csv')

            txtMonitoringCSVFile = params_csv[1][1]
            if not os.path.isfile(txtMonitoringCSVFile):
                txtMonitoringCSVFile = os.path.join(globals.root_dir, 'datasets', 'sensor_data_hanna_35_hours.csv')

            self.txtTrainCSVFile_Text.set(txtTrainCSVFile)
            self.txtMonitoringCSVFile_Text.set(txtMonitoringCSVFile)
            if params_csv[2][1] ==  '1':
                self.btnMonitoringCSVOnOff.select()
                self.txtMonitoringCSVFile.configure(state = 'normal')
                self.btnIniMonitoringCSV.configure(state = 'normal')
                globals.monitoring_mode = True
            else:
                self.btnMonitoringCSVOnOff.deselect()
                self.txtMonitoringCSVFile.configure(state = 'disabled')
                self.btnIniMonitoringCSV.configure(state = 'disabled')
                globals.monitoring_mode = False
                
            self.txtResultCSVFile_Text.set( globals.EXPERIMENT_RESULT_FILE + ' located at ' + os.path.join('{project working directory}', 'reports','csv_out', '{newly created foler}','{sampling rate}','{windows}'))
            
            # Enable / Disable selected data source section
            if params_db_select[0][1] ==  '1':
                self.radSourceDB.select()
                globals.data_from_db = True
            else:
                self.radSourceDB.deselect()
                globals.data_from_db = False

    def btnIniSelectTraining_clicked(self):
        file = filedialog.askopenfilename(filetypes = (('CSV files', '*.csv'), ('All files', '*.*')))
        self.txtTrainCSVFile_Text.set(str(file))

    def btnIniSelectMonitoring_clicked(self):
        file = filedialog.askopenfilename(filetypes = (('CSV files', '*.csv'), ('All files', '*.*')))
        self.txtMonitoringCSVFile_Text.set(str(file))
        
    def btnIniSelect_clicked(self):
        
        file = filedialog.askopenfilename(filetypes = (('Ini files', '*.ini'), ('All files', '*.*')))
        if file:
            try:
                self.txtDBini_text.set(str(file))

                params_db = ini_file.get_db_params_from_in_file(str(file))
                
            
                self.txtHost_text.set(params_db[0][1])
                self.txtPort_text.set(params_db[1][1])
                self.txtDB_text.set(params_db[2][1])
                self.txtTrainTable_text.set(params_db[3][1])
                self.txtMonitoringTable_Text.set(params_db[4][1])

                if params_db[5][1] == '1':
                    self.btnMonitoringOnOff.select()
                    self.txtMonitoringTable.configure(state = 'normal')
                    globals.monitoring_mode = True
                else:
                    self.btnMonitoringOnOff.deselect()
                    self.txtMonitoringTable.configure(state = 'disabled')
                    globals.monitoring_mode = False

                self.txtResultTable_Text.set(params_db[6][1])
                self.txtUser_text.set(params_db[7][1])
                self.txtPwd_text.set(params_db[8][1])

            except:
                messagebox.showinfo('Alert', 'Postgresql credentials from {} are missing \n Please fill your own credentials'.format(str(file)))

    def rdoDataSourceSelectDeselect(self):
        if self.rdoDataSource.get() ==  1:
            
            # Enable database section controls 
            self.txtHost.configure(state = 'normal')
            self.txtPort.configure(state = 'normal')
            self.txtDB.configure(state = 'normal')
            self.txtUser.configure(state = 'normal')
            self.txtPwd.configure(state = 'normal')
            self.txtTable.configure(state = 'normal')
            self.txtMonitoringTable.configure(state = 'normal')
            self.btnMonitoringOnOff.configure(state = 'normal')
            self.txtResultTable.configure(state = 'normal')
            self.btnIni.configure(state = 'normal')
            self.txtDBini.configure(state = 'normal')

            self.lblIni.configure(state = 'normal')
            self.lblHost.configure(state = 'normal')
            self.lblPort.configure(state = 'normal')
            self.lblDB.configure(state = 'normal')
            self.lblUser.configure(state = 'normal')
            self.lblPwd.configure(state = 'normal')
            self.lblTable.configure(state = 'normal')
            self.lblResultTable.configure(state = 'normal')
            
            if self.selectMonitoringActivityOnOff.get() ==  0:
                self.txtMonitoringTable.configure(state = 'disabled')
            else:
                self.txtMonitoringTable.configure(state = 'normal')

            # Disable CSV section controls
            self.lblTrainCSVFile.configure(state = 'disabled') 
            self.txtTrainCSVFile.configure(state = 'disabled')
            self.btnIniTrainingCSV.configure(state = 'disabled') 
            self.btnMonitoringCSVOnOff.configure(state = 'disabled')
            self.txtMonitoringCSVFile.configure(state = 'disabled')
            self.btnIniMonitoringCSV.configure(state = 'disabled')
            self.lblResultCSVFile.configure(state = 'disabled')
            self.txtResultCSV.configure(state = 'disabled')                  

            globals.data_from_db = True

        elif self.rdoDataSource.get() ==  0:
            
            # Disable database section controls 
            self.txtHost.configure(state = 'disabled')
            self.txtPort.configure(state = 'disabled')
            self.txtDB.configure(state = 'disabled')
            self.txtUser.configure(state = 'disabled')
            self.txtPwd.configure(state = 'disabled')
            self.txtTable.configure(state = 'disabled')
            self.txtMonitoringTable.configure(state = 'disabled')
            self.btnMonitoringOnOff.configure(state = 'disabled')
            self.txtResultTable.configure(state = 'disabled')
            self.btnIni.configure(state = 'disabled')
            self.txtDBini.configure(state = 'disabled')

            self.lblIni.configure(state = 'disabled')
            self.lblHost.configure(state = 'disabled')
            self.lblPort.configure(state = 'disabled')
            self.lblDB.configure(state = 'disabled')
            self.lblUser.configure(state = 'disabled')
            self.lblPwd.configure(state = 'disabled')
            self.lblTable.configure(state = 'disabled')
            self.lblResultTable.configure(state = 'disabled')

            # Enable CSV section controls
            self.lblTrainCSVFile.configure(state = 'normal') 
            self.txtTrainCSVFile.configure(state = 'normal') 
            self.btnIniTrainingCSV.configure(state = 'normal')  
            self.btnMonitoringCSVOnOff.configure(state = 'normal') 
            self.txtMonitoringCSVFile.configure(state = 'normal') 
            self.btnIniMonitoringCSV.configure(state = 'normal') 
            self.lblResultCSVFile.configure(state = 'normal') 
            self.txtResultCSV.configure(state = 'normal')

            if self.selectMonitoringCSVActivityOnOff.get() ==  0:
                self.txtMonitoringCSVFile.configure(state = 'disabled')
                self.btnIniMonitoringCSV.configure(state = 'disabled')
            else:
                self.txtMonitoringCSVFile.configure(state = 'normal')
                self.btnIniMonitoringCSV.configure(state = 'normal')

            globals.data_from_db = False

    def monitoringDBSelectDeselect(self):
        if self.selectMonitoringActivityOnOff.get() ==  0:
            self.txtMonitoringTable.configure(state = 'disabled')           
            self.training_tab.txtWindowSimuStride.configure(state = 'disabled')
            globals.monitoring_mode = False
        else:
            self.txtMonitoringTable.configure(state = 'normal')
            self.training_tab.txtWindowSimuStride.configure(state = 'normal')
            globals.monitoring_mode = True
  
    def monitoringCSVSelectDeselect(self):
        if self.selectMonitoringCSVActivityOnOff.get() ==  0:

            self.txtMonitoringCSVFile.configure(state = 'disabled')
            self.btnIniMonitoringCSV.configure(state = 'disabled')
            self.training_tab.txtWindowSimuStride.configure(state = 'disabled')

            # self.training_tab.btnStatics.configure(state = 'disabled')
            # self.training_tab.btnMonitorDist.configure(state = 'disabled')
            # self.training_tab.btnSimulatition.configure(state = 'disabled')
            globals.monitoring_mode = False

        else:
            self.txtMonitoringCSVFile.configure(state = 'normal')
            self.btnIniMonitoringCSV.configure(state = 'normal')
            self.training_tab.txtWindowSimuStride.configure(state = 'normal')

            # self.training_tab.btnStatics.configure(state = 'normal')
            # self.training_tab.btnMonitorDist.configure(state = 'normal')
            # self.training_tab.btnSimulatition.configure(state = 'normal')
            globals.monitoring_mode = True            

    def connect_import_clicked(self):
        
        globals.start_time = datetime.datetime.now()
        data_fetching_successful = False

        if globals.data_from_db:
            self.lblNotification.configure(font = ('Sans', '10', 'bold'), text = 'Connecting to the database and fetching data...')
            
            # Setting table names and values from input form
            globals.experiment_result_table_name = self.txtResultTable_Text.get()
            globals.training_table_name = self.txtTrainTable_text.get() 
            globals.train_valid_data_set_name = globals.training_table_name[:18]

            globals.monitoring_table_name = self.txtMonitoringTable_Text.get()
            globals.monitoring_data_set_name = globals.monitoring_table_name[:18] 
            
            # Reading connection parameters
            params_db = {}
            params_db['host'] = self.txtHost_text.get()
            params_db['port'] = self.txtPort_text.get()
            params_db['database'] = self.txtDB_text.get()
            params_db['user'] = self.txtUser_text.get()
            params_db['password'] = self.txtPwd_text.get()
            globals.params_db = params_db
            
            data_fetching_successful = data_importer.get_training_monitoring_data_from_db()
            if data_fetching_successful:
                self.lblNotification.configure(text = 'Finished data importing!')
           
            if self.selectMonitoringActivityOnOff.get() == 0:
                globals.monitoring_mode = False
            else:
                globals.monitoring_mode = True                

        else:
            self.lblNotification.configure(font = ('Sans', '10', 'bold'), text = 'Reading data from CSV files...')

            # Setting file names and values from input form
            globals.training_valid_test_data_file = self.txtTrainCSVFile_Text.get()                      
            globals.train_valid_data_set_name = os.path.basename(globals.training_valid_test_data_file).replace('.','')
            globals.monitoring_data_file = self.txtMonitoringCSVFile_Text.get()
            globals.monitoring_data_set_name = os.path.basename(globals.monitoring_data_file).replace('.','')

            data_fetching_successful = data_importer.get_training_monitoring_data_from_csv_file()
            if data_fetching_successful:
                self.lblNotification.configure(text = 'Finished data importing!')

            if self.selectMonitoringCSVActivityOnOff.get() ==  0:
                globals.monitoring_mode = False
            else:
                globals.monitoring_mode = True

        if data_fetching_successful:
                
            # Enabling Training tab
            self.master.tab(1, state = 'normal')

            globals.cboActivityValues = pd.Series(np.unique(np.array(globals.train_valid_test_data_frame['label']).tolist()))

            # Reset_activity_cbo_values in the training tab
            self.training_tab.reset_activity_cbo_values()       

            # Update_activity_cbo_values in the training tab
            self.training_tab.update_activity_cbo_values()
            
            # Temporarily disable the monitoring mode buttons
            self.training_tab.disable_monitoring_process_buttons()

            # Predict the original sensor sampling rate
            globals.original_sampling_rate = get_original_sampling_rate()

            # Update db credentials file path in app.ini
            ini_file.update_db_credentials_file_path(self.txtDBini_text.get())
            
            # Update data source selection
            data_source_select_params = {}
            if self.rdoDataSource.get() == 1:
                data_source_select_params['dbisselected'] = '1'
            else:    
                data_source_select_params['dbisselected'] = '0'


            # Update cridentials for database connection
            db_credentials_params = {}
            db_credentials_params['host'] = self.txtHost_text.get()
            db_credentials_params['port'] = self.txtPort_text.get()
            db_credentials_params['database'] = self.txtDB_text.get()
            db_credentials_params['traintable'] = self.txtTrainTable_text.get()
            db_credentials_params['monitortable'] = self.txtMonitoringTable_Text.get()

            if self.selectMonitoringActivityOnOff.get() == 0:
                db_credentials_params['selectmonitoring'] = '0'
            else:
                db_credentials_params['selectmonitoring'] = '1'
                
            db_credentials_params['resulttable'] = self.txtResultTable_Text.get()
            db_credentials_params['user'] = self.txtUser_text.get()
            db_credentials_params['password'] = self.txtPwd_text.get()

            csv_paths_params = {}
            csv_paths_params['trainfile'] = self.txtTrainCSVFile_Text.get()
            csv_paths_params['monitoringfile'] = self.txtMonitoringCSVFile_Text.get()

            if self.selectMonitoringCSVActivityOnOff.get() ==  0:
                csv_paths_params['selectmonitoringfile'] = '0'
            else:
                csv_paths_params['selectmonitoringfile'] = '1'

            # Update import_data_controls_init into db credential ini file
            ini_file.update_import_data_controls_init_params(self.txtDBini_text.get(),data_source_select_params,db_credentials_params,csv_paths_params)
            # Setting focus to the training_tab
            self.master.select(1)

        else:
            self.lblNotification.configure(text = 'Data importing is unsuccessful!')
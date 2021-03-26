import pandas as pd
import numpy as np
import datetime
import math
import winsound
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from tkinter import *
#import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from configparser import ConfigParser
import psycopg2
import json
import pickle
# ----- user defined modules ->
from features import *
from user_defined_functions import *

# ----- user defined modules <-

pd.set_option('display.max_columns', 30)
# pd.set_option('display.max_rows', None)

train_valid_test_db_table = pd.DataFrame()
monitoring_db_table = pd.DataFrame()
monitoring_db_table_resampled_monitor = pd.DataFrame()
monitoring_data_fr = pd.DataFrame()  # grounthTruth label and predicted
predicted_data_fr = pd.DataFrame()
monitoring_time_deviation_fr = pd.DataFrame()
monitoring_error_types_fr = pd.DataFrame()

appConfig = ConfigParser()  # Parser for Apps
iniConfig = ConfigParser()  # Parser for Ini
timestart = datetime.datetime.now()

window = Tk()
window.title('CAR Model Building')
window.geometry('950x572')
window.geometry('+{}+{}'.format(210, 30))
window.resizable(0, 0)
original_sampling_rate = IntVar()
resamplingrate = IntVar()
functions_labels_json = {}
features_in_dictionary = {}
axes_to_apply_functions_list = []
function_set_for_resampling = ['mean']
curr_monitoring_algorithm = ''
curr_monitoring_window_size = IntVar()
curr_monitoring_sampling_rate = IntVar()

params_db = {}
rdoTrainingPhraseOnly = IntVar()
label_set = pd.Series([], dtype="string")
cboValues = pd.Series([], dtype="string")
rdoSimuStartTime = IntVar()
timestampforCSVfiles = ''
monitoring_mode = False
model_comment = 'Comments:'


def write_apps_ini_files():
    appConfig['TRAINING TAB']['selectallactivityonoff'] = str(selectAllActivityOnOff.get())
    appConfig['TRAINING TAB']['btnallfeaturesonoff'] = str(selectAllFeaturesOnOff.get())
    appConfig['TRAINING TAB']['chkbtnmin'] = str(MinVar.get())
    appConfig['TRAINING TAB']['chkbtnmax'] = str(MaxVar.get())
    appConfig['TRAINING TAB']['chkbtnmean'] = str(MeanVar.get())
    appConfig['TRAINING TAB']['chkbtnmedian'] = str(MedianVar.get())
    appConfig['TRAINING TAB']['chkbtnstdev'] = str(StdVar.get())
    appConfig['TRAINING TAB']['chkbtninterquartilerange'] = str(IQRVar.get())
    appConfig['TRAINING TAB']['chkbtnrootmsvar'] = str(rootMSVar.get())
    appConfig['TRAINING TAB']['chkbtnmeancrvar'] = str(meanCRVar.get())
    appConfig['TRAINING TAB']['chkbtnkurtosisvar'] = str(kurtosisVar.get())
    appConfig['TRAINING TAB']['chkbtnskewnessvar'] = str(skewnessVar.get())
    appConfig['TRAINING TAB']['chkbtnenergyvar'] = str(energyVar.get())
    appConfig['TRAINING TAB']['chkbtnpeakfreqvar'] = str(peakFreqVar.get())
    appConfig['TRAINING TAB']['chkbtnfreqdmentropyvar'] = str(freqDmEntropyVar.get())
    appConfig['TRAINING TAB']['chkbtnfr1cpnmagvar'] = str(fr1cpnMagVar.get())
    appConfig['TRAINING TAB']['chkbtnfr2cpnmagvar'] = str(fr2cpnMagVar.get())
    appConfig['TRAINING TAB']['chkbtnfr3cpnmagvar'] = str(fr3cpnMagVar.get())
    appConfig['TRAINING TAB']['chkbtnfr4cpnmagvar'] = str(fr4cpnMagVar.get())
    appConfig['TRAINING TAB']['chkbtnfr5cpnmagvar'] = str(fr5cpnMagVar.get())
    appConfig['TRAINING TAB']['txtwindowsizefrom'] = txtWindowSizeFrom_text.get()
    appConfig['TRAINING TAB']['txtwindowsizeto'] = txtWindowSizeTo_text.get()
    appConfig['TRAINING TAB']['txtwindowstep'] = txtWindowStep_text.get()
    appConfig['TRAINING TAB']['txtwindowstride'] = txtWindowStride_text.get()
    appConfig['TRAINING TAB']['txtwindowsimustride'] = txtWindowSimuStride_text.get()
    appConfig['TRAINING TAB']['cbodownsamplingfrom'] = str(cboDownSamplingFrom.get())
    appConfig['TRAINING TAB']['cbodownsamplingto'] = str(cboDownSamplingTo.get())
    appConfig['TRAINING TAB']['cbodownsamplingstep'] = str(cboDownSamplingStep.get())
    appConfig['TRAINING TAB']['btnchkrandomforest'] = str(RandomForestVar.get())
    appConfig['TRAINING TAB']['btnchkdecisiontree'] = str(DecisionTreeVar.get())
    appConfig['TRAINING TAB']['btnchksvm'] = str(SVMVar.get())
    appConfig['TRAINING TAB']['btnchknaivebayes'] = str(NaiveBayesVar.get())
    appConfig['TRAINING TAB']['cbokfold'] = str(cboKfold.get())
    appConfig['TRAINING TAB']['rdotrainingphraseonly'] = str(rdoTrainingPhraseOnly.get())
    appConfig['TRAINING TAB']['txtsimulationwindowsize'] = txtSimulationWindowSize_text.get()
    appConfig['TRAINING TAB']['txtsimuframedtpoints'] = txtSimuFrameDtPoints_text.get()
    appConfig['TRAINING TAB']['txtsimuframestride'] = txtSimuFrameStride_text.get()
    appConfig['TRAINING TAB']['txtsimuframedelay'] = txtSimuFrameDelay_text.get()
    appConfig['TRAINING TAB']['txtsimuframerepeat'] = txtSimuFrameRepeat_text.get()

    with open('app.ini', 'w') as configfile:  # save
        appConfig.write(configfile)


def reset_cbo_features_values_default():
    cboActivity1['values'] = 'None'
    cboActivity1.current(0)
    cboActivity2['values'] = 'None'
    cboActivity2.current(0)
    cboActivity3['values'] = 'None'
    cboActivity3.current(0)
    cboActivity4['values'] = 'None'
    cboActivity4.current(0)
    cboActivity5['values'] = 'None'
    cboActivity5.current(0)
    cboActivity6['values'] = 'None'
    cboActivity6.current(0)


def update_cbo_function_values():
    # Get the list of all labels from the dataset and push them into the combo boxes
    global cboValues
    for i in cboValues:
        cboActivity1['values'] += (i,)
        cboActivity2['values'] += (i,)
        cboActivity3['values'] += (i,)
        cboActivity4['values'] += (i,)
        cboActivity5['values'] += (i,)
        cboActivity6['values'] += (i,)


def btnIniSelect_clicked():
    global monitoring_mode
    file = filedialog.askopenfilename(filetypes=(("Ini files", "*.ini"), ("All files", "*.*")))
    if file:
        txtDBini_text.set(str(file))
        iniConfig.read(str(txtDBini_text.get()))
        if iniConfig.has_section('postgresql'):
            params_db = iniConfig.items('postgresql')
            txtHost_text.set(params_db[0][1])
            txtPort_text.set(params_db[1][1])
            txtDB_text.set(params_db[2][1])
            txtTrainTable_text.set(params_db[3][1])
            txtMonitoringTable_Text.set(params_db[4][1])
            if params_db[5][1] == '1':
                btnMonitoringOnOff.select()
                monitoring_mode = True
            else:
                btnMonitoringOnOff.deselect()
                monitoring_mode = False
                txtMonitoringTable.configure(state='disabled')
            txtResultTable_Text.set(params_db[6][1])
            txtUser_text.set(params_db[7][1])
            txtPwd_text.set(params_db[8][1])
        else:
            raise Exception('Section {0} not found in the {1} file'.format('postgresql', str(file)))


def connect_import_clicked():
    global train_valid_test_db_table
    global monitoring_db_table
    global monitoring_mode
    global params_db
    global cboValues
    lblNotification.configure(font=('Sans', '10', 'bold'), text='Connecting to the database...')
    if len(txtDBini_text.get()) > 0:
        # -begin connect
        conn = None
        try:
            # read connection parameters
            params_db = {}
            params_db['host'] = txtHost_text.get()
            params_db['port'] = txtPort_text.get()
            params_db['database'] = txtDB_text.get()
            params_db['user'] = txtUser_text.get()
            params_db['password'] = txtPwd_text.get()
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params_db)

            # create a cursor
            cur = conn.cursor()

            # Create result table if not exist ->
            sql_create_table = 'CREATE TABLE IF NOT EXISTS ' + 'public.' + txtResultTable_Text.get() + ' (model_title text COLLATE pg_catalog."default", model_init_name text COLLATE pg_catalog."default", model_binary_content bytea, features_json_content json, model_comments text COLLATE pg_catalog."default", train_table text COLLATE pg_catalog."default", monitor_table text COLLATE pg_catalog."default", no_of_predicted_classes integer, list_of_predicted_classes text COLLATE pg_catalog."default", original_sample_rate_in_hz integer, no_of_original_train_data_points integer, resampled_rate_in_hz integer, no_of_resampled_train_data_points integer, no_of_instances_for_each_class_in_resampled_train_table integer, algorithm text COLLATE pg_catalog."default", no_of_functions integer, list_of_functions text COLLATE pg_catalog."default", no_of_axes integer, list_of_axes text COLLATE pg_catalog."default", window_size integer, window_stride text COLLATE pg_catalog."default", k_fold integer, accuracy_train_valid real, precision_train_valid real, recall_train_valid real, specificity_train_valid real, f1_train_valid real, accuracy_test real, precision_test real, recall_test real, specificity_test real, f1_test real, monitoring_window_stride text COLLATE pg_catalog."default", accuracy_monitor real, precision_monitor real, recall_monitor real, specificity_monitor real, f1_monitor real, start_time timestamp without time zone, end_time timestamp without time zone, running_time_in_minutes text COLLATE pg_catalog."default") WITH (OIDS = FALSE) TABLESPACE pg_default;'
            cur.execute(sql_create_table)
            conn.commit()
            # Create result table if not exist -<

            cols = ['label', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'timestamp', 'cattle_id']

            print('Start fetching data at ' + str(datetime.datetime.now().strftime("%H:%M:%S")))

            sqlQuery = 'SELECT * FROM ' + txtTrainTable_text.get() + ' ORDER BY timestamp ASC '  # limit 60000'
            train_valid_test_db_table = pd.read_sql_query(sqlQuery, con=conn)[cols]
            if monitoring_mode:
                sqlQuery2 = 'SELECT * FROM ' + txtMonitoringTable_Text.get() + ' ORDER BY timestamp ASC'  # limit 10'
                monitoring_db_table = pd.read_sql_query(sqlQuery2, con=conn)[cols]
            print('Finish fetching data at ' + str(datetime.datetime.now().strftime("%H:%M:%S")))

            # print('pls uncomment this section')
            train_valid_test_db_table.loc[:, 'gyrMag'] = train_valid_test_db_table.apply(
                lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis=1)
            train_valid_test_db_table.loc[:, 'accMag'] = train_valid_test_db_table.apply(
                lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis=1)

            if monitoring_mode == True:
                monitoring_db_table.loc[:, 'gyrMag'] = monitoring_db_table.apply(
                    lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis=1)
                monitoring_db_table.loc[:, 'accMag'] = monitoring_db_table.apply(
                    lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis=1)

            print(
                'Finish calculating the magnitudes of Acc and Gyro at ' + datetime.datetime.now().strftime("%H:%M:%S"))

            winsound.Beep(1000, 300)
            cboValues = pd.Series(np.unique(np.array(train_valid_test_db_table['label']).tolist()))
            reset_cbo_features_values_default()
            update_cbo_function_values()

            tab_control.tab(1, state='normal')
            if monitoring_mode is False:
                txtWindowSimuStride.configure(state='disabled')
                btnStatics["state"] = DISABLED
                btnMonitorDist["state"] = DISABLED
                btnSimulatition["state"] = DISABLED
            else:
                txtWindowSimuStride.configure(state="normal")
                btnStatics["state"] = NORMAL
                btnMonitorDist["state"] = NORMAL
                btnSimulatition["state"] = NORMAL

            # Update for database.ini
            appConfig['IMPORT TAB']['dbini'] = txtDBini_text.get()
            with open('app.ini', 'w') as configfile:  # save
                appConfig.write(configfile)

            # update table name for db connection
            iniConfig['postgresql']['host'] = txtHost_text.get()
            iniConfig['postgresql']['port'] = txtPort_text.get()
            iniConfig['postgresql']['database'] = txtDB_text.get()
            iniConfig['postgresql']['traintable'] = txtTrainTable_text.get()
            iniConfig['postgresql']['monitortable'] = txtMonitoringTable_Text.get()
            if monitoring_mode is True:
                iniConfig['postgresql']['selectmonitoring'] = '1'
            else:
                iniConfig['postgresql']['selectmonitoring'] = '0'
            iniConfig['postgresql']['resulttable'] = txtResultTable_Text.get()
            iniConfig['postgresql']['user'] = txtUser_text.get()
            iniConfig['postgresql']['password'] = txtPwd_text.get()
            with open(txtDBini_text.get(), 'w') as configfile:  # save
                iniConfig.write(configfile)

            lblNotification.configure(font=('Sans', '10', 'bold'), text="Tables are fetched successully")


        except (Exception, psycopg2.DatabaseError) as error:
            messagebox.showinfo("Alert", error)
        finally:
            if conn is not None:
                # messagebox.showinfo("Alert", "Imported successully!")
                # close the communication with the PostgreSQL
                conn.close()
                # print('Database connection closed.')
        # --end connect
    else:
        messagebox.showinfo("Alert", "Please select a valid ini file")


# Building tabs for the main form/window
tab_control = ttk.Notebook(window)
tabImport = ttk.Frame(tab_control)
tabTraining = ttk.Frame(tab_control)

tab_control.add(tabImport, text='Database Connect')
tab_control.add(tabTraining, text='Model Building')

# Building Controls for Import Tab =>

# For DB ini file section
txtDBini_text = StringVar()
txtHost_text = StringVar()
txtPort_text = StringVar()
txtDB_text = StringVar()
txtUser_text = StringVar()
txtPwd_text = StringVar()
txtTrainTable_text = StringVar()
txtMonitoringTable_Text = StringVar()
txtResultTable_Text = StringVar()

# y coordinates value
y_co = int(15)

lblIni = Label(tabImport, text='INI Filepath')
lblIni.place(x=20, y=y_co)
txtDBini = Entry(tabImport, width=60, textvariable=txtDBini_text)
txtDBini.place(x=200, y=y_co)

btnIni = Button(tabImport, text="Select another INI file", command=btnIniSelect_clicked)
btnIni.place(x=590, y=y_co - 4)

# Begin DB login form section
lblHost = Label(tabImport, text='Host')
lblHost.place(x=20, y=y_co + 28)
txtHost = Entry(tabImport, width=20, textvariable=txtHost_text)
txtHost.place(x=200, y=y_co + 28)

lblPort = Label(tabImport, text='Port')
lblPort.place(x=20, y=y_co + 55)
txtPort = Entry(tabImport, width=20, textvariable=txtPort_text)
txtPort.place(x=200, y=y_co + 55)

lblDB = Label(tabImport, text='Database')
lblDB.place(x=20, y=y_co + 80)
txtDB = Entry(tabImport, width=20, textvariable=txtDB_text)
txtDB.place(x=200, y=y_co + 80)

lblUser = Label(tabImport, text='User')
lblUser.place(x=20, y=y_co + 105)
txtUser = Entry(tabImport, width=20, textvariable=txtUser_text)
txtUser.place(x=200, y=y_co + 105)

lblPwd = Label(tabImport, text='Password')
lblPwd.place(x=20, y=y_co + 130)
txtPwd = Entry(tabImport, width=20, show="*", textvariable=txtPwd_text)
txtPwd.place(x=200, y=y_co + 130)

lblTable = Label(tabImport, text='Train_Valid_Test table')
lblTable.place(x=20, y=y_co + 155)
txtTable = Entry(tabImport, width=60, textvariable=txtTrainTable_text)
txtTable.place(x=200, y=y_co + 155)

txtMonitoringTable = Entry(tabImport, width=60, textvariable=txtMonitoringTable_Text)
txtMonitoringTable.place(x=200, y=y_co + 180)

selectMonitoringActivityOnOff = IntVar()


def monitoringSelectDeselect():
    global monitoring_mode
    if selectMonitoringActivityOnOff.get() == 0:
        txtMonitoringTable.configure(state='disabled')
        txtWindowSimuStride.configure(state='disabled')
        btnStatics["state"] = DISABLED
        btnMonitorDist["state"] = DISABLED
        btnSimulatition["state"] = DISABLED
        monitoring_mode = False
    elif selectMonitoringActivityOnOff.get() == 1:
        txtMonitoringTable.configure(state="normal")
        txtWindowSimuStride.configure(state="normal")
        btnStatics["state"] = NORMAL
        btnMonitorDist["state"] = NORMAL
        btnSimulatition["state"] = NORMAL
        monitoring_mode = True


btnMonitoringOnOff = Checkbutton(tabImport, command=monitoringSelectDeselect, text="Statistics_Monitoring table",
                                 variable=selectMonitoringActivityOnOff, onvalue=1, offvalue=0, height=1, width=20)
btnMonitoringOnOff.place(x=22, y=y_co + 178)

lblResultTable = Label(tabImport, text='Table to store result')
lblResultTable.place(x=20, y=y_co + 205)
txtResultTable = Entry(tabImport, width=60, textvariable=txtResultTable_Text)
txtResultTable.place(x=200, y=y_co + 205)

lblNotification = Label(tabImport, text='')
lblNotification.place(x=200, y=y_co + 230)
# End DB log in form section

btnImport = Button(tabImport, text="Connect \ Import Data", bg='gold', command=connect_import_clicked)
btnImport.place(x=590, y=y_co + 153)

# ---End Import Tab <=

# Begin building controls for Model Building Tab =>

# Set coordinates for control
x_coordinate = 7
y_coordinate = 7
selectAllActivityOnOff = IntVar()


def allLabelsSelectDeselect():
    if selectAllActivityOnOff.get() == 0:
        cboActivity1["state"] = NORMAL
        cboActivity2["state"] = NORMAL
        cboActivity3["state"] = NORMAL
        cboActivity4["state"] = NORMAL
        cboActivity5["state"] = NORMAL
        cboActivity6["state"] = NORMAL
    elif selectAllActivityOnOff.get() == 1:
        cboActivity1["state"] = DISABLED
        cboActivity2["state"] = DISABLED
        cboActivity3["state"] = DISABLED
        cboActivity4["state"] = DISABLED
        cboActivity5["state"] = DISABLED
        cboActivity6["state"] = DISABLED


cboAct_x_inc = 90

lblLabelsList = Label(tabTraining, text='Select labels for classifying')
lblLabelsList.configure(font=('Sans', '10', 'bold'))
lblLabelsList.place(x=x_coordinate, y=y_coordinate + 5)

btnAllLabelsOnOff = Checkbutton(tabTraining, command=allLabelsSelectDeselect, text="All labels",
                                variable=selectAllActivityOnOff, onvalue=1, offvalue=0, height=2, width=10)
btnAllLabelsOnOff.configure(font=('Sans', '9'))
btnAllLabelsOnOff.place(x=x_coordinate + 5, y=y_coordinate + 25)

cboActivity1 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity1.place(x=x_coordinate + 102, y=y_coordinate + 33)
cboActivity1.current(0)
cboActivity2 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity2.place(x=x_coordinate + 102 + 1 * cboAct_x_inc, y=y_coordinate + 33)
cboActivity2.current(0)
cboActivity3 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity3.place(x=x_coordinate + 102 + 2 * cboAct_x_inc, y=y_coordinate + 33)
cboActivity3.current(0)
cboActivity4 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity4.place(x=x_coordinate + 102 + 3 * cboAct_x_inc, y=y_coordinate + 33)
cboActivity4.current(0)
cboActivity5 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity5.place(x=x_coordinate + 102 + 4 * cboAct_x_inc, y=y_coordinate + 33)
cboActivity5.current(0)
cboActivity6 = ttk.Combobox(tabTraining, width="10", values='None')
cboActivity6.place(x=x_coordinate + 102 + 5 * cboAct_x_inc, y=y_coordinate + 33)
cboActivity6.current(0)

lblLine01 = Label(tabTraining,
                  text='------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
lblLine01.place(x=x_coordinate + 2, y=y_coordinate + 56)
lblLine01.configure(font=('Sans', '5', 'bold'))

lblResample = Label(tabTraining, text='Resampling data')
lblResample.configure(font=('Sans', '10', 'bold'))
lblResample.place(x=x_coordinate, y=y_coordinate + 67)

radKeepOriginalSample = Radiobutton(tabTraining, text='Keep original data', variable=rdoTrainingPhraseOnly, value=0)
radKeepOriginalSample.place(x=x_coordinate + 15, y=y_coordinate + 90)
radKeepOriginalSample.configure(font=('Sans', '11'))

radReSample = Radiobutton(tabTraining, text='Resample data with rates from', variable=rdoTrainingPhraseOnly, value=1)
radReSample.place(x=x_coordinate + 170, y=y_coordinate + 90)
radReSample.configure(font=('Sans', '11'))

cboDownSamplingFrom = ttk.Combobox(tabTraining, width="2", values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
cboDownSamplingFrom.place(x=x_coordinate + 402, y=y_coordinate + 92)
cboDownSamplingFrom.current(0)
lblDownSamplingFromHz = Label(tabTraining, text='Hz')
lblDownSamplingFromHz.place(x=x_coordinate + 437, y=y_coordinate + 92)

lblDownSamplingTo = Label(tabTraining, text='to')
lblDownSamplingTo.configure(font=('Sans', '10', 'bold'))
lblDownSamplingTo.place(x=x_coordinate + 456, y=y_coordinate + 91)

cboDownSamplingTo = ttk.Combobox(tabTraining, width="2", values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
cboDownSamplingTo.place(x=x_coordinate + 476, y=y_coordinate + 92)
cboDownSamplingTo.current(0)

lblDownSamplingToHz = Label(tabTraining, text='Hz')
lblDownSamplingToHz.place(x=x_coordinate + 511, y=y_coordinate + 92)

lblDownSamplingStep = Label(tabTraining, text='step')
lblDownSamplingStep.place(x=x_coordinate + 530, y=y_coordinate + 91)
lblDownSamplingStep.configure(font=('Sans', '10', 'bold'))

cboDownSamplingStep = ttk.Combobox(tabTraining, width="2", values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
cboDownSamplingStep.place(x=x_coordinate + 564, y=y_coordinate + 92)
cboDownSamplingStep.current(0)

lblDownSamplingStepHz = Label(tabTraining, text='Hz')
lblDownSamplingStepHz.place(x=x_coordinate + 599, y=y_coordinate + 92)

btnFeature_x = x_coordinate + 85
btnFeature_y = y_coordinate + 133
btnFeature_x_inc = 60

lblWindowSize = Label(tabTraining, text='Window setting')
lblWindowSize.configure(font=('Sans', '10', 'bold'))
lblWindowSize.place(x=x_coordinate, y=y_coordinate + 117)

lblWindowBasementSize = Label(tabTraining, text='Size from')
lblWindowBasementSize.configure(font=('Sans', '11'))
lblWindowBasementSize.place(x=x_coordinate, y=y_coordinate + 141)

txtWindowSizeFrom_text = StringVar()
txtWindowSizeFrom = Entry(tabTraining, width=6, textvariable=txtWindowSizeFrom_text)
txtWindowSizeFrom.place(x=x_coordinate + 70, y=y_coordinate + 144)

lblWindowSizeMs = Label(tabTraining, text='ms')
lblWindowSizeMs.place(x=x_coordinate + 107, y=y_coordinate + 141)

lblWindowIn = Label(tabTraining, text='to')
lblWindowIn.configure(font=('Sans', '11'))
lblWindowIn.place(x=x_coordinate + 132, y=y_coordinate + 141)

txtWindowSizeTo_text = StringVar()
txtWindowSizeTo = Entry(tabTraining, width=6, textvariable=txtWindowSizeTo_text)
txtWindowSizeTo.place(x=x_coordinate + 151, y=y_coordinate + 144)

lblWindowTo = Label(tabTraining, text='ms')
lblWindowTo.place(x=x_coordinate + 191, y=y_coordinate + 141)

lblWindowIncrement = Label(tabTraining, text='step')
lblWindowIncrement.place(x=x_coordinate + 216, y=y_coordinate + 141)
lblWindowIncrement.configure(font=('Sans', '11'))

txtWindowStep_text = StringVar()
txtWindowStep = Entry(tabTraining, width=5, textvariable=txtWindowStep_text)
txtWindowStep.place(x=x_coordinate + 249, y=y_coordinate + 144)

lblWindowToStep = Label(tabTraining, text='ms')
lblWindowToStep.place(x=x_coordinate + 281, y=y_coordinate + 141)

lblWindowStride = Label(tabTraining, text='Train (Test) stride')
lblWindowStride.configure(font=('Sans', '11'))
lblWindowStride.place(x=x_coordinate, y=y_coordinate + 171)

txtWindowStride_text = StringVar()
txtWindowStride = Entry(tabTraining, width=6, textvariable=txtWindowStride_text)
txtWindowStride.place(x=x_coordinate + 123, y=y_coordinate + 173)

lblWindowStrideMs = Label(tabTraining, text='%')
lblWindowStrideMs.place(x=x_coordinate + 163, y=y_coordinate + 173)

lblWindowTesingStride = Label(tabTraining, text='Monitoring stride')
lblWindowTesingStride.configure(font=('Sans', '11'))
lblWindowTesingStride.place(x=x_coordinate + 215, y=y_coordinate + 171)

txtWindowSimuStride_text = StringVar()
txtWindowSimuStride = Entry(tabTraining, width=6, textvariable=txtWindowSimuStride_text)
txtWindowSimuStride.place(x=x_coordinate + 327, y=y_coordinate + 173)

lblWindowTestingStrideMs = Label(tabTraining, text='%')
lblWindowTestingStrideMs.place(x=x_coordinate + 365, y=y_coordinate + 173)

lblLine02 = Label(tabTraining,
                  text='---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
lblLine02.place(x=x_coordinate + 2, y=btnFeature_y + 60)
lblLine02.configure(font=('Sans', '5', 'bold'))

lblFeaturesList = Label(tabTraining, text='Select functions')
lblFeaturesList.configure(font=('Sans', '10', 'bold'))
lblFeaturesList.place(x=x_coordinate, y=btnFeature_y + 70)

MinVar = IntVar()
chkBtnMin = Checkbutton(tabTraining, text="Min", variable=MinVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnMin.place(x=btnFeature_x + 1 * btnFeature_x_inc - 2, y=btnFeature_y + 83)

MaxVar = IntVar()
chkBtnMax = Checkbutton(tabTraining, text="Max", variable=MaxVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnMax.place(x=btnFeature_x + 2 * btnFeature_x_inc, y=btnFeature_y + 83)

MeanVar = IntVar()
chkBtnMean = Checkbutton(tabTraining, text="Mean", variable=MeanVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnMean.place(x=btnFeature_x + 3 * btnFeature_x_inc, y=btnFeature_y + 83)

MedianVar = IntVar()
chkBtnMedian = Checkbutton(tabTraining, text="Median", variable=MedianVar, onvalue=1, offvalue=0, height=2, width=5)
chkBtnMedian.place(x=btnFeature_x + 4 * btnFeature_x_inc, y=btnFeature_y + 83)

StdVar = IntVar()
chkBtnStd = Checkbutton(tabTraining, text="Stdev", variable=StdVar, onvalue=1, offvalue=0, height=2, width=4)
chkBtnStd.place(x=btnFeature_x + 5 * btnFeature_x_inc + 13, y=btnFeature_y + 83)

IQRVar = IntVar()
btnChkIQR = Checkbutton(tabTraining, text="IntQtlRange", variable=IQRVar, onvalue=1, offvalue=0, height=2, width=10)
btnChkIQR.place(x=btnFeature_x + 6 * btnFeature_x_inc + 17, y=btnFeature_y + 83)

rootMSVar = IntVar()
btnRMS = Checkbutton(tabTraining, text="RootMS", variable=rootMSVar, onvalue=1, offvalue=0, height=2, width=7)
btnRMS.place(x=btnFeature_x + 7 * btnFeature_x_inc + 61, y=btnFeature_y + 83)

meanCRVar = IntVar()
btnMCR = Checkbutton(tabTraining, text="MeanCR", variable=meanCRVar, onvalue=1, offvalue=0, height=2, width=10)
btnMCR.place(x=btnFeature_x + 8 * btnFeature_x_inc + 78, y=btnFeature_y + 83)

kurtosisVar = IntVar()
btnkurt = Checkbutton(tabTraining, text="Kurtosis", variable=kurtosisVar, onvalue=1, offvalue=0, height=2, width=10)
btnkurt.place(x=btnFeature_x + 9 * btnFeature_x_inc + 108, y=btnFeature_y + 83)

skewnessVar = IntVar()
btnskew = Checkbutton(tabTraining, text="Skewness", variable=skewnessVar, onvalue=1, offvalue=0, height=2, width=15)
btnskew.place(x=btnFeature_x + 10 * btnFeature_x_inc + 127, y=btnFeature_y + 83)

energyVar = IntVar()
btnEnergy = Checkbutton(tabTraining, text="Spectral Energy", variable=energyVar, onvalue=1, offvalue=0, height=2,
                        width=11)
btnEnergy.place(x=btnFeature_x + 1 * btnFeature_x_inc, y=btnFeature_y + 113)

peakFreqVar = IntVar()
btnPeakFreq = Checkbutton(tabTraining, text="PeakFreq", variable=peakFreqVar, onvalue=1, offvalue=0, height=2,
                          width=10)
btnPeakFreq.place(x=btnFeature_x + 1 * btnFeature_x_inc + 106, y=btnFeature_y + 113)

freqDmEntropyVar = IntVar()
btnFredDmEntropy = Checkbutton(tabTraining, text="FreqEntropy", variable=freqDmEntropyVar, onvalue=1, offvalue=0,
                               height=2, width=12)
btnFredDmEntropy.place(x=btnFeature_x + 1 * btnFeature_x_inc + 195, y=btnFeature_y + 113)

fr1cpnMagVar = IntVar()
btn1cpnMag = Checkbutton(tabTraining, text="1stCpnMag", variable=fr1cpnMagVar, onvalue=1, offvalue=0, height=2,
                         width=10)
btn1cpnMag.place(x=btnFeature_x + 1 * btnFeature_x_inc + 300, y=btnFeature_y + 113)

fr2cpnMagVar = IntVar()
btn2cpnMag = Checkbutton(tabTraining, text="2ndCpnMag", variable=fr2cpnMagVar, onvalue=1, offvalue=0, height=2,
                         width=10)
btn2cpnMag.place(x=btnFeature_x + 1 * btnFeature_x_inc + 395, y=btnFeature_y + 113)

fr3cpnMagVar = IntVar()
btn3cpnMag = Checkbutton(tabTraining, text="3rdCpnMag", variable=fr3cpnMagVar, onvalue=1, offvalue=0, height=2,
                         width=10)
btn3cpnMag.place(x=btnFeature_x + 1 * btnFeature_x_inc + 490, y=btnFeature_y + 113)

fr4cpnMagVar = IntVar()
btn4cpnMag = Checkbutton(tabTraining, text="4thCpnMag", variable=fr4cpnMagVar, onvalue=1, offvalue=0, height=2,
                         width=10)
btn4cpnMag.place(x=btnFeature_x + 1 * btnFeature_x_inc + 585, y=btnFeature_y + 113)

fr5cpnMagVar = IntVar()
btn5cpnMag = Checkbutton(tabTraining, text="5thCpnMag", variable=fr5cpnMagVar, onvalue=1, offvalue=0, height=2,
                         width=10)
btn5cpnMag.place(x=btnFeature_x + 1 * btnFeature_x_inc + 680, y=btnFeature_y + 113)

selectAllFeaturesOnOff = IntVar()


def featureSelectDeselect():
    if selectAllFeaturesOnOff.get() == 0:
        chkBtnMin.deselect()
        chkBtnMax.deselect()
        chkBtnMean.deselect()
        chkBtnMedian.deselect()
        chkBtnStd.deselect()
        btnChkIQR.deselect()
        btnRMS.deselect()
        btnMCR.deselect()
        btnkurt.deselect()
        btnskew.deselect()
        btnEnergy.deselect()
        btnPeakFreq.deselect()
        btnFredDmEntropy.deselect()
        btn1cpnMag.deselect()
        btn2cpnMag.deselect()
        btn3cpnMag.deselect()
        btn4cpnMag.deselect()
        btn5cpnMag.deselect()
    elif selectAllFeaturesOnOff.get() == 1:
        chkBtnMin.select()
        chkBtnMax.select()
        chkBtnMean.select()
        chkBtnMedian.select()
        chkBtnStd.select()
        btnChkIQR.select()
        btnRMS.select()
        btnMCR.select()
        btnkurt.select()
        btnskew.select()
        btnEnergy.select()
        btnPeakFreq.select()
        btnFredDmEntropy.select()
        btn1cpnMag.select()
        btn2cpnMag.select()
        btn3cpnMag.select()
        btn4cpnMag.select()
        btn5cpnMag.select()


btnAllFeaturesOnOff = Checkbutton(tabTraining, command=featureSelectDeselect, text="(De)Select all",
                                  variable=selectAllFeaturesOnOff, onvalue=1, offvalue=0, height=1, width=15)
btnAllFeaturesOnOff.configure(font=('Sans', '10'))
btnAllFeaturesOnOff.place(x=3, y=btnFeature_y + 93)

btnFeature_y = btnFeature_y + 40
lblAxesList = Label(tabTraining, text='Select axes to be applied')
lblAxesList.configure(font=('Sans', '10', 'bold'))
lblAxesList.place(x=x_coordinate, y=btnFeature_y + 103)

btnFeature_y_inc = 119
btnFeature_x = btnFeature_x - 5
gxVar = IntVar()
chkBtnGx = Checkbutton(tabTraining, text="Gx", variable=gxVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnGx.place(x=btnFeature_x + 1 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
gyVar = IntVar()
chkBtnGy = Checkbutton(tabTraining, text="Gy", variable=gyVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnGy.place(x=btnFeature_x + 2 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
gzVar = IntVar()
chkBtnGz = Checkbutton(tabTraining, text="Gz", variable=gzVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnGz.place(x=btnFeature_x + 3 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
axVar = IntVar()
chkBtnAx = Checkbutton(tabTraining, text="Ax", variable=axVar, onvalue=1, offvalue=0, height=2, width=3)
chkBtnAx.place(x=btnFeature_x + 4 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
ayVar = IntVar()
btnChkAy = Checkbutton(tabTraining, text="Ay", variable=ayVar, onvalue=1, offvalue=0, height=2, width=3)
btnChkAy.place(x=btnFeature_x + 5 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
azVar = IntVar()
btnChkAz = Checkbutton(tabTraining, text="Az", variable=azVar, onvalue=1, offvalue=0, height=2, width=3)
btnChkAz.place(x=btnFeature_x + 6 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)

g_mag_Var = IntVar()
btnChkmag = Checkbutton(tabTraining, text="gyro magnitude", variable=g_mag_Var, onvalue=1, offvalue=0, height=2,
                        width=15)
btnChkmag.place(x=btnFeature_x + 7 * btnFeature_x_inc, y=btnFeature_y + btnFeature_y_inc)
a_mag_Var = IntVar()
btnChkg_amag = Checkbutton(tabTraining, text="acc magnitude", variable=a_mag_Var, onvalue=1, offvalue=0, height=2,
                           width=15)
btnChkg_amag.place(x=btnFeature_x + 8 * btnFeature_x_inc + 70, y=btnFeature_y + btnFeature_y_inc)

lblLine03 = Label(tabTraining,
                  text='-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
lblLine03.place(x=x_coordinate + 2, y=btnFeature_y + 150)
lblLine03.configure(font=('Sans', '5', 'bold'))

lblClassifier = Label(tabTraining, text='Classifiers')
lblClassifier.configure(font=('Sans', '10', 'bold'))
lblClassifier.place(x=x_coordinate, y=btnFeature_y + 170)

RandomForestVar = IntVar()
btnChkRandomForest = Checkbutton(tabTraining, text="Random forest", variable=RandomForestVar, onvalue=1, offvalue=0,
                                 height=2, width=15)
btnChkRandomForest.place(x=x_coordinate + 73, y=btnFeature_y + 162)

DecisionTreeVar = IntVar()
btnChkDecisionTree = Checkbutton(tabTraining, text="Decision Tree", variable=DecisionTreeVar, onvalue=1, offvalue=0,
                                 height=2, width=15)
btnChkDecisionTree.place(x=x_coordinate + 190, y=btnFeature_y + 162)

SVMVar = IntVar()
btnChkSVM = Checkbutton(tabTraining, text="SVM", variable=SVMVar, onvalue=1, offvalue=0, height=2, width=15)
btnChkSVM.place(x=x_coordinate + 295, y=btnFeature_y + 162)

NaiveBayesVar = IntVar()
btnChkNaiveBayes = Checkbutton(tabTraining, text="Naive Bayes", variable=NaiveBayesVar, onvalue=1, offvalue=0, height=2,
                               width=15)
btnChkNaiveBayes.place(x=x_coordinate + 420, y=btnFeature_y + 162)

lblKfold = Label(tabTraining, text='Using')
lblKfold.configure(font=('Sans', '10', 'bold'))
lblKfold.place(x=x_coordinate + 575, y=btnFeature_y + 170)
cboKfold = ttk.Combobox(tabTraining, width="2", values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
cboKfold.place(x=x_coordinate + 619, y=btnFeature_y + 171)
cboKfold.current(2)
lblKfoldValidation = Label(tabTraining, text=' - fold validation')
lblKfoldValidation.configure(font=('Sans', '10'))
lblKfoldValidation.place(x=x_coordinate + 656, y=btnFeature_y + 171)

lblLine04 = Label(tabTraining,
                  text='-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
lblLine04.place(x=x_coordinate + 2, y=btnFeature_y + 200)
lblLine04.configure(font=('Sans', '5', 'bold'))

# For live plotting =>
lblSimulationSetting = Label(tabTraining, text='Monitoring setting ')
lblSimulationSetting.configure(font=('Sans', '10', 'bold'))
lblSimulationSetting.place(x=x_coordinate, y=btnFeature_y + 215)

lblSimulationSampleRate = Label(tabTraining, text='Sampling rate')
lblSimulationSampleRate.configure(font=('Sans', '11'))
lblSimulationSampleRate.place(x=x_coordinate, y=btnFeature_y + 243)

cboSimulationSampleRate = ttk.Combobox(tabTraining, width="2",
                                       values=('10', '9', '8', '7', '6', '5', '4', '3', '2', '1'))
cboSimulationSampleRate.place(x=x_coordinate + 110, y=btnFeature_y + 244)
cboSimulationSampleRate.current(0)

lblSimulationSampleRateHz = Label(tabTraining, text='Hz')
lblSimulationSampleRateHz.place(x=x_coordinate + 145, y=btnFeature_y + 245)

lblSimulationWindowSize = Label(tabTraining, text='Window size')
lblSimulationWindowSize.configure(font=('Sans', '11'))
lblSimulationWindowSize.place(x=x_coordinate + 180, y=btnFeature_y + 242)

txtSimulationWindowSize_text = StringVar()
txtSimulationWindowSize = Entry(tabTraining, width=6, textvariable=txtSimulationWindowSize_text)
txtSimulationWindowSize.place(x=x_coordinate + 276, y=btnFeature_y + 245)

lblSimulationWindowSizeMs = Label(tabTraining, text='ms')
lblSimulationWindowSizeMs.place(x=x_coordinate + 317, y=btnFeature_y + 243)

lblSimuAlgorithms = Label(tabTraining, text='Algorithm')
lblSimuAlgorithms.configure(font=('Sans', '11'))
lblSimuAlgorithms.place(x=x_coordinate + 350, y=btnFeature_y + 242)

cboSimuAlgorithm = ttk.Combobox(tabTraining, width="15",
                                values=('Random_Forest', 'Decision_Tree', 'SVM', 'Naive_Bayes'))
cboSimuAlgorithm.place(x=x_coordinate + 420, y=btnFeature_y + 243)
cboSimuAlgorithm.current(0)

lblLine05 = Label(tabTraining,
                  text='-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
lblLine05.place(x=x_coordinate + 2, y=btnFeature_y + 281)
lblLine05.configure(font=('Sans', '5', 'bold'))

lblSimuPlotFrameTimeStart = Label(tabTraining, text='Plot from ')
lblSimuPlotFrameTimeStart.configure(font=('Sans', '11'))
lblSimuPlotFrameTimeStart.place(x=x_coordinate, y=btnFeature_y + 300)

radSimuFrameStartTime = Radiobutton(tabTraining, text='Beginning', variable=rdoSimuStartTime, value=0)
radSimuFrameStartTime.place(x=x_coordinate + 105, y=btnFeature_y + 300)
radSimuFrameStartTime.configure(font=('Sans', '10',))

radSimuFrameFromTime = Radiobutton(tabTraining, text='Epoch timestamp', variable=rdoSimuStartTime, value=1)
radSimuFrameFromTime.place(x=x_coordinate + 200, y=btnFeature_y + 300)
radSimuFrameFromTime.configure(font=('Sans', '10'))

txtSimuFrameFromTime_text = StringVar()
txtSimuFrameFromTime = Entry(tabTraining, width=20, textvariable=txtSimuFrameFromTime_text)
txtSimuFrameFromTime.place(x=x_coordinate + 335, y=btnFeature_y + 305)

lblSimuFrameDtPoints = Label(tabTraining, text='Data pts in frame')
lblSimuFrameDtPoints.configure(font=('Sans', '10'))
lblSimuFrameDtPoints.place(x=x_coordinate, y=btnFeature_y + 334)

txtSimuFrameDtPoints_text = StringVar()
txtSimuFrameDtPoints = Entry(tabTraining, width=5, textvariable=txtSimuFrameDtPoints_text)
txtSimuFrameDtPoints.place(x=x_coordinate + 110, y=btnFeature_y + 336)

lblSimuFrameStride = Label(tabTraining, text='Frame stride')
lblSimuFrameStride.configure(font=('Sans', '10'))
lblSimuFrameStride.place(x=x_coordinate + 153, y=btnFeature_y + 334)

txtSimuFrameStride_text = StringVar()
txtSimuFrameStride = Entry(tabTraining, width=6, textvariable=txtSimuFrameStride_text)
txtSimuFrameStride.place(x=x_coordinate + 235, y=btnFeature_y + 336)

lblSimuFrameDelay = Label(tabTraining, text='Frame delay')
lblSimuFrameDelay.configure(font=('Sans', '10'))
lblSimuFrameDelay.place(x=x_coordinate + 290, y=btnFeature_y + 334)

txtSimuFrameDelay_text = StringVar()
txtSimuFrameDelay = Entry(tabTraining, width=6, textvariable=txtSimuFrameDelay_text)
txtSimuFrameDelay.place(x=x_coordinate + 370, y=btnFeature_y + 336)

lblSimuFrameDelayms = Label(tabTraining, text='ms')
lblSimuFrameDelayms.place(x=x_coordinate + 415, y=btnFeature_y + 336)

lblSimuFrameRepeat = Label(tabTraining, text='Repeat times')
lblSimuFrameRepeat.configure(font=('Sans', '10'))
lblSimuFrameRepeat.place(x=x_coordinate + 450, y=btnFeature_y + 334)

txtSimuFrameRepeat_text = StringVar()
txtSimuFrameRepeat = Entry(tabTraining, width=6, textvariable=txtSimuFrameRepeat_text)
txtSimuFrameRepeat.place(x=x_coordinate + 533, y=btnFeature_y + 336)
# For live plotting <=

# End building control for Training tab

tab_control.pack(expand=1, fill='both')

# Reading options in ini file
appConfig.read('app.ini')
txtDBini_text.set(appConfig['IMPORT TAB']['dbini'])

if len(appConfig['IMPORT TAB']['dbini']) > 0:
    # read config file
    iniConfig.read(appConfig['IMPORT TAB']['dbini'])

    # Get control default values from ini for [IMPORT TAB] - database log in section
    if iniConfig.has_section('postgresql'):
        params_db = iniConfig.items('postgresql')
        txtHost_text.set(params_db[0][1])
        txtPort_text.set(params_db[1][1])
        txtDB_text.set(params_db[2][1])
        txtTrainTable_text.set(params_db[3][1])
        txtMonitoringTable_Text.set(params_db[4][1])
        if params_db[5][1] == '1':
            btnMonitoringOnOff.select()
            monitoring_mode = True
        else:
            btnMonitoringOnOff.deselect()
            txtMonitoringTable.configure(state='disabled')
            monitoring_mode = False
        txtResultTable_Text.set(params_db[6][1])
        txtUser_text.set(params_db[7][1])
        txtPwd_text.set(params_db[8][1])
    else:
        # raise Exception('Section {0} not found in the {1} file'.format('TRAINING TAB', 'app.ini'))
        messagebox.showinfo("Alert", "Could not find a proper credentials ini file for db connection")

# Geting general setting from ini for the apps
if appConfig.has_section('GENERAL SETTINGS'):
    params_general_setting = appConfig.items('GENERAL SETTINGS')
    if params_general_setting[0][1] == '1':
        binary_mode = True
    else:
        binary_mode = False

    main_label = str(params_general_setting[1][1])
    sub_labels_set = pd.Series(params_general_setting[2][1].split("_"))
    no_of_sub_labels = sub_labels_set.size

    if params_general_setting[3][1] == '1':
        csv_saving = True
    else:
        csv_saving = False

    test_proportion = float(params_general_setting[4][1])
    # print(binary_mode)
    # print(main_label)
    # print(type(main_label))
    # print(sub_labels_set)
    # print(type(sub_labels_set))
    # print(no_of_sub_labels)
    # print(type(no_of_sub_labels))
    # print(csv_saving)
    # print(test_proportion)
    # print(type(test_proportion))
else:
    raise Exception('Section {0} not found in the {1} file'.format('GENERAL SETTINGS', 'app.ini'))

# Get control default values from ini for TRAINING TAB section
if appConfig.has_section('TRAINING TAB'):
    params_training = appConfig.items('TRAINING TAB')
    # for param in params_training
    if params_training[0][1] == '1':
        btnAllLabelsOnOff.select()
    else:
        btnAllLabelsOnOff.deselect()
    allLabelsSelectDeselect()

    if params_training[1][1] == '1':
        btnAllFeaturesOnOff.select()
    else:
        btnAllFeaturesOnOff.deselect()
    featureSelectDeselect()

    if params_training[2][1] == '1':
        chkBtnMin.select()
    else:
        chkBtnMin.deselect()
    if params_training[3][1] == '1':
        chkBtnMax.select()
    else:
        chkBtnMax.deselect()
    if params_training[4][1] == '1':
        chkBtnMean.select()
    else:
        chkBtnMean.deselect()
    if params_training[5][1] == '1':
        chkBtnMedian.select()
    else:
        chkBtnMedian.deselect()
    if params_training[6][1] == '1':
        chkBtnStd.select()
    else:
        chkBtnStd.deselect()
    if params_training[7][1] == '1':
        btnChkIQR.select()
    else:
        btnChkIQR.deselect()
    if params_training[8][1] == '1':
        btnRMS.select()
    else:
        btnRMS.deselect()
    if params_training[9][1] == '1':
        btnMCR.select()
    else:
        btnMCR.deselect()
    if params_training[10][1] == '1':
        btnkurt.select()
    else:
        btnkurt.deselect()
    if params_training[11][1] == '1':
        btnskew.select()
    else:
        btnskew.deselect()
    if params_training[12][1] == '1':
        btnEnergy.select()
    else:
        btnEnergy.deselect()
    if params_training[13][1] == '1':
        btnPeakFreq.select()
    else:
        btnPeakFreq.deselect()
    if params_training[14][1] == '1':
        btnFredDmEntropy.select()
    else:
        btnFredDmEntropy.deselect()
    if params_training[15][1] == '1':
        btn1cpnMag.select()
    else:
        btn1cpnMag.deselect()
    if params_training[16][1] == '1':
        btn2cpnMag.select()
    else:
        btn2cpnMag.deselect()
    if params_training[17][1] == '1':
        btn3cpnMag.select()
    else:
        btn3cpnMag.deselect()
    if params_training[18][1] == '1':
        btn4cpnMag.select()
    else:
        btn4cpnMag.deselect()
    if params_training[19][1] == '1':
        btn5cpnMag.select()
    else:
        btn5cpnMag.deselect()
    txtWindowSizeFrom_text.set(params_training[20][1])
    txtWindowSizeTo_text.set((params_training[21][1]))
    txtWindowStep_text.set(params_training[22][1])
    txtWindowStride_text.set(params_training[23][1])
    txtWindowSimuStride_text.set(params_training[24][1])
    if int(params_training[25][1]) < 11:
        cboDownSamplingFrom.current(int(params_training[25][1]) - 1)
    else:
        cboDownSamplingFrom.current(0)
    if int(params_training[26][1]) < 11:
        cboDownSamplingTo.current(int(params_training[26][1]) - 1)
    else:
        cboDownSamplingTo.current(0)
    if int(params_training[27][1]) < 11:
        cboDownSamplingStep.current(int(params_training[27][1]) - 1)
    else:
        cboDownSamplingStep.current(0)
    if params_training[28][1] == '1':
        btnChkRandomForest.select()
    else:
        btnChkRandomForest.deselect()
    if params_training[29][1] == '1':
        btnChkDecisionTree.select()
    else:
        btnChkDecisionTree.deselect()
    if params_training[30][1] == '1':
        btnChkSVM.select()
    else:
        btnChkSVM.deselect()
    if params_training[31][1] == '1':
        btnChkNaiveBayes.select()
    else:
        btnChkNaiveBayes.deselect()
    cboKfold.current(int(params_training[32][1]) - 1)

    if params_training[33][1] == '1':
        radReSample.select()
    else:
        radReSample.deselect()
    txtSimulationWindowSize_text.set(params_training[34][1])
    txtSimuFrameDtPoints_text.set(params_training[35][1])
    txtSimuFrameStride_text.set(params_training[36][1])
    txtSimuFrameDelay_text.set(params_training[37][1])
    txtSimuFrameRepeat_text.set(params_training[38][1])

else:
    raise Exception('Section {0} not found in the {1} file'.format('TRAINING TAB', 'app.ini'))

# Frequency domain features =>
frPeakFr = lambda x: frPeakFreq(x, resamplingrate)


# Frequency domain features <=


# Begin main section over the fitting =>
def modelsfitting_clicked():
    global train_valid_test_db_table
    global monitoring_db_table
    global test_proportion
    global csv_saving
    global monitoring_mode
    global monitoring_data_fr
    global predicted_data_fr
    global monitoring_time_deviation_fr
    global monitoring_error_types_fr
    global original_sampling_rate
    global resamplingrate
    global functions_labels_json
    global features_in_dictionary
    global axes_to_apply_functions_list
    global function_set_for_resampling
    global curr_monitoring_algorithm
    global curr_monitoring_window_size
    global curr_monitoring_sampling_rate
    global params_db
    global cboValues
    global label_set
    global timestampforCSVfiles
    time_now = datetime.datetime.now()
    timestampforCSVfiles = '%02d' % time_now.hour + 'h' + '%02d' % time_now.minute + 'm' + '%02d' % time_now.second + 's'

    no_of_labels = int(0)
    no_of_functions = int(0)
    no_of_classifiers = int(0)
    no_of_axes = int(0)
    valid_to_running = int(1)

    if selectAllActivityOnOff.get() == 1:
        label_set = cboValues

    else:
        i = int(0)
        label_set = pd.Series([], dtype="string")
        if cboActivity1.get() != 'None':
            label_set.at[i] = cboActivity1.get()
            i = i + 1
        if cboActivity2.get() != 'None':
            label_set.at[i] = cboActivity2.get()
            i = i + 1
        if cboActivity3.get() != 'None':
            label_set.at[i] = cboActivity3.get()
            i = i + 1
        if cboActivity4.get() != 'None':
            label_set.at[i] = cboActivity4.get()
            i = i + 1
        if cboActivity5.get() != 'None':
            label_set.at[i] = cboActivity5.get()
            i = i + 1
        if cboActivity6.get() != 'None':
            label_set.at[i] = cboActivity6.get()
            i = i + 1
        # Select only unique activities values if duplicate
        label_set = pd.Series(np.unique(label_set.values))
        label_set = label_set.sort_values(ascending=True)

    if binary_mode:
        label_set = pd.Series(main_label).append(sub_labels_set).reset_index(drop=True)

    no_of_labels = len(label_set)
    if no_of_labels < 2:
        valid_to_running = 0

    # Collect the aggregate functions list from selected boxes
    agg_function_list_names = []
    if MinVar.get() == 1:
        agg_function_list_names.append('min')
    if MaxVar.get() == 1:
        agg_function_list_names.append('max')
    if MeanVar.get() == 1:
        agg_function_list_names.append('mean')
    if MedianVar.get() == 1:
        agg_function_list_names.append('median')
    if StdVar.get() == 1:
        agg_function_list_names.append('stdev')
    if IQRVar.get() == 1:
        agg_function_list_names.append('IQR')
    if rootMSVar.get() == 1:
        agg_function_list_names.append('RMS')
    if meanCRVar.get() == 1:
        agg_function_list_names.append('MCR')
    if kurtosisVar.get() == 1:
        agg_function_list_names.append('Kurt')
    if skewnessVar.get() == 1:
        agg_function_list_names.append('Skew')
    if energyVar.get() == 1:
        agg_function_list_names.append('Energy')
    if peakFreqVar.get() == 1:
        agg_function_list_names.append('PeakFreq')
    if freqDmEntropyVar.get() == 1:
        agg_function_list_names.append('FreqEntrpy')
    if fr1cpnMagVar.get() == 1:
        agg_function_list_names.append('FirstCpn')
    if fr2cpnMagVar.get() == 1:
        agg_function_list_names.append('SecondCpn')
    if fr3cpnMagVar.get() == 1:
        agg_function_list_names.append('ThirdCpn')
    if fr4cpnMagVar.get() == 1:
        agg_function_list_names.append('FourthCpn')
    if fr5cpnMagVar.get() == 1:
        agg_function_list_names.append('FifthCpn')

    no_of_functions = len(agg_function_list_names)
    if no_of_functions == 0:
        valid_to_running = 0

    # Collect the axes list to apply the aggregate functions
    axes_to_apply_functions_list = []
    if gxVar.get() == 1:
        axes_to_apply_functions_list.append('gx')
    if gyVar.get() == 1:
        axes_to_apply_functions_list.append('gy')
    if gzVar.get() == 1:
        axes_to_apply_functions_list.append('gz')
    if axVar.get() == 1:
        axes_to_apply_functions_list.append('ax')
    if ayVar.get() == 1:
        axes_to_apply_functions_list.append('ay')
    if azVar.get() == 1:
        axes_to_apply_functions_list.append('az')
    if g_mag_Var.get() == 1:
        axes_to_apply_functions_list.append('gyrMag')
    if a_mag_Var.get() == 1:
        axes_to_apply_functions_list.append('accMag')

    no_of_axes = len(axes_to_apply_functions_list)
    if no_of_axes == 0:
        valid_to_running = 0

    if not (txtWindowSizeFrom_text.get().isdigit()):
        valid_to_running = 0

    if not (txtWindowSizeTo_text.get().isdigit()):
        valid_to_running = 0

    if txtWindowSizeFrom_text.get().isdigit() and txtWindowSizeTo_text.get().isdigit():
        if int(txtWindowSizeFrom_text.get()) > int(txtWindowSizeTo_text.get()):
            valid_to_running = 0

    if not (txtWindowStep_text.get().isdigit()):
        valid_to_running = 0

    if int(cboDownSamplingFrom.get()) > int(cboDownSamplingTo.get()):
        valid_to_running = 0

    no_of_classifiers = RandomForestVar.get() + DecisionTreeVar.get() + SVMVar.get() + NaiveBayesVar.get()
    if no_of_classifiers == 0:
        valid_to_running = 0

    # Check valid to running
    if valid_to_running == 0:
        if no_of_labels < 2:
            messagebox.showinfo("Alert", "Please select at least two different labels to proceed")
        elif no_of_axes == 0:
            messagebox.showinfo("Alert", "Please select at least one axis to proceed")
        elif no_of_functions == 0:
            messagebox.showinfo("Alert", "Please select at least one feature to proceed")
        elif not (txtWindowSizeFrom_text.get().isdigit()):
            messagebox.showinfo("Alert", "Please insert a valid Window size in From field")
        elif not (txtWindowSizeTo_text.get().isdigit()):
            messagebox.showinfo("Alert", "Please insert a valid Window size in To field")
        elif int(txtWindowSizeFrom_text.get()) > int(txtWindowSizeTo_text.get()):
            messagebox.showinfo("Alert", "The From Window size should be less than To Window size")
        elif not (txtWindowStep_text.get().isdigit()):
            messagebox.showinfo("Alert", "Please insert a valid Window size in Step field")
        elif int(cboDownSamplingFrom.get()) > int(cboDownSamplingTo.get()):
            messagebox.showinfo("Sampling range", "The From field must be less than To field")
        elif no_of_classifiers == 0:
            messagebox.showinfo("Alert", "Please choose at least one classifier to proceed")
    else:
        # creating json content for features
        functions_labels_json = {}
        for axis in axes_to_apply_functions_list:
            functions_labels_json[axis] = []
            for func in agg_function_list_names:
                functions_labels_json[axis] = functions_labels_json.get(axis) + [func]

        # manual fetures json creation ->
        # functions_labels_json = {'gx': ['median'], 'gy': ['mean'], 'gz': ['mean'], 'accMag': ['median']}
        # functions_labels_json = {'gyrMag': ['mean', 'stdev', 'median', 'IQR', 'RMS', 'MCR', 'Kurt', 'Skew'],'accMag': ['mean', 'stdev', 'median', 'IQR', 'RMS', 'MCR', 'Kurt', 'Skew']}
        # functions_labels_json = {'gx': ['max', 'mean', 'stdev', 'median'], 'gy': ['max', 'stdev', 'median'],
        #  'gz': ['max', 'mean', 'stdev', 'median'], 'ax': ['mean', 'stdev', 'median'],
        #  'ay': ['max', 'mean', 'stdev', 'median'], 'az': ['max', 'mean']}
        # axes_to_apply_functions_list = list(functions_labels_json.keys())
        # no_of_axes = len(axes_to_apply_functions_list)
        # manual json creation <-

        # converting json content into Python dictionary object
        features_in_dictionary = {}
        for x in functions_labels_json:
            features_in_dictionary[x] = []
            for val in functions_labels_json[x]:
                if val == 'min':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + ['min']
                if val == 'max':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + ['max']
                if val == 'mean':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + ['mean']
                if val == 'median':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + ['median']
                if val == 'stdev':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + ['std']
                if val == 'IQR':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [IQR]
                if val == 'RMS':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [RMS]
                if val == 'MCR':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [MCR]
                if val == 'Kurt':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [Kurt]
                if val == 'Skew':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [Skew]
                if val == 'Energy':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frEnergy]
                if val == 'PeakFreq':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frPeakFr]
                if val == 'FreqEntrpy':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frDmEntroPy]
                if val == 'FirstCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frMag1]
                if val == 'SecondCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frMag2]
                if val == 'ThirdCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frMag3]
                if val == 'FourthCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frMag4]
                if val == 'FifthCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frMag5]
        # print('--4 below')
        # print(functions_labels_json)
        # print(type(functions_labels_json))
        # print(features_in_dictionary)
        # print(type(features_in_dictionary))

        conn = None
        try:
                        
            conn = psycopg2.connect(**params_db)                
            # create a cursor
            cur = conn.cursor()

            # Calculate the delay in milliseconds between data points and predict the sample rate ->
            df_temp = train_valid_test_db_table.head(1000).sort_values(by='timestamp', ascending=True)
            first_label = df_temp['label'].iloc[0]
            df_temp = df_temp.loc[df_temp['label'].isin([first_label])]
            list_temp = []
            for i in range(1, len(df_temp.index)):
                list_temp.append(int(df_temp['timestamp'].iloc[i]) - int(df_temp['timestamp'].iloc[i - 1]))
            delay_between_data_points = max(set(list_temp), key=list_temp.count)
            original_sampling_rate = round(1000 / delay_between_data_points)
            # Calculate the delay in milliseconds between data points and predict the sample rate <-

            train_valid_test_data_filtered_labels = train_valid_test_db_table.loc[
                train_valid_test_db_table['label'].isin(label_set)].sort_values(by=['timestamp'], ascending=True)
            if monitoring_mode is True:
                monitoring_data_filtered_labels = monitoring_db_table.loc[
                    monitoring_db_table['label'].isin(label_set)].sort_values(
                    by=['timestamp'], ascending=True)

            # Reset monitoring data everytime user clickes on fitting
            monitoring_data_fr = pd.DataFrame()  # This dataframe includes test column values, grounthTruth label and predicted
            predicted_data_fr = pd.DataFrame()  # This dataframe includes data for the simulation
            monitoring_time_deviation_fr = pd.DataFrame()  # This one include the time deviation in monitoring phase
            monitoring_error_types_fr = pd.DataFrame()  # This one include the error types in monitoring phase

            curr_monitoring_algorithm = ''
            curr_monitoring_window_size = 0
            curr_monitoring_sampling_rate = 0

            clf_rf = RandomForestClassifier(n_estimators=100)
            clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth=10)
            clf_svm = svm.SVC(C=1.0, kernel='rbf', gamma='scale', decision_function_shape='ovo')
            clf_nb = GaussianNB()

            kfold = int(cboKfold.get())

            if binary_mode is True:
                label_set_origin = label_set.copy()

            if monitoring_mode is True:
                monitor_data_generate = True
            else:
                monitor_data_generate = False

            if rdoTrainingPhraseOnly.get() == 0:  # User selected to keep the original sampling rates

                no_of_original_train_valid_test_data_points = len(train_valid_test_data_filtered_labels.index)
                no_of_resampled_train_data_points = no_of_original_train_valid_test_data_points

                resamplingrate = original_sampling_rate  # they are the same in the case of keeping original sampling rate
                print('--------------Running with original sampling rate at ' + str(resamplingrate) + 'Hz ')
                path_with_train_valid_test_table_name = './csv_out/' + str(txtTrainTable_text.get())[
                                                                       :18] + '_at_' + timestampforCSVfiles
                hz_path = '/' + str(resamplingrate) + 'Hz'

                if not os.path.exists(path_with_train_valid_test_table_name):
                    os.mkdir(path_with_train_valid_test_table_name)
                if not os.path.exists(path_with_train_valid_test_table_name + hz_path):
                    os.mkdir(path_with_train_valid_test_table_name + hz_path)

                for window_size in range(int(txtWindowSizeFrom_text.get()), int(txtWindowSizeTo_text.get()) + 1,
                                         int(txtWindowStep_text.get())):

                    timestart = datetime.datetime.now()
                    window_stride_in_ms = math.floor(window_size * int(txtWindowStride_text.get()) / 100)
                    print('==>Begin processing window size ' + str(window_size) + ' ms Stride ' + str(
                        window_stride_in_ms) + ' ms')

                    # Begin calculating features for training phrase ->
                    print('Start calculating features for train data at ' + str(
                        datetime.datetime.now().strftime("%H:%M:%S")))
                    if binary_mode is True:
                        # Because the label_set has changed into Lying and Non-Lying in the last window setting -> get the origin labels
                        label_set = label_set_origin.copy()
                        four_labels_label_set = label_set.copy()
                        four_labels_narray_temp = four_labels_label_set.to_numpy(copy=True)
                        four_labels_narray_temp = np.append(four_labels_narray_temp, 'Non-' + main_label)

                    agg_train_valid_test_unfiltered_unbalanced = aggregated_frame(train_valid_test_data_filtered_labels,
                                                                                  label_set,
                                                                                  features_in_dictionary,
                                                                                  axes_to_apply_functions_list,
                                                                                  window_size,
                                                                                  window_stride_in_ms, 1)

                    print('End calculating features for train data at ' + str(
                        datetime.datetime.now().strftime("%H:%M:%S")))
                    path_window = '/window_' + str(window_size) + 'ms_stride_' + str(window_stride_in_ms) + 'ms'
                    if not os.path.exists(path_with_train_valid_test_table_name + hz_path + path_window):
                        os.mkdir(path_with_train_valid_test_table_name + hz_path + path_window)
                        os.mkdir(
                            path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set')
                        os.mkdir(path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data')
                        os.mkdir(path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold')

                    if csv_saving is True:
                        agg_train_valid_test_unfiltered_unbalanced.to_csv(
                            path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set' + '/01_train_valid_test_imbalanced_set_all_instances.csv',
                            header=True)
                    # End calculating features for training phrase <-

                    # filtering the number of data points for each window ->
                    minimum_count_allowed = round((window_size * resamplingrate / 1000) * 0.8)
                    maximum_count_allowed = round((window_size * resamplingrate / 1000) * 1.2)
                    agg_train_valid_test_filtered_unbalanced = agg_train_valid_test_unfiltered_unbalanced.loc[
                        (agg_train_valid_test_unfiltered_unbalanced['count'] >= minimum_count_allowed) & (
                                agg_train_valid_test_unfiltered_unbalanced['count'] <= maximum_count_allowed)]
                    # filtering the number of data points for each window <-

                    # Splitting the train_valid and test data set with the 7:3 proportion ->
                    unbalanced_train_valid_dataset, test_dataset = train_test_split(
                        agg_train_valid_test_filtered_unbalanced,
                        test_size=test_proportion,
                        shuffle=True,
                        stratify=
                        agg_train_valid_test_filtered_unbalanced[
                            ['label']])
                    # Splitting the train_valid and test data set with the given 7:3 proportion ->

                    # Begin balancing the Train_Valid data set ->
                    minimum_train_valid_instance_for_each_label = unbalanced_train_valid_dataset[
                        'label'].value_counts().min()

                    balanced_train_valid_dataset = pd.DataFrame()

                    # Lying-Non-Lying
                    if binary_mode is True:
                        no_of_labels = 2
                        no_of_main_label_instances = len(unbalanced_train_valid_dataset.loc[
                                                             unbalanced_train_valid_dataset['label'].isin(
                                                                 [main_label])].index)
                        if (
                                no_of_main_label_instances // no_of_sub_labels < minimum_train_valid_instance_for_each_label):
                            no_of_each_sub_label_instances = no_of_main_label_instances // no_of_sub_labels
                        else:
                            no_of_each_sub_label_instances = minimum_train_valid_instance_for_each_label

                        # For the purpose of correctly update into database
                        minimum_train_valid_instance_for_each_label = no_of_each_sub_label_instances

                        # Begin adjusting the proportion of Lying Standing Walking Grazing with proportion 3:1:1:1 =>
                        print('Number of instances for ' + main_label + ' label in Train_Valid set: ' + str(
                            no_of_each_sub_label_instances * no_of_sub_labels))
                        print(
                            'Number of instances for (each) sub label in Train_Valid set: ' + str(
                                no_of_each_sub_label_instances))

                        # Randomly select instances for main_label
                        features_eachlabel = unbalanced_train_valid_dataset.loc[
                            unbalanced_train_valid_dataset['label'] == main_label]
                        random_set = features_eachlabel.sample(n=no_of_each_sub_label_instances * no_of_sub_labels,
                                                               replace=False)
                        balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

                        # Randomly select instances for sub labels
                        for index, value in sub_labels_set.items():
                            features_eachlabel = unbalanced_train_valid_dataset.loc[
                                unbalanced_train_valid_dataset['label'] == value]
                            random_set = features_eachlabel.sample(n=no_of_each_sub_label_instances,
                                                                   replace=False)
                            balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

                        # End adjusting the proportion of Lying Standing Walking Grazing with proportion 3:1:1:1 <=

                        # Change the label set of Train data set into Non-Lying for the other sub labels ->

                        for index, value in sub_labels_set.items():
                            balanced_train_valid_dataset.loc[
                                balanced_train_valid_dataset.label == value, 'label'] = 'Non-' + main_label
                        # Change the label set of Train data set into Non-main Label for the other sub labels <-

                        balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')

                        # Get the root list of four activities in Test data set before change Stehen Gehen Grasen into non-liegen
                        # This is for the confusion matrix latter
                        test_dataset = test_dataset.reset_index(drop=True)
                        four_labels_y_root = test_dataset['label'].to_numpy(copy=True)

                        # Change the label set of Test dataset into Non-Lying for the other sub labels ->
                        for index, value in sub_labels_set.items():
                            test_dataset.loc[test_dataset.label == value, 'label'] = 'Non-' + main_label

                        # Change the label_set into two labels only
                        label_set = pd.Series([main_label, 'Non-' + main_label])
                    else:
                        print('Number of instances for each label in Train_Valid set: ' + str(
                            minimum_train_valid_instance_for_each_label))
                        for eachlabel in label_set:
                            features_eachlabel = unbalanced_train_valid_dataset.loc[
                                unbalanced_train_valid_dataset['label'] == eachlabel]
                            if len(features_eachlabel) == minimum_train_valid_instance_for_each_label:
                                balanced_train_valid_dataset = balanced_train_valid_dataset.append(features_eachlabel)
                            else:
                                random_set = features_eachlabel.sample(n=minimum_train_valid_instance_for_each_label,
                                                                       replace=False)
                                balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)
                        balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')

                    if csv_saving is True:
                        balanced_train_valid_dataset.to_csv(
                            path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set' + '/02_train_valid_balanced_dataset_with_' + str(
                                minimum_train_valid_instance_for_each_label) + '_instances_for_each_class.csv',
                            header=True)
                    # new 1 row
                    balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['cattle_id'], axis=1)
                    balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['count'], axis=1)
                    balanced_train_valid_dataset = balanced_train_valid_dataset.reset_index(drop=True)
                    # End balancing the Train_Valid data set <-

                    # Write infor to a text file ->
                    text_file = open(
                        path_with_train_valid_test_table_name + hz_path + path_window + '/Train_Test_Result.txt', 'w',encoding='utf-8')
                    text_file.write('Train_Valid_Test DB table: ' + txtTrainTable_text.get())
                    text_file.write('\nMonitoring         DB table: ' + txtMonitoringTable_Text.get())
                    text_file.write('\n' + str(path_window))

                    text_file.write("\nLabels to predict: " + label_set.str.cat(sep=' '))
                    text_file.write("\nFunctions list      : " + str(agg_function_list_names))
                    text_file.write("\nAxes list             : " + str(axes_to_apply_functions_list))

                    # Write infor to a text file <-

                    # Preseting some of metrics for the classification
                    rf_accu = 0
                    rf_prec = 0
                    rf_recal = 0
                    rf_spec = 0
                    rf_f1 = 0

                    dt_accu = 0
                    dt_prec = 0
                    dt_recal = 0
                    dt_spec = 0
                    dt_f1 = 0

                    svm_accu = 0
                    svm_prec = 0
                    svm_recal = 0
                    svm_spec = 0
                    svm_f1 = 0

                    nb_accu = 0
                    nb_prec = 0
                    nb_recal = 0
                    nb_spec = 0
                    nb_f1 = 0

                    feature_cols = list(balanced_train_valid_dataset.columns.values)
                    feature_cols.remove('label')

                    X = balanced_train_valid_dataset[feature_cols]  # Features
                    y = balanced_train_valid_dataset['label']  # Target

                    # Stratified k-fold
                    kf = StratifiedKFold(n_splits=kfold, shuffle=True)  # Considering random_state = 0??
                    k_fold_round = int(0)
                    for train_index, valid_index in kf.split(X, y):
                        k_fold_round = k_fold_round + 1
                        # print('Round ' + str(k_fold_round))
                        X_train = pd.DataFrame(X, columns=feature_cols, index=train_index)
                        X_valid = pd.DataFrame(X, columns=feature_cols, index=valid_index)

                        if csv_saving is True:
                            X_train.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                    k_fold_round) + 'th_round_fold_X_train.csv',
                                header=True)
                            X_valid.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                    k_fold_round) + 'th_round_fold_X_validation.csv',
                                header=True)

                        y_train_df = pd.DataFrame(y, columns=['label'], index=train_index)
                        y_train = y_train_df['label']
                        y_valid_df = pd.DataFrame(y, columns=['label'], index=valid_index)
                        y_valid = y_valid_df['label']

                        if csv_saving is True:
                            y_train.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                    k_fold_round) + 'th_round_fold_y_train.csv',
                                header=True)
                            y_valid.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                    k_fold_round) + 'th_round_fold_y_validation.csv',
                                header=True)

                        text_file.write("\n------------------Round " + str(k_fold_round) + "------------------")

                        if RandomForestVar.get() == 1:
                            clf_rf.fit(X_train, y_train)
                            y_pred_rf = clf_rf.predict(X_valid)

                            temp_acc = round(metrics.accuracy_score(y_valid, y_pred_rf), 4)
                            rf_accu = rf_accu + temp_acc
                            text_file.write("\nRandom Forest accuracy: " + str(temp_acc))
                            # print('Random Forest accuracy ' + str(temp_acc))

                            if label_set.size == 2:
                                rf_prec = rf_prec + metrics.precision_score(y_valid, y_pred_rf, average="binary",
                                                                            pos_label=label_set[0])
                                rf_recal = rf_recal + metrics.recall_score(y_valid, y_pred_rf, average="binary",
                                                                           pos_label=label_set[0])
                                rf_f1 = rf_f1 + metrics.recall_score(y_valid, y_pred_rf, average="binary",
                                                                     pos_label=label_set[0])

                        if DecisionTreeVar.get() == 1:
                            clf_dt.fit(X_train, y_train)
                            y_pred_dt = clf_dt.predict(X_valid)

                            temp_acc = round(metrics.accuracy_score(y_valid, y_pred_dt), 4)
                            dt_accu = dt_accu + temp_acc

                            text_file.write("\nDecision Tree accuracy: " + str(temp_acc))
                            # print('Decision Tree accuracy ' + str(temp_acc))

                            if (label_set.size == 2):
                                dt_prec = dt_prec + metrics.precision_score(y_valid, y_pred_dt, average="binary",
                                                                            pos_label=label_set[0])
                                dt_recal = dt_recal + metrics.recall_score(y_valid, y_pred_dt, average="binary",
                                                                           pos_label=label_set[0])
                                dt_f1 = dt_f1 + metrics.recall_score(y_valid, y_pred_dt, average="binary",
                                                                     pos_label=label_set[0])

                        if SVMVar.get() == 1:
                            clf_svm.fit(X_train, y_train)
                            y_pred_svm = clf_svm.predict(X_valid)

                            temp_acc = round(metrics.accuracy_score(y_valid, y_pred_svm), 4)
                            svm_accu = svm_accu + temp_acc

                            text_file.write("\nSVM accuracy: " + str(temp_acc))
                            # print('SVM Accuracy ' + str(temp_acc))

                            if (label_set.size == 2):
                                svm_prec = svm_prec + metrics.precision_score(y_valid, y_pred_svm, average="binary",
                                                                              pos_label=label_set[0])
                                svm_recal = svm_recal + metrics.recall_score(y_valid, y_pred_svm, average="binary",
                                                                             pos_label=label_set[0])
                                svm_f1 = svm_f1 + metrics.recall_score(y_valid, y_pred_svm, average="binary",
                                                                       pos_label=label_set[0])

                        if NaiveBayesVar.get() == 1:
                            clf_nb.fit(X_train, y_train)
                            y_pred_nb = clf_nb.predict(X_valid)

                            temp_acc = round(metrics.accuracy_score(y_valid, y_pred_nb), 4)
                            nb_accu = nb_accu + temp_acc

                            text_file.write("\nNaive Bayes accuracy: " + str(temp_acc))
                            # print('Naive Bayes accuracy ' + str(temp_acc))

                            if label_set.size == 2:
                                nb_prec = nb_prec + metrics.precision_score(y_valid, y_pred_nb, average="binary",
                                                                            pos_label=label_set[0])
                                nb_recal = nb_recal + metrics.recall_score(y_valid, y_pred_nb, average="binary",
                                                                           pos_label=label_set[0])
                                nb_f1 = nb_f1 + metrics.recall_score(y_valid, y_pred_nb, average="binary",
                                                                     pos_label=label_set[0])
                    text_file.write("\n")

                    if RandomForestVar.get() == 1:
                        text_file.write(
                            "\nTrain_Valid " + str(kfold) + "-fold average accuracy of Random Forest: " + str(
                                round((rf_accu / kfold), 4)))
                        print('Train_Valid ' + str(kfold) + '-fold average accuracy of Random Forest ' + str(
                            round((rf_accu / kfold), 4)))

                    if DecisionTreeVar.get() == 1:
                        text_file.write(
                            "\nTrain_Valid " + str(kfold) + "-fold average accuracy of Decision Tree: " + str(
                                round((dt_accu / kfold), 4)))
                        print('Train_Valid ' + str(kfold) + '-fold average accuracy of Decision Tree ' + str(
                            round((dt_accu / kfold), 4)))

                    if SVMVar.get() == 1:
                        text_file.write("\nTrain_Valid " + str(kfold) + "-fold average accuracy of SVM: " + str(
                            round((svm_accu / kfold), 4)))
                        print('Train_Valid ' + str(kfold) + '-fold average accuracy of SVM           ' + str(
                            round((svm_accu / kfold), 4)))

                    if NaiveBayesVar.get() == 1:
                        text_file.write(
                            "\nTrain_Valid " + str(kfold) + "-fold average accuracy of Naive Bayes: " + str(
                                round((nb_accu / kfold), 4)))
                        print('Train_Valid ' + str(kfold) + '-fold average accuracy of Naive Bayes   ' + str(
                            round((nb_accu / kfold), 4)))

                    # Begin generating data for Test dataset =>
                    rf_accu_test = 0
                    rf_prec_test = 0
                    rf_recal_test = 0
                    rf_spec_test = 0
                    rf_f1_test = 0

                    dt_accu_test = 0
                    dt_prec_test = 0
                    dt_recal_test = 0
                    dt_spec_test = 0
                    dt_f1_test = 0

                    svm_accu_test = 0
                    svm_prec_test = 0
                    svm_recal_test = 0
                    svm_spec_test = 0
                    svm_f1_test = 0

                    nb_accu_test = 0
                    nb_prec_test = 0
                    nb_recal_test = 0
                    nb_spec_test = 0
                    nb_f1_test = 0

                    labels_narray_temp = label_set.to_numpy()
                    test_dataset = test_dataset.dropna().set_index('timestamp')

                    if csv_saving is True:
                        test_dataset.to_csv(
                            path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set/03_test_dataset_counts_filtered.csv',
                            header=True)

                    test_dataset = test_dataset.drop(['cattle_id'], axis=1)
                    test_dataset = test_dataset.drop(['count'], axis=1)
                    test_dataset = test_dataset.reset_index(drop=True)

                    X_test = test_dataset[feature_cols]  # Features
                    y_test = test_dataset['label']  # Target

                    if RandomForestVar.get() == 1:
                        y_test_pred_rf = clf_rf.predict(X_test)
                        rf_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_rf), 4)
                        print("------------------------------------------------------")
                        print('Random Forest - Accuracy on Test data:' + str(rf_accu_test))

                        # Begin saving RF Confusion Matrix into text file ->
                        text_file.write("\n------------------------------------------------------")
                        text_file.write("\nAccuracy of Random Forest on Test data: " + str(rf_accu_test))
                        text_file.write("\nRandom Forest Confusion matrix on Test data")
                        text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")

                        confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_rf,
                                                                         labels=labels_narray_temp)
                        lbl_no = 0
                        for line in confusion_matrix_temp:
                            text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                            lbl_no = lbl_no + 1

                        # Begin printing RF Confusion Matrix to console <-
                        print('Random Forest - Confusion matrix on Test data')
                        print('Predicted ' + label_set.str.cat(sep=' '))
                        lbl_no = 0
                        for line in confusion_matrix_temp:
                            print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                            lbl_no = lbl_no + 1
                        # End Printing RF Confusion Matrix to console <-
                        # End Saving RF Confusion Matrix into text file <-

                        # Additional confusion matrix for binary Lying and Nonlying ->
                        if binary_mode is True:
                            text_file.write("\nRandom Forest Confusion matrix on Test data (" + str(
                                no_of_sub_labels + 1) + " labels) ")
                            text_file.write("\nPredicted \u2193" + four_labels_label_set.str.cat(
                                sep=' ') + ' Non-' + main_label + " \u2193")

                            confusion_matrix_temp = metrics.confusion_matrix(four_labels_y_root, y_test_pred_rf,
                                                                             labels=four_labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            print('------------')
                            print('Random Forest - Confusion matrix on Test data (' + str(
                                no_of_sub_labels + 1) + ' labels) ')
                            print('Predicted ' + four_labels_label_set.str.cat(sep=' ') + ' Non-' + main_label)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                print('True ' + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                        # Additional confusion matrix for binary Lying and Nonlying <-

                        if label_set.size == 2:
                            rf_prec_test = round(metrics.precision_score(y_test, y_test_pred_rf, average="binary",
                                                                         pos_label=label_set[0]), 4)
                            rf_recal_test = round(metrics.recall_score(y_test, y_test_pred_rf, average="binary",
                                                                       pos_label=label_set[0]), 4)
                            rf_f1_test = round(
                                rf_f1_test + metrics.recall_score(y_test, y_test_pred_rf, average="binary",
                                                                  pos_label=label_set[0]), 4)
                        else:
                            rf_prec_test = 0
                            rf_recal_test = 0
                            rf_f1_test = 0

                    if DecisionTreeVar.get() == 1:
                        y_test_pred_dt = clf_dt.predict(X_test)
                        dt_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_dt), 4)
                        print("------------------------------------------------------")
                        print('Accuracy of Decistion Tree on test data:' + str(dt_accu_test))

                        # Begin saving DT Confusion Matrix into text file->
                        text_file.write("\n------------------------------------------------------")
                        text_file.write("\nAccuracy of Decision Tree on Test data: " + str(dt_accu_test))
                        text_file.write("\nDecision Tree - Confusion matrix on Test data")
                        text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                        confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_dt,
                                                                         labels=labels_narray_temp)
                        lbl_no = 0
                        for line in confusion_matrix_temp:
                            text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                            lbl_no = lbl_no + 1
                        print(confusion_matrix_temp)
                        # End saving DT Confusion Matrix into text file <-

                        if label_set.size == 2:
                            dt_prec_test = round(metrics.precision_score(y_test, y_test_pred_dt, average="binary",
                                                                         pos_label=label_set[0]), 4)
                            dt_recal_test = round(metrics.recall_score(y_test, y_test_pred_dt, average="binary",
                                                                       pos_label=label_set[0]), 4)
                            dt_f1_test = round(
                                dt_f1_test + metrics.recall_score(y_test, y_test_pred_dt, average="binary",
                                                                  pos_label=label_set[0]), 4)
                        else:
                            dt_prec_test = 0
                            dt_recal_test = 0
                            dt_f1_test = 0

                    if SVMVar.get() == 1:
                        y_test_pred_svm = clf_svm.predict(X_test)
                        svm_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_svm), 4)
                        print("------------------------------------------------------")
                        print('Accuracy of SVM on Test data:' + str(svm_accu_test))

                        # Begin saving SVM Confusion Matrix into text file->
                        text_file.write("\n------------------------------------------------------")
                        text_file.write("\nAccuracy of SVM on Test data: " + str(svm_accu_test))
                        text_file.write("\nSVM - Confusion matrix on Test data")
                        text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                        confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_svm,
                                                                         labels=labels_narray_temp)
                        lbl_no = 0
                        for line in confusion_matrix_temp:
                            text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                            lbl_no = lbl_no + 1
                        print(confusion_matrix_temp)
                        # End saving SVM Confusion Matrix into text file <-

                        if label_set.size == 2:
                            svm_prec_test = round(metrics.precision_score(y_test, y_test_pred_svm, average="binary",
                                                                          pos_label=label_set[0]), 4)
                            svm_recal_test = round(metrics.recall_score(y_test, y_test_pred_svm, average="binary",
                                                                        pos_label=label_set[0]), 4)
                            svm_f1_test = round(
                                svm_f1_test + metrics.recall_score(y_test, y_test_pred_svm, average="binary",
                                                                   pos_label=label_set[0]), 4)
                        else:
                            svm_prec_test = 0
                            svm_recal_test = 0
                            svm_f1_test = 0

                    if NaiveBayesVar.get() == 1:
                        y_test_pred_nb = clf_nb.predict(X_test)
                        nb_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_nb), 4)
                        print("------------------------------------------------------")
                        print('Accuracy of Naive Bayes on Test data:' + str(nb_accu_test))

                        # Begin saving Naive Bayes Confusion Matrix into text file->
                        text_file.write("\n------------------------------------------------------")
                        text_file.write("\nAccuracy of Naive Bayes on Testing data: " + str(nb_accu_test))
                        text_file.write("\nNaive Bayes - Confusion matrix on Test data")
                        text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                        confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_nb,
                                                                         labels=labels_narray_temp)
                        lbl_no = 0
                        for line in confusion_matrix_temp:
                            text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                            lbl_no = lbl_no + 1
                        print(confusion_matrix_temp)
                        # End saving SVM Confusion Matrix into text file <-

                        if label_set.size == 2:
                            nb_prec_test = round(metrics.precision_score(y_test, y_test_pred_nb, average="binary",
                                                                         pos_label=label_set[0]), 4)
                            nb_recal_test = round(metrics.recall_score(y_test, y_test_pred_nb, average="binary",
                                                                       pos_label=label_set[0]), 4)
                            nb_f1_test = round(
                                nb_f1_test + metrics.recall_score(y_test, y_test_pred_nb, average="binary",
                                                                  pos_label=label_set[0]), 4)
                        else:
                            nb_prec_test = 0
                            nb_recal_test = 0
                            nb_f1_test = 0

                    # Checking whether user want to view statistics and monitoring
                    # For monitoring metrics
                    rf_accu_monitor = 0
                    rf_prec_monitor = 0
                    rf_recal_monitor = 0
                    rf_spec_monitor = 0
                    rf_f1_monitor = 0

                    dt_accu_monitor = 0
                    dt_prec_monitor = 0
                    dt_recal_monitor = 0
                    dt_spec_monitor = 0
                    dt_f1_monitor = 0

                    svm_accu_monitor = 0
                    svm_prec_monitor = 0
                    svm_recal_monitor = 0
                    svm_spec_monitor = 0
                    svm_f1_monitor = 0

                    nb_accu_monitor = 0
                    nb_prec_monitor = 0
                    nb_recal_monitor = 0
                    nb_spec_monitor = 0
                    nb_f1_monitor = 0

                    if monitoring_mode is True:
                        print('------------------------------------------------------')
                        print('Begin calculating features for the Monitoring data at ' + str(
                            datetime.datetime.now().strftime("%H:%M:%S")))
                        simulation_window_stride_in_ms = math.floor(
                            window_size * int(txtWindowSimuStride_text.get()) / 100)

                        if binary_mode is True:
                            # Because the label_set has changed into Lying and Non-Lying in the last window setting -> get the origin labels
                            label_set = label_set_origin.copy()

                        # uncomment here
                        # monitor_data_generate = False

                        if monitor_data_generate is True:
                            # agg_monitor dataframe returns data without label (for predict file generation)
                            agg_monitor = aggregated_unseen_data(monitoring_data_filtered_labels, label_set,
                                                                 features_in_dictionary,
                                                                 axes_to_apply_functions_list, window_size,
                                                                 simulation_window_stride_in_ms, 0)

                        print('End calculating features for the Monitoring data at ' + str(
                            datetime.datetime.now().strftime("%H:%M:%S")))
                        # To be set for saving or not ->

                        if monitor_data_generate is True:
                            if csv_saving is True:
                                agg_monitor.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/0_unfiltered_monitoring_set.csv',
                                    header=True)
                        # End calculating the features for the monitoring and simulator  <-

                        # Option of counts_filtered_monitoring_dataset ->
                        # Filtering the number of data points for each window in monitoring data ->
                        # counts_filtered_monitoring_dataset = agg_monitor.loc[
                        #     (agg_monitor['count'] >= minimum_count_allowed) & (
                        #             agg_monitor['count'] <= maximum_count_allowed)]
                        # # Filtering the number of data points for each window in monitoring data<-
                        #
                        # # Saving instances for monitoring ->
                        # counts_filtered_monitoring_dataset.to_csv(
                        #     path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/1_filtered_monitoring_set.csv',
                        #     header=True)
                        # # Saving instances for monitoring <-
                        #
                        # counts_filtered_monitoring_dataset = counts_filtered_monitoring_dataset.dropna().reset_index(drop = True)
                        # Option of counts_filtered_monitoring_dataset <-conn.cursor

                        # This dataframe is for testing on unseen data (from monitoring data table) ->
                        agg_monitor_temp = aggregated_frame(monitoring_data_filtered_labels, label_set,
                                                            features_in_dictionary,
                                                            axes_to_apply_functions_list, window_size,
                                                            window_stride_in_ms, 1)

                        counts_filtered_monitoring_dataset_temp = agg_monitor_temp.loc[
                            (agg_monitor_temp['count'] >= minimum_count_allowed) & (
                                    agg_monitor_temp['count'] <= maximum_count_allowed)]
                        counts_filtered_monitoring_dataset_temp = counts_filtered_monitoring_dataset_temp.dropna().reset_index(
                            drop=True)
                        # To be deleted ->
                        if csv_saving is True:
                            counts_filtered_monitoring_dataset_temp.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/1_filtered_monitoring_set.csv',
                                header=True)
                        # To be deleted <-

                        if binary_mode is True:
                            # Get the root list of four activities in Test data set before change Stehen Gehen Grasen into non-liegen
                            # This is for the confusion matrix latter
                            four_labels_y_root_monitoring_temp = counts_filtered_monitoring_dataset_temp[
                                'label'].to_numpy(copy=True)

                            # Change the label set of Test dataset into Non-main label for the other three labels ->
                            for index, value in sub_labels_set.items():
                                counts_filtered_monitoring_dataset_temp.loc[
                                    counts_filtered_monitoring_dataset_temp.label == value, 'label'] = 'Non-' + main_label

                            # Change the label_set into two labels only
                            label_set = pd.Series([main_label, 'Non-' + main_label])

                        # X_monitor = counts_filtered_monitoring_dataset[feature_cols]  # Features
                        if monitor_data_generate is True:
                            X_monitor = agg_monitor[feature_cols]  # Features
                        y_monitor_temp = counts_filtered_monitoring_dataset_temp['label']
                        X_monitor_temp = counts_filtered_monitoring_dataset_temp[feature_cols]  # Features
                        # y_monitor_temp = agg_monitor_temp['label']
                        # X_monitor_temp = agg_monitor_temp[feature_cols]  # Features

                        # Begin predicting and generate data for monitoring ==================>
                        if RandomForestVar.get() == 1:
                            if monitor_data_generate is True:
                                y_monitor_pred_rf = clf_rf.predict(X_monitor)
                                simu_predicted_df_rf = pd.concat(
                                    [agg_monitor[['timestamp']], pd.DataFrame(y_monitor_pred_rf)],
                                    axis=1)
                                simu_predicted_df_rf.columns = ['timestamp', 'predicted_label']
                                simu_predicted_df_rf.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/2_monitor_predicted_rf.csv',
                                    header=True)

                            y_monitor_pred_rf_temp = clf_rf.predict(X_monitor_temp)
                            rf_accu_monitor = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_rf_temp), 4)
                            print('Random Forest - Accuracy on Monitor data ' + str(rf_accu_monitor))

                            if label_set.size == 2:
                                rf_prec_monitor = round(
                                    metrics.precision_score(y_monitor_temp, y_monitor_pred_rf_temp, average="binary",
                                                            pos_label=label_set[0]), 4)
                                rf_recal_monitor = round(
                                    metrics.recall_score(y_monitor_temp, y_monitor_pred_rf_temp, average="binary",
                                                         pos_label=label_set[0]), 4)
                                rf_f1_monitor = round(
                                    rf_f1_monitor + metrics.recall_score(y_monitor_temp, y_monitor_pred_rf_temp,
                                                                         average="binary",
                                                                         pos_label=label_set[0]), 4)
                            else:
                                rf_prec_monitor = 0
                                rf_recal_monitor = 0
                                rf_f1_monitor = 0

                            # Begin saving RF Confusion Matrix into text file ->
                            text_file.write("\n------------------------------------------------------")
                            text_file.write("\nAccuracy of Random Forest on monitoring data: " + str(rf_accu_monitor))
                            text_file.write("\nRandom Forest Confusion matrix on monitoring data")
                            text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")

                            confusion_matrix_temp = metrics.confusion_matrix(y_monitor_temp, y_monitor_pred_rf_temp,
                                                                             labels=labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1

                            # Begin printing RF Confusion Matrix to console ->
                            print('Random Forest - Confusion matrix on Monitoring data')
                            print('Predicted ' + label_set.str.cat(sep=' '))
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            # End Printing RF Confusion Matrix to console <-
                            # End Saving RF Confusion Matrix into text file <-

                            # Additional confusion matrix for sub labels in case of Lying and Nonlying mode ->
                            if binary_mode is True:
                                # Showing confusion matrix for sub labelss
                                text_file.write("\nRandom Forest Confusion matrix on monitoring data (" + str(
                                    no_of_sub_labels + 1) + " labels) ")
                                text_file.write("\nPredicted \u2193" + four_labels_label_set.str.cat(
                                    sep=' ') + ' Non-' + main_label + " \u2193")

                                confusion_matrix_temp = metrics.confusion_matrix(four_labels_y_root_monitoring_temp,
                                                                                 y_monitor_pred_rf_temp,
                                                                                 labels=four_labels_narray_temp)
                                lbl_no = 0
                                for line in confusion_matrix_temp:
                                    text_file.write("\n\u2192True " + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                    lbl_no = lbl_no + 1

                                # Begin printing RF Confusion Matrix to console ->
                                print('------------')
                                print('Random Forest - Confusion matrix on monitoring data (' + str(
                                    no_of_sub_labels + 1) + ' labels) ')
                                print('Predicted ' + four_labels_label_set.str.cat(sep=' ') + ' Non-' + main_label)
                                lbl_no = 0
                                for line in confusion_matrix_temp:
                                    print('True ' + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                    lbl_no = lbl_no + 1
                            # Additional confusion matrix for sub labels in case of Lying and Nonlying mode <

                        if DecisionTreeVar.get() == 1:
                            if monitor_data_generate is True:
                                y_monitor_pred_dt = clf_dt.predict(X_monitor)
                                simu_predicted_df_dt = pd.concat(
                                    [agg_monitor[['timestamp']], pd.DataFrame(y_monitor_pred_dt)],
                                    axis=1)
                                simu_predicted_df_dt.columns = ['timestamp', 'predicted_label']
                                simu_predicted_df_dt.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/3_monitor_predicted_dt.csv',
                                    header=True)

                            y_monitor_pred_dt_temp = clf_dt.predict(X_monitor_temp)
                            dt_accu_monitor = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_dt_temp), 4)
                            print('Decision Tree accuracy monitor ' + str(dt_accu_monitor))
                            text_file.write("\nDecision Tree accuracy on Monitoring data: " + str(dt_accu_monitor))

                            if label_set.size == 2:
                                dt_prec_monitor = round(
                                    metrics.precision_score(y_monitor_temp, y_monitor_pred_dt_temp, average="binary",
                                                            pos_label=label_set[0]), 4)
                                dt_recal_monitor = round(
                                    metrics.recall_score(y_monitor_temp, y_monitor_pred_dt_temp, average="binary",
                                                         pos_label=label_set[0]), 4)
                                dt_f1_monitor = round(
                                    dt_f1_monitor + metrics.recall_score(y_monitor_temp, y_monitor_pred_dt_temp,
                                                                         average="binary",
                                                                         pos_label=label_set[0]), 4)
                            else:
                                dt_prec_monitor = 0
                                dt_recal_monitor = 0
                                dt_f1_monitor = 0

                        if SVMVar.get() == 1:
                            if monitor_data_generate is True:
                                y_monitor_pred_svm = clf_svm.predict(X_monitor)
                                simu_predicted_df_svm = pd.concat(
                                    [agg_monitor[['timestamp']], pd.DataFrame(y_monitor_pred_svm)],
                                    axis=1)
                                simu_predicted_df_svm.columns = ['timestamp', 'predicted_label']
                                simu_predicted_df_svm.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/4_monitor_predicted_svm.csv',
                                    header=True)

                            y_monitor_pred_svm_temp = clf_svm.predict(X_monitor_temp)
                            svm_accu_monitor = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_svm_temp), 4)
                            print('SVM accuracy monitor ' + str(svm_accu_monitor))
                            text_file.write("\nSVM accuracy on Monitoring data: " + str(svm_accu_monitor))

                            if label_set.size == 2:
                                svm_prec_monitor = round(
                                    metrics.precision_score(y_monitor_temp, y_monitor_pred_svm_temp, average="binary",
                                                            pos_label=label_set[0]), 4)
                                svm_recal_monitor = round(
                                    metrics.recall_score(y_monitor_temp, y_monitor_pred_svm_temp, average="binary",
                                                         pos_label=label_set[0]), 4)
                                svm_f1_monitor = round(
                                    svm_f1_monitor + metrics.recall_score(y_monitor_temp, y_monitor_pred_svm_temp,
                                                                          average="binary",
                                                                          pos_label=label_set[0]), 4)
                            else:
                                svm_prec_monitor = 0
                                svm_recal_monitor = 0
                                svm_f1_monitor = 0

                        if NaiveBayesVar.get() == 1:
                            if monitor_data_generate is True:
                                y_monitor_pred_nb = clf_nb.predict(X_monitor)
                                simu_predicted_df_nb = pd.concat(
                                    [agg_monitor[['timestamp']], pd.DataFrame(y_monitor_pred_nb)],
                                    axis=1)
                                simu_predicted_df_nb.columns = ['timestamp', 'predicted_label']
                                simu_predicted_df_nb.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/5_monitor_predicted_nb.csv',
                                    header=True)

                            y_monitor_pred_nb_temp = clf_nb.predict(X_monitor_temp)
                            nb_accu_monitor = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_nb_temp), 4)
                            print('Naive Bayes accuracy monitor ' + str(nb_accu_monitor))
                            text_file.write("\nNaive Bayes accuracy on Monitoring data: " + str(nb_accu_monitor))
                            if label_set.size == 2:
                                nb_prec_monitor = round(
                                    metrics.precision_score(y_monitor_temp, y_monitor_pred_nb_temp, average="binary",
                                                            pos_label=label_set[0]), 4)
                                nb_recal_monitor = round(
                                    metrics.recall_score(y_monitor_temp, y_monitor_pred_nb_temp, average="binary",
                                                         pos_label=label_set[0]), 4)
                                nb_f1_monitor = round(
                                    nb_f1_monitor + metrics.recall_score(y_monitor_temp, y_monitor_pred_nb_temp,
                                                                         average="binary",
                                                                         pos_label=label_set[0]), 4)
                            else:
                                nb_prec_monitor = 0
                                nb_recal_monitor = 0
                                nb_f1_monitor = 0

                        # End predicting and generate data for monitoring ==================<
                    print('End processing window size ' + str(window_size) + ' ms Stride ' + str(
                        window_stride_in_ms) + ' ms')
                    print("------------------------------------------------------")
                    text_file.close()

                    # Write experiment result into DB =>
                    slqInsertQuery = 'INSERT INTO ' + txtResultTable_Text.get() + '(model_title, model_init_name, model_binary_content, features_json_content, model_comments, train_table, monitor_table, no_of_predicted_classes, list_of_predicted_classes, original_sample_rate_in_hz, no_of_original_train_data_points, resampled_rate_in_hz, no_of_resampled_train_data_points, no_of_instances_for_each_class_in_resampled_train_table, algorithm, no_of_functions, list_of_functions, no_of_axes, list_of_axes, window_size, window_stride,k_fold, accuracy_train_valid, precision_train_valid, recall_train_valid, specificity_train_valid, f1_train_valid, accuracy_test, precision_test, recall_test, specificity_test, f1_test, monitoring_window_stride, accuracy_monitor, precision_monitor, recall_monitor, specificity_monitor, f1_monitor, start_time, end_time, running_time_in_minutes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

                    monitor_table_name_temp = txtMonitoringTable_Text.get()
                    if monitoring_mode is False:
                        monitor_table_name_temp = 'n.a'

                    if RandomForestVar.get() == 1:
                        timeend = datetime.datetime.now()
                        model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                        if binary_mode is True:
                            model_init_Name = model_init_Name + '_Binary'
                        else:
                            model_init_Name = model_init_Name + '_Multi'
                        model_init_Name = model_init_Name + '_RandomForest'
                        model_data_content = pickle.dumps(clf_rf)  # pickle the model

                        duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                        record_to_insert = (
                            '', model_init_Name, psycopg2.Binary(model_data_content), json.dumps(functions_labels_json),
                            model_comment,
                            txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                            label_set.str.cat(sep='_'),
                            original_sampling_rate, no_of_original_train_valid_test_data_points,
                            resamplingrate, no_of_resampled_train_data_points,
                            str(minimum_train_valid_instance_for_each_label), 'Random Forest',
                            no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                            '_'.join(axes_to_apply_functions_list), window_size,
                            txtWindowStride_text.get() + "%", kfold, round(rf_accu / kfold, 4),
                            round(rf_prec / kfold, 4), round(rf_recal / kfold, 4),
                            round(rf_spec / kfold, 4), round(rf_f1 / kfold, 4), rf_accu_test,
                            rf_prec_test, rf_recal_test, rf_spec_test, rf_f1_test,
                            txtWindowSimuStride_text.get() + "%", rf_accu_monitor, rf_prec_monitor,
                            rf_recal_monitor, rf_spec_monitor, rf_f1_monitor,
                            str(timestart), str(timeend), duration)

                        cur.execute(slqInsertQuery, record_to_insert)
                        conn.commit()

                    if DecisionTreeVar.get() == 1:
                        timeend = datetime.datetime.now()
                        model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                        if binary_mode is True:
                            model_init_Name = model_init_Name + '_Binary'
                        else:
                            model_init_Name = model_init_Name + '_Multi'
                        model_init_Name = model_init_Name + '_DecisionTree'
                        model_data_content = pickle.dumps(clf_dt)  # pickle the model

                        duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                        record_to_insert = (
                            '', model_init_Name, psycopg2.Binary(model_data_content), json.dumps(functions_labels_json),
                            model_comment,
                            txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                            label_set.str.cat(sep='_'),
                            original_sampling_rate, no_of_original_train_valid_test_data_points,
                            resamplingrate, no_of_resampled_train_data_points,
                            str(minimum_train_valid_instance_for_each_label), 'Decision Tree',
                            no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                            '_'.join(axes_to_apply_functions_list), window_size,
                            txtWindowStride_text.get() + "%", kfold, round(dt_accu / kfold, 4),
                            round(dt_prec / kfold, 4), round(dt_recal / kfold, 4),
                            round(dt_spec / kfold, 4), round(dt_f1 / kfold, 4), dt_accu_test,
                            dt_prec_test,
                            dt_recal_test, dt_spec_test, dt_f1_test,
                            txtWindowSimuStride_text.get() + "%",
                            dt_accu_monitor, dt_prec_monitor, dt_recal_monitor, dt_spec_monitor,
                            dt_f1_monitor,
                            str(timestart), str(timeend), duration)

                        cur.execute(slqInsertQuery, record_to_insert)
                        conn.commit()

                    if SVMVar.get() == 1:
                        timeend = datetime.datetime.now()
                        model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                        if binary_mode is True:
                            model_init_Name = model_init_Name + '_Binary'
                        else:
                            model_init_Name = model_init_Name + '_Multi'
                        model_init_Name = model_init_Name + '_SVM'
                        model_data_content = pickle.dumps(clf_svm)  # pickle the model

                        duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                        record_to_insert = (
                            '', model_init_Name, psycopg2.Binary(model_data_content), json.dumps(functions_labels_json),
                            model_comment,
                            txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                            label_set.str.cat(sep='_'),
                            original_sampling_rate, no_of_original_train_valid_test_data_points,
                            resamplingrate, no_of_resampled_train_data_points,
                            str(minimum_train_valid_instance_for_each_label), 'SVM',
                            no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                            '_'.join(axes_to_apply_functions_list), window_size,
                            txtWindowStride_text.get() + "%", kfold, round(svm_accu / kfold, 4),
                            round(svm_prec / kfold, 4), round(svm_recal / kfold, 4),
                            round(svm_spec / kfold, 4), round(svm_f1 / kfold, 4), svm_accu_test,
                            svm_prec_test,
                            svm_recal_test, svm_spec_test, svm_f1_test,
                            txtWindowSimuStride_text.get() + "%",
                            svm_accu_monitor, svm_prec_monitor, svm_recal_monitor, svm_spec_monitor,
                            svm_f1_monitor,
                            str(timestart), str(timeend), duration)

                        cur.execute(slqInsertQuery, record_to_insert)
                        conn.commit()

                    if NaiveBayesVar.get() == 1:
                        timeend = datetime.datetime.now()
                        model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                        if binary_mode is True:
                            model_init_Name = model_init_Name + '_Binary'
                        else:
                            model_init_Name = model_init_Name + '_Multi'
                        model_init_Name = model_init_Name + '_NaiveBayes'

                        model_data_content = pickle.dumps(clf_nb)  # pickle the model

                        duration = round(((timeend - timestart).total_seconds()) / 60, 3)
                        record_to_insert = (
                            '', model_init_Name, psycopg2.Binary(model_data_content), json.dumps(functions_labels_json),
                            model_comment,
                            txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                            label_set.str.cat(sep='_'),
                            original_sampling_rate, no_of_original_train_valid_test_data_points,
                            resamplingrate, no_of_resampled_train_data_points,
                            str(minimum_train_valid_instance_for_each_label), 'Naive Bayes',
                            no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                            '_'.join(axes_to_apply_functions_list), window_size,
                            txtWindowStride_text.get() + "%", kfold, round(nb_accu / kfold, 4),
                            round(nb_prec / kfold, 4), round(nb_recal / kfold, 4),
                            round(nb_spec / kfold, 4), round(nb_f1 / kfold, 4), nb_accu_test,
                            nb_prec_test,
                            nb_recal_test, nb_spec_test, nb_f1_test,
                            txtWindowSimuStride_text.get() + "%",
                            nb_accu_monitor, nb_prec_monitor, nb_recal_monitor, nb_spec_monitor,
                            nb_f1_monitor,
                            str(timestart), str(timeend), duration)

                        cur.execute(slqInsertQuery, record_to_insert)
                        conn.commit()
                    print('Finished in ' + str(duration) + ' minutes')
                    # winsound.Beep(1000, 300)
                    # End training phrase for original tables <-

            if rdoTrainingPhraseOnly.get() == 1:  # User selected to test on different sampling rates tables

                no_of_original_train_valid_test_data_points = len(train_valid_test_data_filtered_labels.index)

                path_with_train_valid_test_table_name = './csv_out/' + str(txtTrainTable_text.get())[
                                                                       :18] + '_at_' + timestampforCSVfiles

                if not os.path.exists(path_with_train_valid_test_table_name):
                    os.mkdir(path_with_train_valid_test_table_name)

                # Start testing on different sampling
                for resamplingrate in range(int(str(cboDownSamplingFrom.get())), int(str(cboDownSamplingTo.get())) + 1,
                                            int(str(cboDownSamplingStep.get()))):

                    # Resampling on train_valid_test_data_filtered_labels to get the down sampled data
                    print('------------------------------------------------------')
                    print('Start resampling ' + str(resamplingrate) + 'Hz at ' + str(
                        datetime.datetime.now().strftime("%H:%M:%S")))

                    resample_dict = {}
                    for axis in axes_to_apply_functions_list:
                        resample_dict[axis] = function_set_for_resampling

                    resampled_train_data = resampled_frame(train_valid_test_data_filtered_labels, label_set,
                                                           resample_dict,
                                                           axes_to_apply_functions_list, resamplingrate)

                    if monitoring_mode is True:
                        resampled_monitoring_data = resampled_frame(monitoring_data_filtered_labels, label_set,
                                                                    resample_dict,
                                                                    axes_to_apply_functions_list, resamplingrate)
                    print('End resampling ' + str(resamplingrate) + "Hz at " + str(
                        datetime.datetime.now().strftime("%H:%M:%S")))

                    no_of_resampled_train_data_points = len(resampled_train_data.index)
                    # print('No of data points in resampled train data: ' + str(no_of_resampled_train_data_points))

                    path_with_train_valid_test_table_name = './csv_out/' + str(txtTrainTable_text.get())[
                                                                           :18] + '_at_' + timestampforCSVfiles
                    hz_path = '/' + str(resamplingrate) + 'Hz'
                    if not os.path.exists(path_with_train_valid_test_table_name):
                        os.mkdir(path_with_train_valid_test_table_name)
                    if not os.path.exists(path_with_train_valid_test_table_name + hz_path):
                        os.mkdir(path_with_train_valid_test_table_name + hz_path)

                    for window_size in range(int(txtWindowSizeFrom_text.get()), int(txtWindowSizeTo_text.get()) + 1,
                                             int(txtWindowStep_text.get())):
                        timestart = datetime.datetime.now()
                        window_stride_in_ms = math.floor(window_size * int(txtWindowStride_text.get()) / 100)
                        print('-----Window size ' + str(window_size) + 'ms Stride ' + str(window_stride_in_ms))

                        # Begin calculating features for training phrase <-
                        print('Start calculating features for train data at ' + str(
                            datetime.datetime.now().strftime("%H:%M:%S")))
                        if binary_mode is True:
                            # Because the label_set has changed into Lying and Non-Lying in the last window setting -> get the origin labels
                            label_set = label_set_origin.copy()
                            four_labels_label_set = label_set.copy()
                            four_labels_narray_temp = four_labels_label_set.to_numpy(copy=True)
                            four_labels_narray_temp = np.append(four_labels_narray_temp, 'Non-' + main_label)

                        agg_train_valid_test_unfiltered_unbalanced = aggregated_frame(resampled_train_data, label_set,
                                                                                      features_in_dictionary,
                                                                                      axes_to_apply_functions_list,
                                                                                      window_size,
                                                                                      window_stride_in_ms, 1)

                        path_window = '/window_' + str(window_size) + 'ms_stride_' + str(window_stride_in_ms) + 'ms'
                        if not os.path.exists(path_with_train_valid_test_table_name + hz_path + path_window):
                            os.mkdir(path_with_train_valid_test_table_name + hz_path + path_window)
                            os.mkdir(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set')
                            os.mkdir(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data')
                            os.mkdir(path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold')
                        if csv_saving is True:
                            agg_train_valid_test_unfiltered_unbalanced.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set' + '/01_train_valid_test_imbalanced_unfiltered_set_all_instances.csv',
                                header=True)
                        # End calculating features for training phrase <-

                        # filtering the number of data points for each window ->
                        minimum_count_allowed = round((window_size * resamplingrate / 1000) * 0.8)
                        maximum_count_allowed = round((window_size * resamplingrate / 1000) * 1.2)
                        agg_train_valid_test_filtered_unbalanced = agg_train_valid_test_unfiltered_unbalanced.loc[
                            (agg_train_valid_test_unfiltered_unbalanced['count'] >= minimum_count_allowed) & (
                                    agg_train_valid_test_unfiltered_unbalanced['count'] <= maximum_count_allowed)]
                        # filtering the number of data points for each window <-

                        # Splitting the train_valid and test data set with the given proportion ->
                        unbalanced_train_valid_dataset, test_dataset = train_test_split(
                            agg_train_valid_test_filtered_unbalanced,
                            test_size=test_proportion,
                            shuffle=True,
                            stratify=
                            agg_train_valid_test_filtered_unbalanced[
                                ['label']])
                        # Splitting the train_valid and test data set with the given proportion ->

                        # Begin balancing the Train_Valid data set ->
                        minimum_train_valid_instance_for_each_label = unbalanced_train_valid_dataset[
                            'label'].value_counts().min()
                        # print('Number of instances for each label in Train_Valid set ' + str(minimum_train_valid_instance_for_each_label))

                        balanced_train_valid_dataset = pd.DataFrame()
                        # Lying-Non-Lying
                        if binary_mode is True:
                            no_of_labels = 2
                            no_of_main_label_instances = len(unbalanced_train_valid_dataset.loc[
                                                                 unbalanced_train_valid_dataset['label'].isin(
                                                                     [main_label])].index)
                            if (
                                    no_of_main_label_instances // no_of_sub_labels < minimum_train_valid_instance_for_each_label):
                                no_of_each_sub_label_instances = no_of_main_label_instances // no_of_sub_labels
                            else:
                                no_of_each_sub_label_instances = minimum_train_valid_instance_for_each_label

                            # For the purpose of correctly update into database
                            minimum_train_valid_instance_for_each_label = no_of_each_sub_label_instances

                            # Begin adjusting the proportion of Lying Standing Walking Grazing with proportion 3:1:1:1 =>
                            print('Number of instances for ' + main_label + ' label in Train_Valid set: ' + str(
                                no_of_each_sub_label_instances * no_of_sub_labels))
                            print(
                                'Number of instances for (each) sub label in Train_Valid set: ' + str(
                                    no_of_each_sub_label_instances))

                            # Randomly select instances for main label
                            features_eachlabel = unbalanced_train_valid_dataset.loc[
                                unbalanced_train_valid_dataset['label'] == main_label]
                            random_set = features_eachlabel.sample(n=no_of_each_sub_label_instances * no_of_sub_labels,
                                                                   replace=False)
                            balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

                            for index, value in sub_labels_set.items():
                                features_eachlabel = unbalanced_train_valid_dataset.loc[
                                    unbalanced_train_valid_dataset['label'] == value]
                                random_set = features_eachlabel.sample(n=no_of_each_sub_label_instances,
                                                                       replace=False)
                                balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

                            # End adjusting the proportion of Lying Standing Walking Grazing with proportion 3:1:1:1 <=

                            # Change the label set of Train data set into Non-Lying for the other three labels ->

                            for index, value in sub_labels_set.items():
                                balanced_train_valid_dataset.loc[
                                    balanced_train_valid_dataset.label == value, 'label'] = 'Non-' + main_label

                            # Change the label set of Train data set into Non-Lying for the other three labels <-
                            balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')

                            # Get the root list of four activities in Test data set before change Stehen Gehen Grasen into non-liegen
                            # This is for the confusion matrix latter
                            test_dataset = test_dataset.reset_index(drop=True)
                            four_labels_y_root = test_dataset['label'].to_numpy(copy=True)

                            # Change the label set of Test dataset into Non-main label for the other three labels ->

                            for index, value in sub_labels_set.items():
                                test_dataset.loc[test_dataset.label == value, 'label'] = 'Non-' + main_label

                            # Change the label set of Test dataset into Non-Lying for the other three labels <-
                            # Change the label_set into two labels only
                            label_set = pd.Series([main_label, 'Non-' + main_label])
                        else:
                            print('Number of instances for each label in Train_Valid set: ' + str(
                                minimum_train_valid_instance_for_each_label))
                            for eachlabel in label_set:
                                features_eachlabel = unbalanced_train_valid_dataset.loc[
                                    unbalanced_train_valid_dataset['label'] == eachlabel]
                                if len(features_eachlabel) == minimum_train_valid_instance_for_each_label:
                                    balanced_train_valid_dataset = balanced_train_valid_dataset.append(
                                        features_eachlabel)
                                else:
                                    random_set = features_eachlabel.sample(
                                        n=minimum_train_valid_instance_for_each_label,
                                        replace=False)
                                    balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)
                            balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')

                        if csv_saving is True:
                            balanced_train_valid_dataset.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set' + '/02_train_valid_balanced_filtered_dataset_with_' + str(
                                    minimum_train_valid_instance_for_each_label) + '_instances_for_each_class.csv',
                                header=True)

                        balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['cattle_id'], axis=1)
                        balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['count'], axis=1)
                        balanced_train_valid_dataset = balanced_train_valid_dataset.reset_index(drop=True)
                        # End balancing the Train_Valid data set <-

                        # Write result to a text file ->
                        text_file = open(
                            path_with_train_valid_test_table_name + hz_path + path_window + '/Train_Test_Result.txt',
                            'w', encoding='utf-8')
                        text_file.write('Train_Valid_Test DB table: ' + txtTrainTable_text.get())
                        text_file.write('\nMonitoring         DB table: ' + txtMonitoringTable_Text.get())
                        text_file.write('\n' + str(path_window))

                        text_file.write("\nLabels to predict: " + label_set.str.cat(sep=' '))
                        text_file.write("\nFunctions list      : " + str(agg_function_list_names))
                        text_file.write("\nAxes list             : " + str(axes_to_apply_functions_list))

                        # Preseting some of metrics for the classification
                        rf_accu = 0
                        rf_prec = 0
                        rf_recal = 0
                        rf_spec = 0
                        rf_f1 = 0

                        dt_accu = 0
                        dt_prec = 0
                        dt_recal = 0
                        dt_spec = 0
                        dt_f1 = 0

                        svm_accu = 0
                        svm_prec = 0
                        svm_recal = 0
                        svm_spec = 0
                        svm_f1 = 0

                        nb_accu = 0
                        nb_prec = 0
                        nb_recal = 0
                        nb_spec = 0
                        nb_f1 = 0

                        feature_cols = list(balanced_train_valid_dataset.columns.values)
                        feature_cols.remove('label')

                        X = balanced_train_valid_dataset[feature_cols]  # Features
                        y = balanced_train_valid_dataset['label']  # Target

                        # Stratified k-fold
                        kf = StratifiedKFold(n_splits=kfold, shuffle=True)  # Considering random_state = 0??
                        k_fold_round = int(0)
                        for train_index, valid_index in kf.split(X, y):
                            k_fold_round = k_fold_round + 1
                            # print('Round ' + str(k_fold_round))
                            X_train = pd.DataFrame(X, columns=feature_cols, index=train_index)
                            X_valid = pd.DataFrame(X, columns=feature_cols, index=valid_index)

                            if csv_saving is True:
                                X_train.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                        k_fold_round) + 'th_round_fold_X_train.csv',
                                    header=True)
                                X_valid.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                        k_fold_round) + 'th_round_fold_X_validation.csv',
                                    header=True)

                            y_train_df = pd.DataFrame(y, columns=['label'], index=train_index)
                            y_train = y_train_df['label']
                            y_valid_df = pd.DataFrame(y, columns=['label'], index=valid_index)
                            y_valid = y_valid_df['label']

                            if csv_saving is True:
                                y_train.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                        k_fold_round) + 'th_round_fold_y_train.csv',
                                    header=True)
                                y_valid.to_csv(
                                    path_with_train_valid_test_table_name + hz_path + path_window + '/3_kfold/' + str(
                                        k_fold_round) + 'th_round_fold_y_validation.csv',
                                    header=True)

                            text_file.write("\n------------------Round " + str(k_fold_round) + "------------------")

                            if RandomForestVar.get() == 1:
                                clf_rf.fit(X_train, y_train)
                                y_pred_rf = clf_rf.predict(X_valid)

                                temp_acc = round(metrics.accuracy_score(y_valid, y_pred_rf), 4)
                                rf_accu = rf_accu + temp_acc
                                text_file.write("\nRandom Forest accuracy: " + str(temp_acc))
                                # print('Random Forest accuracy ' + str(temp_acc))

                                if (label_set.size == 2):
                                    rf_prec = rf_prec + metrics.precision_score(y_valid, y_pred_rf, average="binary",
                                                                                pos_label=label_set[0])
                                    rf_recal = rf_recal + metrics.recall_score(y_valid, y_pred_rf, average="binary",
                                                                               pos_label=label_set[0])
                                    rf_f1 = rf_f1 + metrics.recall_score(y_valid, y_pred_rf, average="binary",
                                                                         pos_label=label_set[0])

                            if DecisionTreeVar.get() == 1:
                                clf_dt.fit(X_train, y_train)
                                y_pred_dt = clf_dt.predict(X_valid)

                                temp_acc = round(metrics.accuracy_score(y_valid, y_pred_dt), 4)
                                dt_accu = dt_accu + temp_acc

                                text_file.write("\nDecision Tree accuracy: " + str(temp_acc))
                                # print('Decision Tree accuracy ' + str(temp_acc))

                                if (label_set.size == 2):
                                    dt_prec = dt_prec + metrics.precision_score(y_valid, y_pred_dt, average="binary",
                                                                                pos_label=label_set[0])
                                    dt_recal = dt_recal + metrics.recall_score(y_valid, y_pred_dt, average="binary",
                                                                               pos_label=label_set[0])
                                    dt_f1 = dt_f1 + metrics.recall_score(y_valid, y_pred_dt, average="binary",
                                                                         pos_label=label_set[0])

                            if SVMVar.get() == 1:
                                clf_svm.fit(X_train, y_train)
                                y_pred_svm = clf_svm.predict(X_valid)

                                temp_acc = round(metrics.accuracy_score(y_valid, y_pred_svm), 4)
                                svm_accu = svm_accu + temp_acc

                                text_file.write("\nSVM accuracy: " + str(temp_acc))
                                # print('SVM Accuracy ' + str(temp_acc))

                                if (label_set.size == 2):
                                    svm_prec = svm_prec + metrics.precision_score(y_valid, y_pred_svm, average="binary",
                                                                                  pos_label=label_set[0])
                                    svm_recal = svm_recal + metrics.recall_score(y_valid, y_pred_svm, average="binary",
                                                                                 pos_label=label_set[0])
                                    svm_f1 = svm_f1 + metrics.recall_score(y_valid, y_pred_svm, average="binary",
                                                                           pos_label=label_set[0])

                            if NaiveBayesVar.get() == 1:
                                clf_nb.fit(X_train, y_train)
                                y_pred_nb = clf_nb.predict(X_valid)

                                temp_acc = round(metrics.accuracy_score(y_valid, y_pred_nb), 4)
                                nb_accu = nb_accu + temp_acc

                                text_file.write("\nNaive Bayes accuracy: " + str(temp_acc))
                                # print('Naive Bayes accuracy ' + str(temp_acc))

                                if (label_set.size == 2):
                                    nb_prec = nb_prec + metrics.precision_score(y_valid, y_pred_nb, average="binary",
                                                                                pos_label=label_set[0])
                                    nb_recal = nb_recal + metrics.recall_score(y_valid, y_pred_nb, average="binary",
                                                                               pos_label=label_set[0])
                                    nb_f1 = nb_f1 + metrics.recall_score(y_valid, y_pred_nb, average="binary",
                                                                         pos_label=label_set[0])

                        text_file.write("\n------------------------------------------------------")
                        if RandomForestVar.get() == 1:
                            text_file.write(
                                "\nTrain_Valid average accuracy of Random Forest: " + str(round((rf_accu / kfold), 4)))
                            print('Train_Valid average accuracy of Random Forest ' + str(round((rf_accu / kfold), 4)))

                        if DecisionTreeVar.get() == 1:
                            text_file.write(
                                "\nTrain_Valid average accuracy of Decision Tree: " + str(round((dt_accu / kfold), 4)))
                            print('Train_Valid average accuracy of Decision Tree ' + str(round((dt_accu / kfold), 4)))

                        if SVMVar.get() == 1:
                            text_file.write(
                                "\nTrain_Valid average accuracy of SVM: " + str(round((svm_accu / kfold), 4)))
                            print('Train_Valid average accuracy of SVM           ' + str(round((svm_accu / kfold), 4)))

                        if NaiveBayesVar.get() == 1:
                            text_file.write(
                                "\nTrain_Valid Average accuracy of Naive Bayes: " + str(round((nb_accu / kfold), 4)))
                            print('Train_Valid average accuracy of Naive Bayes   ' + str(round((nb_accu / kfold), 4)))

                        # Begin generating data for Test dataset =>
                        print("----------------------Test Result---------------------")

                        rf_accu_test = 0
                        rf_prec_test = 0
                        rf_recal_test = 0
                        rf_spec_test = 0
                        rf_f1_test = 0

                        dt_accu_test = 0
                        dt_prec_test = 0
                        dt_recal_test = 0
                        dt_spec_test = 0
                        dt_f1_test = 0

                        svm_accu_test = 0
                        svm_prec_test = 0
                        svm_recal_test = 0
                        svm_spec_test = 0
                        svm_f1_test = 0

                        nb_accu_test = 0
                        nb_prec_test = 0
                        nb_recal_test = 0
                        nb_spec_test = 0
                        nb_f1_test = 0

                        labels_narray_temp = label_set.to_numpy()
                        test_dataset = test_dataset.dropna().set_index('timestamp')
                        if csv_saving is True:
                            test_dataset.to_csv(
                                path_with_train_valid_test_table_name + hz_path + path_window + '/1_train_valid_test_set/03_test_dataset_counts_filtered.csv',
                                header=True)

                        test_dataset = test_dataset.drop(['cattle_id'], axis=1)
                        test_dataset = test_dataset.drop(['count'], axis=1)
                        test_dataset = test_dataset.reset_index(drop=True)

                        X_test = test_dataset[feature_cols]  # Features
                        y_test = test_dataset['label']  # Target

                        if RandomForestVar.get() == 1:
                            y_test_pred_rf = clf_rf.predict(X_test)
                            rf_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_rf), 4)

                            print('Accuracy of Random Forest on Test data:' + str(rf_accu_test))

                            # Begin saving RF Confusion Matrix into text file ->
                            text_file.write("\n------------------------------------------------------")
                            text_file.write("\nAccuracy of Random Forest on Test data: " + str(rf_accu_test))
                            text_file.write("\nRandom Forest - Confusion matrix on Test data")
                            text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")

                            confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_rf,
                                                                             labels=labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            # End Saving RF Confusion Matrix into text file <-

                            # Begin printing RF Confusion Matrix to console ->
                            print('Confusion matrix on Test data')
                            print('Predicted ' + label_set.str.cat(sep=' '))
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            # End Printing RF Confusion Matrix to console <-

                            if label_set.size == 2:
                                rf_prec_test = round(metrics.precision_score(y_test, y_test_pred_rf, average="binary",
                                                                             pos_label=label_set[0]), 4)
                                rf_recal_test = round(metrics.recall_score(y_test, y_test_pred_rf, average="binary",
                                                                           pos_label=label_set[0]), 4)
                                rf_f1_test = round(metrics.recall_score(y_test, y_test_pred_rf, average="binary",
                                                                        pos_label=label_set[0]), 4)
                            else:
                                rf_prec_test = 0
                                rf_recal_test = 0
                                rf_f1_test = 0

                        if DecisionTreeVar.get() == 1:
                            y_test_pred_dt = clf_dt.predict(X_test)
                            dt_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_dt), 4)

                            print('Accuracy of Decistion Tree on Test data:' + str(dt_accu_test))

                            # Begin saving DT Confusion Matrix into text file ->
                            text_file.write("\n------------------------------------------------------")
                            text_file.write("\nAccuracy of Decistion Tree on Test data: " + str(dt_accu_test))
                            text_file.write("\nDecision Tree - Confusion matrix on Test data")
                            text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                            confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_dt,
                                                                             labels=labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            print(confusion_matrix_temp)
                            # End saving DT Confusion Matrix into text file <-

                            if (label_set.size == 2):
                                dt_prec_test = round(metrics.precision_score(y_test, y_test_pred_dt, average="binary",
                                                                             pos_label=label_set[0]), 4)
                                dt_recal_test = round(metrics.recall_score(y_test, y_test_pred_dt, average="binary",
                                                                           pos_label=label_set[0]), 4)
                                dt_f1_test = round(metrics.recall_score(y_test, y_test_pred_dt, average="binary",
                                                                        pos_label=label_set[0]), 4)
                            else:
                                dt_prec_test = 0
                                dt_recal_test = 0
                                dt_f1_test = 0

                        if SVMVar.get() == 1:
                            y_test_pred_svm = clf_svm.predict(X_test)
                            svm_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_svm), 4)

                            print('Accuracy of SVM on Test data:' + str(svm_accu_test))

                            # Begin saving SVM Confusion Matrix into text file ->
                            text_file.write("\n------------------------------------------------------")
                            text_file.write("\nAccuracy of SVM on Test data: " + str(svm_accu_test))
                            text_file.write("\nSVM - Confusion matrix on Test data")
                            text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                            confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_svm,
                                                                             labels=labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            print(confusion_matrix_temp)
                            # End saving SVM Confusion Matrix into text file <-

                            if (label_set.size == 2):
                                svm_prec_test = round(metrics.precision_score(y_test, y_test_pred_svm, average="binary",
                                                                              pos_label=label_set[0]), 4)
                                svm_recal_test = round(metrics.recall_score(y_test, y_test_pred_svm, average="binary",
                                                                            pos_label=label_set[0]), 4)
                                svm_f1_test = round(metrics.recall_score(y_test, y_test_pred_svm, average="binary",
                                                                         pos_label=label_set[0]), 4)
                            else:
                                svm_prec_test = 0
                                svm_recal_test = 0
                                svm_f1_test = 0

                        if NaiveBayesVar.get() == 1:
                            y_test_pred_nb = clf_nb.predict(X_test)
                            nb_accu_test = round(metrics.accuracy_score(y_test, y_test_pred_nb), 4)

                            print('Accuracy of Naive Bayes on Test data:' + str(nb_accu_test))

                            # Begin saving Naive Bayes Confusion Matrix into text file->
                            text_file.write("\n------------------------------------------------------")
                            text_file.write("\nAccuracy of Naive Bayes on Testing data: " + str(nb_accu_test))
                            text_file.write("\nNaive Bayes - Confusion matrix on Test data")
                            text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")
                            confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_nb,
                                                                             labels=labels_narray_temp)
                            lbl_no = 0
                            for line in confusion_matrix_temp:
                                text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                lbl_no = lbl_no + 1
                            print(confusion_matrix_temp)
                            # End saving SVM Confusion Matrix into text file <-

                            if (label_set.size == 2):
                                nb_prec_test = round(metrics.precision_score(y_test, y_test_pred_nb, average="binary",
                                                                             pos_label=label_set[0]), 4)
                                nb_recal_test = round(metrics.recall_score(y_test, y_test_pred_nb, average="binary",
                                                                           pos_label=label_set[0]), 4)
                                nb_f1_test = round(metrics.recall_score(y_test, y_test_pred_nb, average="binary",
                                                                        pos_label=label_set[0]), 4)
                            else:
                                nb_prec_test = 0
                                nb_recal_test = 0
                                nb_f1_test = 0
                        print("------------------------------------------------------")

                        # --------checking whether user want to monitor and show staticstics
                        # For monitoring metrics
                        rf_accu_monitor = 0
                        rf_prec_monitor = 0
                        rf_recal_monitor = 0
                        rf_spec_monitor = 0
                        rf_f1_monitor = 0

                        dt_accu_monitor = 0
                        dt_prec_monitor = 0
                        dt_recal_monitor = 0
                        dt_spec_monitor = 0
                        dt_f1_monitor = 0

                        svm_accu_monitor = 0
                        svm_prec_monitor = 0
                        svm_recal_monitor = 0
                        svm_spec_monitor = 0
                        svm_f1_monitor = 0

                        nb_accu_monitor = 0
                        nb_prec_monitor = 0
                        nb_recal_monitor = 0
                        nb_spec_monitor = 0
                        nb_f1_monitor = 0

                        if monitoring_mode is True:
                            # Begin calculating the features for the monitoring and simulator  ->
                            print('--------------------------------------------')
                            print('Begin calculating features for the Monitoring data at ' + str(
                                datetime.datetime.now().strftime("%H:%M:%S")))
                            simulation_window_stride_in_ms = math.floor(
                                window_size * int(txtWindowSimuStride_text.get()) / 100)

                            if binary_mode is True:
                                # Because the label_set has changed into Lying and Non-Lying in the last window setting -> get the origin labels
                                label_set = label_set_origin.copy()

                            if monitor_data_generate is True:
                                # agg_monitor dataframe returns data without label (for predict file generation)
                                agg_monitor = aggregated_unseen_data(resampled_monitoring_data, label_set,
                                                                     features_in_dictionary,
                                                                     axes_to_apply_functions_list, window_size,
                                                                     simulation_window_stride_in_ms, 0)

                            # To be set for saving or not ->
                            if monitor_data_generate is True:
                                if csv_saving is True:
                                    agg_monitor.to_csv(
                                        path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/0_unfiltered_monitoring_set.csv',
                                        header=True)
                            # End calculating the features for the monitoring and simulator  <-

                            # Filtering the number of data points for each window in test and monitoring data ->
                            # counts_filtered_monitoring_dataset = agg_monitor.loc[
                            #     (agg_monitor['count'] >= minimum_count_allowed) & (
                            #             agg_monitor['count'] <= maximum_count_allowed)]
                            # Filtering the number of data points for each window in test and monitoring data<-

                            # Saving instances for monitoring ->
                            # if csv_saving is True:
                            #     counts_filtered_monitoring_dataset.to_csv(
                            #         path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/1_filtered_monitoring_set.csv',
                            #         header=True)
                            # Saving instances for monitoring <-

                            # counts_filtered_monitoring_dataset = counts_filtered_monitoring_dataset.dropna().reset_index(
                            #     drop=True)

                            # This dataframe is for testing on unseen data (from monitoring data table) ->
                            agg_monitor_temp = aggregated_frame(resampled_monitoring_data, label_set,
                                                                features_in_dictionary,
                                                                axes_to_apply_functions_list, window_size,
                                                                window_stride_in_ms, 1)

                            print('End calculating features for the Monitoring data at ' + str(
                                datetime.datetime.now().strftime("%H:%M:%S")))

                            counts_filtered_monitoring_dataset_temp = agg_monitor_temp.loc[
                                (agg_monitor_temp['count'] >= minimum_count_allowed) & (
                                        agg_monitor_temp['count'] <= maximum_count_allowed)]
                            counts_filtered_monitoring_dataset_temp = counts_filtered_monitoring_dataset_temp.dropna().reset_index(
                                drop=True)

                            if binary_mode is True:
                                # Get the root list of four activities in Test data set before change Stehen Gehen Grasen into non-liegen
                                # This is for the confusion matrix latter
                                four_labels_y_root_monitoring_temp = counts_filtered_monitoring_dataset_temp[
                                    'label'].to_numpy(copy=True)

                                # Change the label set of Test dataset into Non-Lying for the other sub labels ->

                                for index, value in sub_labels_set.items():
                                    counts_filtered_monitoring_dataset_temp.loc[
                                        counts_filtered_monitoring_dataset_temp.label == value, 'label'] = 'Non-' + main_label

                                # Change the label set of Test dataset into Non-Lying for the other three labels <-

                                # Change the label_set into two labels only
                                label_set = pd.Series([main_label, 'Non-' + main_label])

                            # X_monitor = counts_filtered_monitoring_dataset[feature_cols]  # Features
                            if monitor_data_generate is True:
                                X_monitor = agg_monitor[feature_cols]  # Features
                            y_monitor_temp = counts_filtered_monitoring_dataset_temp['label']
                            X_monitor_temp = counts_filtered_monitoring_dataset_temp[feature_cols]  # Features
                            # y_monitor_temp = agg_monitor_temp['label']
                            # X_monitor_temp = agg_monitor_temp[feature_cols]  # Features

                            # Monitoring part ==================>
                            # Predict and generate data for monitoring
                            if RandomForestVar.get() == 1:
                                if monitor_data_generate is True:
                                    y_monitor_pred_rf = clf_rf.predict(X_monitor)
                                    simu_predicted_df_rf = pd.concat(
                                        [agg_monitor[['timestamp']],
                                         pd.DataFrame(y_monitor_pred_rf)],
                                        axis=1)
                                    simu_predicted_df_rf.columns = ['timestamp', 'predicted_label']
                                    simu_predicted_df_rf.to_csv(
                                        path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/2_monitor_predicted_rf.csv',
                                        header=True)

                                y_monitor_pred_rf_temp = clf_rf.predict(X_monitor_temp)
                                rf_accu_monitor = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_rf_temp),
                                                        4)
                                print('Random Forest - Accuracy on Monitor data ' + str(rf_accu_monitor))

                                if label_set.size == 2:
                                    rf_prec_monitor = round(
                                        metrics.precision_score(y_monitor_temp, y_monitor_pred_rf_temp,
                                                                average="binary",
                                                                pos_label=label_set[0]), 4)
                                    rf_recal_monitor = round(
                                        metrics.recall_score(y_monitor_temp, y_monitor_pred_rf_temp, average="binary",
                                                             pos_label=label_set[0]), 4)
                                    rf_f1_monitor = round(
                                        rf_f1_monitor + metrics.recall_score(y_monitor_temp, y_monitor_pred_rf_temp,
                                                                             average="binary",
                                                                             pos_label=label_set[0]), 4)
                                else:
                                    rf_prec_monitor = 0
                                    rf_recal_monitor = 0
                                    rf_f1_monitor = 0

                                # Begin saving RF Confusion Matrix into text file ->
                                text_file.write("\n------------------------------------------------------")
                                text_file.write(
                                    "\nAccuracy of Random Forest on monitoring data: " + str(rf_accu_monitor))
                                text_file.write("\nRandom Forest Confusion matrix on monitoring data")
                                text_file.write("\nPredicted \u2193" + label_set.str.cat(sep=' ') + " \u2193")

                                confusion_matrix_temp = metrics.confusion_matrix(y_monitor_temp,
                                                                                 y_monitor_pred_rf_temp,
                                                                                 labels=labels_narray_temp)
                                lbl_no = 0
                                for line in confusion_matrix_temp:
                                    text_file.write("\n\u2192True " + labels_narray_temp[lbl_no] + ' ' + str(line))
                                    lbl_no = lbl_no + 1

                                # Begin printing RF Confusion Matrix to console ->
                                print('Random Forest - Confusion matrix on Monitoring data')
                                print('Predicted ' + label_set.str.cat(sep=' '))
                                lbl_no = 0
                                for line in confusion_matrix_temp:
                                    print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                                    lbl_no = lbl_no + 1
                                # End Printing RF Confusion Matrix to console <-
                                # End Saving RF Confusion Matrix into text file <-

                                # Additional confusion matrix for sub labels in case of Lying and Nonlying mode ->
                                if binary_mode is True:
                                    # Showing confusion matrix for sub labelss
                                    text_file.write(
                                        "\nRandom Forest Confusion matrix on monitoring data (" + str(
                                            no_of_sub_labels + 1) + " labels) ")
                                    text_file.write("\nPredicted \u2193" + four_labels_label_set.str.cat(
                                        sep=' ') + ' Non-' + main_label + " \u2193")

                                    confusion_matrix_temp = metrics.confusion_matrix(
                                        four_labels_y_root_monitoring_temp, y_monitor_pred_rf_temp,
                                        labels=four_labels_narray_temp)
                                    lbl_no = 0
                                    for line in confusion_matrix_temp:
                                        text_file.write(
                                            "\n\u2192True " + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                        lbl_no = lbl_no + 1

                                    # Begin printing RF Confusion Matrix to console ->
                                    print('------------')
                                    print('Random Forest - Confusion matrix on monitoring data (' + str(
                                        no_of_sub_labels + 1) + ' labels) ')
                                    print('Predicted ' + four_labels_label_set.str.cat(sep=' ') + ' Non-liegen')
                                    lbl_no = 0
                                    for line in confusion_matrix_temp:
                                        print('True ' + four_labels_narray_temp[lbl_no] + ' ' + str(line))
                                        lbl_no = lbl_no + 1
                                # Additional confusion matrix for sub labels in case of Lying and Nonlying mode <

                            if DecisionTreeVar.get() == 1:
                                if monitor_data_generate is True:
                                    y_monitor_pred_dt = clf_dt.predict(X_monitor)
                                    simu_predicted_df_dt = pd.concat(
                                        [agg_monitor[['timestamp']],
                                         pd.DataFrame(y_monitor_pred_dt)],
                                        axis=1)
                                    simu_predicted_df_dt.columns = ['timestamp', 'predicted_label']
                                    simu_predicted_df_dt.to_csv(
                                        path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/3_monitor_predicted_dt.csv',
                                        header=True)

                            if SVMVar.get() == 1:
                                if monitor_data_generate is True:
                                    y_monitor_pred_svm = clf_svm.predict(X_monitor)
                                    simu_predicted_df_svm = pd.concat(
                                        [agg_monitor[['timestamp']],
                                         pd.DataFrame(y_monitor_pred_svm)],
                                        axis=1)
                                    simu_predicted_df_svm.columns = ['timestamp', 'predicted_label']
                                    simu_predicted_df_svm.to_csv(
                                        path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/4_monitor_predicted_svm.csv',
                                        header=True)

                            if NaiveBayesVar.get() == 1:
                                if monitor_data_generate is True:
                                    y_monitor_pred_nb = clf_nb.predict(X_monitor)
                                    simu_predicted_df_nb = pd.concat(
                                        [agg_monitor[['timestamp']],
                                         pd.DataFrame(y_monitor_pred_nb)],
                                        axis=1)
                                    simu_predicted_df_nb.columns = ['timestamp', 'predicted_label']
                                    simu_predicted_df_nb.to_csv(
                                        path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/5_monitor_predicted_nb.csv',
                                        header=True)
                                # Monitoring part ==================<

                        text_file.close()

                        # Write/ insert experiment result into DB =>
                        slqInsertQuery = 'INSERT INTO ' + txtResultTable_Text.get() + '(model_title, model_init_name, model_binary_content, features_json_content, model_comments, train_table, monitor_table, no_of_predicted_classes, list_of_predicted_classes, original_sample_rate_in_hz, no_of_original_train_data_points, resampled_rate_in_hz, no_of_resampled_train_data_points, no_of_instances_for_each_class_in_resampled_train_table, algorithm, no_of_functions, list_of_functions, no_of_axes, list_of_axes, window_size, window_stride,k_fold, accuracy_train_valid, precision_train_valid, recall_train_valid, specificity_train_valid, f1_train_valid, accuracy_test, precision_test, recall_test, specificity_test, f1_test, monitoring_window_stride, accuracy_monitor, precision_monitor, recall_monitor, specificity_monitor, f1_monitor, start_time, end_time, running_time_in_minutes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

                        monitor_table_name_temp = txtMonitoringTable_Text.get()
                        if monitoring_mode is False:
                            monitor_table_name_temp = 'n.a'

                        if RandomForestVar.get() == 1:
                            timeend = datetime.datetime.now()
                            model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                            if binary_mode is True:
                                model_init_Name = model_init_Name + '_Binary'
                            else:
                                model_init_Name = model_init_Name + '_Multi'
                            model_init_Name = model_init_Name + '_RandomForest'
                            model_data_content = pickle.dumps(clf_rf)  # pickle the model

                            duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                            record_to_insert = (
                                '', model_init_Name, psycopg2.Binary(model_data_content),
                                json.dumps(functions_labels_json),
                                model_comment,
                                txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                                label_set.str.cat(sep='_'),
                                original_sampling_rate, no_of_original_train_valid_test_data_points,
                                resamplingrate, no_of_resampled_train_data_points,
                                str(minimum_train_valid_instance_for_each_label), 'Random Forest',
                                no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                                '_'.join(axes_to_apply_functions_list), window_size,
                                txtWindowStride_text.get() + "%", kfold, round(rf_accu / kfold, 4),
                                round(rf_prec / kfold, 4), round(rf_recal / kfold, 4),
                                round(rf_spec / kfold, 4), round(rf_f1 / kfold, 4), rf_accu_test,
                                rf_prec_test,
                                rf_recal_test, rf_spec_test, rf_f1_test,
                                txtWindowSimuStride_text.get() + "%",
                                rf_accu_monitor, rf_prec_monitor, rf_recal_monitor, rf_spec_monitor,
                                rf_f1_monitor,
                                str(timestart), str(timeend), duration)

                            cur.execute(slqInsertQuery, record_to_insert)
                            conn.commit()

                        if DecisionTreeVar.get() == 1:
                            timeend = datetime.datetime.now()
                            model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                            if binary_mode is True:
                                model_init_Name = model_init_Name + '_Binary'
                            else:
                                model_init_Name = model_init_Name + '_Multi'
                            model_init_Name = model_init_Name + '_DecisionTree'

                            model_data_content = pickle.dumps(clf_dt)  # pickle the model
                            duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                            record_to_insert = (
                                '', model_init_Name, psycopg2.Binary(model_data_content),
                                json.dumps(functions_labels_json),
                                model_comment,
                                txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                                label_set.str.cat(sep='_'),
                                original_sampling_rate, no_of_original_train_valid_test_data_points,
                                resamplingrate, no_of_resampled_train_data_points,
                                str(minimum_train_valid_instance_for_each_label), 'Decision Tree',
                                no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                                '_'.join(axes_to_apply_functions_list), window_size,
                                txtWindowStride_text.get() + "%", kfold, round(dt_accu / kfold, 4),
                                round(dt_prec / kfold, 4), round(dt_recal / kfold, 4),
                                round(dt_spec / kfold, 4), round(dt_f1 / kfold, 4), dt_accu_test,
                                dt_prec_test,
                                dt_recal_test, dt_spec_test, dt_f1_test,
                                txtWindowSimuStride_text.get() + "%",
                                dt_accu_monitor, dt_prec_monitor, dt_recal_monitor, dt_spec_monitor,
                                dt_f1_monitor,
                                str(timestart), str(timeend), duration)

                            cur.execute(slqInsertQuery, record_to_insert)
                            conn.commit()

                        if SVMVar.get() == 1:
                            timeend = datetime.datetime.now()
                            model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                            if binary_mode is True:
                                model_init_Name = model_init_Name + '_Binary'
                            else:
                                model_init_Name = model_init_Name + '_Multi'
                            model_init_Name = model_init_Name + '_SVM'
                            model_data_content = pickle.dumps(clf_svm)  # pickle the model

                            duration = round(((timeend - timestart).total_seconds()) / 60, 2)
                            record_to_insert = (
                                '', model_init_Name, psycopg2.Binary(model_data_content),
                                json.dumps(functions_labels_json),
                                model_comment,
                                txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                                label_set.str.cat(sep='_'),
                                original_sampling_rate, no_of_original_train_valid_test_data_points,
                                resamplingrate, no_of_resampled_train_data_points,
                                str(minimum_train_valid_instance_for_each_label), 'SVM',
                                no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                                '_'.join(axes_to_apply_functions_list), window_size,
                                txtWindowStride_text.get() + "%", kfold, round(svm_accu / kfold, 4),
                                round(svm_prec / kfold, 4), round(svm_recal / kfold, 4),
                                round(svm_spec / kfold, 4), round(svm_f1 / kfold, 4), svm_accu_test,
                                svm_prec_test,
                                svm_recal_test, svm_spec_test, svm_f1_test,
                                txtWindowSimuStride_text.get() + "%",
                                svm_accu_monitor, svm_prec_monitor, svm_recal_monitor, svm_spec_monitor,
                                svm_f1_monitor,
                                str(timestart), str(timeend), duration)

                            cur.execute(slqInsertQuery, record_to_insert)
                            conn.commit()

                        if NaiveBayesVar.get() == 1:
                            timeend = datetime.datetime.now()
                            model_init_Name = '%04d' % timeend.year + '%02d' % timeend.month + '%02d' % timeend.day + '_' + '%02d' % timeend.hour + '%02d' % timeend.minute + '%02d' % timeend.second + '_' + txtUser_text.get()
                            if binary_mode is True:
                                model_init_Name = model_init_Name + '_Binary'
                            else:
                                model_init_Name = model_init_Name + '_Multi'
                            model_init_Name = model_init_Name + '_NaiveBayes'

                            model_data_content = pickle.dumps(clf_nb)  # pickle the model
                            duration = round(((timeend - timestart).total_seconds()) / 60, 3)
                            record_to_insert = (
                                '', model_init_Name, psycopg2.Binary(model_data_content),
                                json.dumps(functions_labels_json),
                                model_comment,
                                txtTrainTable_text.get(), monitor_table_name_temp, no_of_labels,
                                label_set.str.cat(sep='_'),
                                original_sampling_rate, no_of_original_train_valid_test_data_points,
                                resamplingrate, no_of_resampled_train_data_points,
                                str(minimum_train_valid_instance_for_each_label), 'Naive Bayes',
                                no_of_functions, '_'.join(agg_function_list_names), no_of_axes,
                                '_'.join(axes_to_apply_functions_list), window_size,
                                txtWindowStride_text.get() + "%", kfold, round(nb_accu / kfold, 4),
                                round(nb_prec / kfold, 4), round(nb_recal / kfold, 4),
                                round(nb_spec / kfold, 4), round(nb_f1 / kfold, 4), nb_accu_test,
                                nb_prec_test,
                                nb_recal_test, nb_spec_test, nb_f1_test,
                                txtWindowSimuStride_text.get() + "%",
                                nb_accu_monitor, nb_prec_monitor, nb_recal_monitor, nb_spec_monitor,
                                nb_f1_monitor,
                                str(timestart), str(timeend), duration)

                            cur.execute(slqInsertQuery, record_to_insert)
                            conn.commit()
                            # End training phrase for original tables <-

            # close the communication with the PostgreSQL
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            messagebox.showinfo("Database", error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')
        write_apps_ini_files()
        winsound.Beep(1000, 300)
        print('End at ' + str(datetime.datetime.now().strftime("%H:%M:%S")))


# <= end main section for fitting

# Generate prediction data frame for simulating dataset
def monitoring_data_generate(simulation_sample_rate, window_size, predicted_df):
    global monitoring_db_table
    global monitoring_db_table_resampled_monitor
    global original_sampling_rate
    global label_set
    global binary_mode
    global function_set_for_resampling

    if binary_mode is True:
        for index, value in sub_labels_set.items():
            monitoring_db_table.loc[
                monitoring_db_table.label == value, 'label'] = 'Non-' + main_label

    temp_monitor_data_set = monitoring_db_table.loc[monitoring_db_table['label'].isin(label_set)].sort_values(
        by=['timestamp'],
        ascending=True)
    axes_to_apply_functions_list = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
    monitoring_data_set = temp_monitor_data_set[['label'] + axes_to_apply_functions_list + ['timestamp']]

    resample_dict = {}
    for axis in axes_to_apply_functions_list:
        resample_dict[axis] = function_set_for_resampling

    if original_sampling_rate != simulation_sample_rate:
        monitoring_db_table_resampled_monitor = resampled_frame(monitoring_data_set, label_set,
                                                                resample_dict,
                                                                axes_to_apply_functions_list,
                                                                simulation_sample_rate)
    else:
        monitoring_db_table_resampled_monitor = monitoring_data_set

    # print("len of: monitoring_db_table_resampled_monitor")
    # print(len(monitoring_db_table_resampled_monitor.index))

    result = pd.DataFrame()

    for index, row in predicted_df.iterrows():
        start_time = row['timestamp']
        predicted_temp = row['predicted_label']
        df_temp = monitoring_db_table_resampled_monitor.loc[
            (monitoring_db_table_resampled_monitor['timestamp'] >= start_time) & (
                    monitoring_db_table_resampled_monitor['timestamp'] < start_time + window_size)]
        df_temp = df_temp.assign(predicted_label=predicted_temp)
        result = result.append(df_temp)
    # print("len of monitoring_data_fr")
    # print(len(result.index))
    if binary_mode is True:
        for index, value in sub_labels_set.items():
            result.loc[
                result.label == value, 'label'] = 'Non-' + main_label

    return result


def validate_monitoring_run():
    global monitoring_data_fr
    global predicted_data_fr
    global monitoring_time_deviation_fr
    global monitoring_error_types_fr
    global curr_monitoring_algorithm
    global curr_monitoring_window_size
    global curr_monitoring_sampling_rate
    global timestampforCSVfiles
    global label_set

    selected_monitor_window_size = int(txtSimulationWindowSize_text.get())
    selected_monitoring_sample_rate = int(cboSimulationSampleRate.get())
    algorithm = str(cboSimuAlgorithm.get())

    validtorun = True
    if ((len(monitoring_data_fr.index) == 0) or (len(predicted_data_fr.index) == 0) or (
            curr_monitoring_window_size != selected_monitor_window_size) or (
            curr_monitoring_algorithm != algorithm) or (
            curr_monitoring_sampling_rate != selected_monitoring_sample_rate)):
        # User changed one of these parameters for the simulation

        window_stride_in_ms = math.floor(selected_monitor_window_size * int(txtWindowStride_text.get()) / 100)
        path_with_train_valid_test_table_name = './csv_out/' + str(txtTrainTable_text.get())[
                                                               :18] + '_at_' + timestampforCSVfiles
        hz_path = '/' + str(selected_monitoring_sample_rate) + 'Hz'
        path_window = '/window_' + str(selected_monitor_window_size) + 'ms_stride_' + str(window_stride_in_ms) + 'ms'

        predicted_file = ''
        if algorithm == 'Random_Forest':
            predicted_file = '/2_monitor_predicted_rf.csv'
        if algorithm == 'Decision_Tree':
            predicted_file = '/3_monitor_predicted_dt.csv'
        if algorithm == 'SVM':
            predicted_file = '/4_monitor_predicted_svm.csv'
        if algorithm == 'Naive_Bayes':
            predicted_file = '/5_monitor_predicted_nb.csv'

        file_name = path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data' + predicted_file

        if os.path.isfile(file_name):
            # Get the predicted data file location for the monitoring/staticstics
            predicted_data_fr = pd.read_csv(file_name).sort_values(by=['timestamp'], ascending=True)
            write_apps_ini_files()
            # Recalculate the monitoring_data_fr
            monitoring_data_fr = monitoring_data_generate(selected_monitoring_sample_rate, selected_monitor_window_size,
                                                          predicted_data_fr)

            # ---> For checking, can be removed ->
            if csv_saving is True:
                monitoring_data_fr[['label', 'gx', 'timestamp', 'predicted_label']].to_csv(
                    path_with_train_valid_test_table_name + hz_path + path_window + '/2_monitoring_data/monitoring_data.csv',
                    header=True)
            # ---> For checking, can be removed <-

            curr_monitoring_algorithm = algorithm
            curr_monitoring_window_size = selected_monitor_window_size
            curr_monitoring_sampling_rate = selected_monitoring_sample_rate
        else:
            messagebox.showinfo("Alert", 'Could not locate the file ' + file_name + ' for the plot!')
            validtorun = False
    return validtorun


def simulation_clicked():
    global monitoring_data_fr
    global label_set

    if (rdoSimuStartTime.get() == 0) and (validate_monitoring_run() is True):
        simulation_show(label_set, monitoring_data_fr, int(txtSimuFrameDtPoints_text.get()),
                        int(txtSimuFrameStride_text.get()), int(txtSimuFrameDelay_text.get()),
                        int(txtSimuFrameRepeat_text.get()))


def statics_clicked():
    global monitoring_data_fr
    global monitoring_db_table_resampled_monitor
    global curr_monitoring_sampling_rate

    if validate_monitoring_run() is True:
        # print(len(monitoring_db_table_resampled_monitor))
        # print(len(monitoring_data_fr))
        statistics_metrics_show(monitoring_db_table_resampled_monitor, monitoring_data_fr,
                                curr_monitoring_sampling_rate)


def monitoringdist_clicked():
    global monitoring_data_fr
    global curr_monitoring_sampling_rate

    if validate_monitoring_run() is True:
        monitoring_show(monitoring_data_fr, curr_monitoring_sampling_rate)


btnFitting = Button(tabTraining, text="Model fitting", bg='gold', command=modelsfitting_clicked, height=2, width=10)
# btnFitting.configure(font=('Sans','9','bold'))
btnFitting.place(x=820, y=340)

btnStatics = Button(tabTraining, text="Statics", bg='gold', command=statics_clicked, height=2, width=10)
btnStatics.place(x=701, y=407)

btnMonitorDist = Button(tabTraining, text="Monitoring", bg='gold', command=monitoringdist_clicked, height=2, width=10)
btnMonitorDist.place(x=820, y=407)

btnSimulatition = Button(tabTraining, text="Simulator", bg='#b269ff', command=simulation_clicked, height=2, width=10)
btnSimulatition.place(x=820, y=484)

# monitoringSelectDeselect()

tab_control.tab(1, state='disable')
# tab_control.tab(2, state='disable')
window.mainloop()

print('Thank you for your patience!')

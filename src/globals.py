
import os
import datetime
import psycopg2
from configparser import ConfigParser
from tkinter import messagebox

def log_message(message):
    print(str(datetime.datetime.now().strftime('%H:%M:%S')) +' - ' + message)

def stop_app(message):
    log_message(message)
    messagebox.showinfo('Alert', message)

def start_db_connection():
    global params_db
    global conn
    global cur
    try:
        conn = psycopg2.connect(**params_db)
        cur = conn.cursor()
    except (Exception, psycopg2.DatabaseError) as error:
            messagebox.showinfo("Database", error)
            return False
    return True        

def close_db_connection():
    global conn
    try:
        if conn is not None:
            conn.close()
            log_message('Database connection closed')
    except Exception as err:
        log_message('Error: {0}'.format(err))
        messagebox.showinfo('Error',err)
        return False
    return True

def init():

    global dir_path
    global root_dir

    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    
    global APP_INI_FILE
    APP_INI_FILE = 'app.ini'

    global start_time
    start_time = None
    
    global app_config
    app_config = ConfigParser()
    global db_config
    db_config = ConfigParser()

    global binary_mode
    binary_mode = None
    
    # Database connection section
    global params_db
    params_db = {}
    global conn
    conn = None
    global cur
    cur  = None
    
    global data_from_db
    data_from_db = False

    global COLUMS
    COLUMS = ['label', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'timestamp', 'cattle_id']

    global EXPERIMENT_RESULT_FILE
    EXPERIMENT_RESULT_FILE = 'Experiment_Result.txt'

    global experiment_result_table_name
    experiment_result_table_name = None


    global train_valid_data_set_name
    train_valid_data_set_name = None
    global training_table_name
    training_table_name = None  
    global training_valid_test_data_file
    training_valid_test_data_file = None


    global monitoring_data_set_name
    monitoring_data_set_name = None
    global monitoring_table_name
    monitoring_table_name = None
    global monitoring_data_file
    monitoring_data_file = None

    global train_valid_test_data_frame
    train_valid_test_data_frame= None
    
    global monitoring_data_frame
    monitoring_data_frame = None

    global monitoring_data_frame_resampled_monitor
    monitoring_data_frame_resampled_monitor = None


    # For binary classification params
    global main_label
    main_label = None  
    global sub_labels_set
    sub_labels_set = None
    global no_of_sub_labels
    no_of_sub_labels = None
    
    global csv_saving
    csv_saving = None
    
    global test_proportion
    test_proportion = None
    
    global monitoring_mode
    monitoring_mode = False
    
    global label_set
    label_set = None
    
    global cboActivityValues
    cboActivityValues = None

    global json_axes_functions
    json_axes_functions = None

    global list_agg_function_names
    list_agg_function_names = None

    global list_axes_to_apply_functions
    list_axes_to_apply_functions = None

    global features_in_dictionary  
    features_in_dictionary = None
    
    global original_sampling_rate
    original_sampling_rate = None
    global resampling_rate
    resampling_rate = None

    global data_point_filter_rate
    data_point_filter_rate = 0.2
    
    global main_and_non_main_labels_y_root
    main_and_non_main_labels_y_root = None

    global minimum_train_valid_instance_for_each_label
    minimum_train_valid_instance_for_each_label = None

    global function_set_for_resampling
    function_set_for_resampling = ['mean']

    global csv_txt_file_exporter
    csv_txt_file_exporter = None
    

    # This group of variables is for reports and statics
    global monitoring_data_fr
    monitoring_data_fr = None
    global predicted_data_fr
    predicted_data_fr = None
    
    # global monitoring_time_deviation_fr
    # monitoring_time_deviation_fr = None

    global monitoring_error_types_fr
    monitoring_error_types_fr = None    
    global curr_monitoring_algorithm
    curr_monitoring_algorithm = None
    global curr_monitoring_window_size
    curr_monitoring_window_size = None
    global curr_monitoring_sampling_rate
    curr_monitoring_sampling_rate = None
    global timestampforCSVfiles
    timestampforCSVfiles = None

    
    # rdoTrainingPhraseOnly = IntVar()
    # rdoSimuStartTime = IntVar()


  


    
    
    # global model_comment
    # model_comment = 'Comments:'
import pandas as pd
import psycopg2
import math
from tkinter import messagebox
import globals
from globals import log_message

def get_training_monitoring_data_from_db():

    # Connect to the database and get the data for training and monitoring tables
    # Calculate the two derived columns 
    return_status = globals.start_db_connection()
    if return_status:
        try:

            # Create result table if not exist ->       
            sql_create_table = 'CREATE TABLE IF NOT EXISTS ' + 'public.' + globals.experiment_result_table_name + ' (model_title text COLLATE pg_catalog."default", model_init_name text COLLATE pg_catalog."default", model_binary_content bytea, features_json_content json, model_comments text COLLATE pg_catalog."default", train_table text COLLATE pg_catalog."default", monitor_table text COLLATE pg_catalog."default", no_of_predicted_classes integer, list_of_predicted_classes text COLLATE pg_catalog."default", original_sample_rate_in_hz integer, no_of_original_train_data_points integer, resampled_rate_in_hz integer, no_of_resampled_train_data_points integer, no_of_instances_for_each_class_in_resampled_train_table integer, algorithm text COLLATE pg_catalog."default", no_of_functions integer, list_of_functions text COLLATE pg_catalog."default", no_of_axes integer, list_of_axes text COLLATE pg_catalog."default", window_size integer, window_stride text COLLATE pg_catalog."default", k_fold integer, accuracy_train_valid real, precision_train_valid real, recall_train_valid real, specificity_train_valid real, f1_train_valid real, accuracy_test real, precision_test real, recall_test real, specificity_test real, f1_test real, monitoring_window_stride text COLLATE pg_catalog."default", accuracy_monitor real, precision_monitor real, recall_monitor real, specificity_monitor real, f1_monitor real, start_time timestamp without time zone, end_time timestamp without time zone, running_time_in_minutes text COLLATE pg_catalog."default") WITH (OIDS = FALSE) TABLESPACE pg_default;'
            globals.cur.execute(sql_create_table)
            globals.conn.commit()
            
            log_message('Connecting to the PostgreSQL database...')
            log_message('Start fetching data')

            sqlQuery = 'SELECT * FROM ' + globals.training_table_name + ' ORDER BY timestamp ASC '
            
            cols = globals.COLUMS
            globals.train_valid_test_data_frame = pd.read_sql_query(sqlQuery, con = globals.conn)[cols]

            if globals.monitoring_mode:
                sqlQuery2 = 'SELECT * FROM ' + globals.monitoring_table_name + ' ORDER BY timestamp ASC'
                globals.monitoring_data_frame = pd.read_sql_query(sqlQuery2, con = globals.conn)[cols]

            log_message('Finish fetching data')
            log_message('Start calculating the magnitudes of Acc and Gyro')        
            
            globals.train_valid_test_data_frame.loc[:, 'gyrMag'] = globals.train_valid_test_data_frame.apply(
                lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis = 1)
            globals.train_valid_test_data_frame.loc[:, 'accMag'] = globals.train_valid_test_data_frame.apply(
                lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis = 1)

            if globals.monitoring_mode:
                globals.monitoring_data_frame.loc[:, 'gyrMag'] = globals.monitoring_data_frame.apply(
                    lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis = 1)
                globals.monitoring_data_frame.loc[:, 'accMag'] = globals.monitoring_data_frame.apply(
                    lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis = 1)
                
                # Copy for the case user switches from binary to multi-class classification
                globals.monitoring_data_frame_origin = globals.monitoring_data_frame.copy()    

            log_message('Finish calculating the magnitudes of Acc and Gyro')
            log_message('Finish data importing!')
            
            return globals.close_db_connection()

        except Exception as err:
            log_message('Error: {0}'.format(err))
            messagebox.showinfo('Error',err)
            return False
    else:
        return False

def get_training_monitoring_data_from_csv_file():
    try:
        log_message('Start reading CSV data')
        
        cols = globals.COLUMS
        globals.train_valid_test_data_frame = pd.read_csv(globals.training_valid_test_data_file)[cols]

        if globals.monitoring_mode:
            globals.monitoring_data_frame = pd.read_csv(globals.monitoring_data_file)[cols]

        log_message('Start calculating the magnitudes of Acc and Gyro')
        globals.train_valid_test_data_frame.loc[:, 'gyrMag'] = globals.train_valid_test_data_frame.apply(
            lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis = 1)
        globals.train_valid_test_data_frame.loc[:, 'accMag'] = globals.train_valid_test_data_frame.apply(
            lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis = 1)

        if globals.monitoring_mode:
            globals.monitoring_data_frame.loc[:, 'gyrMag'] = globals.monitoring_data_frame.apply(
                lambda x: math.sqrt(x.gx * x.gx + x.gy * x.gy + x.gz * x.gz), axis = 1)
            globals.monitoring_data_frame.loc[:, 'accMag'] = globals.monitoring_data_frame.apply(
                lambda x: math.sqrt(x.ax * x.ax + x.ay * x.ay + x.az * x.az), axis = 1)
           
            # Copy for the case when user switch from binary to multi-class classification
            globals.monitoring_data_frame_origin = globals.monitoring_data_frame.copy()                    
            
        log_message('Finish calculating the magnitudes of Acc and Gyro')

        return True    
        
    except Exception as err:
        log_message('Error: {0}'.format(err))
        messagebox.showinfo('Error',err)

        return False

    


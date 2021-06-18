import os
import globals

class CsvTxtFileExporter():
    
    def __init__(self, samping_rate):
        
        csv_folder = os.path.join(globals.root_dir,'csv_out')      
        if not os.path.exists(csv_folder):
            os.mkdir(csv_folder)
        
        csv_folder = os.path.join(csv_folder, globals.train_valid_data_set_name + '_at_' + globals.timestampforCSVfiles)
        os.mkdir(csv_folder)     

        self.folder_path_with_each_model = os.path.join(csv_folder, str(samping_rate) + 'Hz')    
        os.mkdir(self.folder_path_with_each_model)


    def create_window_size_stride_folder (self, window_size, window_stride_in_ms):       
        self.str_window_size_stride = 'window_' + str(window_size) + 'ms_stride_' + str(window_stride_in_ms) + 'ms'
        
        self.window_size_stride_path = os.path.join(self.folder_path_with_each_model, self.str_window_size_stride)    
        os.mkdir(self.window_size_stride_path)
        os.mkdir(os.path.join(self.window_size_stride_path, '1_train_valid_test_set'))
        os.mkdir(os.path.join(self.window_size_stride_path, '2_monitoring_data'))
        os.mkdir(os.path.join(self.window_size_stride_path, '3_kfold'))


    def save_into_csv_file(self, data_frame, folder_name, file_name):
        data_frame.to_csv(os.path.join(self.window_size_stride_path, folder_name, file_name), header = True)

    
    def create_experiment_result_txt_file(self):
        self.text_file = open(os.path.join(self.window_size_stride_path, globals.EXPERIMENT_RESULT_FILE), 'w', encoding='utf-8')

        if globals.data_from_db:
            self.text_file.write('Train_Valid_Test DB table: ' + globals.training_table_name)
            if globals.monitoring_mode:
                self.text_file.write('\nMonitoring         DB table: ' + globals.monitoring_table_name)
        else:
            self.text_file.write('Train_Valid_Test CSV file: ' + globals.training_valid_test_data_file)
            if globals.monitoring_mode:
                self.text_file.write('\nMonitoring         CSV file: ' + globals.monitoring_data_file)

        self.text_file.write('\n' + self.str_window_size_stride)
        self.text_file.write('\nLabels to predict: ' + globals.label_set.str.cat(sep=' '))
        self.text_file.write('\nFunctions list      : ' + str(globals.list_agg_function_names))
        self.text_file.write('\nAxes list             : ' + str(globals.list_axes_to_apply_functions))

    def close_experiment_result_txt_file(self):
        self.text_file.close()

    def write_single_line(self, line_to_write):
        self.text_file.write(line_to_write)

        



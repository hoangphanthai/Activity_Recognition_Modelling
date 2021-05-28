import os
import datetime
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import globals
from globals import log_message
from models import classifiers
from data import ini_file, data_exporter, data_proccessor, data_resampling
from visualisation import plot


class TabTraining(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        
        self.build_controls()
        self.fill_init_data_into_controls()
 
    def validate_monitoring_run(self):
        
        txtWindowStride = self.txtWindowStride_text.get()
        selected_monitor_window_size = int(self.txtSimulationWindowSize_text.get())
        selected_monitoring_sample_rate = int(self.cboSimulationSampleRate.get())
        algorithm = str(self.cboSimuAlgorithm.get())
        
        return data_proccessor.validate_monitoring_run(txtWindowStride, selected_monitor_window_size, selected_monitoring_sample_rate, algorithm)


    def statics_clicked(self):
        if self.validate_monitoring_run():
            plot.statistics_metrics_show(globals.monitoring_data_frame_resampled_monitor, globals.monitoring_data_fr, globals.curr_monitoring_sampling_rate)


    def monitoring_dist_clicked(self):
        if self.validate_monitoring_run():
            plot.monitoring_show(globals.monitoring_data_fr, globals.curr_monitoring_sampling_rate)
    

    def simulation_clicked(self):
        if (self.rdoSimuStartTime.get() == 0) and (self.validate_monitoring_run()):
            plot.simulation_show(globals.label_set, globals.monitoring_data_fr, int(self.txtSimuFrameDtPoints_text.get()),
                            int(self.txtSimuFrameStride_text.get()), int(self.txtSimuFrameDelay_text.get()),
                            int(self.txtSimuFrameRepeat_text.get()))


    def models_fitting_clicked(self):
        print(' ')
        print(' ')
        log_message('START BUILDING NEW MODEL(S)')

        time_now = datetime.datetime.now()
        globals.timestampforCSVfiles = '%02d' % time_now.hour + 'h' + '%02d' % time_now.minute + 'm' + '%02d' % time_now.second + 's'
        
        # This step check validity of input data and set global default values
        valid_input_data, error_message = self.check_input_data()
        
        if not valid_input_data:
            messagebox.showinfo('Alert', error_message)
        else:
            # Start new db connection for experiment result updating
            if globals.data_from_db:
                connection_status = globals.start_db_connection()
                if not connection_status:
                    globals.stop_app('Database connection for saving experiment result is unsuccessful')
                    return    
            
            input_training_settings = self.get_input_training_settings_from_UI()

            # Update input_data_setting into app.in         
            ini_file.update_training_tab_layout_data(input_training_settings)
        
            self.convert_features_into_json_content()

            # Because the binary_mode will changes the non-main labels into 'Non-...'
            globals.label_set = self.label_set_origin.copy()
            if globals.monitoring_mode:
                globals.monitoring_data_frame = globals.monitoring_data_frame_origin.copy()

            if globals.binary_mode:
                log_message('BINARY classification with main label: ' + globals.main_label)
            else:
                log_message('MULTI-CLASS classification  '+ globals.label_set.str.cat(sep=' '))

            train_valid_test_data_filtered_by_selected_labels = globals.train_valid_test_data_frame.loc[
                globals.train_valid_test_data_frame['label'].isin(globals.label_set)].sort_values(by=['timestamp'], ascending=True)

            if globals.monitoring_mode:
                monitoring_data_filtered_by_selected_labels = globals.monitoring_data_frame.loc[
                    globals.monitoring_data_frame['label'].isin(globals.label_set)].sort_values(
                    by=['timestamp'], ascending=True)

                # The activity existed in monitoring data must cover the labels in training, so that the metrics are properly calculated
                monitoring_data_label_set = pd.Series(np.unique(np.array(monitoring_data_filtered_by_selected_labels['label']).tolist()))
                monitoring_data_label_set = monitoring_data_label_set.sort_values(ascending=True)

                globals.monitoring_mode = monitoring_data_label_set.equals(globals.label_set)

                if not globals.monitoring_mode:
                    messagebox.showinfo('Alert', 'The activities (labels) list in monitoring data is not matched with that from training data. Please re-select the datasets')
                    return
            else: 
                monitoring_data_filtered_by_selected_labels = pd.DataFrame()

            # Reset monitoring data everytime user clickes on fitting
            globals.monitoring_data_fr = pd.DataFrame()  # This dataframe includes test column values, grounthTruth label and predicted
            globals.predicted_data_fr = pd.DataFrame()  # This dataframe includes data for the simulation
            globals.curr_monitoring_algorithm = ''
            globals.curr_monitoring_window_size = 0
            globals.curr_monitoring_sampling_rate = 0

            self.random_forest = classifiers.CLS(RandomForestClassifier(n_estimators = 100),'Random_Forest')
            self.decision_tree = classifiers.CLS(DecisionTreeClassifier(criterion='entropy', max_depth=10),'Decision_Tree')
            self.support_vector_machine = classifiers.CLS(svm.SVC(C=1.0, kernel='rbf', gamma='scale', decision_function_shape='ovo'),'SVM')
            self.naive_bayes = classifiers.CLS(GaussianNB(),'Naive_Bayes')
            self.kfold = int(self.cboKfold.get())

            self.no_of_original_train_valid_test_data_points = len(train_valid_test_data_filtered_by_selected_labels.index)   
            
            if self.rdoResampling.get() == 0: # User wants to keep the original sampling rates

                # These variables are the same in the case of keeping original sampling rate
                self.no_of_resampled_train_data_points = self.no_of_original_train_valid_test_data_points
                globals.resampling_rate = globals.original_sampling_rate 
                
                log_message('--------------------------------------------------------Running with original sampling rate at ' + str(globals.resampling_rate) + 'Hz ')
                
                globals.csv_txt_file_exporter = data_exporter.CsvTxtFileExporter(globals.resampling_rate)

                for window_size in range(int(self.txtWindowSizeFrom_text.get()), int(self.txtWindowSizeTo_text.get()) + 1,
                                         int(self.txtWindowStep_text.get())):

                    window_stride_in_ms = math.floor(window_size * int(self.txtWindowStride_text.get()) / 100)
                    log_message('Begin train-valid-test processing at window size ' + str(window_size) + 'ms with stride of ' + str(window_stride_in_ms) + 'ms')
                    log_message('Start calculating features for train data')
                    self.process_at_window_size_stride (train_valid_test_data_filtered_by_selected_labels, monitoring_data_filtered_by_selected_labels, window_size, window_stride_in_ms)

            else: # User wants to resample the input data
                for resampling_rate in range(int(str(self.cboDownSamplingFrom.get())), int(str(self.cboDownSamplingTo.get())) + 1,
                                            int(str(self.cboDownSamplingStep.get()))):
                    
                    globals.resampling_rate = resampling_rate                    
                    log_message('--------------------------------------------------------Running with resampled rate at ' + str(globals.resampling_rate) + 'Hz ')
                    log_message('Sensor data is being resampled')
                    resampling_dict = {}
                    for axis in globals.list_axes_to_apply_functions:
                        resampling_dict[axis] = globals.function_set_for_resampling

                    resampled_train_valid_data = data_resampling.resampled_frame(train_valid_test_data_filtered_by_selected_labels, globals.label_set,
                                                           resampling_dict,
                                                           globals.list_axes_to_apply_functions, resampling_rate)

                    self.no_of_resampled_train_data_points = len(resampled_train_valid_data.index)

                    if globals.monitoring_mode:
                        resampled_monitoring_data = data_resampling.resampled_frame(monitoring_data_filtered_by_selected_labels, globals.label_set,
                                                           resampling_dict,
                                                           globals.list_axes_to_apply_functions, resampling_rate)
                    else:
                        resampled_monitoring_data = pd.DataFrame()

                    globals.csv_txt_file_exporter = data_exporter.CsvTxtFileExporter(globals.resampling_rate)
                    for window_size in range(int(self.txtWindowSizeFrom_text.get()), int(self.txtWindowSizeTo_text.get()) + 1,
                                         int(self.txtWindowStep_text.get())):

                        window_stride_in_ms = math.floor(window_size * int(self.txtWindowStride_text.get()) / 100)
                        log_message('Begin train-valid-test processing at window size ' + str(window_size) + 'ms with stride of ' + str(window_stride_in_ms) + 'ms')
                        log_message('Start calculating features for train data')
                        self.process_at_window_size_stride (resampled_train_valid_data, resampled_monitoring_data, window_size, window_stride_in_ms)

                    # winsound.Beep(1000, 300)

            if globals.data_from_db:
                connection_status = globals.close_db_connection()
                if not connection_status:
                    log_message('Database connection closing is unsuccessful')

            # Monitoring window size is set to the begining window size (for the purpose of better UI experience)
            self.txtSimulationWindowSize_text.set(self.txtWindowSizeFrom_text.get())
      
        if globals.monitoring_mode:
            self.enable_monitoring_process_buttons()
        else:
            self.disable_monitoring_process_buttons()    


    def process_at_window_size_stride (self, train_valid_test_data_frame, monitoring_data_frame, window_size, window_stride_in_ms):
     
        main_and_non_main_labels_set = None
        main_and_non_main_labels_narray_temp = None
        

        if globals.binary_mode:
            # Because the label_set has changed into Main and Non-Main in the last window setting -> get the origin labels
            globals.label_set = self.label_set_origin.copy()
            main_and_non_main_labels_set = globals.label_set.copy()
            main_and_non_main_labels_narray_temp = main_and_non_main_labels_set.to_numpy(copy=True)
            main_and_non_main_labels_narray_temp = np.append(main_and_non_main_labels_narray_temp, 'Non-' + globals.main_label)

        agg_train_valid_test_unfiltered_unbalanced = data_resampling.aggregated_frame(train_valid_test_data_frame,
                                                                        globals.label_set,
                                                                        globals.features_in_dictionary,
                                                                        globals.list_axes_to_apply_functions,
                                                                        window_size,
                                                                        window_stride_in_ms)

        log_message('End calculating features for train data')
        
        globals.csv_txt_file_exporter.create_window_size_stride_folder(window_size, window_stride_in_ms)

        if globals.csv_saving:
            globals.csv_txt_file_exporter.save_into_csv_file(agg_train_valid_test_unfiltered_unbalanced,'1_train_valid_test_set','01_train_valid_test_imbalanced_set_all_instances.csv')    
        # End calculating features for training phrase <-
        
        # Calculate the range of data points allowed in a window
        minimum_count_allowed = round((window_size * globals.resampling_rate / 1000) * (1 - globals.data_point_filter_rate))
        maximum_count_allowed = round((window_size * globals.resampling_rate / 1000) * (1 + globals.data_point_filter_rate))

        # Filtering the number of data points for each window ->
        agg_train_valid_test_filtered_unbalanced = agg_train_valid_test_unfiltered_unbalanced.loc[
            (agg_train_valid_test_unfiltered_unbalanced['count'] >= minimum_count_allowed) & (
                    agg_train_valid_test_unfiltered_unbalanced['count'] <= maximum_count_allowed)]

        # Getting balancing Train_Valid and Test data set
        balanced_train_valid_dataset, test_dataset, return_status = data_proccessor.get_balanced_train_valid_test_data_set(agg_train_valid_test_filtered_unbalanced)
        
        if not return_status:
            globals.stop_app('Training data is insufficient, having not enough labels/activities for models fitting')                       
            return

        if globals.csv_saving:
            globals.csv_txt_file_exporter.save_into_csv_file(balanced_train_valid_dataset, '1_train_valid_test_set', '02_train_valid_balanced_dataset_with_' + str(
            globals.minimum_train_valid_instance_for_each_label) + '_instances_for_each_class.csv')                                
        
        balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['cattle_id'], axis = 1)
        balanced_train_valid_dataset = balanced_train_valid_dataset.drop(['count'], axis = 1)
        balanced_train_valid_dataset = balanced_train_valid_dataset.reset_index(drop = True)

        # Open and write infor to a text file ->
        globals.csv_txt_file_exporter.create_experiment_result_txt_file()
        
        # Initialising some of metrics for the classification
        self.random_forest.init_metrics()
        self.decision_tree.init_metrics()
        self.support_vector_machine.init_metrics()
        self.naive_bayes.init_metrics()

        feature_cols = list(balanced_train_valid_dataset.columns.values)
        feature_cols.remove('timestamp_human')
        feature_cols.remove('label')

        X = balanced_train_valid_dataset[feature_cols]  # Features
        y = balanced_train_valid_dataset['label']  # Target

        # Stratified k-fold
        kf = StratifiedKFold(n_splits=self.kfold, shuffle=True)  # Considering random_state = 0??

        k_fold_round = int(0)
        for train_index, valid_index in kf.split(X, y):
            k_fold_round += 1
            X_train = pd.DataFrame(X, columns = feature_cols, index = train_index)
            X_valid = pd.DataFrame(X, columns = feature_cols, index = valid_index)

            y_train_df = pd.DataFrame(y, columns = ['label'], index = train_index)
            y_train = y_train_df['label']
            y_valid_df = pd.DataFrame(y, columns = ['label'], index = valid_index)
            y_valid = y_valid_df['label']

            if globals.csv_saving is True:
                globals.csv_txt_file_exporter.save_into_csv_file(X_train, '3_kfold', str(k_fold_round) + 'th_round_fold_X_train.csv')
                globals.csv_txt_file_exporter.save_into_csv_file(X_valid, '3_kfold', str(k_fold_round) + 'th_round_fold_X_validation.csv')
                globals.csv_txt_file_exporter.save_into_csv_file(y_train, '3_kfold', str(k_fold_round) + 'th_round_fold_y_train.csv')                                
                globals.csv_txt_file_exporter.save_into_csv_file(y_valid, '3_kfold', str(k_fold_round) + 'th_round_fold_y_validation.csv')     
                
            globals.csv_txt_file_exporter.write_single_line('\n------------------Round ' + str(k_fold_round) + '------------------')

            self.random_forest.train_validate(self.RandomForestVar.get(), X_train, y_train, X_valid, y_valid)
            self.decision_tree.train_validate(self.DecisionTreeVar.get(), X_train, y_train, X_valid, y_valid)
            self.support_vector_machine.train_validate(self.SVMVar.get(), X_train, y_train, X_valid, y_valid)
            self.naive_bayes.train_validate(self.NaiveBayesVar.get(), X_train, y_train, X_valid, y_valid)

        globals.csv_txt_file_exporter.write_single_line('\n')
        
        print('-------------------------------------------------------------------------')
        self.random_forest.calc_and_save_train_valid_result(self.RandomForestVar.get(), self.kfold)
        self.decision_tree.calc_and_save_train_valid_result(self.DecisionTreeVar.get(), self.kfold)
        self.support_vector_machine.calc_and_save_train_valid_result(self.SVMVar.get(), self.kfold)
        self.naive_bayes.calc_and_save_train_valid_result(self.NaiveBayesVar.get(), self.kfold)
        
        # Begin processing the test phrase
        labels_narray_temp = globals.label_set.to_numpy()
        test_dataset = test_dataset.dropna().set_index('timestamp')

        if globals.csv_saving:
            globals.csv_txt_file_exporter.save_into_csv_file(y_valid, '1_train_valid_test_set', '03_test_dataset_counts_filtered.csv')      

        test_dataset = test_dataset.drop(['cattle_id'], axis = 1)
        test_dataset = test_dataset.drop(['count'], axis = 1)
        test_dataset = test_dataset.reset_index(drop = True)

        X_test = test_dataset[feature_cols]  # Features
        y_test = test_dataset['label']  # Target

        self.random_forest.predict_test_data(self.RandomForestVar.get(), X_test, y_test, labels_narray_temp, main_and_non_main_labels_set, main_and_non_main_labels_narray_temp)
        self.decision_tree.predict_test_data(self.DecisionTreeVar.get(), X_test, y_test, labels_narray_temp, main_and_non_main_labels_set, main_and_non_main_labels_narray_temp)
        self.support_vector_machine.predict_test_data(self.SVMVar.get(), X_test, y_test, labels_narray_temp, main_and_non_main_labels_set, main_and_non_main_labels_narray_temp)
        self.naive_bayes.predict_test_data(self.NaiveBayesVar.get(), X_test, y_test, labels_narray_temp, main_and_non_main_labels_set, main_and_non_main_labels_narray_temp)


        if globals.monitoring_mode:
            print('-------------------------------------------------------------------------')
            monitoring_window_stride_in_ms = math.floor(window_size * int(self.txtWindowSimuStride_text.get()) / 100)
            log_message('Begin calculating features for the Monitoring data with window size ' + str(window_size) + 'ms and stride of ' + str(monitoring_window_stride_in_ms) + 'ms')

            if globals.binary_mode:
                # Because the label_set has changed into Main and Non-Main in the last window setting -> get the origin labels
                globals.label_set = self.label_set_origin.copy()
                
            # This dataframe is for testing on unseen data (from monitoring data table) ->
            agg_monitor = data_resampling.aggregated_frame(monitoring_data_frame, globals.label_set,
                                                globals.features_in_dictionary,
                                                globals.list_axes_to_apply_functions, window_size, monitoring_window_stride_in_ms)
            
            
            globals.csv_txt_file_exporter.save_into_csv_file(agg_monitor, '2_monitoring_data', '0_unfiltered_monitoring_data_set.csv')    
                    
            agg_monitor_counts_filtered = agg_monitor.loc[
                (agg_monitor['count'] >= minimum_count_allowed) & (agg_monitor['count'] <= maximum_count_allowed)]
            agg_monitor_counts_filtered = agg_monitor_counts_filtered.dropna().reset_index(drop = True)

            # To be deleted ->
            if globals.csv_saving:
                globals.csv_txt_file_exporter.save_into_csv_file(agg_monitor_counts_filtered, '2_monitoring_data', '1_filtered_monitoring_set.csv')

            if globals.binary_mode:
                # Get the root list of all activities in test data set before changing sub labels/activities into non-... label. This is for the confusion matrix latter.
                globals.main_and_non_main_labels_y_root_monitoring_temp = agg_monitor_counts_filtered['label'].to_numpy(copy=True)

                # Change the label set of Test dataset into Non-main label for the other non main labels
                for _, value in globals.sub_labels_set.items():
                    agg_monitor_counts_filtered.loc[
                        agg_monitor_counts_filtered.label == value, 'label'] = 'Non-' + globals.main_label

                # Change the label_set into two labels only
                globals.label_set = pd.Series([globals.main_label, 'Non-' + globals.main_label])

            X_monitor = agg_monitor[feature_cols]  # Features
            y_monitor_temp = agg_monitor_counts_filtered['label']
            X_monitor_temp = agg_monitor_counts_filtered[feature_cols]

            # Begin predicting and generate data for monitoring
            self.random_forest.predict_monitoring_data(self.RandomForestVar.get(), X_monitor_temp, y_monitor_temp, main_and_non_main_labels_set, labels_narray_temp, main_and_non_main_labels_narray_temp)
            self.random_forest.generate_monitor_prediction_file(self.RandomForestVar.get(), X_monitor, agg_monitor)

            self.decision_tree.predict_monitoring_data(self.DecisionTreeVar.get(), X_monitor_temp, y_monitor_temp, main_and_non_main_labels_set, labels_narray_temp, main_and_non_main_labels_narray_temp)
            self.decision_tree.generate_monitor_prediction_file(self.DecisionTreeVar.get(), X_monitor, agg_monitor)

            self.support_vector_machine.predict_monitoring_data(self.SVMVar.get(), X_monitor_temp, y_monitor_temp, main_and_non_main_labels_set, labels_narray_temp, main_and_non_main_labels_narray_temp)
            self.support_vector_machine.generate_monitor_prediction_file(self.SVMVar.get(), X_monitor, agg_monitor)

            self.naive_bayes.predict_monitoring_data(self.NaiveBayesVar.get(), X_monitor_temp, y_monitor_temp, main_and_non_main_labels_set, labels_narray_temp, main_and_non_main_labels_narray_temp)
            self.naive_bayes.generate_monitor_prediction_file(self.NaiveBayesVar.get(), X_monitor, agg_monitor)
            

        log_message('End processing window size ' + str(window_size) + ' ms Stride ' + str(window_stride_in_ms) + ' ms')
        print('-------------------------------------------------------------------------')
        
        globals.csv_txt_file_exporter.close_experiment_result_txt_file()
        
        # Update experiment result to database
        if globals.data_from_db:
            self.random_forest.save_experiment_result_into_db(self.RandomForestVar.get(), self.no_of_original_train_valid_test_data_points, self.no_of_resampled_train_data_points, window_size, self.kfold, self.txtWindowStride_text.get(),  self.txtWindowSimuStride_text.get())
            self.decision_tree.save_experiment_result_into_db(self.DecisionTreeVar.get(), self.no_of_original_train_valid_test_data_points, self.no_of_resampled_train_data_points, window_size, self.kfold, self.txtWindowStride_text.get(),  self.txtWindowSimuStride_text.get())
            self.support_vector_machine.save_experiment_result_into_db(self.SVMVar.get(), self.no_of_original_train_valid_test_data_points, self.no_of_resampled_train_data_points, window_size, self.kfold, self.txtWindowStride_text.get(),  self.txtWindowSimuStride_text.get())
            self.naive_bayes.save_experiment_result_into_db(self.NaiveBayesVar.get(), self.no_of_original_train_valid_test_data_points, self.no_of_resampled_train_data_points, window_size, self.kfold, self.txtWindowStride_text.get(),  self.txtWindowSimuStride_text.get())


    def get_input_training_settings_from_UI(self):
        input_training_settings = {}
        input_training_settings['selectallactivityonoff'] = str(self.selectAllActivityOnOff.get())
        input_training_settings['btnallfeaturesonoff'] = str(self.selectAllFeaturesOnOff.get())
        input_training_settings['chkbtnmin'] = str(self.MinVar.get())
        input_training_settings['chkbtnmax'] = str(self.MaxVar.get())
        input_training_settings['chkbtnmean'] = str(self.MeanVar.get())
        input_training_settings['chkbtnmedian'] = str(self.MedianVar.get())
        input_training_settings['chkbtnstdev'] = str(self.StdVar.get())
        input_training_settings['chkbtninterquartilerange'] = str(self.IQRVar.get())
        input_training_settings['chkbtnrootmsvar'] = str(self.rootMSVar.get())
        input_training_settings['chkbtnmeancrvar'] = str(self.meanCRVar.get())
        input_training_settings['chkbtnkurtosisvar'] = str(self.kurtosisVar.get())
        input_training_settings['chkbtnskewnessvar'] = str(self.skewnessVar.get())
        input_training_settings['chkbtnenergyvar'] = str(self.energyVar.get())
        input_training_settings['chkbtnpeakfreqvar'] = str(self.peakFreqVar.get())
        input_training_settings['chkbtnfreqdmentropyvar'] = str(self.freqDmEntropyVar.get())
        input_training_settings['chkbtnfr1cpnmagvar'] = str(self.fr1cpnMagVar.get())
        input_training_settings['chkbtnfr2cpnmagvar'] = str(self.fr2cpnMagVar.get())
        input_training_settings['chkbtnfr3cpnmagvar'] = str(self.fr3cpnMagVar.get())
        input_training_settings['chkbtnfr4cpnmagvar'] = str(self.fr4cpnMagVar.get())
        input_training_settings['chkbtnfr5cpnmagvar'] = str(self.fr5cpnMagVar.get())
        input_training_settings['txtwindowsizefrom'] = self.txtWindowSizeFrom_text.get()
        input_training_settings['txtwindowsizeto'] = self.txtWindowSizeTo_text.get()
        input_training_settings['txtwindowstep'] = self.txtWindowStep_text.get()
        input_training_settings['txtwindowstride'] = self.txtWindowStride_text.get()
        input_training_settings['txtwindowsimustride'] = self.txtWindowSimuStride_text.get()
        input_training_settings['cbodownsamplingfrom'] = str(self.cboDownSamplingFrom.get())
        input_training_settings['cbodownsamplingto'] = str(self.cboDownSamplingTo.get())
        input_training_settings['cbodownsamplingstep'] = str(self.cboDownSamplingStep.get())
        input_training_settings['btnchkrandomforest'] = str(self.RandomForestVar.get())
        input_training_settings['btnchkdecisiontree'] = str(self.DecisionTreeVar.get())
        input_training_settings['btnchksvm'] = str(self.SVMVar.get())
        input_training_settings['btnchknaivebayes'] = str(self.NaiveBayesVar.get())
        input_training_settings['cbokfold'] = str(self.cboKfold.get())
        input_training_settings['rdoresampling'] = str(self.rdoResampling.get())
        input_training_settings['txtsimulationwindowsize'] = self.txtSimulationWindowSize_text.get()
        input_training_settings['txtsimuframedtpoints'] = self.txtSimuFrameDtPoints_text.get()
        input_training_settings['txtsimuframestride'] = self.txtSimuFrameStride_text.get()
        input_training_settings['txtsimuframedelay'] = self.txtSimuFrameDelay_text.get()
        input_training_settings['txtsimuframerepeat'] = self.txtSimuFrameRepeat_text.get()
        input_training_settings['chkbtngx'] = str(self.gxVar.get())
        input_training_settings['chkbtngy'] = str(self.gyVar.get())
        input_training_settings['chkbtngz'] = str(self.gzVar.get())
        input_training_settings['chkbtnax'] = str(self.axVar.get())
        input_training_settings['chkbtnay'] = str(self.ayVar.get())
        input_training_settings['chkbtnaz'] = str(self.azVar.get())
        input_training_settings['chkbtngmag'] = str(self.g_mag_Var.get())
        input_training_settings['chkbtnamag'] = str(self.a_mag_Var.get())


        return input_training_settings


    def enable_monitoring_process_buttons(self):
            self.btnStatics.configure(state = 'normal')
            self.btnMonitorDist.configure(state = 'normal')
            self.btnSimulatition.configure(state = 'normal')


    def disable_monitoring_process_buttons(self):
        self.btnStatics.configure(state = 'disabled')
        self.btnMonitorDist.configure(state = 'disabled')
        self.btnSimulatition.configure(state = 'disabled')    


    def check_input_data(self):
        no_of_labels = int(0)
        no_of_functions = int(0)
        no_of_classifiers = int(0)
        no_of_axes = int(0)
        valid_to_running = True

        # Verify the number of activities user selected
        label_set = pd.Series([], dtype='string')
        i = int(0)

        if globals.binary_mode:
            sub_labels_set = pd.Series([], dtype='string')
            if self.cboBinMainActivity.get() != 'None':
                globals.main_label = self.cboBinMainActivity.get()
                label_set.at[i] = self.cboBinMainActivity.get()               
                i = i + 1
            if self.cboBinNonMainActivity1.get() != 'None':
                label_set.at[i] = self.cboBinNonMainActivity1.get()
                sub_labels_set.at[i-1] = self.cboBinNonMainActivity1.get()
                i = i + 1
            if self.cboBinNonMainActivity2.get() != 'None':
                label_set.at[i] = self.cboBinNonMainActivity2.get()
                sub_labels_set.at[i-1] = self.cboBinNonMainActivity2.get()
                i = i + 1
            if self.cboBinNonMainActivity3.get() != 'None':
                label_set.at[i] = self.cboBinNonMainActivity3.get()
                sub_labels_set.at[i-1] = self.cboBinNonMainActivity3.get()
                i = i + 1
            if self.cboBinNonMainActivity4.get() != 'None':
                label_set.at[i] = self.cboBinNonMainActivity4.get()
                sub_labels_set.at[i-1] = self.cboBinNonMainActivity4.get()

            globals.sub_labels_set = pd.Series(np.unique(sub_labels_set.values))
            globals.no_of_sub_labels = len(globals.sub_labels_set)

        else: # Multi class classification is selected
            if self.selectAllActivityOnOff.get() == 1:
                label_set = globals.cboActivityValues
            else:          
                if self.cboActivity1.get() != 'None':
                    label_set.at[i] = self.cboActivity1.get()
                    i = i + 1
                if self.cboActivity2.get() != 'None':
                    label_set.at[i] = self.cboActivity2.get()
                    i = i + 1
                if self.cboActivity3.get() != 'None':
                    label_set.at[i] = self.cboActivity3.get()
                    i = i + 1
                if self.cboActivity4.get() != 'None':
                    label_set.at[i] = self.cboActivity4.get()
                    i = i + 1
                if self.cboActivity5.get() != 'None':
                    label_set.at[i] = self.cboActivity5.get()
                    i = i + 1
                if self.cboActivity6.get() != 'None':
                    label_set.at[i] = self.cboActivity6.get()
          
        # Select only unique activities values if duplicate
        globals.label_set = pd.Series(np.unique(label_set.values))
        globals.label_set = globals.label_set.sort_values(ascending=True)
        self.label_set_origin = globals.label_set.copy()

        no_of_labels = len(globals.label_set)
        if no_of_labels < 2:
            valid_to_running = False

        # Collect the aggregate functions list from selected boxes
        list_agg_function_names = []
        if self.MinVar.get() == 1:
            list_agg_function_names.append('min')
        if self.MaxVar.get() == 1:
            list_agg_function_names.append('max')
        if self.MeanVar.get() == 1:
            list_agg_function_names.append('mean')
        if self.MedianVar.get() == 1:
            list_agg_function_names.append('median')
        if self.StdVar.get() == 1:
            list_agg_function_names.append('stdev')
        if self.IQRVar.get() == 1:
            list_agg_function_names.append('IQR')
        if self.rootMSVar.get() == 1:
            list_agg_function_names.append('RMS')
        if self.meanCRVar.get() == 1:
            list_agg_function_names.append('MCR')
        if self.kurtosisVar.get() == 1:
            list_agg_function_names.append('Kurt')
        if self.skewnessVar.get() == 1:
            list_agg_function_names.append('Skew')
        if self.energyVar.get() == 1:
            list_agg_function_names.append('Energy')
        if self.peakFreqVar.get() == 1:
            list_agg_function_names.append('PeakFreq')
        if self.freqDmEntropyVar.get() == 1:
            list_agg_function_names.append('FreqEntrpy')
        if self.fr1cpnMagVar.get() == 1:
            list_agg_function_names.append('FirstCpn')
        if self.fr2cpnMagVar.get() == 1:
            list_agg_function_names.append('SecondCpn')
        if self.fr3cpnMagVar.get() == 1:
            list_agg_function_names.append('ThirdCpn')
        if self.fr4cpnMagVar.get() == 1:
            list_agg_function_names.append('FourthCpn')
        if self.fr5cpnMagVar.get() == 1:
            list_agg_function_names.append('FifthCpn')
       
        globals.list_agg_function_names = list_agg_function_names

        no_of_functions = len(list_agg_function_names)
        if no_of_functions == 0:
            valid_to_running = False

        # Collect the axes list to apply the aggregate functions
        list_axes_to_apply_functions = []
        if self.gxVar.get() == 1:
            list_axes_to_apply_functions.append('gx')
        if self.gyVar.get() == 1:
            list_axes_to_apply_functions.append('gy')
        if self.gzVar.get() == 1:
            list_axes_to_apply_functions.append('gz')
        if self.axVar.get() == 1:
            list_axes_to_apply_functions.append('ax')
        if self.ayVar.get() == 1:
            list_axes_to_apply_functions.append('ay')
        if self.azVar.get() == 1:
            list_axes_to_apply_functions.append('az')
        if self.g_mag_Var.get() == 1:
            list_axes_to_apply_functions.append('gyrMag')
        if self.a_mag_Var.get() == 1:
            list_axes_to_apply_functions.append('accMag')
        
        globals.list_axes_to_apply_functions = list_axes_to_apply_functions
        
        no_of_axes = len(list_axes_to_apply_functions)
        if no_of_axes == 0:
            valid_to_running = False

        if not (self.txtWindowSizeFrom_text.get().isdigit()):
            valid_to_running = False

        if not (self.txtWindowSizeTo_text.get().isdigit()):
            valid_to_running = False

        if self.txtWindowSizeFrom_text.get().isdigit() and self.txtWindowSizeTo_text.get().isdigit():
            if int(self.txtWindowSizeFrom_text.get()) > int(self.txtWindowSizeTo_text.get()):
                valid_to_running = False

        if not (self.txtWindowStep_text.get().isdigit()):
            valid_to_running = False

        if int(self.cboDownSamplingFrom.get()) > int(self.cboDownSamplingTo.get()):
            valid_to_running = False

        no_of_classifiers = self.RandomForestVar.get() + self.DecisionTreeVar.get() + self.SVMVar.get() + self.NaiveBayesVar.get()
        if no_of_classifiers == 0:
            valid_to_running = False

        # Check valid to running
        error_message = ''
        if not valid_to_running:
            if no_of_labels < 2:
                error_message = 'Please select at least two different labels to proceed'
            elif no_of_axes == 0:
                error_message = 'Please select at least one axis to proceed'
            elif no_of_functions == 0:
                error_message = 'Please select at least one feature to proceed'
            elif not (self.txtWindowSizeFrom_text.get().isdigit()):
                error_message = 'Please insert a valid Window size in From field'
            elif not (self.txtWindowSizeTo_text.get().isdigit()):
                error_message = 'Please insert a valid Window size in To field'
            elif int(self.txtWindowSizeFrom_text.get()) > int(self.txtWindowSizeTo_text.get()):
                error_message = 'The From Window size should be less than To Window size'
            elif not (self.txtWindowStep_text.get().isdigit()):
                error_message = 'Please insert a valid Window size in Step field'
            elif int(self.cboDownSamplingFrom.get()) > int(self.cboDownSamplingTo.get()):
                error_message = 'The From field must be less than To field in resampling section'
            elif no_of_classifiers == 0:
                error_message = 'Please choose at least one classifier to proceed'
        return valid_to_running, error_message


    def convert_features_into_json_content(self):
        from models import features
        # Frequency domain features =>
        frPeakFr = lambda x: features.frPeakFreq(x, globals.resampling_rate) 

        json_axes_functions = {}
        for axis in globals.list_axes_to_apply_functions:
            json_axes_functions[axis] = []
            for func in globals.list_agg_function_names:
                json_axes_functions[axis] = json_axes_functions.get(axis) + [func]
        
        # manual fetures json creation ->
        # json_axes_functions = {'gx': ['median'], 'gy': ['mean'], 'gz': ['mean'], 'accMag': ['median']}
        # json_axes_functions = {'gyrMag': ['mean', 'stdev', 'median', 'IQR', 'RMS', 'MCR', 'Kurt', 'Skew'],'accMag': ['mean', 'stdev', 'median', 'IQR', 'RMS', 'MCR', 'Kurt', 'Skew']}
        # json_axes_functions = {'gx': ['max', 'mean', 'stdev', 'median'], 'gy': ['max', 'stdev', 'median'],
        #  'gz': ['max', 'mean', 'stdev', 'median'], 'ax': ['mean', 'stdev', 'median'],
        #  'ay': ['max', 'mean', 'stdev', 'median'], 'az': ['max', 'mean']}
        # list_axes_to_apply_functions = list(json_axes_functions.keys())
        # no_of_axes = len(list_axes_to_apply_functions)
        # manual json creation <-

        # converting json content into Python dictionary object
        features_in_dictionary = {}
        for x in json_axes_functions:
            features_in_dictionary[x] = []
            for val in json_axes_functions[x]:
                # if val == 'min':
                #     features_in_dictionary[x] = features_in_dictionary.get(x) + ['min']
                # if val == 'max':
                #     features_in_dictionary[x] = features_in_dictionary.get(x) + ['max']
                # if val == 'mean':
                #     features_in_dictionary[x] = features_in_dictionary.get(x) + ['mean']
                # if val == 'median':
                #     features_in_dictionary[x] = features_in_dictionary.get(x) + ['median']
                # if val == 'stdev':
                #     features_in_dictionary[x] = features_in_dictionary.get(x) + ['std']
                if val == 'min':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [np.min]
                if val == 'max':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [np.max]
                if val == 'mean':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [np.mean]
                if val == 'median':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [np.median]
                if val == 'stdev':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [np.std]
                if val == 'IQR':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.IQR]
                if val == 'RMS':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.RMS]
                if val == 'MCR':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.MCR]
                if val == 'Kurt':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.Kurt]
                if val == 'Skew':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.Skew]
                if val == 'Energy':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frEnergy]
                if val == 'PeakFreq':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [frPeakFr]
                if val == 'FreqEntrpy':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frDmEntroPy]
                if val == 'FirstCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frMag1]
                if val == 'SecondCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frMag2]
                if val == 'ThirdCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frMag3]
                if val == 'FourthCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frMag4]
                if val == 'FifthCpn':
                    features_in_dictionary[x] = features_in_dictionary.get(x) + [features.frMag5]
        
        globals.json_axes_functions = json_axes_functions
        globals.features_in_dictionary = features_in_dictionary

        # log_message("tab_training")
        # log_message('--4 below')
        # print(json_axes_functions)
        # print(type(json_axes_functions))
        # print(features_in_dictionary)
        # print(type(features_in_dictionary))


    def re_sampling_select(self):
        
        if self.rdoResampling.get() == 1:

            self.cboDownSamplingFrom.configure(state = 'normal')            
            self.lblDownSamplingFromHz.configure(state = 'normal')
            self.lblDownSamplingTo.configure(state = 'normal')
            self.cboDownSamplingTo.configure(state = 'normal')
            self.lblDownSamplingToHz.configure(state = 'normal')
            self.lblDownSamplingStep.configure(state = 'normal')
            self.cboDownSamplingStep.configure(state = 'normal')
            self.lblDownSamplingStepHz.configure(state = 'normal')
        else:
            self.cboDownSamplingFrom.configure(state = 'disabled')            
            self.lblDownSamplingFromHz.configure(state = 'disabled')
            self.lblDownSamplingTo.configure(state = 'disabled')
            self.cboDownSamplingTo.configure(state = 'disabled')
            self.lblDownSamplingToHz.configure(state = 'disabled')
            self.lblDownSamplingStep.configure(state = 'disabled')
            self.cboDownSamplingStep.configure(state = 'disabled')
            self.lblDownSamplingStepHz.configure(state = 'disabled')


    def multi_class_all_labels_select_deselect(self):
        if self.selectAllActivityOnOff.get() == 0:
            self.cboActivity1.configure(state = 'normal')
            self.cboActivity2.configure(state = 'normal')
            self.cboActivity3.configure(state = 'normal')
            self.cboActivity4.configure(state = 'normal')
            self.cboActivity5.configure(state = 'normal')
            self.cboActivity6.configure(state = 'normal')
            
        elif self.selectAllActivityOnOff.get() == 1:
            self.cboActivity1.configure(state = 'disabled')
            self.cboActivity2.configure(state = 'disabled')
            self.cboActivity3.configure(state = 'disabled')
            self.cboActivity4.configure(state = 'disabled')
            self.cboActivity5.configure(state = 'disabled')
            self.cboActivity6.configure(state = 'disabled')
            

    def features_select_deselect(self):
        if self.selectAllFeaturesOnOff.get() == 0:
            self.chkBtnMin.deselect()
            self.chkBtnMax.deselect()
            self.chkBtnMean.deselect()
            self.chkBtnMedian.deselect()
            self.chkBtnStd.deselect()
            self.btnChkIQR.deselect()
            self.btnRMS.deselect()
            self.btnMCR.deselect()
            self.btnkurt.deselect()
            self.btnskew.deselect()
            self.btnEnergy.deselect()
            self.btnPeakFreq.deselect()
            self.btnFredDmEntropy.deselect()
            self.btn1cpnMag.deselect()
            self.btn2cpnMag.deselect()
            self.btn3cpnMag.deselect()
            self.btn4cpnMag.deselect()
            self.btn5cpnMag.deselect()

        elif self.selectAllFeaturesOnOff.get() == 1:
            self.chkBtnMin.select()
            self.chkBtnMax.select()
            self.chkBtnMean.select()
            self.chkBtnMedian.select()
            self.chkBtnStd.select()
            self.btnChkIQR.select()
            self.btnRMS.select()
            self.btnMCR.select()
            self.btnkurt.select()
            self.btnskew.select()
            self.btnEnergy.select()
            self.btnPeakFreq.select()
            self.btnFredDmEntropy.select()
            self.btn1cpnMag.select()
            self.btn2cpnMag.select()
            self.btn3cpnMag.select()
            self.btn4cpnMag.select()
            self.btn5cpnMag.select()


    def rdoMultiBinaryClsSelectDeselect(self):
        if self.intRadioMultiBinaryClsSelect.get() == 1:

            # Enable multi-class classification section controls     
            # self.lblLabelsList.configure(state = 'normal')
            self.btnAllLabelsOnOff.configure(state = 'normal')
            self.multi_class_all_labels_select_deselect()

            # Disable binary classification section controls  
            self.lblBinaryMainLabel.configure(state = 'disabled')
            self.cboBinMainActivity.configure(state = 'disabled')
          
            self.lblBinaryNonMainLabels.configure(state = 'disabled')
            self.cboBinNonMainActivity1.configure(state = 'disabled')
            self.cboBinNonMainActivity2.configure(state = 'disabled')
            self.cboBinNonMainActivity3.configure(state = 'disabled')
            self.cboBinNonMainActivity4.configure(state = 'disabled')

            globals.binary_mode = False

        elif self.intRadioMultiBinaryClsSelect.get() ==  0:
            
            # Disable multi-class classification section controls 
            # self.lblLabelsList.configure(state = 'disabled')
            self.btnAllLabelsOnOff.configure(state = 'disabled')
            self.cboActivity1.configure(state = 'disabled')
            self.cboActivity2.configure(state = 'disabled')
            self.cboActivity3.configure(state = 'disabled')
            self.cboActivity4.configure(state = 'disabled')
            self.cboActivity5.configure(state = 'disabled')
            self.cboActivity6.configure(state = 'disabled')
           
            # Enable binary classification section controls
            self.lblBinaryMainLabel.configure(state = 'normal')
            self.cboBinMainActivity.configure(state = 'normal')

            self.lblBinaryNonMainLabels.configure(state = 'normal')
            self.cboBinNonMainActivity1.configure(state = 'normal')
            self.cboBinNonMainActivity2.configure(state = 'normal')
            self.cboBinNonMainActivity3.configure(state = 'normal')
            self.cboBinNonMainActivity4.configure(state = 'normal')                      
            
            globals.binary_mode = True


    def reset_activity_cbo_values(self):
        self.cboActivity1['values'] = 'None'
        self.cboActivity1.current(0)
        self.cboActivity2['values'] = 'None'
        self.cboActivity2.current(0)
        self.cboActivity3['values'] = 'None'
        self.cboActivity3.current(0)
        self.cboActivity4['values'] = 'None'
        self.cboActivity4.current(0)
        self.cboActivity5['values'] = 'None'
        self.cboActivity5.current(0)
        self.cboActivity6['values'] = 'None'
        self.cboActivity6.current(0)

        self.cboBinMainActivity['values'] = 'None'
        self.cboBinMainActivity.current(0)
        self.cboBinNonMainActivity1['values'] = 'None'
        self.cboBinNonMainActivity1.current(0)
        self.cboBinNonMainActivity2['values'] = 'None'
        self.cboBinNonMainActivity2.current(0)
        self.cboBinNonMainActivity3['values'] = 'None'
        self.cboBinNonMainActivity3.current(0)
        self.cboBinNonMainActivity4['values'] = 'None'
        self.cboBinNonMainActivity4.current(0)


    def update_activity_cbo_values(self):
        # Get the list of all labels from the dataset and push them into the combo boxes
        for i in globals.cboActivityValues:
            self.cboActivity1['values'] += (i,)
            self.cboActivity2['values'] += (i,)
            self.cboActivity3['values'] += (i,)
            self.cboActivity4['values'] += (i,)
            self.cboActivity5['values'] += (i,)
            self.cboActivity6['values'] += (i,)

            self.cboBinMainActivity['values'] += (i,)
            self.cboBinNonMainActivity1['values'] += (i,)
            self.cboBinNonMainActivity2['values'] += (i,)
            self.cboBinNonMainActivity3['values'] += (i,)
            self.cboBinNonMainActivity4['values'] += (i,)
       

    def fill_init_data_into_controls(self):

        params_training = ini_file.get_training_tab_layout_data()

        if params_training[0][1] == '1':
            self.btnAllLabelsOnOff.select()
        else:
            self.btnAllLabelsOnOff.deselect()

        self.multi_class_all_labels_select_deselect()

        if globals.binary_mode:
            self.radMulticlassSelect.deselect()          
        else:
            self.radMulticlassSelect.select()
        
        # Disable the mutual controls
        self.rdoMultiBinaryClsSelectDeselect()    

        if params_training[1][1] == '1':
            self.btnAllFeaturesOnOff.select()
        else:
            self.btnAllFeaturesOnOff.deselect()
        
        self.features_select_deselect()

        if params_training[2][1] == '1':
            self.chkBtnMin.select()
        else:
            self.chkBtnMin.deselect()

        if params_training[3][1] == '1':
            self.chkBtnMax.select()
        else:
            self.chkBtnMax.deselect()

        if params_training[4][1] == '1':
            self.chkBtnMean.select()
        else:
            self.chkBtnMean.deselect()

        if params_training[5][1] == '1':
            self.chkBtnMedian.select()
        else:
            self.chkBtnMedian.deselect()

        if params_training[6][1] == '1':
            self.chkBtnStd.select()
        else:
            self.chkBtnStd.deselect()

        if params_training[7][1] == '1':
            self.btnChkIQR.select()
        else:
            self.btnChkIQR.deselect()

        if params_training[8][1] == '1':
            self.btnRMS.select()
        else:
            self.btnRMS.deselect()

        if params_training[9][1] == '1':
            self.btnMCR.select()
        else:
            self.btnMCR.deselect()

        if params_training[10][1] == '1':
            self.btnkurt.select()
        else:
            self.btnkurt.deselect()

        if params_training[11][1] == '1':
            self.btnskew.select()
        else:
            self.btnskew.deselect()

        if params_training[12][1] == '1':
            self.btnEnergy.select()
        else:
            self.btnEnergy.deselect()

        if params_training[13][1] == '1':
            self.btnPeakFreq.select()
        else:
            self.btnPeakFreq.deselect()

        if params_training[14][1] == '1':
            self.btnFredDmEntropy.select()
        else:
            self.btnFredDmEntropy.deselect()

        if params_training[15][1] == '1':
            self.btn1cpnMag.select()
        else:
            self.btn1cpnMag.deselect()

        if params_training[16][1] == '1':
            self.btn2cpnMag.select()
        else:
            self.btn2cpnMag.deselect()

        if params_training[17][1] == '1':
            self.btn3cpnMag.select()
        else:
            self.btn3cpnMag.deselect()

        if params_training[18][1] == '1':
            self.btn4cpnMag.select()
        else:
            self.btn4cpnMag.deselect()

        if params_training[19][1] == '1':
            self.btn5cpnMag.select()
        else:
            self.btn5cpnMag.deselect()

        self.txtWindowSizeFrom_text.set(params_training[20][1])
        self.txtWindowSizeTo_text.set((params_training[21][1]))
        self.txtWindowStep_text.set(params_training[22][1])
        self.txtWindowStride_text.set(params_training[23][1])
        self.txtWindowSimuStride_text.set(params_training[24][1])

        if int(params_training[25][1]) < 11:
            self.cboDownSamplingFrom.current(int(params_training[25][1]) - 1)
        else:
            self.cboDownSamplingFrom.current(0)

        if int(params_training[26][1]) < 11:
            self.cboDownSamplingTo.current(int(params_training[26][1]) - 1)
        else:
            self.cboDownSamplingTo.current(0)
        
        if int(params_training[27][1]) < 11:
            self.cboDownSamplingStep.current(int(params_training[27][1]) - 1)
        else:
            self.cboDownSamplingStep.current(0)
        
        if params_training[28][1] == '1':
            self.btnChkRandomForest.select()
        else:
            self.btnChkRandomForest.deselect()

        if params_training[29][1] == '1':
            self.btnChkDecisionTree.select()
        else:
            self.btnChkDecisionTree.deselect()
        
        if params_training[30][1] == '1':
            self.btnChkSVM.select()
        else:
            self.btnChkSVM.deselect()
        
        if params_training[31][1] == '1':
            self.btnChkNaiveBayes.select()
        else:
            self.btnChkNaiveBayes.deselect()

        self.cboKfold.current(int(params_training[32][1]) - 1)

        if params_training[33][1] == '1':
            self.radReSample.select()
        else:
            self.radReSample.deselect()
        self.re_sampling_select()            

        self.txtSimulationWindowSize_text.set(params_training[34][1])
        self.txtSimuFrameDtPoints_text.set(params_training[35][1])
        self.txtSimuFrameStride_text.set(params_training[36][1])
        self.txtSimuFrameDelay_text.set(params_training[37][1])
        self.txtSimuFrameRepeat_text.set(params_training[38][1])


        if params_training[39][1] == '1':
            self.chkBtnGx.select()
        else:
            self.chkBtnGx.deselect()

        if params_training[40][1] == '1':
            self.chkBtnGy.select()
        else:
            self.chkBtnGy.deselect()

        if params_training[41][1] == '1':
            self.chkBtnGz.select()
        else:
            self.chkBtnGz.deselect()

        if params_training[42][1] == '1':
            self.chkBtnAx.select()
        else:
            self.chkBtnAx.deselect()

        if params_training[43][1] == '1':
            self.chkBtnAy.select()
        else:
            self.chkBtnAy.deselect()

        if params_training[44][1] == '1':
            self.chkBtnAz.select()
        else:
            self.chkBtnAz.deselect()

        if params_training[45][1] == '1':
            self.btnChk_gmag.select()
        else:
            self.btnChk_gmag.deselect()

        if params_training[46][1] == '1':
            self.btnChk_amag.select()
        else:
            self.btnChk_amag.deselect()
    

    def build_controls(self):

        self.intRadioMultiBinaryClsSelect = tk.IntVar()
        self.rdoResampling = tk.IntVar()
        self.rdoSimuStartTime = tk.IntVar()
        self.selectAllActivityOnOff = tk.IntVar()

        self.lbl_frame_cls_problems = tk.LabelFrame(self, text = 'Classification problem',font = ('Sans', '10', 'bold'))
        self.lbl_frame_cls_problems.grid (row = 0, column = 0, padx = 20, pady = 8, sticky = tk.W)

        # Multi-class section
        self.radMulticlassSelect = tk.Radiobutton(self.lbl_frame_cls_problems, text = 'Multi-class',  command = self.rdoMultiBinaryClsSelectDeselect, variable = self.intRadioMultiBinaryClsSelect, value = 1)
        self.radMulticlassSelect.grid(row = 0, column = 0, sticky = tk.W)
        self.radMulticlassSelect.configure(font = ('Sans', '10', 'bold'))
        
        self.btnAllLabelsOnOff = tk.Checkbutton(self.lbl_frame_cls_problems, command = self.multi_class_all_labels_select_deselect, text = 'All activities',
                                        variable = self.selectAllActivityOnOff, onvalue = 1, offvalue = 0)
        self.btnAllLabelsOnOff.configure(font = ('Sans', '9'))
        self.btnAllLabelsOnOff.grid(row = 0, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.cboActivity1 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity1.grid(row = 0, column = 2, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity1.current(0)
        self.cboActivity2 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity2.grid(row = 0, column = 3, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity2.current(0)
        self.cboActivity3 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity3.grid(row = 0, column = 4, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity3.current(0)
        self.cboActivity4 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity4.grid(row = 0, column = 5, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity4.current(0)
        self.cboActivity5 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity5.grid(row = 0, column = 6, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity5.current(0)
        self.cboActivity6 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboActivity6.grid(row = 0, column = 7, padx = 5, pady = 2, sticky = tk.W)
        self.cboActivity6.current(0)

        self.radBinaryclassSelect = tk.Radiobutton(self.lbl_frame_cls_problems, text = 'Binary',  command = self.rdoMultiBinaryClsSelectDeselect, variable = self.intRadioMultiBinaryClsSelect, value = 0)
        self.radBinaryclassSelect.grid(row = 1, column = 0, sticky = tk.W)
        self.radBinaryclassSelect.configure(font = ('Sans', '10', 'bold'))

        # Binary classification section
        self.lblBinaryMainLabel = tk.Label(self.lbl_frame_cls_problems, text = 'Main activity')
        self.lblBinaryMainLabel.configure(font = ('Sans', '10'))
        self.lblBinaryMainLabel.grid(row = 1, column = 1, padx = 5, pady = 2, sticky = tk.W)

        self.cboBinMainActivity = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboBinMainActivity.grid(row = 1, column = 2, padx = 5, pady = 2, sticky = tk.W)
        self.cboBinMainActivity.current(0)

        self.lblBinaryNonMainLabels = tk.Label(self.lbl_frame_cls_problems, text = 'Non-main activities')
        self.lblBinaryNonMainLabels.grid(row = 2, column = 1, padx = 5, pady = 3, sticky = tk.W)
        self.lblBinaryNonMainLabels.configure(font = ('Sans', '10'))
        self.cboBinNonMainActivity1 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboBinNonMainActivity1.grid(row = 2, column = 2, padx = 5, pady = 3, sticky = tk.W)
        self.cboBinNonMainActivity1.current(0)
        self.cboBinNonMainActivity2 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboBinNonMainActivity2.grid(row = 2, column = 3, padx = 5, pady = 3, sticky = tk.W)
        self.cboBinNonMainActivity2.current(0)
        self.cboBinNonMainActivity3 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboBinNonMainActivity3.grid(row = 2, column = 4, padx = 5, pady = 3, sticky = tk.W)
        self.cboBinNonMainActivity3.current(0)
        self.cboBinNonMainActivity4 = ttk.Combobox(self.lbl_frame_cls_problems, width = '13', values = 'None')
        self.cboBinNonMainActivity4.grid(row = 2, column = 5, padx = 5, pady = 3, sticky = tk.W)
        self.cboBinNonMainActivity4.current(0)

        self.lbl_frame_resampling = tk.LabelFrame(self, text = 'Data resampling',font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_frame_resampling.grid (row = 1, column = 0, padx = 20, pady = 5, sticky = tk.W)

        self.radKeepOriginalSample = tk.Radiobutton(self.lbl_frame_resampling , text = 'Keep original data',  command = self.re_sampling_select, variable = self.rdoResampling, value = 0)
        self.radKeepOriginalSample.grid(row = 0, column = 0, sticky = tk.W)
        self.radKeepOriginalSample.configure(font = ('Sans', '11'))

        self.radReSample = tk.Radiobutton(self.lbl_frame_resampling , text = 'Resample data from',  command = self.re_sampling_select, variable = self.rdoResampling, value = 1)
        self.radReSample.grid(row = 0, column = 1, sticky = tk.W)
        self.radReSample.configure(font = ('Sans', '11'))

        self.cboDownSamplingFrom = ttk.Combobox(self.lbl_frame_resampling , width = '2', values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        self.cboDownSamplingFrom.grid(row = 0, column = 2, padx = 5,  pady = 3, sticky = tk.W)
        self.cboDownSamplingFrom.current(0)

        self.lblDownSamplingFromHz = tk.Label(self.lbl_frame_resampling , text = 'Hz')
        self.lblDownSamplingFromHz.grid(row = 0, column = 3, sticky = tk.W)

        self.lblDownSamplingTo = tk.Label(self.lbl_frame_resampling , width = '5', text = 'to', justify = tk.RIGHT)
        self.lblDownSamplingTo.configure(font = ('Sans', '10', 'bold'))
        self.lblDownSamplingTo.grid(row = 0, column = 4,   pady = 3, sticky = tk.E)

        self.cboDownSamplingTo = ttk.Combobox(self.lbl_frame_resampling , width = '2', values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        self.cboDownSamplingTo.grid(row = 0, column = 5, padx = 5,  pady = 3, sticky = tk.W)
        self.cboDownSamplingTo.current(0)

        self.lblDownSamplingToHz = tk.Label(self.lbl_frame_resampling , text = 'Hz')
        self.lblDownSamplingToHz.grid(row = 0, column = 6, sticky = tk.W)

        self.lblDownSamplingStep = tk.Label(self.lbl_frame_resampling ,  width = '5', text = 'step', justify = tk.RIGHT)
        self.lblDownSamplingStep.grid(row = 0, column = 7, sticky = tk.E)
        self.lblDownSamplingStep.configure(font = ('Sans', '10', 'bold'))

        self.cboDownSamplingStep = ttk.Combobox(self.lbl_frame_resampling , width = '2', values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        self.cboDownSamplingStep.grid(row = 0, column = 8, padx = 5,  pady = 3, sticky = tk.W)
        self.cboDownSamplingStep.current(0)

        self.lblDownSamplingStepHz = tk.Label(self.lbl_frame_resampling , text = 'Hz')
        self.lblDownSamplingStepHz.grid(row = 0, column = 9, sticky = tk.W)

        # Window settings
        self.lbl_window_settings = tk.LabelFrame(self, text = 'Window settings', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_window_settings.grid (row = 2, column = 0, padx = 20, pady = 5, sticky = tk.W)

        self.lblWindowBasementSize = tk.Label(self.lbl_window_settings, text = 'Size from')
        self.lblWindowBasementSize.configure(font = ('Sans', '11'))
        self.lblWindowBasementSize.grid(row = 0, column = 0, padx = 5, pady = 3, sticky = tk.W)

        self.txtWindowSizeFrom_text = tk.StringVar()
        self.txtWindowSizeFrom = tk.Entry(self.lbl_window_settings, width = 8, textvariable = self.txtWindowSizeFrom_text)
        self.txtWindowSizeFrom.grid(row = 0, column = 1, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowSizeMs = tk.Label(self.lbl_window_settings, text = 'ms', justify = tk.LEFT)
        self.lblWindowSizeMs.grid(row = 0, column = 2, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowIn = tk.Label(self.lbl_window_settings, text = 'to', justify = tk.RIGHT)
        self.lblWindowIn.configure(font = ('Sans', '10', 'bold'))
        self.lblWindowIn.grid(row = 0, column = 4,   pady = 3, sticky = tk.E)

        self.txtWindowSizeTo_text = tk.StringVar()
        self.txtWindowSizeTo = tk.Entry(self.lbl_window_settings, width = 8, textvariable = self.txtWindowSizeTo_text)
        self.txtWindowSizeTo.grid(row = 0, column = 5, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowTo = tk.Label(self.lbl_window_settings, text = 'ms', justify = tk.LEFT)
        self.lblWindowTo.grid(row = 0, column = 6, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowIncrement = tk.Label(self.lbl_window_settings, text = 'step', justify = tk.RIGHT)
        self.lblWindowIncrement.grid(row = 0, column = 7, padx = 5, pady = 3, sticky = tk.W)
        self.lblWindowIncrement.configure(font = ('Sans', '10', 'bold'))

        self.txtWindowStep_text = tk.StringVar()
        self.txtWindowStep = tk.Entry(self.lbl_window_settings, width = 8, textvariable = self.txtWindowStep_text)
        self.txtWindowStep.grid(row = 0, column = 8, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowToStep = tk.Label(self.lbl_window_settings, text = 'ms')
        self.lblWindowToStep.grid(row = 0, column = 9, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowStride = tk.Label(self.lbl_window_settings, justify = tk.RIGHT, text = 'Train (Test) stride')
        self.lblWindowStride.configure(font = ('Sans', '10'))
        self.lblWindowStride.grid(row = 0, column = 10, padx = 5, pady = 3, sticky = tk.W)

        self.txtWindowStride_text = tk.StringVar()
        self.txtWindowStride = tk.Entry(self.lbl_window_settings, width = 8, textvariable = self.txtWindowStride_text)
        self.txtWindowStride.grid(row = 0, column = 11, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowStrideMs = tk.Label(self.lbl_window_settings, justify = tk.LEFT, text = '%')
        self.lblWindowStrideMs.grid(row = 0, column = 12, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowTesingStride = tk.Label(self.lbl_window_settings, justify = tk.RIGHT, text = 'Monitoring stride')
        self.lblWindowTesingStride.configure(font = ('Sans', '10'))
        self.lblWindowTesingStride.grid(row = 0, column = 13, padx = 5, pady = 3, sticky = tk.W)

        self.txtWindowSimuStride_text = tk.StringVar()
        self.txtWindowSimuStride = tk.Entry(self.lbl_window_settings, width = 8, textvariable = self.txtWindowSimuStride_text)
        self.txtWindowSimuStride.grid(row = 0, column = 14, padx = 5, pady = 3, sticky = tk.W)

        self.lblWindowTestingStrideMs = tk.Label(self.lbl_window_settings, justify = tk.LEFT, text = '%')
        self.lblWindowTestingStrideMs.grid(row = 0, column = 15, padx = 5, pady = 3, sticky = tk.W)

        # Functions select
        self.lbl_functions_select = tk.LabelFrame(self, text = 'Functions select', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_functions_select.grid (row = 3, column = 0, padx = 18, pady = 5, sticky = tk.W)

        self.selectAllFeaturesOnOff = tk.IntVar()
        self.btnAllFeaturesOnOff = tk.Checkbutton(self.lbl_functions_select, command = self.features_select_deselect, text = 'Select all',
                                        variable = self.selectAllFeaturesOnOff, onvalue = 1, offvalue = 0)
        self.btnAllFeaturesOnOff.configure(font = ('Sans', '10'))
        self.btnAllFeaturesOnOff.grid(row = 0, column = 0, sticky = tk.W)

        self.MinVar = tk.IntVar()
        self.chkBtnMin = tk.Checkbutton(self.lbl_functions_select, text = 'Min', variable = self.MinVar, onvalue = 1, offvalue = 0, height = 1)
        self.chkBtnMin.grid(row = 0, column = 1, sticky = tk.W)

        self.MaxVar = tk.IntVar()
        self.chkBtnMax = tk.Checkbutton(self.lbl_functions_select, text = 'Max', variable = self.MaxVar, onvalue = 1, offvalue = 0, height = 1)
        self.chkBtnMax.grid(row = 0, column = 2, sticky = tk.W)

        self.MeanVar = tk.IntVar()
        self.chkBtnMean = tk.Checkbutton(self.lbl_functions_select, text = 'Mean', variable = self.MeanVar, onvalue = 1, offvalue = 0, height = 1)
        self.chkBtnMean.grid(row = 0, column = 3, sticky = tk.W)

        self.MedianVar = tk.IntVar()
        self.chkBtnMedian = tk.Checkbutton(self.lbl_functions_select, text = 'Median', variable = self.MedianVar, onvalue = 1, offvalue = 0, height = 1)
        self.chkBtnMedian.grid(row = 0, column = 4, sticky = tk.W)

        self.StdVar = tk.IntVar()
        self.chkBtnStd = tk.Checkbutton(self.lbl_functions_select, text = 'Stdev', variable = self.StdVar, onvalue = 1, offvalue = 0, height = 1)
        self.chkBtnStd.grid(row = 0, column = 5, sticky = tk.W)

        self.IQRVar = tk.IntVar()
        self.btnChkIQR = tk.Checkbutton(self.lbl_functions_select, text = 'IntQtlRange', variable = self.IQRVar, onvalue = 1, offvalue = 0, height = 1)
        self.btnChkIQR.grid(row = 0, column = 6, sticky = tk.W)

        self.rootMSVar = tk.IntVar()
        self.btnRMS = tk.Checkbutton(self.lbl_functions_select, text = 'RootMS', variable = self.rootMSVar, onvalue = 1, offvalue = 0, height = 1)
        self.btnRMS.grid(row = 0, column = 7, sticky = tk.W)

        self.meanCRVar = tk.IntVar()
        self.btnMCR = tk.Checkbutton(self.lbl_functions_select, text = 'MeanCR', variable = self.meanCRVar, onvalue = 1, offvalue = 0, height = 1)
        self.btnMCR.grid(row = 0, column = 8, sticky = tk.W)

        self.kurtosisVar = tk.IntVar()
        self.btnkurt = tk.Checkbutton(self.lbl_functions_select, text = 'Kurtosis', variable = self.kurtosisVar, onvalue = 1, offvalue = 0, height = 1)
        self.btnkurt.grid(row = 0, column = 9, sticky = tk.W)

        self.skewnessVar = tk.IntVar()
        self.btnskew = tk.Checkbutton(self.lbl_functions_select, text = 'Skewness', variable = self.skewnessVar, onvalue = 1, offvalue = 0, height = 2)
        self.btnskew.grid(row = 1, column = 1, sticky = tk.W)

        self.energyVar = tk.IntVar()
        self.btnEnergy = tk.Checkbutton(self.lbl_functions_select, text = 'Spectral Energy', variable = self.energyVar, onvalue = 1, offvalue = 0, height = 2)
        self.btnEnergy.grid(row = 1, column = 2, sticky = tk.W)

        self.peakFreqVar = tk.IntVar()
        self.btnPeakFreq = tk.Checkbutton(self.lbl_functions_select, text = 'PeakFreq', variable = self.peakFreqVar, onvalue = 1, offvalue = 0, height = 2)
        self.btnPeakFreq.grid(row = 1, column = 3, sticky = tk.W)

        self.freqDmEntropyVar = tk.IntVar()
        self.btnFredDmEntropy = tk.Checkbutton(self.lbl_functions_select, text = 'FreqEntropy', variable = self.freqDmEntropyVar, onvalue = 1, offvalue = 0, height = 2)
        self.btnFredDmEntropy.grid(row = 1, column = 4, sticky = tk.W)

        self.fr1cpnMagVar = tk.IntVar()
        self.btn1cpnMag = tk.Checkbutton(self.lbl_functions_select, text = '1stCpnMag', variable = self.fr1cpnMagVar, onvalue = 1, offvalue = 0, height = 2)
        self.btn1cpnMag.grid(row = 1, column = 5, sticky = tk.W)

        self.fr2cpnMagVar = tk.IntVar()
        self.btn2cpnMag = tk.Checkbutton(self.lbl_functions_select, text = '2ndCpnMag', variable = self.fr2cpnMagVar, onvalue = 1, offvalue = 0, height = 2)
        self.btn2cpnMag.grid(row = 1, column = 6, sticky = tk.W)

        self.fr3cpnMagVar = tk.IntVar()
        self.btn3cpnMag = tk.Checkbutton(self.lbl_functions_select, text = '3rdCpnMag', variable = self.fr3cpnMagVar, onvalue = 1, offvalue = 0, height = 2)
        self.btn3cpnMag.grid(row = 1, column = 7, sticky = tk.W)

        self.fr4cpnMagVar = tk.IntVar()
        self.btn4cpnMag = tk.Checkbutton(self.lbl_functions_select, text = '4thCpnMag', variable = self.fr4cpnMagVar, onvalue = 1, offvalue = 0, height = 2)
        self.btn4cpnMag.grid(row = 1, column = 8, sticky = tk.W)

        self.fr5cpnMagVar = tk.IntVar()
        self.btn5cpnMag = tk.Checkbutton(self.lbl_functions_select, text = '5thCpnMag', variable = self.fr5cpnMagVar, onvalue = 1, offvalue = 0, height = 2)
        self.btn5cpnMag.grid(row = 1, column = 9, sticky = tk.W)

        # Axes
        self.lbl_axes_settings = tk.LabelFrame(self, text = 'Axes to be applied', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_axes_settings.grid (row = 4, column = 0, padx = 20, pady = 5, sticky = tk.W)
        
        self.gxVar = tk.IntVar()
        self.chkBtnGx = tk.Checkbutton(self.lbl_axes_settings, text = 'Gx', variable = self.gxVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnGx.grid(row = 0, column = 0, sticky = tk.W)
        
        self.gyVar = tk.IntVar()
        self.chkBtnGy = tk.Checkbutton(self.lbl_axes_settings, text = 'Gy', variable = self.gyVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnGy.grid(row = 0, column = 1, sticky = tk.W)
        
        self.gzVar = tk.IntVar()
        self.chkBtnGz = tk.Checkbutton(self.lbl_axes_settings, text = 'Gz', variable = self.gzVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnGz.grid(row = 0, column = 2, sticky = tk.W)
        
        self.axVar = tk.IntVar()
        self.chkBtnAx = tk.Checkbutton(self.lbl_axes_settings, text = 'Ax', variable = self.axVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnAx.grid(row = 0, column = 3, sticky = tk.W)
        
        self.ayVar = tk.IntVar()
        self.chkBtnAy = tk.Checkbutton(self.lbl_axes_settings, text = 'Ay', variable = self.ayVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnAy.grid(row = 0, column = 4, sticky = tk.W)
        
        self.azVar = tk.IntVar()
        self.chkBtnAz = tk.Checkbutton(self.lbl_axes_settings, text = 'Az', variable = self.azVar, onvalue = 1, offvalue = 0, height = 2, width = 4)
        self.chkBtnAz.grid(row = 0, column = 5, sticky = tk.W)

        self.g_mag_Var = tk.IntVar()
        self.btnChk_gmag = tk.Checkbutton(self.lbl_axes_settings, text = 'gyro magnitude', variable = self.g_mag_Var, onvalue = 1, offvalue = 0, height = 2,
                                width = 15)
        self.btnChk_gmag.grid(row = 0, column = 6, sticky = tk.W)
        
        self.a_mag_Var = tk.IntVar()
        self.btnChk_amag = tk.Checkbutton(self.lbl_axes_settings, text = 'acc magnitude', variable = self.a_mag_Var, onvalue = 1, offvalue = 0, height = 2,
                                width = 15)
        self.btnChk_amag.grid(row = 0, column = 7, sticky = tk.W)
        
        # Classifiers
        self.lbl_classifiers = tk.LabelFrame(self, text = 'Classifiers', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_classifiers.grid (row = 5, column = 0, padx = 20, pady = 5 , sticky = tk.W)

        self.RandomForestVar = tk.IntVar()
        self.btnChkRandomForest = tk.Checkbutton(self.lbl_classifiers, text = 'Random Forest', variable = self.RandomForestVar, onvalue = 1, offvalue = 0, height = 2)
        self.btnChkRandomForest.grid(row = 0, column = 0, sticky = tk.W)

        self.DecisionTreeVar = tk.IntVar()
        self.btnChkDecisionTree = tk.Checkbutton(self.lbl_classifiers, text = 'Decision Tree', variable = self.DecisionTreeVar, onvalue = 1, offvalue = 0,
                                        height = 2, width = 13)
        self.btnChkDecisionTree.grid(row = 0, column = 1, sticky = tk.W)

        self.SVMVar = tk.IntVar()
        self.btnChkSVM = tk.Checkbutton(self.lbl_classifiers, text = 'SVM', variable = self.SVMVar, onvalue = 1, offvalue = 0, height = 2, width = 13)
        self.btnChkSVM.grid(row = 0, column = 2, sticky = tk.W)

        self.NaiveBayesVar = tk.IntVar()
        self.btnChkNaiveBayes = tk.Checkbutton(self.lbl_classifiers, text = 'Naive Bayes', variable = self.NaiveBayesVar, onvalue = 1, offvalue = 0, height = 2,
                                    width = 13)
        self.btnChkNaiveBayes.grid(row = 0, column = 3, sticky = tk.W)

        self.lblKfold = tk.Label(self.lbl_classifiers, text = 'Using', width = 10)
        self.lblKfold.configure(font = ('Sans', '10', 'bold'))
        self.lblKfold.grid(row = 0, column = 4, sticky = tk.W)
        
        self.cboKfold = ttk.Combobox(self.lbl_classifiers, width = '2', values=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        self.cboKfold.grid(row = 0, column = 5, sticky = tk.W)
        self.cboKfold.current(9)
        
        self.lblKfoldValidation = tk.Label(self.lbl_classifiers, text = ' - fold validation')
        self.lblKfoldValidation.configure(font = ('Sans', '10'))
        self.lblKfoldValidation.grid(row = 0, column = 6, sticky = tk.W)

        # Monitoring
        self.lbl_monitoring_settings = tk.LabelFrame(self, text = 'Monitoring settings', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_monitoring_settings.grid (row = 6, column = 0, padx = 20, pady = 5 , sticky = tk.W)

        self.lblSimulationSampleRate = tk.Label(self.lbl_monitoring_settings, text = 'Sampling rate', height = 2)
        self.lblSimulationSampleRate.configure(font = ('Sans', '11'))
        self.lblSimulationSampleRate.grid(row = 0, column = 0, sticky = tk.W)

        self.cboSimulationSampleRate = ttk.Combobox(self.lbl_monitoring_settings, width = '5',
                                            values=('10', '9', '8', '7', '6', '5', '4', '3', '2', '1'))
        self.cboSimulationSampleRate.grid(row = 0, column = 1, padx = 5,  pady = 3, sticky = tk.W)
        self.cboSimulationSampleRate.current(0)

        self.lblSimulationSampleRateHz = tk.Label(self.lbl_monitoring_settings, text = 'Hz')
        self.lblSimulationSampleRateHz.grid(row = 0, column = 2, sticky = tk.W)

        self.lblSimulationWindowSize = tk.Label(self.lbl_monitoring_settings, text = 'Window size')
        self.lblSimulationWindowSize.configure(font = ('Sans', '11'))
        self.lblSimulationWindowSize.grid(row = 0, column = 3, padx = 5, pady = 3, sticky = tk.W)

        self.txtSimulationWindowSize_text = tk.StringVar()
        self.txtSimulationWindowSize = tk.Entry(self.lbl_monitoring_settings, width = 8, textvariable = self.txtSimulationWindowSize_text)
        self.txtSimulationWindowSize.grid(row = 0, column = 4, padx = 5, pady = 3, sticky = tk.W)

        self.lblSimulationWindowSizeMs = tk.Label(self.lbl_monitoring_settings, text = 'ms')
        self.lblSimulationWindowSizeMs.grid(row = 0, column = 5, sticky = tk.W)

        self.lblSimuAlgorithms = tk.Label(self.lbl_monitoring_settings, text = 'Algorithm')
        self.lblSimuAlgorithms.configure(font = ('Sans', '11'))
        self.lblSimuAlgorithms.grid(row = 0, column = 6, padx = 5, pady = 3, sticky = tk.W)

        self.cboSimuAlgorithm = ttk.Combobox(self.lbl_monitoring_settings, width = '15',
                                        values=('Random_Forest', 'Decision_Tree', 'SVM', 'Naive_Bayes'))
        self.cboSimuAlgorithm.grid(row = 0, column = 7, padx = 5, pady = 3, sticky = tk.W)
        self.cboSimuAlgorithm.current(0)

        self.lbl_live_plot = tk.LabelFrame(self, text = 'Live plotting', font = ('Sans', '10', 'bold'), borderwidth = 1, highlightthickness = 0)
        self.lbl_live_plot.grid (row = 7, column = 0, padx = 20, pady = 5 , sticky = tk.W)

        self.lblSimuPlotFrameTimeStart = tk.Label(self.lbl_live_plot, text = 'Plot from ')
        self.lblSimuPlotFrameTimeStart.configure(font = ('Sans', '10'))
        self.lblSimuPlotFrameTimeStart.grid(row = 0, column = 0, padx = 5, pady = 3, sticky= tk.W)

        self.radSimuFrameStartTime = tk.Radiobutton(self.lbl_live_plot, text = 'Beginning', variable = self.rdoSimuStartTime, value = 0)
        self.radSimuFrameStartTime.grid(row = 0, column = 1, padx = 5, pady = 3, sticky= tk.W)
        self.radSimuFrameStartTime.configure(font = ('Sans', '10',))

        self.radSimuFrameFromTime = tk.Radiobutton(self.lbl_live_plot, text = 'Epoch timestamp', variable = self.rdoSimuStartTime, value = 1)
        self.radSimuFrameFromTime.grid(row = 0, column = 3, padx = 5, pady = 3, sticky= tk.W)
        self.radSimuFrameFromTime.configure(font = ('Sans', '10'))

        self.txtSimuFrameFromTime_text = tk.StringVar()
        self.txtSimuFrameFromTime = tk.Entry(self.lbl_live_plot, width = 20, textvariable = self.txtSimuFrameFromTime_text)
        self.txtSimuFrameFromTime.grid(row = 0, column = 4, padx = 5, pady = 3, sticky= tk.W)

        self.lblSimuFrameDtPoints = tk.Label(self.lbl_live_plot, text = 'Data pts in a frame')
        self.lblSimuFrameDtPoints.configure(font = ('Sans', '10'))
        self.lblSimuFrameDtPoints.grid(row = 1, column = 0, padx = 5, pady = 3, sticky= tk.W)

        self.txtSimuFrameDtPoints_text = tk.StringVar()
        self.txtSimuFrameDtPoints = tk.Entry(self.lbl_live_plot, width = 15, textvariable = self.txtSimuFrameDtPoints_text)
        self.txtSimuFrameDtPoints.grid(row = 1, column = 1, padx = 5, pady = 3, sticky= tk.W)

        self.lblSimuFrameStride = tk.Label(self.lbl_live_plot, text = 'Frame stride')
        self.lblSimuFrameStride.configure(font = ('Sans', '10'))
        self.lblSimuFrameStride.grid(row = 1, column = 3, padx = 5, pady = 3, sticky= tk.W)

        self.txtSimuFrameStride_text = tk.StringVar()
        self.txtSimuFrameStride = tk.Entry(self.lbl_live_plot, width = 20, textvariable = self.txtSimuFrameStride_text)
        self.txtSimuFrameStride.grid(row = 1, column = 4, padx = 5, pady = 3, sticky= tk.W)

        self.lblSimuFrameDelay = tk.Label(self.lbl_live_plot, text = 'Frame delay')
        self.lblSimuFrameDelay.configure(font = ('Sans', '10'))
        self.lblSimuFrameDelay.grid(row = 2, column = 0, padx = 5, pady = 3, sticky= tk.W)

        self.txtSimuFrameDelay_text = tk.StringVar()
        self.txtSimuFrameDelay = tk.Entry(self.lbl_live_plot, width = 15, textvariable = self.txtSimuFrameDelay_text)
        self.txtSimuFrameDelay.grid(row = 2, column = 1, padx = 5, pady = 3, sticky= tk.W)

        self.lblSimuFrameDelayms = tk.Label(self.lbl_live_plot, text = 'ms')
        self.lblSimuFrameDelayms.grid(row = 2, column = 2, padx = 5, pady = 3, sticky= tk.W)

        self.lblSimuFrameRepeat = tk.Label(self.lbl_live_plot, text = 'Repeat times')
        self.lblSimuFrameRepeat.configure(font = ('Sans', '10'))
        self.lblSimuFrameRepeat.grid(row = 2, column = 3, padx = 5, pady = 3, sticky= tk.W)

        self.txtSimuFrameRepeat_text = tk.StringVar()
        self.txtSimuFrameRepeat = tk.Entry(self.lbl_live_plot, width = 20, textvariable = self.txtSimuFrameRepeat_text)
        self.txtSimuFrameRepeat.grid(row = 2, column = 4, padx = 4, pady = 3, sticky= tk.W)

        self.btnSimulatition = tk.Button(self.lbl_live_plot, text='Simulator',  command = self.simulation_clicked,  width = 13)
        self.btnSimulatition.configure(font=('bold'))
        self.btnSimulatition.grid(row = 2, column = 5, padx = 8, pady = 3, sticky= tk.W)
        # For live plotting <=

        # self.btnFitting = tk.Button(self, text='Model fitting', bg='black', fg='white',  command = self.models_fitting_clicked, height = 2, width = 10)
        # self.btnFitting.place(x=830, y=410)

        # self.btnStatics = tk.Button(self, text='Statics', bg='red', fg='white', command = self.statics_clicked, height = 2, width = 10)
        # self.btnStatics.place(x=830, y=460)    

        self.btnFitting = tk.Button(self, text='Model fitting', bg='gold',  command = self.models_fitting_clicked, height = 2, width = 10)
        self.btnFitting.place(x=830, y=410)

        self.btnStatics = tk.Button(self, text='Statics', bg='gold', command = self.statics_clicked, height = 2, width = 10)
        self.btnStatics.place(x=830, y=460)

        self.btnMonitorDist = tk.Button(self, text='Monitoring', bg='gold', command = self.monitoring_dist_clicked, height = 2, width = 10)
        self.btnMonitorDist.place(x=830, y=510)
        
        # End building control for Training tab            

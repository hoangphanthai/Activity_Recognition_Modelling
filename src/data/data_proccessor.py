import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tkinter import messagebox
import globals
from globals import log_message
from data import data_resampling


def validate_monitoring_run(txtWindowStride, selected_monitor_window_size, selected_monitoring_sample_rate, algorithm):

    valid_to_run = True
    if ((len(globals.monitoring_data_fr.index) == 0) or (len(globals.predicted_data_fr.index) == 0) or (
            globals.curr_monitoring_window_size != selected_monitor_window_size) or (
            globals.curr_monitoring_algorithm != algorithm) or (
            globals.curr_monitoring_sampling_rate != selected_monitoring_sample_rate)):
            # User changed one of these parameters for the simulation
        
        window_stride_in_ms = math.floor(selected_monitor_window_size * int(txtWindowStride) / 100)
        
        path_with_train_valid_test_table_name = os.path.join(globals.root_dir, 'csv_out', globals.train_valid_data_set_name + '_at_' + globals.timestampforCSVfiles)

        predicted_file = ''
        if algorithm == 'Random_Forest':
            predicted_file = '2_monitor_predicted_Random_Forest.csv'
        if algorithm == 'Decision_Tree':
            predicted_file = '3_monitor_predicted_Decision_Tree.csv'
        if algorithm == 'SVM':
            predicted_file = '4_monitor_predicted_SVM.csv'
        if algorithm == 'Naive_Bayes':
            predicted_file = '5_monitor_predicted_NB.csv'

        hz_path = str(selected_monitoring_sample_rate) + 'Hz'
        window_path = 'window_' + str(selected_monitor_window_size) + 'ms_stride_' + str(window_stride_in_ms) + 'ms'

        file_name = os.path.join(path_with_train_valid_test_table_name, hz_path, window_path, '2_monitoring_data', predicted_file)
            

        if os.path.isfile(file_name):
            # Get the predicted data file location for the monitoring/staticstics
            globals.predicted_data_fr = pd.read_csv(file_name).sort_values(by=['timestamp'], ascending=True)
            
            # Recalculate the monitoring_data_fr
            globals.monitoring_data_fr = monitoring_data_generate(selected_monitoring_sample_rate, selected_monitor_window_size, globals.predicted_data_fr)

            globals.curr_monitoring_algorithm = algorithm
            globals.curr_monitoring_window_size = selected_monitor_window_size
            globals.curr_monitoring_sampling_rate = selected_monitoring_sample_rate
        else:
            messagebox.showinfo("Alert", 'Could not locate the file ' + file_name + ' for the plot!')
            valid_to_run = False

    return valid_to_run


# Generate prediction data frame for simulating dataset
def monitoring_data_generate(simulation_sample_rate, window_size, predicted_df):

    if globals.binary_mode:
        for _, value in globals.sub_labels_set.items():
            globals.monitoring_data_frame.loc[
                globals.monitoring_data_frame.label == value, 'label'] = 'Non-' + globals.main_label

    temp_monitor_data_set = globals.monitoring_data_frame.loc[globals.monitoring_data_frame['label'].isin(globals.label_set)].sort_values(
        by=['timestamp'], ascending=True)

    axes_to_apply_functions_list = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
    monitoring_data_set = temp_monitor_data_set[['label'] + axes_to_apply_functions_list + ['timestamp']]

    resample_dict = {}
    for axis in axes_to_apply_functions_list:
        resample_dict[axis] = globals.function_set_for_resampling

    if globals.original_sampling_rate != simulation_sample_rate:
        globals.monitoring_data_frame_resampled_monitor = data_resampling.resampled_frame(monitoring_data_set, globals.label_set,
                                                                resample_dict,
                                                                axes_to_apply_functions_list,
                                                                simulation_sample_rate)
    else:
        globals.monitoring_data_frame_resampled_monitor = monitoring_data_set

    result = pd.DataFrame()

    for _, row in predicted_df.iterrows():
        start_time = row['timestamp']
        predicted_temp = row['predicted_label']
        df_temp = globals.monitoring_data_frame_resampled_monitor.loc[
            (globals.monitoring_data_frame_resampled_monitor['timestamp'] >= start_time) & (
                    globals.monitoring_data_frame_resampled_monitor['timestamp'] < start_time + window_size)]
        df_temp = df_temp.assign(predicted_label = predicted_temp)
        result = result.append(df_temp)
    if globals.binary_mode:
        for _, value in globals.sub_labels_set.items():
            result.loc[
                result.label == value, 'label'] = 'Non-' + globals.main_label

    return result


def get_balanced_train_valid_test_data_set(agg_train_valid_test_filtered_unbalanced):
    
    # Splitting the train_valid and test data set with the 7:3 proportion ->
    unbalanced_train_valid_dataset, test_dataset = train_test_split(
        agg_train_valid_test_filtered_unbalanced, 
        test_size = globals.test_proportion, shuffle = True, stratify = agg_train_valid_test_filtered_unbalanced[['label']])

    # Check the amount of training data, that the 70 percentage will have sufficient labels/activities
    train_data_label_set = pd.Series(np.unique(np.array(unbalanced_train_valid_dataset['label']).tolist()))
    train_data_label_set = train_data_label_set.sort_values(ascending=True)
    if not train_data_label_set.equals(globals.label_set):
        return None, None, False
        

    # Begin balancing the Train_Valid data set ->
    globals.minimum_train_valid_instance_for_each_label = unbalanced_train_valid_dataset['label'].value_counts().min()

    balanced_train_valid_dataset = pd.DataFrame()

    if globals.binary_mode:
        no_of_main_label_instances = len(unbalanced_train_valid_dataset.loc[
                                                unbalanced_train_valid_dataset['label'].isin(
                                                    [globals.main_label])].index)
        if (
                no_of_main_label_instances // globals.no_of_sub_labels < globals.minimum_train_valid_instance_for_each_label):
            no_of_each_sub_label_instances = no_of_main_label_instances // globals.no_of_sub_labels
        else:
            no_of_each_sub_label_instances = globals.minimum_train_valid_instance_for_each_label

        # For the purpose of correctly updating into database
        globals.minimum_train_valid_instance_for_each_label = no_of_each_sub_label_instances

        # Begin adjusting the proportion of Lying Standing Walking Grazing with proportion 3:1:1:1 =>
        print('Number of instances for ' + globals.main_label + ' label in Train_Valid set: ' + str(
            no_of_each_sub_label_instances * globals.no_of_sub_labels))
        print(
            'Number of instances for (each) sub label in Train_Valid set: ' + str(
                no_of_each_sub_label_instances))

        # Randomly select instances for main label
        features_eachlabel = unbalanced_train_valid_dataset.loc[
            unbalanced_train_valid_dataset['label'] == globals.main_label]
        random_set = features_eachlabel.sample(n=no_of_each_sub_label_instances * globals.no_of_sub_labels,
                                                replace=False)
        balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

        # Randomly select instances for sub labels
        for _, value in globals.sub_labels_set.items():
            features_eachlabel = unbalanced_train_valid_dataset.loc[
                unbalanced_train_valid_dataset['label'] == value]
            random_set = features_eachlabel.sample(n = no_of_each_sub_label_instances,replace=False)
            balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)

        # End adjusting the proportion of Main label and Non-main labels with proportion 3:1:1:1 <=

        # Change the label set of Train data set into Non-Main for the other sub labels ->
        for _, value in globals.sub_labels_set.items():
            balanced_train_valid_dataset.loc[balanced_train_valid_dataset.label == value, 'label'] = 'Non-' + globals.main_label

        balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')

        # Get the root list of all activities in test data set before change sub labels into non-main label. This is for the confusion matrix latter.
        test_dataset = test_dataset.reset_index(drop=True)
        globals.main_and_non_main_labels_y_root = test_dataset['label'].to_numpy(copy=True)

        # Change the label set of test dataset into Non-main label for the other sub labels ->
        for _, value in globals.sub_labels_set.items():
            test_dataset.loc[test_dataset.label == value, 'label'] = 'Non-' + globals.main_label

        # Change the label_set into two labels only
        globals.label_set = pd.Series([globals.main_label, 'Non-' + globals.main_label])
    else:
        log_message('Number of instances for each label in Train_Valid set: ' + str(globals.minimum_train_valid_instance_for_each_label))
        for eachlabel in globals.label_set:
            features_eachlabel = unbalanced_train_valid_dataset.loc[
                unbalanced_train_valid_dataset['label'] == eachlabel]

            if len(features_eachlabel) == globals.minimum_train_valid_instance_for_each_label:
                balanced_train_valid_dataset = balanced_train_valid_dataset.append(features_eachlabel)
            else:
                random_set = features_eachlabel.sample(n=globals.minimum_train_valid_instance_for_each_label,
                                                        replace=False)
                balanced_train_valid_dataset = balanced_train_valid_dataset.append(random_set)
        balanced_train_valid_dataset = balanced_train_valid_dataset.dropna().set_index('timestamp')
    
    return balanced_train_valid_dataset, test_dataset, True
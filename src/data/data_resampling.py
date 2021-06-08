import pandas as pd
import pandas as pd
import numpy as np

import globals

# This function is trying to predict the sampling rate from the input sensor dataset
def get_original_sampling_rate():
    
    # Get the first cattle_id and label
    first_cattle = globals.train_valid_test_data_frame['cattle_id'].iloc[0]
    first_label = globals.train_valid_test_data_frame['label'].iloc[0]

    first_1000_data_points = globals.train_valid_test_data_frame.loc[(globals.train_valid_test_data_frame['cattle_id'] == first_cattle) & (globals.train_valid_test_data_frame['label'] == first_label)].head(1000).sort_values(by='timestamp', ascending=True)

    # Calculate the delay in milliseconds between data points to predict the sampling rate
    list_temp = []
    for i in range(1, len(first_1000_data_points.index)):
        list_temp.append(int(first_1000_data_points['timestamp'].iloc[i]) - int(first_1000_data_points['timestamp'].iloc[i - 1]))
    
    # Get the value which is mostly occured
    delay_between_data_points = max(set(list_temp), key=list_temp.count)
    
    return round(1000 / delay_between_data_points)


# This function is to resample a dataset to a specific sampling rate (with function defined in globals.function_set_for_resampling)
def resampled_frame(sensor_data_frame, label_set, features_dict, axes_to_apply_functions_list, resampled_rate):
    agg_result = pd.DataFrame()
    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(sensor_data_frame['cattle_id']).tolist()))

    for each_cow in cow_list:
        sensor_data_of_each_cow = sensor_data_frame.loc[sensor_data_frame['cattle_id'] == each_cow]

        for each_label in label_set:
            data_of_one_label_each_cow = sensor_data_of_each_cow.loc[sensor_data_of_each_cow['label'] == each_label]
            data_of_one_label_each_cow = data_of_one_label_each_cow[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by = ['timestamp'], ascending = True)
            no_stride_items = int(0)

            start_time = int(data_of_one_label_each_cow['timestamp'].iloc[0])
            last_time = int(data_of_one_label_each_cow['timestamp'].iloc[len(data_of_one_label_each_cow.index) - 1])
            while start_time < last_time:
                # There is a minor bug regarding the case when stride size is too large. Eg 2000%
                sensor_data_of_one_window = data_of_one_label_each_cow.loc[
                    (data_of_one_label_each_cow['timestamp'] >= start_time) & (data_of_one_label_each_cow['timestamp'] < start_time + round(1000 / resampled_rate))]
                if len(sensor_data_of_one_window.index) > 0:
                    features_vector_of_one_window = pd.DataFrame()

                    # sensor_data_of_one_window contains the sensor data of one window
                    # features_vector_of_one_window contains 1 feature vector of that window

                    for axis_name in features_dict:
                        df_temp = sensor_data_of_one_window.agg({axis_name: features_dict[axis_name]})
                        for func_name in df_temp.columns:
                            features_vector_of_one_window[func_name] = df_temp[func_name].to_numpy()

                    features_vector_of_one_window = features_vector_of_one_window.round(3)
                    features_vector_of_one_window['count'] = len(sensor_data_of_one_window.index)
                    features_vector_of_one_window['label'] = each_label
                    features_vector_of_one_window['cattle_id'] = each_cow
                    features_vector_of_one_window['timestamp'] = start_time
                    agg_result = agg_result.append(features_vector_of_one_window)

                    # get the next timestamp for the loop
                    stride_len = len(data_of_one_label_each_cow.loc[
                                         (data_of_one_label_each_cow['timestamp'] >= start_time) & (
                                                 data_of_one_label_each_cow['timestamp'] < start_time + round(1000 / resampled_rate))])
                    no_stride_items = no_stride_items + stride_len
                    start_time = start_time + round(1000 / resampled_rate)
                else:
                    if no_stride_items < len(data_of_one_label_each_cow.index):
                        start_time = int(data_of_one_label_each_cow['timestamp'].iloc[no_stride_items])
                    else:
                        start_time = int(data_of_one_label_each_cow['timestamp'].iloc[len(data_of_one_label_each_cow.index) - 1])

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result


# This function is to generate the features vectors for the model to consume
def aggregated_frame(sensor_data_frame, label_set, features_dict, axes_to_apply_functions_list,
                     window_size, window_stride_in_ms):
    features_vectors_of_input_data = pd.DataFrame()

    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(sensor_data_frame['cattle_id']).tolist()))

    for each_cow in cow_list:
        sensor_data_of_each_cow = sensor_data_frame.loc[sensor_data_frame['cattle_id'] == each_cow]

        for each_label in label_set:
            data_of_one_label_each_cow = sensor_data_of_each_cow.loc[sensor_data_of_each_cow['label'] == each_label]
            data_of_one_label_each_cow = data_of_one_label_each_cow[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by=['timestamp'],
                                                                                            ascending=True)
            no_stride_items = int(0)

            start_time = int(data_of_one_label_each_cow['timestamp'].iloc[0])
            last_time = int(data_of_one_label_each_cow['timestamp'].iloc[len(data_of_one_label_each_cow.index) - 1])
            while start_time < last_time:
                # There is a minor bug regarding the case Stride size is too large. Eg 2000%
                sensor_data_of_one_window = data_of_one_label_each_cow.loc[
                    (data_of_one_label_each_cow['timestamp'] >= start_time) & (data_of_one_label_each_cow['timestamp'] < start_time + window_size)]
                if len(sensor_data_of_one_window.index) > 0:
                    features_vector_of_one_window = pd.DataFrame()
                    # sensor_data_of_one_window contains the sensor data of the derived (1) window
                    # features_vector_of_one_window contains 1 feature vector of the derived (1) window

                    for axis_name in features_dict:
                        # For each of axis, we calculate the coresponding feature function
                        features_vector_of_one_axis = pd.DataFrame()
                        for a_func in features_dict[axis_name]:
                            features_vector_of_one_axis[axis_name+'-'+a_func.__name__] = [a_func(sensor_data_of_one_window[axis_name].tolist())]
                    
                        for func_name in features_vector_of_one_axis.columns:
                            features_vector_of_one_window[func_name] = features_vector_of_one_axis[func_name].to_numpy()

                    features_vector_of_one_window = features_vector_of_one_window.round(3)
                    features_vector_of_one_window['count'] = len(sensor_data_of_one_window.index)
                    features_vector_of_one_window['label'] = each_label
                    features_vector_of_one_window['cattle_id'] = each_cow
                    features_vector_of_one_window['timestamp'] = start_time
                    features_vector_of_one_window['timestamp_human'] = pd.to_datetime(start_time, unit='ms')    

                    features_vectors_of_input_data = features_vectors_of_input_data.append(features_vector_of_one_window)

                    # get the next timestamp for the loop
                    stride_len = len(data_of_one_label_each_cow.loc[
                                         (data_of_one_label_each_cow['timestamp'] >= start_time) & (
                                                 data_of_one_label_each_cow['timestamp'] < start_time + window_stride_in_ms)])
                    no_stride_items = no_stride_items + stride_len
                    start_time = start_time + window_stride_in_ms
                else:
                    if no_stride_items < len(data_of_one_label_each_cow.index):
                        start_time = int(data_of_one_label_each_cow['timestamp'].iloc[no_stride_items])
                    else:
                        start_time = int(data_of_one_label_each_cow['timestamp'].iloc[len(data_of_one_label_each_cow.index) - 1])
    
    # In case using the lambda function frEnergy in the feature list
    features_vectors_of_input_data = features_vectors_of_input_data.rename(columns=lambda x: x.replace('<lambda>', 'frPeakFreq') if x.find('<lambda>')>=0 else x)
    features_vectors_of_input_data = features_vectors_of_input_data.dropna().sort_values(by='timestamp', ascending=True)
    features_vectors_of_input_data = features_vectors_of_input_data.reset_index(drop=True)
    return features_vectors_of_input_data

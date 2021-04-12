import pandas as pd
import pandas as pd
import numpy as np

import globals

# This function is to predict the sampling rate from a sensor data set

def get_original_sampling_rate():
    
    #Calculate the delay in milliseconds between data points and predict the sample rate
    df_temp = globals.train_valid_test_data_frame.head(1000).sort_values(by='timestamp', ascending=True)
    first_label = df_temp['label'].iloc[0]
    df_temp = df_temp.loc[df_temp['label'].isin([first_label])]
    list_temp = []
    for i in range(1, len(df_temp.index)):
        list_temp.append(int(df_temp['timestamp'].iloc[i]) - int(df_temp['timestamp'].iloc[i - 1]))
    
    delay_between_data_points = max(set(list_temp), key=list_temp.count)
    return round(1000 / delay_between_data_points)


# This function is to resample a dataset to a specific sampling rate (with function defined in globals.function_set_for_resampling)
def resampled_frame(df, label_set, features_dict, axes_to_apply_functions_list, resampled_rate):
    agg_result = pd.DataFrame()

    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(df['cattle_id']).tolist()))

    for each_cow in cow_list:
        df_cow_temp = df.loc[df['cattle_id'] == each_cow]

        for each_label in label_set:
            df2 = df_cow_temp.loc[df_cow_temp['label'] == each_label]
            df2 = df2[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by = ['timestamp'], ascending = True)
            no_stride_items = int(0)

            start_time = int(df2['timestamp'].iloc[0])
            last_time = int(df2['timestamp'].iloc[len(df2.index) - 1])
            while start_time < last_time:
                # There is a minor bug regarding the case when stride size is too large. Eg 2000%
                df3 = df2.loc[
                    (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + round(1000 / resampled_rate))]
                if len(df3.index) > 0:
                    df4 = pd.DataFrame()

                    # df3 contains the sensor data of the derived (1) window
                    # df4 contains 1 feature vector of the derived (1) window

                    for x in features_dict:
                        # The new method iterate by axis, not function as last version
                        df_temp = df3.agg({x: features_dict[x]})
                        for func_name in df_temp.columns:
                            df4[func_name] = df_temp[func_name].to_numpy()
                    df4 = df4.round(3)
                    df4['count'] = len(df3.index)
                    df4['label'] = each_label
                    df4['cattle_id'] = each_cow
                    df4['timestamp'] = start_time
                    agg_result = agg_result.append(df4)

                    # get the next timestamp for the loop
                    stride_len = len(df2.loc[
                                         (df2['timestamp'] >= start_time) & (
                                                 df2['timestamp'] < start_time + round(1000 / resampled_rate))])
                    no_stride_items = no_stride_items + stride_len
                    start_time = start_time + round(1000 / resampled_rate)
                else:
                    if no_stride_items < len(df2.index):
                        start_time = int(df2['timestamp'].iloc[no_stride_items])
                    else:
                        start_time = int(df2['timestamp'].iloc[len(df2.index) - 1])

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result

# This function is to apply calculate features (for each window size and stride)
# according to select functions in agg_function_set param
def aggregated_frame(df, label_set, features_dict, axes_to_apply_functions_list,
                     window_size, window_stride_in_ms):
    agg_result = pd.DataFrame()

    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(df['cattle_id']).tolist()))

    for each_cow in cow_list:
        df_cow_temp = df.loc[df['cattle_id'] == each_cow]

        for each_label in label_set:
            df2 = df_cow_temp.loc[df_cow_temp['label'] == each_label]
            df2 = df2[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by=['timestamp'],
                                                                                            ascending=True)
            no_stride_items = int(0)

            start_time = int(df2['timestamp'].iloc[0])
            last_time = int(df2['timestamp'].iloc[len(df2.index) - 1])
            while start_time < last_time:
                # There is a minor bug regarding the case Stride size is too large. Eg 2000%
                df3 = df2.loc[
                    (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + window_size)]
                if len(df3.index) > 0:
                    df4 = pd.DataFrame()
                    # df3 contains the sensor data of the derived (1) window
                    # df4 contains 1 feature vector of the derived (1) window

                    for x in features_dict:
                        df_temp = df3.agg({x: features_dict[x]})
                        df_temp2 = df_temp.transpose()
                        
                        axis_name = df_temp2.index.values[0]
                        list_rename = []
                        for func_name in df_temp2.columns:                            
                            list_rename.append(axis_name + '-' + func_name)
                        df_temp2.columns = list_rename

                        for col in df_temp2.columns:
                            df4[col] = df_temp2[col].to_numpy()

                    df4 = df4.round(3)
                    df4['count'] = len(df3.index)
                    df4['label'] = each_label
                    df4['cattle_id'] = each_cow
                    df4['timestamp'] = start_time
                    df4['timestamp_human'] = pd.to_datetime(start_time, unit='ms')    

                    agg_result = agg_result.append(df4)

                    # get the next timestamp for the loop
                    stride_len = len(df2.loc[
                                         (df2['timestamp'] >= start_time) & (
                                                 df2['timestamp'] < start_time + window_stride_in_ms)])
                    no_stride_items = no_stride_items + stride_len
                    start_time = start_time + window_stride_in_ms
                else:
                    if no_stride_items < len(df2.index):
                        start_time = int(df2['timestamp'].iloc[no_stride_items])
                    else:
                        start_time = int(df2['timestamp'].iloc[len(df2.index) - 1])
    
    # In case using the lambda function frEnergy in the feature list
    agg_result = agg_result.rename(columns=lambda x: x.replace('<lambda>', 'PeakFreq') if x.find('<lambda>')>=0 else x)
    
    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result




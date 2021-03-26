import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator


# This function is to resample a dataset at a specific rate (with Mean function)
def resampled_frame_old(df, label_set, agg_function_set, axes_to_apply_functions_list, resampled_rate):
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
                df3 = df2.loc[
                    (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + round(1000 / resampled_rate))]
                if len(df3.index) > 0:
                    df4 = pd.DataFrame()

                    for func in agg_function_set:
                        agg_dict = {}
                        for column in axes_to_apply_functions_list:
                            agg_dict[column] = func
                        df_temp = df3.agg(agg_dict)
                        df_temp2 = df_temp.to_frame().transpose()
                        for col in df_temp2.columns:
                            df4[col] = df_temp2[col].to_numpy()

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

# This function is to resample a dataset at a specific rate (with Mean function)
def resampled_frame(df, label_set, features_dict, axes_to_apply_functions_list, resampled_rate):
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
                    (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + round(1000 / resampled_rate))]
                if len(df3.index) > 0:
                    df4 = pd.DataFrame()

                    # df3 contains the sensor data of the derived (1) window
                    # df4 contains 1 feature vector of the derived (1) window

                    for x in features_dict:
                        # The new method iterate by axis, not function as last version
                        df_temp = df3.agg({x: features_dict[x]})
                        for col in df_temp.columns:
                            df4[col] = df_temp[col].to_numpy()
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
def aggregated_frame_old(df, label_set, agg_function_set, agg_function_list_names, axes_to_apply_functions_list,
                     window_size, window_stride_in_ms, output_timestamp_type):

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
                    i = int(0)  # For the purpose of getting the name of function
                    for func in agg_function_set:
                        agg_dict = {}
                        for column in axes_to_apply_functions_list:
                            agg_dict[column] = func
                        df_temp = df3.agg(agg_dict)
                        df_temp2 = df_temp.to_frame().transpose()
                        list_rename = []
                        for col in df_temp2.columns:
                            list_rename.append(col + '_' + agg_function_list_names[i])
                        i = i + 1
                        df_temp2.columns = list_rename
                        for col in df_temp2.columns:
                            df4[col] = df_temp2[col].to_numpy()
                    df4 = df4.round(3)
                    df4['count'] = len(df3.index)
                    df4['label'] = each_label
                    df4['cattle_id'] = each_cow
                    if output_timestamp_type == 1:
                        df4['timestamp'] = pd.to_datetime(start_time, unit='ms')
                    else:
                        df4['timestamp'] = start_time

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

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result


# This function is to apply calculate features (for each window size and stride)
# according to select functions in agg_function_set param
def aggregated_frame(df, label_set, features_dict, axes_to_apply_functions_list,
                     window_size, window_stride_in_ms, output_timestamp_type):
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

                        function_name = df_temp2.index.values[0]
                        list_rename = []
                        for col in df_temp2.columns:
                            list_rename.append(function_name + '-' + col)
                        df_temp2.columns = list_rename

                        for col in df_temp2.columns:
                            df4[col] = df_temp2[col].to_numpy()

                    df4 = df4.round(3)
                    df4['count'] = len(df3.index)
                    df4['label'] = each_label
                    df4['cattle_id'] = each_cow
                    if output_timestamp_type == 1:
                        df4['timestamp'] = pd.to_datetime(start_time, unit='ms')
                    else:
                        df4['timestamp'] = start_time

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

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result


def aggregated_unseen_data_old(df, label_set, agg_function_set, agg_function_list_names, axes_to_apply_functions_list,
                           window_size, window_stride_in_ms, output_timestamp_type):
    # label is not included here anymore in the result of the function
    # notice if the distance between two derived timestamps is less than window size then it is mixed label windows

    agg_result = pd.DataFrame()
    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(df['cattle_id']).tolist()))

    for each_cow in cow_list:
        df_cow_temp = df.loc[df['cattle_id'] == each_cow]

        df2 = df_cow_temp.loc[df_cow_temp['label'].isin(label_set)].sort_values(by=['timestamp'], ascending=True)
        df2 = df2[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by=['timestamp'],
                                                                                        ascending=True)
        no_stride_items = int(0)
        # This function is different from the above function is because now it does not need to sort by labels
        start_time = int(df2['timestamp'].iloc[0])
        last_time = int(df2['timestamp'].iloc[len(df2.index) - 1])
        while start_time < last_time:
            df3 = df2.loc[
                (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + window_size)]
            if len(df3.index) > 0:
                df4 = pd.DataFrame()
                i = int(0)  # For the purpose of getting the name of function
                for func in agg_function_set:
                    agg_dict = {}
                    for column in axes_to_apply_functions_list:
                        agg_dict[column] = func
                    df_temp = df3.agg(agg_dict)
                    df_temp2 = df_temp.to_frame().transpose()
                    list_rename = []
                    for col in df_temp2.columns:
                        list_rename.append(col + '_' + agg_function_list_names[i])
                    i = i + 1
                    df_temp2.columns = list_rename
                    for col in df_temp2.columns:
                        df4[col] = df_temp2[col].to_numpy()
                df4 = df4.round(3)
                df4['count'] = len(df3.index)
                df4['cattle_id'] = each_cow
                if output_timestamp_type == 1:
                    df4['timestamp'] = pd.to_datetime(start_time, unit='ms')
                else:
                    df4['timestamp'] = start_time

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

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result

def aggregated_unseen_data(df, label_set, features_dict, axes_to_apply_functions_list,
                           window_size, window_stride_in_ms, output_timestamp_type):
    # label is not included here anymore in the result of the function
    # notice if the distance between two derived timestamps is less than window size then it is mixed label windows

    agg_result = pd.DataFrame()
    # Get the list of different cows
    cow_list = pd.Series(np.unique(np.array(df['cattle_id']).tolist()))

    for each_cow in cow_list:
        df_cow_temp = df.loc[df['cattle_id'] == each_cow]

        df2 = df_cow_temp.loc[df_cow_temp['label'].isin(label_set)].sort_values(by=['timestamp'], ascending=True)
        df2 = df2[['label'] + axes_to_apply_functions_list + ['timestamp']].sort_values(by=['timestamp'],
                                                                                        ascending=True)
        no_stride_items = int(0)
        # This function is different from the above function is because now it does not need to sort by labels
        start_time = int(df2['timestamp'].iloc[0])
        last_time = int(df2['timestamp'].iloc[len(df2.index) - 1])
        while start_time < last_time:
            df3 = df2.loc[
                (df2['timestamp'] >= start_time) & (df2['timestamp'] < start_time + window_size)]
            if len(df3.index) > 0:
                df4 = pd.DataFrame()
                # df3 contains the sensor data of the derived (1) window
                # df4 contains 1 feature vector of the derived (1) window

                for x in features_dict:
                    df_temp = df3.agg({x: features_dict[x]})
                    df_temp2 = df_temp.transpose()

                    function_name = df_temp2.index.values[0]
                    list_rename = []
                    for col in df_temp2.columns:
                        list_rename.append(function_name + '-' + col)
                    df_temp2.columns = list_rename

                    for col in df_temp2.columns:
                        df4[col] = df_temp2[col].to_numpy()

                # print('-----df4---\n')
                # print(df4.head(1))
                df4 = df4.round(3)
                df4['count'] = len(df3.index)
                df4['cattle_id'] = each_cow
                if output_timestamp_type == 1:
                    df4['timestamp'] = pd.to_datetime(start_time, unit='ms')
                else:
                    df4['timestamp'] = start_time

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

    agg_result = agg_result.dropna().sort_values(by='timestamp', ascending=True)
    agg_result = agg_result.reset_index(drop=True)
    return agg_result

def simulation_show(label_set, df1, length, stride, delay_ms, repeat):
    # stride and length is the numbering index of the dataframe not the time in epoch
    # input df1 should contain timestamp label, prediced label and axayazgzgygz
    cols = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
    df = df1[cols]
    df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    df['label'] = df1['label']
    df['timestamp'] = df1['timestamp'].apply(str)
    df['timestamp'] = df['timestamp'].apply(lambda x: x[-7:])
    df['predicted'] = df1['predicted_label']
    df = df.round(3)

    agg_dict = {}
    i = int(1)
    for each_label in label_set:
        agg_dict[each_label] = i
        i = i + 1
    df['label'] = df['label'].map(agg_dict)
    df['predicted'] = df['predicted'].map(agg_dict)

    labels = label_set.tolist()
    lenoflabels = len(label_set)
    labels.insert(0, '')
    labels.insert(len(labels), '')

    fig = plt.figure()
    acc = fig.add_subplot(3, 1, 1)
    gro = fig.add_subplot(3, 1, 2)
    pred = fig.add_subplot(3, 1, 3)

    def animate(i):
        # stride and length is the numbering index of the dataframe not the time in epoch
        stepf = i * stride
        stept = i * stride + length
        xs = df.iloc[stepf:stept]['timestamp'].to_numpy()
        ax = df.iloc[stepf:stept]['ax'].to_numpy()
        ay = df.iloc[stepf:stept]['ay'].to_numpy()
        az = df.iloc[stepf:stept]['az'].to_numpy()
        gx = df.iloc[stepf:stept]['gx'].to_numpy()
        gy = df.iloc[stepf:stept]['gy'].to_numpy()
        gz = df.iloc[stepf:stept]['gz'].to_numpy()
        ground_true = list(df.iloc[stepf:stept]['label'])
        predicted = list(df.iloc[stepf:stept]['predicted'])

        acc.clear()
        plt.setp(acc.get_xticklabels(), visible=False)
        acc.set_ylim([-1, 1])
        acc.plot(xs, ax)
        acc.plot(xs, ay)
        acc.plot(xs, az)
        acc.set_title('Accelerometer', fontsize=10)

        gro.clear()
        plt.setp(gro.get_xticklabels(), visible=False)
        gro.set_ylim([-1, 1])
        gro.plot(xs, gx)
        gro.plot(xs, gy)
        gro.plot(xs, gz)
        gro.set_title('Gyroscope', fontsize=10)

        pred.clear()
        plt.setp(pred.xaxis.get_majorticklabels(), fontsize=7, rotation=45)
        pred.set_ylim([0, lenoflabels + 1])
        pred.plot(xs, predicted, color="red")
        pred.plot(xs, ground_true, color="blue")
        pred.set_yticklabels(labels)
        pred.set_title('Prediction and Ground Truth', fontsize=10)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, repeat), interval=delay_ms)

    # # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('lines.mp4', writer=writer)

    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.show()


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=False, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.
    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """
    plt.subplot(121)
    ny = len(data[0])
    ind = list(range(ny))
    axes = []
    cum_size = np.zeros(ny)
    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=colors[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label, fontsize=12)

    plt.legend(prop={'size': 9})

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2,
                         value_format.format(h), ha="center",
                         va="center")
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
    plt.xlabel('Data', fontsize=12)
    plt.title('Activity Pattern Comparision', fontsize=13)
    # plt.title('Temperature \n Humidity', fontsize=10)


def statistics_metrics_show(test_dataset_resampled_monitor, monitoring_data, monitoring_sampling_rate):
    # Begin calculating statistics percentage plot on the left ->
    original_activity_percentage = pd.DataFrame(
        test_dataset_resampled_monitor['label'].value_counts(normalize=True) * 100)
    original_activity_percentage.columns = ['Ground_truth']

    predicted_activity_percentage = pd.DataFrame(monitoring_data['predicted_label'].value_counts(normalize=True) * 100)
    predicted_activity_percentage.columns = ['Model_based']

    df = pd.concat([original_activity_percentage, predicted_activity_percentage], axis=1, sort=False)
    df = df.rename_axis('label').reset_index()
    category_labels = ['Ground_truth', 'Model_based']
    series_labels = df['label'].to_numpy()
    data = df[["Ground_truth", "Model_based"]].to_numpy()
    if len(series_labels) == 2:
        colors_list = ['tab:orange', 'tab:green']
    if len(series_labels) == 3:
        colors_list = ['tab:orange', 'tab:green', 'tab:cyan']
    if len(series_labels) == 4:
        colors_list = ['tab:orange', 'tab:green', 'tab:cyan', 'tab:blue']
    if len(series_labels) == 5:
        colors_list = ['tab:orange', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:red']
    if len(series_labels) == 6:
        colors_list = ['tab:orange', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:white']

    plot_stacked_bar(
        data,
        series_labels,
        category_labels=category_labels,
        show_values=True,
        value_format="{:.2f}%",
        colors=colors_list,
        y_label="Percentage"
    )
    # End calculating statistics percentage plot on the left <-

    # Begin calculating absolute plot on the right ->

    # The obsolute should be calculated on the grouth_truth data of sampled data, not in the orignial from test_dataset_resampled_monitor
    # This is because it will cause large deviation and it is not meaningful.
    # The data frame test_dataset_resampled_monitor is replaced by monitoring_data as a result
    # original_activity_absolute = pd.DataFrame(
    #     test_dataset_resampled_monitor['label'].value_counts() * round(1 / (60 * monitoring_sampling_rate), 3))
    # monitoring_data['label'].value_counts() * round(1 / (60 * monitoring_sampling_rate), 3))

    original_activity_absolute = pd.DataFrame(monitoring_data['label'].value_counts())
    original_activity_absolute.columns = ['Ground_truth']
    original_activity_absolute = original_activity_absolute.div(60 * monitoring_sampling_rate).round(
        3)  # Calculate minutes

    predicted_activity_absolute = pd.DataFrame(monitoring_data['predicted_label'].value_counts())
    predicted_activity_absolute.columns = ['Model_based']
    predicted_activity_absolute = predicted_activity_absolute.div(60 * monitoring_sampling_rate).round(
        3)  # Calculate minutes

    absolute_df = pd.concat([original_activity_absolute, predicted_activity_absolute], axis=1, sort=False)
    absolute_df.loc[:, 'Deviation'] = absolute_df.apply(lambda x: (x.Model_based - x.Ground_truth), axis=1)
    absolute_df = absolute_df.rename_axis('label')
    absolute_df = absolute_df.round(2)
    colors_list = ['#002097', '#4d88d5', 'r']

    ax2 = plt.subplot(122)
    # df1 = result.div(result.sum(1), axis=0)
    # df1=(df1*100).round(2)
    absolute_df.plot(ax=ax2, kind='bar', color=colors_list, edgecolor=None)

    plt.legend(prop={'size': 9})
    plt.legend(labels=absolute_df.columns, )
    for tick in ax2.get_xticklabels():
        tick.set_rotation(0)
    # ax2.set_yticks(range(0,101,10))
    # ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_title("Time deviation of activity", size=13)
    ax2.set_ylabel('Duration (Minutes)', fontsize=12)
    ax2.set_xlabel('Activity', fontsize=12)

    # Add this loop to add the annotations
    for p in ax2.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        # ax2.annotate('{:.0%}'.format(height), (x, y + height + 0.01))
        ax2.annotate('{:.1f}'.format(height), (x + 0.012, y + height + 0.02))
    # End calculating absolute plot on the right <-

    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    fig = plt.gcf()
    fig.canvas.set_window_title('Statistics')
    plt.show()


def monitoring_show(monitoring_data, sampling_rate):
    no_late_detection = 0
    no_early_detection = 0
    no_wrong_detection = 0
    no_other_detection = 0
    previous_ground_truth_label = monitoring_data['label'].iloc[0]
    last_ground_truth_label = monitoring_data['label'].iloc[0]

    previous_predicted_label = monitoring_data['predicted_label'].iloc[0]
    last_predicted_label = monitoring_data['predicted_label'].iloc[0]

    time_deviation = []
    count = 0
    current_difference = False
    for index, row in monitoring_data.iterrows():
        label = row['label']
        predicted_label = row['predicted_label']
        if (label == predicted_label) and (current_difference is True):
            # Ending of the wrong segment of prediction
            time_deviation.append(count)
            count = 0
            current_difference = False
            # Calculate the no of deviation types ->
            if label_of_wrong_segment == predicted_label:
                no_early_detection = no_early_detection + 1
            elif label_of_wrong_segment == last_ground_truth_label:
                no_late_detection = no_late_detection + 1
            elif predicted_label == last_predicted_label:
                no_wrong_detection = no_wrong_detection + 1
            else:
                no_other_detection = no_other_detection + 1
            # Calculate the no of deviation types <-

        elif label != predicted_label:
            if current_difference is False:
                # Begin a new wrong segment of prediction
                count = 1
                current_difference = True
                label_of_wrong_segment = predicted_label
                last_ground_truth_label = previous_ground_truth_label
                last_predicted_label = previous_predicted_label
            else:
                count = count + 1
        previous_ground_truth_label = label
        previous_predicted_label = predicted_label

    time_deviation_plot = [round(i / sampling_rate, 1) for i in time_deviation]

    # print()
    if round(max(time_deviation_plot)) < 11:
        tick_list = list(np.arange(round(min(time_deviation_plot)), 10, 0.5))
        bins_no = round((max(time_deviation_plot) - min(time_deviation_plot) + 1) * 10)
    elif round(max(time_deviation_plot)) < 21:
        tick_list = list(np.arange(round(min(time_deviation_plot)), 10, 1)) + list(
            np.arange(10, round(max(time_deviation_plot)) + 4, 4))
        bins_no = round((max(time_deviation_plot) - min(time_deviation_plot) + 1) * 5)
    elif round(max(time_deviation_plot)) < 41:
        tick_list = list(np.arange(round(min(time_deviation_plot)), 10, 2)) + list(
            np.arange(10, round(max(time_deviation_plot)) + 9, 10))
        bins_no = round((max(time_deviation_plot) - min(time_deviation_plot) + 1) * 5)
    elif round(max(time_deviation_plot)) < 81:
        tick_list = list(np.arange(round(min(time_deviation_plot)), 10, 3)) + list(
            np.arange(10, round(max(time_deviation_plot)), 20))
        bins_no = round((max(time_deviation_plot) - min(time_deviation_plot) + 1) * 5)
    else:
        tick_list = list(np.arange(round(min(time_deviation_plot)), 10, 3)) + list(
            np.arange(10, round(max(time_deviation_plot)), 40))
        bins_no = round((max(time_deviation_plot) - min(time_deviation_plot) + 1) * 5)

    if len(time_deviation_plot) < 1:
        bins_no = 1
        time_deviation_plot = [0]

    # print(time_deviation_plot)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(time_deviation_plot, bins=bins_no)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_xticks(tick_list)
    axes[0].tick_params(labelsize=8)
    # axes[0].set_xticklabels(tick_list, fontsize=6)
    # axes[0].set_xticks(range(round(min(time_deviation_plot)),round(max(time_deviation_plot))+1))
    # axes[0].set_tick_params(labelsize=5)
    axes[0].set_xlabel('Deviation (seconds)', fontsize=18)
    axes[0].set_ylabel('Frequency (times)', fontsize=18)
    axes[0].set_title("Distribution of error by time deviation", size=18)

    # Sub plot on the right ->
    detection_dict = {'Early': no_early_detection, 'Late': no_late_detection, 'Wrong': no_wrong_detection,
                      'Other': no_other_detection}

    barlist = axes[1].bar(range(len(detection_dict)), list(detection_dict.values()), align='center')
    barlist[0].set_color('#3498DB')
    barlist[1].set_color('#FFC300')
    barlist[2].set_color('#E74C3C')
    barlist[3].set_color('#D2B4DE')

    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xticks(range(len(detection_dict)))
    axes[1].set_xticklabels(list(detection_dict.keys()))  # , fontsize=12)
    # axes[1].set_xlim(0, 10)

    axes[1].set_xlabel('Detection Error', fontsize=18)
    axes[1].set_ylabel('Frequency (Times)', fontsize=18)
    axes[1].set_title("Error Type Statistics", size=18)

    #     new
    # Add this loop to add the annotations
    for p in axes[1].patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        axes[1].annotate('{}'.format(height), (x + 0.35, y + height + 0.05))
    # End calculating absolute plot on the right <-
    # new
    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    fig = plt.gcf()
    fig.canvas.set_window_title('Monitoring')

    plt.show()

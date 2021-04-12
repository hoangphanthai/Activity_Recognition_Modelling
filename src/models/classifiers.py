
import datetime
import psycopg2
from sklearn import metrics
import pickle
import json
import globals

class CLS:
    def __init__(self, classifier, name):
        
        self.clf = classifier
        self.name = name       
        self.init_metrics()


    def train_validate(self, to_run, X_train, y_train, X_valid, y_valid):
        if to_run == 1:
            self.clf.fit(X_train, y_train)
            y_pred_rf = self.clf.predict(X_valid)

            temp_acc = round(metrics.accuracy_score(y_valid, y_pred_rf), 4)
            self.train_valid_metrics['accu'] += temp_acc

            output_string = '{:<35}{:>8}'.format('\n' + self.name + ' accuracy: ', str(temp_acc))
            # globals.csv_txt_file_exporter.write_single_line( + str(temp_acc))
            globals.csv_txt_file_exporter.write_single_line(output_string)

            if globals.label_set.size == 2:
                self.train_valid_metrics['prec'] +=  round(metrics.precision_score(y_valid, y_pred_rf, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.train_valid_metrics['recal'] +=  round(metrics.recall_score(y_valid, y_pred_rf, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.train_valid_metrics['f1'] +=  round(metrics.recall_score(y_valid, y_pred_rf, average = 'binary', pos_label = globals.label_set[0]), 4)
    
    def calc_and_save_train_valid_result(self, to_run, kfold):
        if to_run == 1:
            self.train_valid_metrics['accu'] /= kfold
            output_string = '{:<60}{:>7}'.format('Train_Valid ' + str(kfold) + '-fold average accuracy of ' + self.name + ' ' , str(round(self.train_valid_metrics['accu'], 4)))
            globals.csv_txt_file_exporter.write_single_line('\n')
            globals.csv_txt_file_exporter.write_single_line(output_string)
            print(output_string)

            if globals.label_set.size == 2:
                self.train_valid_metrics['prec'] /= kfold
                self.train_valid_metrics['recal'] /= kfold
                self.train_valid_metrics['f1'] /= kfold

    def predict_test_data(self, to_run, X_test, y_test, labels_narray_temp, main_and_non_main_labels_set, main_and_non_main_labels_narray_temp):
        if to_run == 1:
            y_test_pred_rf = self.clf.predict(X_test)
            self.test_metrics['accu'] = round(metrics.accuracy_score(y_test, y_test_pred_rf), 4)
            print('-------------------------------------------------------------------------')
            print(self.name + ' - Accuracy on Test data: ' + str(self.test_metrics['accu']))

            # Begin saving Confusion Matrix into text file ->
            globals.csv_txt_file_exporter.write_single_line('\n-------------------------------------------------------------------------')
            globals.csv_txt_file_exporter.write_single_line('\nAccuracy of ' + self.name + ' on Test data: ' + str(self.test_metrics['accu']))
            globals.csv_txt_file_exporter.write_single_line('\n' + self.name + ' Confusion matrix on Test data')
            globals.csv_txt_file_exporter.write_single_line('\nPredicted \u2193' + globals.label_set.str.cat(sep=' ') + ' \u2193')

            confusion_matrix_temp = metrics.confusion_matrix(y_test, y_test_pred_rf, labels = labels_narray_temp)
            
            lbl_no = 0
            for line in confusion_matrix_temp:
                globals.csv_txt_file_exporter.write_single_line('\n\u2192True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                lbl_no = lbl_no + 1
            # End Saving Confusion Matrix into text file <-

            # Begin printing Confusion Matrix to console <-
            print(self.name + ' - Confusion matrix on Test data')
            print('Predicted ' + globals.label_set.str.cat(sep=' '))
            lbl_no = 0
            for line in confusion_matrix_temp:
                print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                lbl_no = lbl_no + 1
            # End Printing Confusion Matrix to console <-


            # Additional confusion matrix for binary classification ->
            if globals.binary_mode:
                globals.csv_txt_file_exporter.write_single_line('\n' + self.name + '(' + str(globals.no_of_sub_labels + 1) + ' labels) ')
                globals.csv_txt_file_exporter.write_single_line('\nPredicted \u2193' + main_and_non_main_labels_set.str.cat(
                    sep = ' ') + ' Non-' + globals.main_label + ' \u2193')

                confusion_matrix_temp = metrics.confusion_matrix(globals.main_and_non_main_labels_y_root, y_test_pred_rf, labels = main_and_non_main_labels_narray_temp)
                lbl_no = 0
                for line in confusion_matrix_temp:
                    globals.csv_txt_file_exporter.write_single_line('\n\u2192True ' + main_and_non_main_labels_narray_temp[lbl_no] + ' ' + str(line))
                    lbl_no = lbl_no + 1
                
                print('------------')
                print(self.name + ' - Confusion matrix on Test data (' + str(globals.no_of_sub_labels + 1) + ' labels) ')
                print('Predicted ' + main_and_non_main_labels_set.str.cat(sep = ' ') + ' Non-' + globals.main_label)
                
                lbl_no = 0
                for line in confusion_matrix_temp:
                    print('True ' + main_and_non_main_labels_narray_temp[lbl_no] + ' ' + str(line))
                    lbl_no = lbl_no + 1
            # Additional confusion matrix for binary classification <-

            if globals.label_set.size == 2:
                self.test_metrics['prec'] = round(metrics.precision_score(y_test, y_test_pred_rf, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.test_metrics['recal'] = round(metrics.recall_score(y_test, y_test_pred_rf, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.test_metrics['f1'] = round(metrics.f1_score(y_test, y_test_pred_rf, average = 'binary',pos_label = globals.label_set[0]), 4)

    def generate_monitor_prediction_file (self, to_run, X_monitor, agg_monitor):
        if to_run == 1:        
            import pandas as pd
            y_monitor_pred_rf = self.clf.predict(X_monitor)
            simu_predicted_df_rf = pd.concat([agg_monitor[['timestamp']], pd.DataFrame(y_monitor_pred_rf)],axis=1)
            simu_predicted_df_rf.columns = ['timestamp', 'predicted_label']
            globals.csv_txt_file_exporter.save_into_csv_file(simu_predicted_df_rf,'2_monitoring_data','2_monitor_predicted_' + self.name + '.csv')    

    def predict_monitoring_data(self, to_run, X_monitor_temp, y_monitor_temp, main_and_non_main_labels_set, labels_narray_temp, main_and_non_main_labels_narray_temp):

        if to_run == 1:
            y_monitor_pred_rf_temp = self.clf.predict(X_monitor_temp)
            self.monitor_metrics['accu'] = round(metrics.accuracy_score(y_monitor_temp, y_monitor_pred_rf_temp), 4)
            
            print(self.name + ' - Accuracy on Monitor data ' + str(self.monitor_metrics['accu']))

            if globals.label_set.size == 2:
                self.monitor_metrics['prec'] = round(metrics.precision_score(y_monitor_temp, y_monitor_pred_rf_temp, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.monitor_metrics['recal'] = round(metrics.recall_score(y_monitor_temp, y_monitor_pred_rf_temp, average = 'binary', pos_label = globals.label_set[0]), 4)
                self.monitor_metrics['f1'] = round(metrics.f1_score(y_monitor_temp, y_monitor_pred_rf_temp,average = 'binary', pos_label = globals.label_set[0]), 4)

            # Begin saving Confusion Matrix into text file ->
            globals.csv_txt_file_exporter.write_single_line('\n------------------------------------------------------')
            globals.csv_txt_file_exporter.write_single_line('\nAccuracy of ' + self.name + ' on monitoring data: ' + str(self.monitor_metrics['accu']))
            globals.csv_txt_file_exporter.write_single_line('\n' + self.name + ' Confusion matrix on monitoring data')
            globals.csv_txt_file_exporter.write_single_line('\nPredicted \u2193' + globals.label_set.str.cat(sep=' ') + ' \u2193')

            confusion_matrix_temp = metrics.confusion_matrix(y_monitor_temp, y_monitor_pred_rf_temp, labels = labels_narray_temp)
            lbl_no = 0
            for line in confusion_matrix_temp:
                globals.csv_txt_file_exporter.write_single_line('\n\u2192True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                lbl_no = lbl_no + 1
            # End Saving Confusion Matrix into text file <-
            
            # Begin printing Confusion Matrix to console ->
            print(self.name + ' - Confusion matrix on Monitoring data')
            print('Predicted ' + globals.label_set.str.cat(sep=' '))
            lbl_no = 0
            for line in confusion_matrix_temp:
                print('True ' + labels_narray_temp[lbl_no] + ' ' + str(line))
                lbl_no = lbl_no + 1
            

            # Additional confusion matrix for sub labels in case of Lying and Nonlying mode ->
            if globals.binary_mode is True:
                # Showing confusion matrix for sub labelss
                globals.csv_txt_file_exporter.write_single_line('\n' + self.name + ' Confusion matrix on monitoring data (' + str(globals.no_of_sub_labels + 1) + ' labels) ')
                globals.csv_txt_file_exporter.write_single_line('\nPredicted \u2193' + main_and_non_main_labels_set.str.cat(sep = ' ') + ' Non-' + globals.main_label + ' \u2193')

                confusion_matrix_temp = metrics.confusion_matrix(globals.main_and_non_main_labels_y_root_monitoring_temp, y_monitor_pred_rf_temp, labels = main_and_non_main_labels_narray_temp)
                
                lbl_no = 0
                for line in confusion_matrix_temp:
                    globals.csv_txt_file_exporter.write_single_line('\n\u2192True ' + main_and_non_main_labels_narray_temp[lbl_no] + ' ' + str(line))
                    lbl_no = lbl_no + 1

                # Begin printing Confusion Matrix to console ->
                print('------------')
                print(self.name + ' - Confusion matrix on monitoring data (' + str(globals.no_of_sub_labels + 1) + ' labels) ')
                print('Predicted ' + main_and_non_main_labels_set.str.cat(sep=' ') + ' Non-' + globals.main_label)
                lbl_no = 0
                for line in confusion_matrix_temp:
                    print('True ' + main_and_non_main_labels_narray_temp[lbl_no] + ' ' + str(line))
                    lbl_no = lbl_no + 1
            # Additional confusion matrix for sub labels in case of Lying and Nonlying mode <
            print('-------------------------------------------------------------------------')
            
    def save_experiment_result_into_db(self, to_run, no_of_original_train_valid_test_data_points, no_of_resampled_train_data_points, window_size, kfold, txtWindowStride_text, txtWindowSimuStride_text):
        
        if to_run == 1:
            slqInsertQuery = 'INSERT INTO ' + globals.experiment_result_table_name + '(model_title, model_init_name, model_binary_content, features_json_content, model_comments, train_table, monitor_table, no_of_predicted_classes, list_of_predicted_classes, original_sample_rate_in_hz, no_of_original_train_data_points, resampled_rate_in_hz, no_of_resampled_train_data_points, no_of_instances_for_each_class_in_resampled_train_table, algorithm, no_of_functions, list_of_functions, no_of_axes, list_of_axes, window_size, window_stride,k_fold, accuracy_train_valid, precision_train_valid, recall_train_valid, specificity_train_valid, f1_train_valid, accuracy_test, precision_test, recall_test, specificity_test, f1_test, monitoring_window_stride, accuracy_monitor, precision_monitor, recall_monitor, specificity_monitor, f1_monitor, start_time, end_time, running_time_in_minutes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

            monitor_table_name_temp = globals.monitoring_data_set_name
            if not globals.monitoring_mode:
                monitor_table_name_temp = 'n.a'
            model_comment = 'Comments:'

            end_time = datetime.datetime.now()
            model_init_name = '%04d' % end_time.year + '%02d' % end_time.month + '%02d' % end_time.day + '_' + '%02d' % end_time.hour + '%02d' % end_time.minute + '%02d' % end_time.second + '_' + globals.params_db['user']
           
            if globals.binary_mode:
                model_init_name += '_Binary_'
            else:
                model_init_name += '_Multi_'
           
            model_init_name += self.name

            model_data_content = pickle.dumps(self.clf)  # pickle the model

            duration = round(((end_time - globals.start_time).total_seconds()) / 60, 2)
            
            record_to_insert = (
                '', model_init_name, psycopg2.Binary(model_data_content), json.dumps(globals.json_axes_functions), model_comment,
                globals.train_valid_data_set_name, monitor_table_name_temp, len(globals.label_set), globals.label_set.str.cat(sep='_'),
                globals.original_sampling_rate , no_of_original_train_valid_test_data_points, globals.resampling_rate, no_of_resampled_train_data_points,
                str(globals.minimum_train_valid_instance_for_each_label), self.name, len(globals.list_agg_function_names), '_'.join(globals.list_agg_function_names),
                len(globals.list_axes_to_apply_functions), '_'.join(globals.list_axes_to_apply_functions), window_size, txtWindowStride_text + '%', kfold, 
                self.train_valid_metrics['accu'],
                self.train_valid_metrics['prec'], self.train_valid_metrics['recal'],
                self.train_valid_metrics['spec'], self.train_valid_metrics['f1'],
                self.test_metrics['accu'],
                self.test_metrics['prec'], self.test_metrics['recal'],
                self.test_metrics['spec'], self.test_metrics['f1'],
                txtWindowSimuStride_text + '%',
                self.monitor_metrics['accu'],
                self.monitor_metrics['prec'], self.monitor_metrics['recal'],
                self.monitor_metrics['spec'], self.monitor_metrics['f1'],               
                str(globals.start_time), str(end_time), duration)

            globals.cur.execute(slqInsertQuery, record_to_insert)
            globals.conn.commit()

    def init_metrics(self):
       
        self.train_valid_metrics = {}
        self.train_valid_metrics['accu'] = 0
        self.train_valid_metrics['prec'] = 0
        self.train_valid_metrics['recal'] = 0
        self.train_valid_metrics['f1'] = 0
        self.train_valid_metrics['spec'] = 0
        

        self.test_metrics = {}
        self.test_metrics['accu'] = 0
        self.test_metrics['prec'] = 0
        self.test_metrics['recal'] = 0
        self.test_metrics['f1'] = 0
        self.test_metrics['spec'] = 0
        
        self.monitor_metrics = {}
        self.monitor_metrics['accu'] = 0
        self.monitor_metrics['prec'] = 0
        self.monitor_metrics['recal'] = 0
        self.monitor_metrics['f1'] = 0
        self.monitor_metrics['spec'] = 0
        
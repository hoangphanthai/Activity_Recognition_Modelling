import os
import globals
from globals import log_message

def initialize_general_settings():
   
    try:
        globals.app_config.read(os.path.join(globals.dir_path, globals.APP_INI_FILE))

        params_global_setting = globals.app_config.items('GLOBAL SETTINGS')
        if params_global_setting[0][1] == '1':
            globals.binary_mode = True
        else:
            globals.binary_mode = False

        globals.test_proportion = float(params_global_setting[1][1])

        if params_global_setting[2][1] == '1':
            globals.csv_saving = True
        else:
            globals.csv_saving = False

    except OSError as err:
        log_message('OS error: {0}'.format(err))             
        raise Exception('Section {0} not found or invalid in the {1} file'.format('GENERAL SETTINGS', globals.APP_INI_FILE))


def update_db_credentials_file_path(iniFileNameFullPath):
   
    try:
        globals.app_config['IMPORT TAB']['dbini'] = iniFileNameFullPath
        with open(os.path.join(globals.dir_path, globals.APP_INI_FILE), 'w') as configfile:
            globals.app_config.write(configfile)

    except Exception as err:
        log_message('OS error: {0}'.format(err))


def get_db_params_from_in_file(iniFileNameFullPath):
 
    try:
        globals.db_config.read(iniFileNameFullPath)
        return globals.db_config.items('POSTGRESQL')
    except Exception as err:
        log_message('OS error: {0}'.format(err))
        return {}    


def update_import_data_controls_init_params(iniFileNameFullPath, data_source_select_params, db_credentials_params, csv_paths_params):
    
    try:       
        for item in data_source_select_params:
            globals.db_config['DATA SOURCE'][item] = data_source_select_params[item]

        for item in db_credentials_params:
            globals.db_config['POSTGRESQL'][item] = db_credentials_params[item]

        for item in csv_paths_params:
            globals.db_config['CSV PATH'][item] = csv_paths_params[item]            

        with open(iniFileNameFullPath, 'w') as configfile:  # save
            globals.db_config.write(configfile)                       
        return True

    except Exception as err:
        log_message('OS error: {0}'.format(err))
        return False


def get_import_data_controls_init_params():
    try:
        globals.app_config.read(os.path.join(globals.dir_path, globals.APP_INI_FILE))

        db_credentials_file_path = globals.app_config['IMPORT TAB']['dbini']
        globals.db_config.read(db_credentials_file_path)

        return db_credentials_file_path, globals.db_config.items('DATA SOURCE'), globals.db_config.items('POSTGRESQL'),  globals.db_config.items('CSV PATH')

    except Exception as err:
        log_message('OS error: {0}'.format(err))    
        return '', [], [], []


def update_training_tab_layout_data(input_training_settings):

    try:
        if globals.binary_mode:
            globals.app_config['GLOBAL SETTINGS']['binarymode'] = '1'
        else:
            globals.app_config['GLOBAL SETTINGS']['binarymode'] = '0'

        for item in input_training_settings:
            globals.app_config['TRAINING TAB'][item] = input_training_settings[item]
        
        with open(os.path.join(globals.dir_path, globals.APP_INI_FILE), 'w') as configfile: 
            globals.app_config.write(configfile)

    except Exception as err:
        log_message('OS error: {0}'.format(err)) 


def get_training_tab_layout_data():
    try:
        globals.app_config.read(os.path.join(globals.dir_path, globals.APP_INI_FILE))
        return globals.app_config.items('TRAINING TAB')
    
    except Exception as err:
        log_message('OS error: {0}'.format(err))            
        return []
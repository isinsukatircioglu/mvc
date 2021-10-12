import matplotlib
matplotlib.use('Agg')
import os, sys, time, shutil, importlib
sys.path.insert(0, './')
import IPython
import generic_train
from time import gmtime, strftime
import importlib.util
import torch

def evalContinueTraining(argv, file_path):
    if len(argv)>1:
        continueFrom = "--continue" in [str(v) for v in argv]
        #predictPoses = str(argv[1]) == "--predict"
        if continueFrom:
            config_instance_path = '/'.join(file_path.split('/')[:-1])
            folder = config_instance_path
            continueTrainingFrom = {'path': folder, 'iteration': '200000'}#'last_val'}
            return continueTrainingFrom
    return None

def getSavePath(config_orig_path, directory_path):
    config_save_path = '{}/{}'.format(save_path, os.path.basename(config_orig_path))
    return config_save_path

def savePythonFile(config_orig_path, directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    config_save_path = getSavePath(config_orig_path, directory_path)
    if not os.path.exists(config_save_path):
        shutil.copy(config_orig_path, config_save_path)
        print('copying {} to {}'.format(config_orig_path, config_save_path))

def loadModule(module_path_and_name):
    # if contained in module it would be a oneliner: 
    # config_dict_module = importlib.import_module(dict_module_name) 
    module_child_name = module_path_and_name.split('/')[-1].replace('.py','')
    spec = importlib.util.spec_from_file_location(module_child_name, module_path_and_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    dict_module_path_and_name = sys.argv[1]

    # load the specified module
    config_dict_module = loadModule(dict_module_path_and_name)
    config_dict = config_dict_module.config_dict

    continueTrainingFrom = evalContinueTraining(sys.argv, dict_module_path_and_name)
    if continueTrainingFrom:
        print("In if continueTrainingFrom:")
        old_class_file_name = config_dict['config_class_file'].split('/')[-1]
        config_dict['config_class_file'] = continueTrainingFrom['path'] + '/' + old_class_file_name
                
    # load the respective class as well
    class_module_path = config_dict['config_class_file']
    config_class_module = loadModule(class_module_path)
    config_class = config_class_module.Config_class

    # instanciate config with loaded params
    config_instance = config_class(config_dict)
    # save a copy of the config for future inspection
    if not continueTrainingFrom:
        time_stamp = strftime("%Y-%m-%d_%H-%M", gmtime())
        save_path = config_instance.get_parameter_description() + time_stamp
        savePythonFile(config_dict_module.__file__, save_path)
        savePythonFile(config_class_module.__file__, save_path)
    else:
        save_path = continueTrainingFrom['path']
        config_instance.continueTrainingFrom = continueTrainingFrom
        
    print(config_dict)
        
    # finally run the training
    optimization_flag = True
    if 'freeze_all' in config_dict.keys():
        if config_dict['freeze_all']:
            optimization_flag = False
    generic_train.main(save_path, config_instance, log=True, optimization_flag=optimization_flag)

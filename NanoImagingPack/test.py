# -*- coding: utf-8 -*-

from .config import __DEFAULTS__;
import numpy as np;
import inspect;
import itertools;
from .util import struct;

def test(func, test_para_dict, log_file_path = None):
    """
    function for testing functions
        
    :param func: The function to be tested
    :param test_para_dict:  a dictionary with the function parameters as keys and a list of potential arguments as values. If the function "func" has parameters with default values, these can (but do not need to ) be given    
    :param log_file_path: log file path (txt -file) of the test result. If none, the default value is used
    :return: a tuple: 1: list of strings for the successful test results, 2: a dictionary containing the results: the keys are the strings from the list (c.f.1), the values are the return values of func
    
    Note: The dictionary that is returned might contain many images! Only access values via keys 
    
    Example:
        # results are logged in \\nanoimagingpack.test_log.txt
        import NanoImagingPack as nip;
    
        # Define the test parameters in this way:
        test_para_dict = {
        'img' : [nip.readim('erika')],     # no default value in DampEdge: not giving this will result in an error!
        'width' : [None, 10],
 #       'rwidth' : [None, 0.1],          # e.g. this can be commented out, as it has a default value
        'axes' : [None, -1,-2,0,(-2,)],
        'func' : [nip.coshalf],
        'method' : ['zero', 'moisan', 'damp'],
        'sigma' : [4.0]
        }
        test_dict, keys = test(func = nip.DampEdge, test_para_dict = test_para_dict);
        res = test_dict[keys[0]]; # test result of the first test result
        
        
        
    """

    if log_file_path is None:
        log_file_path = __DEFAULTS__['TEST_LOG_FILE'];
    print('Test results saved in '+log_file_path)
    log_string = '';
    arg_names = inspect.getfullargspec(func).args;
    def_vals = inspect.getfullargspec(func).defaults;
    non_default_arg_number = len(arg_names)-len(def_vals);
    default_dict = dict(zip(arg_names[non_default_arg_number:], def_vals)); # Default values of the dictionary
    log_string += 'Log Test for function      '+ func.__name__+'\n';
    log_string += '==================================================\n\n';
    log_string += 'DEFAULT VALUES:\n';
    test_dict = {};
    
    def __get_log_string__(name, value):
        if isinstance(value, np.ndarray):
            if value.size > 10:
                to_log = 'array of shape '+str(value.shape);
            else:
                to_log = str(value);
        elif isinstance(value, list):
            to_log = 'list of size '+str(len(value));
        elif isinstance(value, dict):
            to_log = 'dictionary';
        elif isinstance(value, struct):
            to_log = 'nip struct';
        elif isinstance(value, tuple):
            to_log = 'tuple of size '+str(len(value));
        else:
            to_log = str(value);
        return('\t '+name+' = '+to_log);
    
    for n in arg_names:
        if n in test_para_dict:
            test_dict[n] = test_para_dict[n];
        elif n in default_dict:
            test_dict[n] = [default_dict[n]];
            log_string += __get_log_string__(n, default_dict[n])+'\n';
        else:
            raise ValueError('parameter '+n+' not given in test dictionary!');
            
    test_list =  [];
    for d in test_dict:
        test_list.append(test_dict[d]);
    para_test_list = list(itertools.product(*test_list));
    
    results_dict = {}
    keys = [];
    log_string += '\nRESULTS:\n========\n';
    for test_set in para_test_list:
        para_string = '';
        for a,t in zip(arg_names, test_set):
            para_string += __get_log_string__(a,t)+' ';
        log_string += para_string +'  ';
        print('Testing: '+para_string);    
        try:
            results_dict[para_string] = func(*test_set);
            keys += [para_string]
            log_string += ' SUCCESSFUL \n';
        except Exception as e: 
            log_string +=  str(e)+ '\n';
            print('FAILED');
    text_file = open(log_file_path, "w")
    text_file.write(log_string);
    text_file.close()
    
    return(results_dict,keys);

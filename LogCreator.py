import os
import datetime


def dict_to_csv(log_dict):

    line = []
    for key in log_dict:
        line.append(str(log_dict[key]))
    return ';'.join(line)+'\n'


def Add_to_Log(log_dict, file_path):

    #Check if file exists
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

    str2write = dict_to_csv(log_dict)

    f = open(file_path, 'a')
    f.write(str2write)
    f.close()
    







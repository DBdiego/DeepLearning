import os
import datetime


def dict_to_csv(log_dict):

    lines = []
    for key in log_dict:
        info = log_dict[key]
        
        line = []
        for parameter in info:
            line.append(str(info[parameter]))
        lines.append(';'.join(line))

    return '\n'.join(lines)+'\n'


def Add_to_Log(log_dict, file_path):

    #Check if file exists
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

    str2write = dict_to_csv(log_dict)

    f = open(file_path, 'a')
    f.write(str2write)
    f.close()
    







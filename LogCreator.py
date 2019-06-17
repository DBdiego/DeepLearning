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

    str2append = dict_to_csv(log_dict)

    f = open(file_path, 'a')
    f.write(str2append)
    f.close()

    
def get_run_id(status = 'read_current'):
    file_path = './Logs/Run_IDs.txt'
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    if len(lines) == 0:
        last_id = -1
    else:
        last_id = int(lines[-1].replace('\n', ''))
    
    if status == 'create_new':
        last_id += 1
        f = open(file_path, 'a')
        f.write('\n'+str(last_id))
        f.close()

    return last_id
            
    
    






import os
import datetime


def dict_to_csv(log_dict):

    line = []
    for key in log_dict:
        line.append(str(log_dict[key]))
    return '\n' + ';'.join(line)


def Add_to_Log(log_dict, file_path):

    #Check if file exists
    if not os.path.isfile(file_path):
        f = open(file_path, 'w')
        f.write(';'.join([key for key in log_dict.keys()]))
        f.close()

    str2append = dict_to_csv(log_dict)

    f = open(file_path, 'a')
    f.write(str2append)
    f.close()

    
def get_run_id(status='read_current'):
    file_path = './Logs/Run_IDs.txt'
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

    
    if status == 'create_new':
        last_id = (datetime.datetime.now()+ datetime.timedelta(hours=2)).strftime('%Y%m%d%H%M%S')
        f = open(file_path, 'a')
        f.write('\n' + last_id)
        f.close()

    else:
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        last_id = int(lines[-1].replace('\n', ''))

    return last_id
            
    
    






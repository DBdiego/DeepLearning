import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

'''
Columns in logs:  start_time
                  run_id
                  gpu
                  generation
                  network
                  n_conv
                  dim1
                  kernel_conv
                  stride_conv
                  kernel_pool
                  stride_pool
                  n_layers
                  dim2
                  train_time
                  end_time
                  accuracy
                  num_epochs
                  loss_log
                  test_imgs
                  errors
'''

def to_datetime(input_string):
    datetime_element = datetime.datetime.strptime(input_string, '%Y-%m-%d %H:%M:%S.%f')
    return datetime_element
    
def GPU_Usage(logs, show=False, save=False):


    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('GPU Usage')

    num_gpus_used = len(logs.groupby('gpu'))
    colors = ['#FF0000', '#00FF00']
    first_used = to_datetime(logs.iloc[ 0]['start_time'])
    last_used  = to_datetime(logs.iloc[-1]['end_time'  ])

    total_available = last_used - first_used

    for i in range(num_gpus_used):
        gpu_name = 'cuda:'+str(i)

        ax = fig.add_subplot('21'+str(i+1))
        ax.plot([first_used, first_used], [0, 0])
        ax.plot([ last_used, last_used ], [0, 0])
        
        gpu_logs = logs[logs['gpu']==gpu_name]

        num_rows, num_cols = gpu_logs.shape

        prev_end_time = first_used
        gpu_active_time = datetime.timedelta(hours=0)
        for row_index in range(num_rows):
            start_time = to_datetime(gpu_logs.iloc[row_index]['start_time'])
            end_time   = to_datetime(gpu_logs.iloc[row_index]['end_time'  ])
            gpu_active_time += (end_time-start_time)
            
            ax.plot([prev_end_time, start_time], [0, 0], c='r', label='dead time')
            ax.plot([start_time, start_time, end_time, end_time], [0, 1, 1, 0], c='g', label='active time')
            
            prev_end_time = end_time

        relative_gpu_activity = round(gpu_active_time/total_available*100, 1)
        ax.plot([end_time, last_used], [0, 0], c='r')
        ax.set_title('Activity of GPU '+str(i)+f' ({relative_gpu_activity}%)')
        

    short_stay_line = mlines.Line2D([],[], color='r', label='dead time')
    long_stay_line  = mlines.Line2D([],[], color='g', label='active time' )
    fig.legend(handles=[short_stay_line, long_stay_line],loc='upper right')
    
    if save:
        run_ID = logs.iloc[0]['run_id']
        plt.savefig(f'./Plots/GPU_Usage_{run_ID}.pdf')

    if show:
        plt.show()
    else:
        plt.close()


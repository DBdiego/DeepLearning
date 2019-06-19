import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Plotters.GPU_Usage_Plotter import GPU_Usage

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

file_name = '20190619010243.csv'
logs = pd.read_csv('./Logs/Backup_Logs/'+file_name, sep=';')

columns = logs.columns


# Fitness vs Generations
#Generation_Fitness(logs, show=1, save=1)

# GPU Usage over time
#GPU_Usage(logs, show=0, save=1)

f = open('./tit.txt', 'a')
f.write('')
f.close()













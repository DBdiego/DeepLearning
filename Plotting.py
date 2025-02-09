import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Plotters.AlexNet_Loss       import alexnet_loss
#from Plotters.Generation_Fitness import generation_fitness
from Plotters.Generation_Fitness_2_runs import generation_fitness
from Plotters.GPU_Usage_Plotter  import gpu_usage

trained_id   = '20190619030718'
untrained_id = '20190619150000'

run_id = untrained_id
#run_id = trained_id

file_name_all_logs = f'{run_id}.csv'

file_name_enn_logs_t = f'{trained_id}_ENN_fitness.csv'
file_name_enn_logs_u = f'{untrained_id}_ENN_fitness.csv'

# 1. Importing data
all_logs = pd.read_csv('./Logs/Backup_Logs/'     + file_name_all_logs, sep=';')

enn_logs_t = pd.read_csv('./Logs/Generation_Logs/' + file_name_enn_logs_t, sep=';')
enn_logs_u = pd.read_csv('./Logs/Generation_Logs/' + file_name_enn_logs_u, sep=';')

alx_logs = pd.read_csv('./Logs/AlexNet_Logs.csv', sep=';')


# 2. Plotting Data
# --> Fitness vs Generations
generation_fitness([enn_logs_t, enn_logs_u], show=1, save=1)

# --> GPU Usage over time
#gpu_usage(all_logs, show=1, save=1)

# --> AlexNet Loss vs epch
#alexnet_loss(alx_logs, show=1, save=1)











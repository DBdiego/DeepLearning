import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Plotters.GPU_Usage_Plotter  import gpu_usage
from Plotters.Generation_Fitness import generation_fitness


run_id = '20190619030718'

file_name_all_logs = f'{run_id}.csv'
file_name_enn_logs = f'{run_id}_ENN_fitness.csv'

all_logs = pd.read_csv('./Logs/Backup_Logs/'     + file_name_all_logs, sep=';')
enn_logs = pd.read_csv('./Logs/Generation_Logs/' + file_name_enn_logs, sep=';')


# Fitness vs Generations
generation_fitness(enn_logs, show=0, save=1)

# GPU Usage over time
gpu_usage(all_logs, show=1, save=1)














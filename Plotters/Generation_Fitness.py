import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

'''
Columns in logs:  Run ID
                  Generation
                  Maximum Fitness
                  Minimum Fitness
                  Average Fitness
                  Chromosome
                  Best Solution
'''

    
def generation_fitness(logs, show=False, save=False):


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    num_generations = len(logs)

    min_fits = []
    avg_fits = []
    max_fits = []
    for i in range(num_generations):
        min_fits.append(logs.iloc[i]['Minimum Fitness'])
        avg_fits.append(logs.iloc[i]['Average Fitness'])
        max_fits.append(logs.iloc[i]['Maximum Fitness'])

    ax.plot(range(num_generations), min_fits, c = 'r', label='Minimum Fitness')
    ax.plot(range(num_generations), avg_fits, c = 'b', label='Average Fitness')
    ax.plot(range(num_generations), max_fits, c = 'g', label='Maximum Fitness')

    ax.set_title('Fitness Accross Generations')
        
    ax.legend(loc='lower right')
    ax.set_ylabel('Accuracy [%]')
    ax.set_xlabel('Generations')
    if save:
        run_ID = logs.iloc[0]['Run ID']
        plt.savefig(f'./Plots/{run_ID}_Genertaion_Fitness.pdf')

    if show:
        plt.show()
    else:
        plt.close()


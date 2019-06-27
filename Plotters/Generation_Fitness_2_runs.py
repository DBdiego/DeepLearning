import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#plt.rc('font', family='serif')

'''
Columns in logs:  Run ID
                  Generation
                  Maximum Fitness
                  Minimum Fitness
                  Average Fitness
                  Chromosome
                  Best Solution
'''

    
def generation_fitness(log_dicts, show=False, save=False):


    fig = plt.figure(figsize=(8, 4), dpi=400)
    ax = fig.add_subplot(111)

    lss = ['-', '--']
    alphas = [1, 0.7]
    titles = ['Trained', 'Untrained']
    for i in range(len(log_dicts)):
        logs = log_dicts[i]
        num_generations = len(logs)

        min_fits = []
        avg_fits = []
        max_fits = []
        for j in range(num_generations):
            min_fits.append(logs.iloc[j]['Minimum Fitness'])
            avg_fits.append(logs.iloc[j]['Average Fitness'])
            max_fits.append(logs.iloc[j]['Maximum Fitness'])

        lw = 1
        ax.plot(range(num_generations), min_fits, c = 'r', label='Minimum Fitness '+titles[i], lw = lw, ls = lss[i], alpha=alphas[i])
        ax.plot(range(num_generations), avg_fits, c = 'b', label='Average Fitness '+titles[i], lw = lw, ls = lss[i], alpha=alphas[i])
        ax.plot(range(num_generations), max_fits, c = 'g', label='Maximum Fitness '+titles[i], lw = lw, ls = lss[i], alpha=alphas[i])

    ax.set_title('Fitness Accross Generations')
        
    ax.legend(loc='lower right')
    ax.set_ylabel('Accuracy [%]')
    ax.set_xlabel('Generations')
    ax.set_xlim([0, 20])

    ax.set_xticks(range(num_generations))
    ax.set_xticklabels(range(num_generations), fontsize=10)

    plt.grid(ls = '--', alpha = 0.7, zorder = 0)   
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.90, wspace=0.1, hspace=0.4)

    
    if save:
        #run_ID = logs.iloc[0]['Run ID']
        plt.savefig(f'./Plots/Genertaion_Fitness_T_U.jpg')

    if show:
        plt.show()
    else:
        plt.close()


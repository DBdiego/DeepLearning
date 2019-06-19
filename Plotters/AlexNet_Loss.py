import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

'''
Columns in logs:   start_time    ,
                   epoch         ,
                   loss          ,
                   n_minibatches ,
                   avg_loss_imgs
'''

    
def alexnet_loss(logs, show=False, save=False):


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    num_epochs = len(logs)-1

    losses = []
    start = 3
    for i in range(start, num_epochs):
        losses.append(logs.iloc[i]['loss'])


    ax.plot(range(start, num_epochs), losses, c = 'b', label='Epoch Loss', lw = 0.4)

    ax.set_title(f'AlexNet Loss Accross Epochs {start} to {num_epochs}')
        
    ax.legend(loc='upper right')
    ax.set_ylabel('Loss ')
    ax.set_xlabel('Epochs')
    if save:
        run_ID = '20190618102448'#logs.iloc[0]['run_id']
        plt.savefig(f'./Plots/{run_ID}_AlexNet_Loss.pdf')

    if show:
        plt.show()
    else:
        plt.close()


import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from cnn2 import Net
from dataloader import CustomDataset
from LogCreator import Add_to_Log

# torch.manual_seed(0)

class CNN:

    def __init__(self, network_index, generation_index, gpu_index, trainset, testset,
                 n_conv,
                 dim1,
                 kernel_conv,
                 stride_conv,
                 kernel_pool,
                 stride_pool,
                 n_layers,
                 dim2):

        log_dict = {'start_time'  : datetime.datetime.now()   ,
                    'gpu'         : 'cuda:'+str(gpu_index)    ,
                    'generation'  : generation_index          ,
                    'network'     : 'Net_'+str(network_index) , 
                    'n_conv'      : n_conv                    ,
                    'dim1'        : dim1                      ,
                    'kernel_conv' : kernel_conv               ,
                    'stride_conv' : stride_conv               ,
                    'kernel_pool' : kernel_pool               ,
                    'stride_pool' : stride_pool               ,
                    'n_layers'    : n_layers                  ,
                    'dim2'        : dim2                      } 

        # --------------------------------------
        # Parameters:
        MAXTRAINTIME = 15*60  # seconds, not sure if this is a good time. Note that testing time is not included, this is (often) slightly less than 1 epoch time.
        BATCH_SIZE = 20
        LR = 1*1E-2
        MOMENTUM = 0.7
        CONVERGENCE = 1E-5  # Not sure if this is a good value (smaller change than 0.001%)
        MIN_EPOCH = 15  # should be 6 or higher, it can have less epochs in results if the MAXTRAINTIME is exceeded.

        #print([n_conv, dim1, kernel_conv, stride_conv, kernel_pool, stride_pool, n_layers, dim2])
        # --------------------------------------
        # Data loading:
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
        testloader = DataLoader(testset,
                                shuffle=False, num_workers=1)

        # Display image
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        #print('\n=========== NEW NETWORK ===========')
        # --------------------------------------
        # CNN:
        use_gpu = torch.cuda.is_available()
        net = Net(n_conv, dim1, kernel_conv, stride_conv, kernel_pool, stride_pool, n_layers, dim2)

        if use_gpu:
            net = net.cuda()
            if 0 and torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)

        device = torch.device('cuda:'+str(gpu_index) if torch.cuda.is_available() else "cpu")
        #print('running on : ', 'cuda:'+str(gpu_index))
        net.to(device)


        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCEWithLogitsLoss()
        
        # with optim, can also use e.g. Adam
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
        #optimizer = optim.Adam(net.parameters(), lr=LR)

        # --------------------------------------
        print('\tTraining Network', network_index, 'on GPU #',gpu_index)
        # Training:
        losslst = []
        starttime = time.time()
        traintime = time.time() - starttime
        epoch = 0

        # for epoch in range(nr_epochs):  # loop over the dataset multiple times
        while traintime < MAXTRAINTIME:
            epoch = epoch + 1
            running_loss_epoch = 0.0  # reset running loss per epoch
            running_loss       = 0.0  # reset running loss per batch
            
            if epoch > MIN_EPOCH:  # minimum number of epochs
                try:
                    rule = abs(np.mean(np.diff(losslst[-5:]))) / losslst[-5:][0]
                except:
                    print('losslst: ', losslst[-5:])
                    rule = CONVERGENCE/10 #making the following if-statement true and breaking the loop

                    
                if rule < CONVERGENCE:
                    self.losslst = losslst
                    self.realtime = time.time() - starttime
                    break
                
            for i, data in enumerate(trainloader, 0):  # for every batch, start at 0

                # get the inputs and labels
                if traintime > MAXTRAINTIME:
                    break
            
                inputs, labels = data
                    
                inputs = inputs.to(device)#cuda()
                labels = labels.to(device)#cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)  # labels need to be of type: torch.long
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                every_x_minibatches = 20 # print every X mini-batches
                if i % every_x_minibatches == (every_x_minibatches-1):  
                    #print(f'\t N{network_index}:   [{epoch}, {i + 1}] avg. loss: {np.round(running_loss / every_x_minibatches, 4)}')
                    #print(outputs, labels)
                    running_loss_epoch += running_loss
                    running_loss = 0.0

                if epoch == 1 and i == 0:
                    batchtime = time.time() - starttime
                traintime = time.time() - starttime + batchtime  # + batchtime estimates the time for the next batch
                self.realtime = time.time() - starttime
                
            losslst.append(running_loss_epoch)
            self.tot_epoch = epoch
            self.losslst = losslst

            if len(losslst) > 5:
                print(abs(np.average(np.diff(np.array(losslst[-5:])))), LR*np.average(np.array(losslst[-5:])))
            if len(losslst) > 5 and abs(np.average(np.diff(np.array(losslst[-5:])))) < LR*np.average(np.array(losslst[-5:])):
                print(f'\t N{network_index}: epoch {epoch} reducing LR from {LR} to {LR/10}')
                LR = LR/10
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
                #optimizer = optim.Adam(net.parameters(), lr=LR)
                

            print(f'\t N{network_index}: epoch {epoch} loss:', running_loss_epoch)

        train_time = round(time.time()-starttime, 5)
        log_dict.update({'train_time':train_time})
            
        print('\tTraining Network', network_index, 'on GPU #', gpu_index,'DONE ('+str(round(train_time, 1))+'s)')
        print('\tTesting Network' , network_index, 'on GPU #', gpu_index)

        # --------------------------------
        # Testing:
        # Whole test data set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                
                images, labels = data
                images = images.to(device)#.cuda()
                labels = labels.to(device)#.cuda()
                outputs = net(images)
                a, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        self.accuracy = 100 * correct / total
        print('\t --> Accuracy of network', network_index,'on the '+str(total)+' test images: %d %%' % (
            self.accuracy))

        
        # Filling Logging Information and Saving the log
        log_dict.update({'train_time' : train_time,
                         'end_time'   : datetime.datetime.now(),
                         'accuracy'   : self.accuracy          ,
                         'epochs'     : epoch                  ,
                         'loss_log'   : losslst                ,
                         'test_imgs'  : total                  })

        Add_to_Log(log_dict, './Logs/GPU_Logs/Logs_Cuda_'+str(gpu_index)+'.txt')






























        #print('\t --> Finished Training')

        #print('Testing Network: DONE')
        
        #print('===================================\n')

'''
# ---------------------------------------------------------------------

# Main:
# Those 3 variables are set in the class itself look # parameters
# BATCH_SIZE = 10
# LR = 0.001
# MOMENTUM = 0.9


# n_conv = 3
# dim1 = [6, 16, 32]
# kernel_conv = [3, 3, 3]
# stride_conv = [1, 1, 1]
# kernel_pool = [2, 2, 2]
# stride_pool = [2, 2, 2]
# n_layers = 3
# dim2 = [120, 84, 40]
# print('LOADING DATA...')
#
# image_path = 'database/'
# normalise = True  # Will transform [0, 255] to [0, 1]
# # Load data set and organise into batch size and right input for Net()
# dataset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
# lengths = [10000, 10778]  # train data and test data
# train_dataset, test_dataset = random_split(dataset, lengths)  # 20778
# trial = CNN(train_dataset,
#             test_dataset,
#             n_conv,
#             dim1,
#             kernel_conv,
#             stride_conv,
#             kernel_pool,
#             stride_pool,
#             n_layers,
#             dim2)
=======
#BATCH_SIZE = 10
#LR = 0.001
#MOMENTUM = 0.9

n_conv = 3
dim1 = [6, 19, 32]
kernel_conv = [3, 3, 3]
stride_conv = [2, 2, 2]
kernel_pool = [3, 3, 3]
stride_pool = [2, 2, 2]
n_layers = 10
dim2 = [120, 111, 102, 93, 84, 75, 66, 57, 48, 40]


print('LOADING DATA...')



image_path = 'database/'
normalise = True # Will transform [0, 255] to [0, 1]
# Load data set and organise into batch size and right input for Net()
dataset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
lengths = [10000,10778] #train data and test data
train_dataset, test_dataset = random_split(dataset,lengths) # 20778
trial = CNN(train_dataset, test_dataset,
    n_conv,
    dim1,
    kernel_conv,
    stride_conv,
    kernel_pool,
    stride_pool,
    n_layers,
    dim2)
'''
# important self.variables are those three:
# print((trial.losslst)) # list of loss after each epoch
# print(trial.realtime)
# print(trial.accuracy)

# important self.variables are those three:
# print((trial.losslst)) # list of loss after each epoch
# print(trial.realtime)
# print(trial.accuracy)

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
from LogCreator import Add_to_Log, get_run_id

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
        
        run_id = get_run_id()
        log_dict = {'start_time'  : (datetime.datetime.now()+datetime.timedelta(hours=2))   ,
                    'run_id'      : run_id                    ,
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
        MAXTRAINTIME = 2*60*60  # seconds, not sure if this is a good time. Note that testing time is not included, this is (often) slightly less than 1 epoch time.
        BATCH_SIZE = 40
        LR = 5*1E-4
        MOMENTUM = 0.9
        CONVERGENCE = 1E-5  # Not sure if this is a good value (smaller change than 0.001%)
        MIN_EPOCH = int(1e3) # should be 6 or higher, it can have less epochs in results if the MAXTRAINTIME is exceeded.
        WEIGHT_DECAY = 1e-5

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
        #print(net)
        if use_gpu:
            net = net.cuda()
            if 0 and torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)

        device = torch.device('cuda:'+str(gpu_index) if torch.cuda.is_available() else "cpu")

        try: 
            net.to(device)
            train_the_network = 1 #True
            test_the_network  = 1
            
        except RuntimeError:
            train_the_network = 0
            test_the_network  = 0
            print('\t!!! GPU Memory Overload !!!')
            
            # Filling Logging Information and Saving the log
            log_dict.update({'train_time' : 0,
                             'end_time'   : (datetime.datetime.now()+datetime.timedelta(hours=2)),
                             'accuracy'   : 0                      ,
                             'epochs'     : 0                      ,
                             'loss_log'   : 1e4                    ,
                             'test_imgs'  : 0                      ,
                             'errors'     : 'GPU Memory Overload'  })
            

        starttime = time.time()
        traintime = time.time() - starttime
        train_time = 0
        losslst = []
        epoch = 0
        if train_the_network:
            # Loss function
            criterion = nn.CrossEntropyLoss()
            #criterion = nn.BCEWithLogitsLoss()
            
            # with optim, can also use e.g. Adam
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay = WEIGHT_DECAY)

            # --------------------------------------
            print('\tTraining Network', network_index, 'on GPU #',gpu_index)
            # Training:


            # for epoch in range(nr_epochs):  # loop over the dataset multiple times
            while traintime < MAXTRAINTIME:
                epoch = epoch + 1
                running_loss_epoch = 0.0  # reset running loss per epoch
                running_loss       = 0.0  # reset running loss per batch

                '''
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
                '''
                    
                for i, data in enumerate(trainloader, 0):  # for every batch, start at 0

                    # get the inputs and labels
                    if traintime > MAXTRAINTIME:
                        break
                
                    inputs, labels = data
                    
                    inputs = inputs.to(device) #cuda()
                    labels = labels.to(device) #cuda()

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

                # Last number is average loss value per minibatch in this epoch
                print(outputs[0])
                print(labels[0])
                print(f'\t N{network_index}: epoch {epoch} loss:', round(running_loss_epoch, 5), f'on {i} minibatches {(running_loss_epoch/i)/BATCH_SIZE}')

            train_time = round(time.time()-starttime, 5)
                
            print('\tTraining Network', network_index, 'on GPU #', gpu_index,'DONE ('+str(round(train_time, 1))+'s)')
            
            
            # Testing:
        if test_the_network:
            print('\tTesting Network' , network_index, 'on GPU #', gpu_index)
            correct = 0
            total   = 0
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
                             'end_time'   : (datetime.datetime.now()+datetime.timedelta(hours=2)),
                             'accuracy'   : self.accuracy          ,
                             'num_epochs' : epoch+1                ,
                             'loss_log'   : losslst                ,
                             'test_imgs'  : total                  ,
                             'errors'     : 'no errors'            })

        Add_to_Log(log_dict, './Logs/GPU_Logs/Logs_Cuda_'+str(gpu_index)+'.csv')

        self.all_info = log_dict





















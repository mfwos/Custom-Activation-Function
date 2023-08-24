import openml
import seaborn as sbn
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

PROJECT_PATH = '...' # choose your project path here
os.chdir(PROJECT_PATH) 

from network_classes import actvf_fc_nn

rand_seed = 100                     # random seed for reproducibility
model_device = torch.device('cuda') # Define device on which to compute


### Get dataset for creating the costum activation function;
### the dataset used is covertype 

data = openml.datasets.get_dataset(44121).get_data()[0]

dict_outer_nn_size = {'small': [48,24],
                      'medium':[48,24,16],
                      'large':   [96,48,32]}

dict_activ_nn_size = {'small': [16,8],
                      'medium':[32,16],
                      'large':   [32,16,8]}

dict_base_activation = {'tanh': torch.nn.Tanh(),
                        'relu': torch.nn.ReLU(),
                        'gelu': torch.nn.GELU()}

l_outer_nn_layers = ['small','medium','large']
l_activ_nn_layers = ['small','medium','large']
l_base_activation = ['tanh','relu','gelu']



for outer_size in l_outer_nn_layers:
    for activ_size in l_activ_nn_layers:
        for base_activ in l_base_activation:

            shuffle_state = np.random.RandomState(rand_seed)
            torch.manual_seed(rand_seed**3)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            X_train, X_valid, y_train, y_valid = train_test_split(data.drop('Y', axis = 1),data['Y'],test_size = 50000, random_state = rand_seed)

            ### Normalize data
            
            X_valid = X_valid / X_train.std() - X_train.mean()
            X_train = X_train / X_train.std() - X_train.mean()

            ### Define activation function-flexible NN
            
            nn_layers = dict_outer_nn_size[outer_size]
            actv_layers = dict_activ_nn_size[activ_size]
            base_actv = dict_base_activation[base_activ]

            model_name = 'tab_classification_covertype_' + base_activ + '_' + outer_size + '_' + activ_size

            model = actvf_fc_nn(X_train.shape[1], True, len(nn_layers), nn_layers, base_actv, 
                        len(actv_layers), actv_layers, True, 1, torch.nn.Sigmoid()).to(model_device)

            y_train_torch = torch.tensor(np.array(y_train.values, dtype = int), dtype = torch.float32).to(model_device)
            y_valid_torch = torch.tensor(np.array(y_valid.values, dtype = int), dtype = torch.float32).to(model_device)

            optim = torch.optim.Adam(model.parameters(), lr = 0.005)

            n_max_epochs = 400
            early_stopping_epochs = 10
            batch_size = 20000 # the combination 'large'/'large'/'gelu' had to be run with batch_size = 10000 due to memory issues of my GPU
            no_batches = int(np.ceil(len(X_train) / batch_size))
            
            best_score = 0
            epochs_no_improv = 0

            for n in range(n_max_epochs):
                new_idxs = np.arange(len(X_train))
                shuffle_state.shuffle(new_idxs)
                X_train = X_train.iloc[new_idxs,:]
                y_train_torch = y_train_torch[new_idxs]
                
                for m in range(no_batches):
                    
                    start_idx = m*batch_size
                    end_idx = (m+1) * batch_size if m != no_batches else len(X_train)
                
                    output = model(torch.tensor(X_train.iloc[start_idx : end_idx, :].values, dtype = torch.float32).to(model_device))
                    loss = torch.nn.BCELoss(reduction = 'none')(output, y_train_torch[start_idx:end_idx]).sum() / batch_size
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    #print(m)
                    
                model.eval()
                output_eval = np.ones(50000)
                output_eval[:25000] = model(torch.tensor(X_valid.values[:25000,:], dtype = torch.float32).to(model_device)).detach().clone().cpu().numpy()
                output_eval[25000:] = model(torch.tensor(X_valid.values[25000:,:], dtype = torch.float32).to(model_device)).detach().clone().cpu().numpy()
                eval_score = roc_auc_score(y_valid_torch.detach().clone().cpu().numpy(), output_eval)
                
                
                if best_score < eval_score:
                    best_score = eval_score
                    epochs_no_improv = 0
                else:
                    epochs_no_improv += 1
                
                
                print("Finished epoch ", n, " of training. Validation BCELoss this epoch: ", eval_score)
                
                # Evaluate the Activation Function and plot its graph
                actv_func = model.activation_func.eval()
                
                x = np.arange(-3,3,0.1)
                actv_y = actv_func(torch.tensor(x.reshape(-1,1), dtype = torch.float32).to(model_device)).squeeze().detach().clone().cpu().numpy()
                plt.figure()
                sbn.lineplot(x = x, y = actv_y).set(title = 'Tabular classification with base activation ' + base_activ + '\nouter nn: ' + outer_size + '     activation nn: ' + activ_size)
                
                model.train()
                
                # if end of training is reached: Save the plot of the activation functions graph
                if epochs_no_improv >= early_stopping_epochs:
                    print("There were ", early_stopping_epochs, " epochs without improvement. Training will be stopped.")
                    plt.savefig(PROJECT_PATH + 'nn_activation_functions/' + model_name + '.png')
                    break
                
                else:
                    plt.show()
            
            # Save activation function for further use if desired.
            actv_func = model.activation_func.eval()
            torch.save({
                'model_state_dict': actv_func.state_dict(),
                'actv_func_layers': actv_layers,
                'training_dataset': 'openml_covertype_44121',
                'training_outer_nn': nn_layers,
                'training_eval_score': eval_score}
                , PROJECT_PATH + 'nn_activation_functions/' + model_name + '.pth')
            
            print("Finished calculation of activation function for combination ",
                  outer_size," ",activ_size, " ", base_activ)
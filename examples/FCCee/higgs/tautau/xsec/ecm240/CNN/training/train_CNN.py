import sys,os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import uproot
#from root_pandas import read_root, to_root
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve, auc
import ROOT
import joblib
import glob
from matplotlib import rc
import pprint
from higgs_config import *
from topVts_config import *
from model import CNN_Model
from model import DNN
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import data_prepper

model_struct = 'DNN'

train = False

higgs_path = '/ceph/awiedl/FCCee/HiggsCP/R5-tag/stage2_241025_BDT/'
topVts_path = '/ceph/xzuo/FCC_ntuples/topVts/training_samples/'

topVts_cat = [#'dilep',
            #'semilep_heavy',
            'semilep_light'
            ]

higgs_cats = [#'NuNu',
              'QQ'
              ]

higgs_subcats = [#'LL',
                 #'LH',
                 'HH'
                 ]

def higgs_train_and_test():
    for c in higgs_cats:
        for s in higgs_subcats:
            train_loader, test_dataset = data_prepper.higgs_data_prepper(higgs_path+c+'/'+s+'/', globals()[f'vars_list_{c+s}'])
            # Definiere das Modell
            if model_struct == 'DNN':
                model = DNN(globals()[f'num_vars_{c+s}'])
            else:
                model = CNN_Model(globals()[f'num_vars_{c+s}'])

            # Loss-Funktion und Optimierer
            criterion = nn.BCELoss() #reduction='none')
            optimizer = optim.Adam(model.parameters(), lr=0.001)
                
            #Fit the model
            print("Training model")
            # Trainingsloop

            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()  # Setzt das Modell in den Trainingsmodus
                running_loss = 0.0

                for batch_idx, (inputs, labels, weights) in enumerate(train_loader):
                    # Eingabe-Daten (mit zusätzlicher Dimension für Kanäle)
                    inputs = inputs.unsqueeze(1)  # Form wird jetzt (batch_size, 1, num_params)
                    labels = labels.unsqueeze(1)
                    # Forward-Pass
                    outputs = model(inputs)
                    
                    # Wende die Gewichtung der einzelnen Events an
                    loss = criterion(outputs, labels)
                    #weighted_loss = loss * event_weights

                    # Verlust für den Batch berechnen (Summe der gewichteten Verluste)
                    batch_loss = (weights*loss).mean()
                    
                    # Backward-Pass und Optimierung
                    optimizer.zero_grad()  # Gradienten zurücksetzen
                    batch_loss.backward()  # Gradienten berechnen
                    optimizer.step()  # Parameter aktualisieren

                    # Verlust kumulieren
                    running_loss += batch_loss.item()
                    
                # Ausgabe des Verlusts nach jeder Epoche
                print(f'Epoche [{epoch+1}/{num_epochs}], Verlust: {running_loss/len(train_loader):.4f}')
            
            #Saving Model
            model.eval()
            torch_input = torch.randn(64,1,globals()[f'num_vars_{c+s}'])
            #torch_weights = torch.randn(64)
            torch.onnx.export(model, torch_input,'/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/CNN/models/model_'+c+s+'.onnx')
            torch.save(model.state_dict(), '/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/CNN/models/model_'+c+s+'.pt')

            print('Testing model')
            corr = 0
            false = 0
            for data, label in test_dataset:
                data = data.unsqueeze(0).unsqueeze(1)
                y_pred = model(data).flatten()

                if (y_pred >= 0.5 and label == 1):
                    print('sig')
                    corr += 1
                elif (y_pred <= 0.5 and label == 0):
                    print('bkg')
                    corr += 1 
                else: 
                    print('falsch')
                    false += 1
            print('Acc:', corr/(corr+false))

            x_test, y_test = test_dataset.tensors
            x_test = x_test.unsqueeze(1)
            pred_test = model(x_test)
            y_test = y_test.detach().numpy()
            pred_test = pred_test.detach().numpy()
            # Calculate FPR, TPR, and AUC
            fpr, tpr, thresholds = roc_curve(y_test, pred_test, pos_label=1)
            roc_auc = auc(fpr, tpr)

            # Create the figure and plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title('FCC-ee Simulation IDEA Delphes', loc='right', fontsize=20)

            # Plot the ROC curve
            plt.plot(fpr, tpr, lw=1.5, color="k", label=f'Htautau ROC (area = {roc_auc:.3f}) for NuNuHH')

            # Plot the baseline for random classifier
            plt.plot([0., 1.], [0., 1.], linestyle="--", color="k", label='50/50')

            # Set limits and labels
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.ylabel('Background rejection', fontsize=30)  # 1 - FPR
            plt.xlabel('Signal efficiency', fontsize=30)  # TPR

            # Adjust ticks and legend
            ax.tick_params(axis='both', which='major', labelsize=25)
            plt.legend(loc="lower left", fontsize=20)
            plt.grid()
            plt.tight_layout()

            # Save the figure
            fig.savefig(f"/web/awiedl/public_html/ML/CNN/higgs_ROC_{c+s}.pdf")

def topVts_train_and_test():
    for c in topVts_cat:
        if train == True:
            train_loader = data_prepper.topVts_data_prepper(topVts_path, c, 'train')
            # Definiere das Modell
            if model_struct == 'DNN':
                model = DNN(num_train_vars[c])
                print(num_train_vars[c])
            else:
                model = CNN_Model(num_train_vars[c])

            # Loss-Funktion und Optimierer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
                        
            #Fit the model
            print("Training model")
            # Trainingsloop
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()  # Setzt das Modell in den Trainingsmodus
                running_loss = 0.0

                for batch_idx, (inputs, labels, weights) in enumerate(train_loader):
                    # Eingabe-Daten (mit zusätzlicher Dimension für Kanäle)
                    #inputs = inputs.unsqueeze(1)  # Form wird jetzt (batch_size, 1, num_params)
                    labels = labels.unsqueeze(1)
                    # Forward-Pass
                    outputs = model(inputs)
                            
                    # Wende die Gewichtung der einzelnen Events an
                    loss = criterion(outputs, labels)
                    #weighted_loss = loss * event_weights

                    # Verlust für den Batch berechnen (Summe der gewichteten Verluste)
                    batch_loss = (weights*loss).mean()
                    
                    # Backward-Pass und Optimierung
                    optimizer.zero_grad()  # Gradienten zurücksetzen
                    batch_loss.backward()  # Gradienten berechnen
                    optimizer.step()  # Parameter aktualisieren

                    # Verlust kumulieren
                    running_loss += batch_loss.item()

                # Ausgabe des Verlusts nach jeder Epoche
                print(f'Epoche [{epoch+1}/{num_epochs}], Verlust: {running_loss/len(train_loader):.4f}')

            #Saving Model
            model.eval()
            torch_input = torch.randn(64,1,num_train_vars[c])
            #torch_weights = torch.randn(64)
            torch.onnx.export(model, torch_input,'/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+model_struct+c+'.onnx')
            torch.save(model.state_dict(), '/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+model_struct+c+'.pt')

        print('Testing model')

        if train == False:
            model = DNN(num_train_vars[c])
            model.load_state_dict(torch.load('/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+model_struct+c+'.pt'))
            model.eval()
        corr = 0
        false = 0
        x_test, y_test = data_prepper.topVts_data_prepper(topVts_path, c, 'test')
        dataset = TensorDataset(x_test, y_test)
        for data, label in dataset:
            if model_struct == 'CNN':
                data = data.unsqueeze(0).unsqueeze(1)
            y_pred = model(data).flatten()
            if (y_pred >= 0.5 and label == 1):
                corr += 1
            elif (y_pred <= 0.5 and label == 0):
                corr += 1 
            else: 
                false += 1
        print('Acc:', corr/(corr+false))

        x_test, y_test = dataset.tensors
        x_test = x_test.unsqueeze(1)
        pred_test = model(x_test)
        y_test = y_test.detach().numpy()
        pred_test = pred_test.detach().numpy().flatten()
        # Calculate FPR, TPR, and AUC
        fpr, tpr, thresholds = roc_curve(y_test, pred_test, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Create the figure and plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('FCC-ee Simulation IDEA Delphes', loc='right', fontsize=20)

        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=1.5, color="k", label=f'Htautau ROC (area = {roc_auc:.3f}) for NuNuHH')

        # Plot the baseline for random classifier
        plt.plot([0., 1.], [0., 1.], linestyle="--", color="k", label='50/50')

        # Set limits and labels
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.ylabel('Background rejection', fontsize=30)  # 1 - FPR
        plt.xlabel('Signal efficiency', fontsize=30)  # TPR

        # Adjust ticks and legend
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(loc="lower left", fontsize=20)
        plt.grid()
        plt.tight_layout()

        # Save the figure
        fig.savefig(f"/web/awiedl/public_html/ML/CNN/topVts_ROC_{model_struct+c}.pdf")

if __name__ == '__main__':
    topVts_train_and_test()
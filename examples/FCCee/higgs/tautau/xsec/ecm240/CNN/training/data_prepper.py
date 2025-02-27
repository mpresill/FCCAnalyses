import sys,os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import ROOT
import joblib
import glob
from matplotlib import rc
import pprint
from higgs_config import *
from topVts_config import *
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_entries(infilepath: str) -> tuple[int, int]:
    '''
    Get number of original entries and number of actual entries in the file
    '''
    events_processed = 0
    events_in_ttree = 0

    with ROOT.TFile(infilepath, 'READ') as infile:
        try:
            events_processed_obj = infile.Get('eventsProcessed')
            if events_processed_obj:
                events_processed = events_processed_obj.GetVal()
            events_ttree = infile.Get("events")
            if events_ttree:
                events_in_ttree = events_ttree.GetEntries()
            else:
                return None, None

        except AttributeError:
            return None, None

    return events_processed, events_in_ttree

def higgs_data_prepper(path, vars_list):
    N = {}
    N_gen = {}
    eff = {}
    weight = {}
    N_bkg = 0
    N_bkg_gen = 0
    tot_weight_bkg = 0
    N_sig = 0
    N_sig_gen = 0
    tot_weight_sig = 0
    #get gen number of events for each signal and backgorund file
    for i in sigs+bkgs:
        files = glob.glob(path + i + '/chunk_*.root')
        N[i] = 0
        N_gen[i] = 0
        eff[i] = 0
        for f in files:
            #getting the raw number of events in this way only works if there are some events in the trees themselves 
            #but here it doesn't matter as we don't use those samples anyway
            events_processed, events_in_ttree = get_entries(f)
            if events_processed is not None and events_in_ttree is not None:
                N[i] += events_in_ttree
                N_gen[i] += events_processed

        #calculate efficiency of each sample    
        if N_gen[i]!=0:
            eff[i] = N[i] / N_gen[i]
        weight[i] = xsec[i] * eff[i] * 10.8e6

        #commulative number of events for background
        if i in bkgs: 
            N_bkg += N[i]
            N_bkg_gen += N_gen[i]
            tot_weight_bkg += weight[i]
        if N_bkg_gen!=0:
            eff_tot_bkg = N_bkg / N_bkg_gen
        if i in sigs: 
            N_sig += N[i]
            N_sig_gen += N_gen[i]
            tot_weight_sig += weight[i]
        if N_sig_gen!=0:
            eff_tot_sig = N_sig / N_sig_gen

    #minumum number between the events in the samples and the one we expect to have in the signal composition
    N_min = {}
    N_sig_new = N_sig
    for i in sigs:
        N_min[i] = min(N[i], N_sig * weight[i] / tot_weight_sig) 
        if N_min[i]==N[i] and weight[i]>0 and N[i]>0:
            N_sig_new = N_min[i] * tot_weight_sig / weight[i]

    #upload signals into a dataframe
    df_sig = pd.DataFrame()
    for q in sigs:
        prev = len(df_sig)
        target_events = int(N_sig_new * weight[q] / tot_weight_sig)
            
        # Only takes the samples that actually have any events remaining  
        if N[q] > 0: 
            files = glob.glob(path + q + '/chunk_*.root')
            df = pd.DataFrame()

            for file in files:
                f = uproot.open(file)
                tree = f["events"]
                temp_df = tree.arrays(expressions=vars_list, library="pd")
                df = pd.concat([df, temp_df])

                # Check if we have enough events to meet the target
                if len(df) >= target_events:
                    break 

            df = df.head(target_events)
            df_sig = pd.concat([df_sig, df])
        
    #now for backgrounds
    df_bkg = pd.DataFrame()
    for q in bkgs:
        prev = len(df_bkg)
        target_events = int(N_sig_new * weight[q] / tot_weight_bkg)
            
        # Only takes the samples that actually have any events remaining  
        if N[q] > target_events and target_events>0: 
            files = glob.glob(path + q + '/chunk_*.root')
            df = pd.DataFrame()

            for file in files:
                f = uproot.open(file)
                tree = f["events"]
                temp_df = tree.arrays(expressions=vars_list, library="pd")
                df = pd.concat([df, temp_df])

                # Check if we have enough events to meet the target
                if len(df) >= target_events:
                    break 

            df = df.head(target_events)
            df_bkg = pd.concat([df_bkg, df])
        
    #set Signal and background labels
    df_sig["label"] = 1
    df_bkg["label"] = 0
        
    #save some data for testing later
    df_sig = df_sig.sample(frac=1, random_state=1)
    df_bkg = df_bkg.sample(frac=1, random_state=1)
    train_sig, test_sig = train_test_split(df_sig, test_size=0.3)
    train_bkg, test_bkg = train_test_split(df_bkg, test_size=0.3)
        
    #Combine the datasets
    df_train = pd.concat([train_sig,train_bkg])
    #shuffle the rows so they are mixed between signal and background
    df_train = df_train.sample(frac=1)
    df_test = pd.concat([test_sig,test_bkg])

    #Split into class label (y) and training vars (x)
    y = df_train["label"]
    x = df_train[vars_list]

    y = y.to_numpy()
    y2D = np.array([[1, 0] if x == 0 else [0, 1] for x in y])
    x = x.to_numpy()

    y_test = df_test["label"]
    x_test = df_test[vars_list]

    y_test = y_test.to_numpy()
    y_test2D = np.array([[1, 0] if x == 0 else [0, 1] for x in y_test])
    x_test = x_test.to_numpy()
            
    #Sample weights to balance the classes
    weights = compute_sample_weight(class_weight='balanced', y=y)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Umwandlung in PyTorch Tensoren
    x_tensor = torch.tensor(x, dtype=torch.float32)  # Eingabedaten
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Labels 
    x_testtensor = torch.tensor(x_test, dtype=torch.float32)  # Eingabedaten
    y_testtensor = torch.tensor(y_test, dtype=torch.float32)  # Labels 

    # Erstellen des Datasets und DataLoader
    dataset = TensorDataset(x_tensor, y_tensor, weights_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # Batch-Größe 64
    
    test_dataset = TensorDataset(x_testtensor, y_testtensor)

    return train_loader, test_dataset

def topVts_data_prepper(path, sig, purpose):
    vars_list = train_vars[sig]
    cat_sel = cat_sel_base

    ## load df for sig and bkg
    df_sig = pd.read_pickle(f"{path}/sigs_{pkl_version}/{sig}_{purpose}.pkl")
    df_sig = df_sig.query(cat_sel[sig])
    df_sig["label"] = 1
    sumW_sig = df_sig["evt_wgt"].sum()
    print (f"{sig}:  {sumW_sig}")

    df_bkg = {}
    sumW_bkg = 0
    for bkg in bkg_list:
        df_bkg[bkg] = pd.read_pickle(f"{path}/bkgs_{pkl_version}/{bkg}_{purpose}.pkl")
        df_bkg[bkg] = df_bkg[bkg].query(cat_sel[sig])
        df_bkg[bkg]["label"] = 0
        ## overwrite bkg weight so minor bkgs (WW, ZZ, ZH) do not have huge weights
        df_bkg[bkg]["evt_wgt"]=1
        sumW_bkg += df_bkg[bkg]["evt_wgt"].sum()
        print (f"{bkg}:  {sumW_bkg}")

    ## balance event weights
    df_sig["evt_wgt"] = df_sig["evt_wgt"] * sumW_bkg / sumW_sig

    ## prepare inputs for training
    df_tot = df_sig
    for bkg in bkg_list:
        df_tot = pd.concat([df_tot, df_bkg[bkg]])

    y = df_tot["label"]
    x = df_tot[vars_list]
    
    w = df_tot["evt_wgt"]

    y = y.to_numpy()
    x = x.to_numpy()
    w = w.to_numpy()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    w_tensor = torch.tensor(w, dtype=torch.float32)
    if purpose == 'test':
        return x_tensor, y_tensor
    dataset = TensorDataset(x_tensor, y_tensor, w_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return train_loader
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
from training_variables import *
from event_wgts import *

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


def topVts_data_prepper_DNN(path, cat, purpose):

    output_path = '/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/training/output.txt'
    vars_list = training_vars[cat]
    cat_sel = sig_filter[cat]
    liste = vars_list + [cat_sel]
    samples = [] #sample_list['training']['sig'] + sample_list['training']['bkg']
    samples_sig = [] #sample_list['training']['sig']
    samples_bkg = []

    with open(output_path, "a") as file:
        file.write(f"{cat}\n")

    for sig in sample_list['training']['sig']:
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            processed, entries = get_entries(f'{path}{cat}/{sig}/chunk0.root')
            if 'events;1' in file.keys() and entries > 2:
                samples += [sig]
                samples_sig += [sig]
            else:
                with open(output_path, "a") as file:
                    file.write(f"       {sig}: 0\n")

    for bkg in sample_list['training']['bkg']:
        with uproot.open(f'{path}{cat}/{bkg}/chunk0.root') as file:
            processed, entries = get_entries(f'{path}{cat}/{bkg}/chunk0.root')
            if 'events;1' in file.keys() and entries > 2:
                samples += [bkg]
                samples_bkg += [bkg]
            else:
                with open(output_path, "a") as file:
                    file.write(f"       {bkg}: 0\n")

    ## load df for sig and bkg
    df = {}
    W_bkg = 0
    N_bkg = 0

    for bkg in samples_bkg:
        with uproot.open(f'{path}{cat}/{bkg}/chunk0.root') as file:
            events = file['events']
            temp_df = events.arrays(expressions=vars_list, library="pd")
            temp_df["label"] = 0
            #if(bkg=='p8_ee_WW_ecm365' or bkg=='p8_ee_ZZ_ecm365'):
            #    if cat == 'dilep_1tau':
            #        temp_df["evt_wgt"]=1
            #    else:
            temp_df["evt_wgt"]=evt_wgt['bkgs'][bkg]
            N = len(temp_df)
            #else:    
            #    temp_df["evt_wgt"]=1
            sumW_bkg = temp_df["evt_wgt"].sum()
        df[bkg] = temp_df
        print (f"{bkg}:  {sumW_bkg}")
        print (f'Size {bkg}:  {N}')
        with open(output_path, "a") as file:
            file.write(f"       {bkg}: {N}\n")
        W_bkg += sumW_bkg
        N_bkg += N
        ## balance event weights
        #df[bkg]["evt_wgt"] = df[bkg]["evt_wgt"] * W_sig / W_bkg

    with open(output_path, "a") as file:
        file.write(f"       Wgt All bkg: {W_bkg}\n")
        file.write(f"       Size All bkg: {N_bkg}\n")

    W_sig = 0
    N_sig = 0


    for sig in ['wzp6_ee_SM_tt_tWsTWb_tlepTall_ecm365','wzp6_ee_SM_tt_tWbTWs_tallTlep_ecm365','wzp6_ee_SM_tt_tWbTWs_tallTlight_ecm365','wzp6_ee_SM_tt_tWbTWs_tallTheavy_ecm365','wzp6_ee_SM_tt_tWsTWb_tlightTall_ecm365','wzp6_ee_SM_tt_tWsTWb_theavyTall_ecm365']:
        if(sig not in samples_sig):
            continue
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            events = file['events']
            temp_df = events.arrays(expressions=liste, library="pd").query(f'{cat_sel} == 1')
        if len(temp_df) < 2:
            samples.remove(sig)
            samples_sig.remove(sig)
        else:
            temp_df["label"] = 1
            temp_df["evt_wgt"]= evt_wgt['sigs'][sig]
            temp_df, plot_df = train_test_split(temp_df, train_size=0.5, test_size=0.5, random_state=12)
            sumW_sig = temp_df["evt_wgt"].sum()
            df[sig] = temp_df
            N = len(df[sig])
            print (f"{sig}:  {sumW_sig}")
            print (f"Size {sig}:  {N}")
            with open(output_path, "a") as file:
                file.write(f"       {sig}: {N}\n")
            W_sig += sumW_sig
            N_sig += N
            
    with open(output_path, "a") as file:
        file.write(f"       Size All sig: {N_sig}\n")
        file.write(f"       All sig: {W_sig}\n")
        file.write(f"       sig wgt: {W_bkg/W_sig}\n")

    for sig in samples_sig:
        df[sig]["evt_wgt"] = df[sig]["evt_wgt"] * W_bkg / W_sig

    ## prepare inputs for training
    df_tot = pd.DataFrame()
    df_tot_test = pd.DataFrame()
    for s in samples:
        train, test = train_test_split(df[s], train_size=0.7, test_size=0.3, random_state=12)
        #train['evt_wgt'] = train["evt_wgt"] * W_bkg / W_sig
        #test['evt_wgt'] = test["evt_wgt"] * W_bkg / W_sig
        df_tot = pd.concat([df_tot, train])
        df_tot_test = pd.concat([df_tot_test, test])


    y = df_tot["label"]
    y_test = df_tot_test["label"]
    x = df_tot[vars_list]
    x_test = df_tot_test[vars_list]
    w = df_tot["evt_wgt"]

    y = y.to_numpy()
    x = x.to_numpy()
    w = w.to_numpy()
    y_test = y_test.to_numpy()
    x_test = x_test.to_numpy()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    w_tensor = torch.tensor(w, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    if purpose == 'test':
        return x_tensor, y_tensor, x_test_tensor, y_test_tensor
    dataset = TensorDataset(x_tensor, y_tensor, w_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return train_loader, x_test_tensor, y_test_tensor
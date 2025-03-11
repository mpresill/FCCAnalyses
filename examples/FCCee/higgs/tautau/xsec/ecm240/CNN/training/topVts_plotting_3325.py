import sys, os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import xgboost as xgb
import joblib
from ROOT import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from model import CNN_Model
from model import DNN
from training_variables import *
import data_prepper
from event_wgts import *

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

from topVts_config import *
model_struct = 'DNN'
def run(vars, cat, doBDT):
    vars_list = training_vars[cat]
    cat_sel = sig_filter[cat]
    liste = vars_list + [cat_sel]
    samples_sig = []
    samples_bkg = []
    # Load trained model
    if model_struct == 'CNN':
        cnn = CNN_Model(num_training_vars[cat])
        cnn.load_state_dict(torch.load(f"/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_{model_struct+cat}.pt"))
    else:
        cnn = DNN(num_training_vars[cat])
        cnn.load_state_dict(torch.load('/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+ model_struct+cat + '_5325.pt'))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/stage2_R5_for_training_20250302/'
    vars_list = training_vars[cat]

    cat_sel = sig_filter[cat]


    df_sig = {}
    df_bkg = {}


    for sig in sample_list['training']['sig']:
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{sig}:  0")
                continue
            samples_sig.append(sig)
            events = file['events']
            temp_df = events.arrays(expressions=liste, library="pd").query(cat_sel)
            temp_df["label"] = 1
            temp_df["evt_wgt"] = 1 #evt_wgt['sigs'][sig]
        df_sig[sig] = temp_df
        x = df_sig[sig][vars_list].to_numpy()
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.nelement()==0:
            df_sig[sig]["CNN"] = []
            continue
        pred = cnn(x_tensor)
        pred = pred.detach().numpy()
        df_sig[sig]["CNN"] = pred
        print (f"{sig}: raw number {len(temp_df)}")


    for bkg in sample_list['training']['bkg']:
        with uproot.open(f'{path}{cat}/{bkg}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{bkg}:  0")
                continue
            samples_bkg.append(bkg)
            events = file['events']
            temp_df = events.arrays(expressions=liste, library="pd").query(cat_sel)
            temp_df["label"] = 0
            temp_df["evt_wgt"] = 1 #evt_wgt['bkgs'][bkg]
        df_bkg[bkg] = temp_df
        x = df_bkg[bkg][vars_list].to_numpy()
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.nelement()==0:
            df_bkg[bkg]["CNN"] = []
            continue
        pred = cnn(x_tensor)
        print(pred)
        pred = pred.detach().numpy()
        df_bkg[bkg]["CNN"] = pred

        print (f"{bkg}: raw number {len(temp_df)}")


    df_tot_sig = pd.DataFrame()
    df_tot_bkg = pd.DataFrame()
    for s in samples_sig:
        df_tot_sig = pd.concat([df_tot_sig, df_sig[s]])
    for b in samples_bkg:
        df_tot_bkg = pd.concat([df_tot_bkg, df_bkg[b]])

    plot_vars = training_vars[cat]
    plot_vars.append("CNN")
    binning = {}
    for pv in plot_vars:
        binning[pv] = {}
        binning[pv]['max'] = 0
        binning[pv]['min'] = 0
        for s in samples_sig:
            if df_sig[s][pv].max() > binning[pv]['max']:
                binning[pv]['max'] = df_sig[s][pv].max()
            if df_sig[s][pv].min() < binning[pv]['min']:
                binning[pv]['min'] = df_sig[s][pv].min()
        for b in samples_bkg:
            if df_bkg[b][pv].max() > binning[pv]['max']:
                binning[pv]['max'] = df_bkg[b][pv].max()
            if df_bkg[b][pv].min() < binning[pv]['min']:
                binning[pv]['min'] = df_bkg[b][pv].min()


    for pv in plot_vars:
        plt.figure(figsize=(10, 6))
        plt.hist(df_tot_sig[pv], bins = np.linspace(binning[pv]['min'],binning[pv]['max'],200), density = True, alpha = 0.5, label = s, color = 'g', stacked = True, histtype = 'barstacked')
        plt.hist(df_tot_bkg[pv], bins = np.linspace(binning[pv]['min'],binning[pv]['max'],200), density = True, alpha = 0.5, label = b, color = 'r', stacked = True, histtype = 'barstacked')
    
    # Hinzufügen von Labels und Titel
        plt.title(f'{pv}')
        plt.xlabel(f'{pv}')
        plt.yscale('log')
        #plt.legend()
        plt.savefig(f'/web/awiedl/public_html/ML/DNN/{cat}/{cat}_{pv}.pdf')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot xgb model for Bc -> tau nu vs. Z -> qq, cc, bb')
    parser.add_argument("--Vars", choices=["normal","vtx"],required=False,help="Event-level vars (normal) or added vertex vars (vtx)",default="vtx")
    parser.add_argument("--Channel", choices=['dilep_0tau', 'dilep_1tau', 'dilep_2tau', 'semilep_0tau_ud', 'semilep_0tau_cs','semilep_1tau_ud', 'semilep_1tau_cs',
        'dihad_ud_only', 'dihad_udcs', 'dihad_cs_only'],required=False,help="Which event category to train",default='dilep_1tau')
    parser.add_argument("--doBDT", choices=[True, False],required=False,help="Whether to use BDT",default=True)
    args = parser.parse_args()

    run(args.Vars, args.Channel, args.doBDT)

if __name__ == '__main__':
    main()

import sys, os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
#import xgboost as xgb
#import joblib
import ROOT
import torch
from sklearn.model_selection import train_test_split
#from torch.utils.data import DataLoader, TensorDataset
#import torch.optim as optim
#import torch.nn as nn
from model import CNN_Model
from model import DNN
from training_variables import *
#import data_prepper
from event_wgts import *

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

#from topVts_config import *
model_struct = 'DNN'

def run(vars, cat, doBDT):
    vars_list = training_vars[cat]
    cat_sel = sig_filter[cat]
    liste = vars_list + [cat_sel] + ['Is_dihad_ud_only','Is_dihad_cs_only','Is_dihad_udcs','Is_dihad_CKMmix','Is_dilep_0tau','Is_dilep_1tau','Is_dilep_2tau','Is_semilep_1tau_cs','Is_semilep_1tau_cs','Is_semilep_0tau_ud','Is_semilep_0tau_ud']
    samples_sig = []
    samples_bkg = []
    # Load trained model
    if model_struct == 'CNN':
        cnn = CNN_Model(num_training_vars[cat])
        cnn.load_state_dict(torch.load(f"/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_{model_struct+cat}.pt"))
    else:
        cnn = DNN(num_training_vars[cat])
        cnn.load_state_dict(torch.load('/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+ model_struct+cat + '_17325.pt'))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/stage2_R5_for_training_20250317/'
    vars_list = training_vars[cat]

    cat_sel = sig_filter[cat]


    df_sig = {}
    df_bkg = {}
    samples_sig = []
    samples_bkg = []

    ##CHANGE0318
    ##pop tallTlep signal
    if cat in ['dilep_1tau', 'dilep_2tau']:
        sample_list['fitting']['sig'].remove('wzp6_ee_SM_tt_tWbTWs_tallTlep_ecm365')

    for sig in sample_list['fitting']['sig']:
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{sig}:  0")
                continue
            samples_sig.append(sig)
            events = file['events']
            temp_df = events.arrays(library="pd") #.query(f'{cat_sel} == 1')
            temp_df["label"] = 1
            ##CHANGE0318
            ##double tlepTall weight
            if cat in ['dilep_1tau', 'dilep_2tau'] and sig == 'wzp6_ee_SM_tt_tWsTWb_tlepTall_ecm365':
                temp_df["evt_wgt"] = evt_wgt['sigs'][sig]*4
            else:
                temp_df["evt_wgt"] = evt_wgt['sigs'][sig]*2
            if len(temp_df) <=2:
                samples_sig.remove(sig)
        df_sig[sig] = temp_df
        x = df_sig[sig][vars_list].to_numpy()
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.nelement()==0:
            df_sig[sig]["DNN"] = []
            continue
        pred = cnn(x_tensor)
        pred = pred.detach().numpy()
        df_sig[sig]["DNN"] = pred
        print (f"{sig}: raw number {len(temp_df)}")


    for bkg in sample_list['fitting']['bkg']:
        with uproot.open(f'{path}{cat}/{bkg}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{bkg}:  0")
                continue
            samples_bkg.append(bkg)
            events = file['events']
            temp_df = events.arrays(library="pd")
            temp_df["label"] = 0
            temp_df["evt_wgt"] = evt_wgt['bkgs'][bkg]
            if len(temp_df) <=2:
                samples_bkg.remove(bkg)
        df_bkg[bkg] = temp_df
        x = df_bkg[bkg][vars_list].to_numpy()
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.nelement()==0:
            df_bkg[bkg]["DNN"] = []
            continue
        pred = cnn(x_tensor)
        pred = pred.detach().numpy()
        df_bkg[bkg]["DNN"] = pred
        print (f"{bkg}: raw number {len(temp_df)}")


    procs = [cat, 'dihad', 'dilep', 'semilep_ud', 'semilep_cs', 'others']
    procs_label = {cat: f'{cat} Sig', 
                    'dihad':'dihad Bkg', 
                    'dilep':'dilep Bkg', 
                    'semilep_ud':f'semilep_ud Bkg', 
                    'semilep_cs':'semilep_cs Bkg', 
                    'others':'other Bkgs'}
    tt = ['dihad', 'dilep', 'semilep_ud', 'semilep_cs']

    modes_cut = {
        'dihad': 'Is_dihad_ud_only==1 or Is_dihad_cs_only ==1 or Is_dihad_udcs==1 or Is_dihad_CKMmix==1',
        'dilep': 'Is_dilep_0tau == 1 or Is_dilep_1tau == 1 or Is_dilep_2tau == 1',
        'semilep_cs': 'Is_semilep_0tau_cs==1 or Is_semilep_1tau_cs==1',
        'semilep_ud': 'Is_semilep_0tau_ud==1 or Is_semilep_1tau_ud==1'
    }

    df_train = {}
    df_test = {}
    for m in procs:
        df_train[m] = pd.DataFrame()
        df_test[m] = pd.DataFrame()

    for s in samples_sig:
        train, test = train_test_split(df_sig[s], train_size=0.5, test_size=0.5, random_state=12)
        df_train[cat] = pd.concat([df_train[cat], train])
        df_test[cat] = pd.concat([df_test[cat], test])
    df_train[cat] = df_train[cat].query(f'{sig_filter[cat]} == 1')
    df_test[cat] = df_test[cat].query(f'{sig_filter[cat]} == 1')

    for b in samples_bkg:
        train, test = train_test_split(df_bkg[b], train_size=0.7, test_size=0.3, random_state=12)
        if b in ttbar:
            for t in tt:
                df_train[t] = pd.concat([df_train[t],pd.DataFrame(train).query(modes_cut[t])]) 
                df_test[t] = pd.concat([df_test[t],pd.DataFrame(test).query(modes_cut[t])]) 
        else:
            df_train['others'] = pd.concat([df_train['others'], train])
            df_test['others'] = pd.concat([df_test['others'], test])

    N_train = {}
    N_test = {}
    modes = []
    for m in procs:
        if m == 'others':
            N_train[m] = df_train[m]['evt_wgt'].sum() 
            N_test[m] = df_test[m]['evt_wgt'].sum()           
        else:
            N_train[m] = len(df_train[m])
            N_test[m] = len(df_test[m])
        if N_train[m] < 100 or N_test[m] < 100:
            continue
        else:
            modes.append(m)

    print(N_train)
    print(N_test)
    ##CHANGE0318
    ##bkg colors
    modes_color = {
        cat: 'black',
        'dihad':'gray',
        'dilep':'olive',
        'semilep_cs':'brown',
        'semilep_ud':'red',
        'others':'cyan',
    }

    eff_train = {}
    eff_test  = {}
    DNN_cuts = np.linspace(0.05,5.05,500)
    cut_vals = []
    for m in modes:
      eff_train[m] = []
      eff_test [m] = []
    for x in DNN_cuts:
      cut_val = float(x)
      cut_vals.append(cut_val)
      cut_val = 1 - pow(10, -cut_val)
      for m in modes:
        if m  == 'others':
            eff_train[m].append( max( 1e-3, float(df_train[m].query("DNN > %s" % cut_val)['evt_wgt'].sum())) / N_train[m] )
            eff_test [m].append( max( 1e-3, float(df_test [m].query("DNN > %s" % cut_val)['evt_wgt'].sum())) / N_test [m] )
        else:
            eff_train[m].append( max( 1e-3, float(len(df_train[m].query("DNN > %s" % cut_val)))) / N_train[m] )
            eff_test [m].append( max( 1e-3, float(len(df_test [m].query("DNN > %s" % cut_val)))) / N_test [m] )


    fig, ax = plt.subplots(figsize=(12,8))

    for m in modes:
        plt.plot(cut_vals, eff_train[m], color=modes_color[m], label=f'Train {procs_label[m]}')
    for m in modes:
        plt.plot(cut_vals, eff_test [m], color=modes_color[m], label=f'Test {procs_label[m]}', linestyle='dashed')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title( FCC_label, loc='right', fontsize=20)
    plt.xlim(0,2.1)
    plt.xlabel("1 - DNN score",fontsize=30)
    plt.ylabel("Efficiency",fontsize=30)
    plt.xticks([0, 1, 2], ["$10^0$", "$10^{-1}$", "$10^{-2}$"])
    plt.yscale('log')
    ymin,ymax = plt.ylim()
    plt.ylim(10e-4,2)
    plt.legend(fontsize=18, loc="lower left", ncol=2)
    plt.grid(alpha=0.4,which="both")
    plt.tight_layout()
    fig.savefig(f"/web/awiedl/public_html/ML/DNN/overtraining/DNN_{cat}_overtrain_eff_19325.pdf")

def main():
    parser = argparse.ArgumentParser(description='Plot xgb model for Bc -> tau nu vs. Z -> qq, cc, bb')
    parser.add_argument("--Vars", choices=["normal","vtx"],required=False,help="Event-level vars (normal) or added vertex vars (vtx)",default="vtx")
    parser.add_argument("--Channel", choices=['dilep_0tau', 'dilep_1tau', 'dilep_2tau', 'semilep_0tau_ud', 'semilep_0tau_cs','semilep_1tau_ud', 'semilep_1tau_cs',
        'dihad_ud_only', 'dihad_udcs', 'dihad_cs_only'],required=False,help="Which event category to train",default='dihad_udcs')
    parser.add_argument("--doBDT", choices=[True, False],required=False,help="Whether to use BDT",default=True)
    args = parser.parse_args()

    run(args.Vars, args.Channel, args.doBDT)

if __name__ == '__main__':
    main()

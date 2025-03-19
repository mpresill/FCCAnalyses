import sys, os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import xgboost as xgb
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from model import CNN_Model
from model import DNN

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

from topVts_config import *
model_struct = 'DNN'
def run(vars, cat):

    # Load trained model
    cnn = DNN(num_training_vars[cat])
    cnn.load_state_dict(torch.load(f'/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+ model_struct + cat + '_14325.pt'))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/training_samples/'
    vars_list = training_vars[sig]
    cat_sel = sig_filter[cat]

    #Load samples
    processes = {"sig_"+sig:    [sig,       "p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU", "#b2182b", 'WbWs ' + sig],
                 "bkg_dilep":   ["dilep",   "p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU", "#3e9682", 'WbWb dilep'],
                 "bkg_semilep": ["semilep", "p8_ee_Zuds_ecm91",                        "#77bfa9", "WbWb semilep"],
                 #"bkg_dihad":   ["dihad",   "p8_ee_Zcc_ecm91",                         "#bce6d8", "WbWb dihad"],
                 "bkg_WW":      ["WW",      "p8_ee_Zbb_ecm91",                         "#2166ac", "WW"],
                 "bkg_ZZ":      ["ZZ",      "p8_ee_Zbb_ecm91",                         "#92c5de", "ZZ"],
                 "bkg_ZH":      ["ZH",      "p8_ee_Zbb_ecm91",                         "#d1e5f0", "ZH"]}
    df_train = {}
    df_test = {}
    N_train = {}
    N_test  = {}

    path_train_sig = f"{path}sigs_{pkl_version}"
    path_train_bkg = f"{path}bkgs_{pkl_version}"

    empty_procs = []
    for proc in processes:
      # If using pkl from process_sig_bkg_samples_for_xgb.py
      path_train = path_train_bkg
      if "sig" in proc :
          path_train = path_train_sig
      df_train[proc] = pd.read_pickle(f"{path_train}/{processes[proc][0]}_train.pkl")
      df_train[proc] = df_train[proc].query(cat_sel[sig])
      #df_train[proc]["label"] = 0
      df_train[proc] = df_train[proc][vars_list]
      #print(df_train[proc])
      x = df_train[proc].to_numpy()
      if model_struct == 'CNN':
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
      else:
        x_tensor = torch.tensor(x, dtype=torch.float32) 
      if x_tensor.nelement()==0:
        df_train[proc]["CNN"] = []
        continue
      pred = cnn(x_tensor)
      pred = pred.detach().numpy()
      df_train[proc]["CNN"] = pred
      N_train [proc] = len(df_train[proc])

      df_test[proc] = pd.read_pickle(f"{path_train}/{processes[proc][0]}_test.pkl")
      df_test[proc] = df_test[proc].query(cat_sel[sig])
#      df_test[proc]["label"] = 0
      df_test[proc] = df_test[proc][vars_list]
      x = df_test[proc].to_numpy()
      if model_struct == 'CNN':
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
      else:
        x_tensor = torch.tensor(x, dtype=torch.float32)
      if x_tensor.nelement()==0:
        df_test[proc]["CNN"] = []
        continue
      pred = cnn(x_tensor)
      pred = pred.detach().numpy()
      df_test[proc]["CNN"] = pred
      N_test [proc] = len(df_test[proc])

      print (f'{proc}:  train: {N_train[proc]},  test: {N_test[proc]} to {len(df_test[proc])}')
      if N_train[proc] < 100 or N_test[proc] < 100: empty_procs.append(proc)

    for proc in empty_procs:
        processes.pop(proc)

    #Plot efficiency as a function of BDT cut in each sample
    eff_train = {}
    eff_test  = {}
    BDT_cuts = np.linspace(0.05,5.05,500)
    cut_vals = []
    for proc in processes:
      eff_train[proc] = []
      eff_test [proc] = []
    for x in BDT_cuts:
      cut_val = float(x)
      cut_vals.append(cut_val)
      cut_val = 1 - pow(10, -cut_val)
      for proc in processes:
        eff_train[proc].append( max( 1e-3, float(len(df_train[proc].query("CNN > %s" % cut_val)))) / N_train[proc] )
        eff_test [proc].append( max( 1e-3, float(len(df_test [proc].query("CNN > %s" % cut_val)))) / N_test [proc] )


    fig, ax = plt.subplots(figsize=(12,8))

    for proc in processes:
      plt.plot(cut_vals, eff_train[proc], color=processes[proc][2], label=f'Train {processes[proc][3]}')
    for proc in processes:
      plt.plot(cut_vals, eff_test [proc], color=processes[proc][2], label=f'Test {processes[proc][3]}', linestyle='dashed')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title( FCC_label, loc='right', fontsize=20)
    plt.xlim(0,4.1)
    plt.xlabel("1 - CNN score",fontsize=30)
    plt.ylabel("Efficiency",fontsize=30)
    plt.xticks([0, 1, 2, 3, 4], ["$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$"])
    plt.yscale('log')
    ymin,ymax = plt.ylim()
    plt.ylim(10e-6,2)
    plt.legend(fontsize=18, loc="lower left", ncol=2)
    plt.grid(alpha=0.4,which="both")
    plt.tight_layout()
    fig.savefig(f"/web/awiedl/public_html/ML/DNN/DNN_{sig}_{bdt_suffix}_overtrain_eff.pdf")


#    for var in vars_list:
#        fig, ax = plt.subplots(figsize=(12,8))
#        xmin = min( df_train["sig_"+sig][var].quantile(0.01), df_train["bkg_dilep"][var].quantile(0.01), df_train["bkg_semilep"][var].quantile(0.01) )
#        xmax = max( df_train["sig_"+sig][var].quantile(0.99), df_train["bkg_dilep"][var].quantile(0.99), df_train["bkg_semilep"][var].quantile(0.99) )
#
#        for proc in processes:
#            plt.hist(df_train[proc][var], bins=binnings[var][0],density=True,color=processes[proc][2],histtype='step',linewidth=1.5,label=f'Train {processes[proc][3]}', fill=(proc == "sig_"+sig),  alpha=0.3 if (proc=="sig_"+sig) else 1.0)
#        for proc in processes:
#            plt.hist(df_test[proc][var],  bins=binnings[var][0],density=True,color=processes[proc][2],histtype='step',linewidth=2.5,label=f'Test {processes[proc][3]}', linestyle='dashed')
#
#        ax.tick_params(axis='both', which='major', labelsize=20)
#        ax.set_title( FCC_label, loc='right', fontsize=20)
#
#        plt.xlabel(binnings[var][1],fontsize=30)
#        plt.ylabel("Normalised yield (a.u.)",fontsize=30)
#        plt.xlim(min(binnings[var][0]), max(binnings[var][0]))
#        ymin,ymax = plt.ylim()
#        plt.ylim(ymin,1.5*ymax)
#        plt.legend(fontsize=18, loc="upper right", ncol=2)
#        plt.tight_layout()
#        fig.savefig(f"{path}/figs/BDT/{pkl_version}/{sig}_vars/{var}.png")
#        fig.savefig(f"{path}/figs/BDT/{pkl_version}/{sig}_vars/{var}.pdf")
#        plt.close(fig)




def main():
    parser = argparse.ArgumentParser(description='Plot xgb model for Bc -> tau nu vs. Z -> qq, cc, bb')
    parser.add_argument("--Vars", choices=["normal","vtx"],required=False,help="Event-level vars (normal) or added vertex vars (vtx)",default="vtx")
    parser.add_argument("--Channel", choices=["dilep", "semilep_heavy", "semilep_light"],required=False,help="Which event category to train",default="semilep_light")
    args = parser.parse_args()
    run(args.Vars, args.Channel)

if __name__ == '__main__':
    main()

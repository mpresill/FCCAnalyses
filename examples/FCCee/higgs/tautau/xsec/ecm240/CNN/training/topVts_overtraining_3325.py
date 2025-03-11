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
from training_variables import *
import data_prepper
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

from topVts_config import *
model_struct = 'DNN'
def run(vars, sig):

    # Load trained model
    if model_struct == 'CNN':
      cnn = CNN_Model(num_training_vars[sig])
      cnn.load_state_dict(torch.load(f"/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_{sig}.pt"))
    else:
      cnn = DNN(num_training_vars[sig])
      cnn.load_state_dict(torch.load('/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+ model_struct+sig + '3325.pt'))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/stage2_R5_for_training_20250302/'

    df_train = {}
    df_test = {}
    N_train = {}
    N_test  = {}

    x, y, x_test, y_test = data_prepper.topVts_data_prepper_DNN(path, sig, 'test')

    pred = cnn(x)
    pred = pred.detach().numpy()
    df_train["CNN"] = pred
    N_train = len(x)

    pred = cnn(x_test)
    pred = pred.detach().numpy()
    df_test["CNN"] = pred
    N_test = len(x_test)

    print (f'train: {N_train},  test: {N_test}')

    #Plot efficiency as a function of BDT cut in each sample
    eff_train = []
    eff_test  = []
    BDT_cuts = np.linspace(0.05,5.05,500)
    cut_vals = []

    for x in BDT_cuts:
      cut_val = float(x)
      cut_vals.append(cut_val)
      cut_val = 1 - pow(10, -cut_val)
      eff_train.append( max( 1e-3, float(len(list(filter(lambda j: j>cut_val,df_train['CNN'])))))/ N_train)
      eff_test.append( max( 1e-3, float(len(list(filter(lambda j: j>cut_val,df_test['CNN'])))))/ N_test)
      #eff_train.append( max( 1e-3, float(len(df_train.query("CNN > %s" % cut_val)))) / N_train)
      #eff_test.append( max( 1e-3, float(len(df_test.query("CNN > %s" % cut_val)))) / N_test)


    fig, ax = plt.subplots(figsize=(12,8))


    plt.plot(cut_vals, eff_train, label='Train')

    plt.plot(cut_vals, eff_test, label=f'Test', linestyle='dashed')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title( FCC_label, loc='right', fontsize=20)
    plt.xlim(0,4.1)
    plt.xlabel("1 - DNN score",fontsize=30)
    plt.ylabel("Efficiency",fontsize=30)
    plt.xticks([0, 1, 2, 3, 4], ["$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$"])
    plt.yscale('log')
    ymin,ymax = plt.ylim()
    plt.ylim(10e-6,2)
    plt.legend(fontsize=18, loc="lower left", ncol=2)
    plt.grid(alpha=0.4,which="both")
    plt.tight_layout()
    fig.savefig(f"/web/awiedl/public_html/ML/DNN/DNN_{sig}_overtrain_eff.pdf")


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
    parser.add_argument("--Channel", choices=['dilep_0tau', 'dilep_1tau', 'dilep_2tau', 'semilep_0tau_ud', 'semilep_0tau_cs','semilep_1tau_ud', 'semilep_1tau_cs','dihad_ud_only', 'dihad_udcs', 'dihad_cs_only'],required=False,help="Which event category to train",default='dilep_1tau')
    args = parser.parse_args()
    run(args.Vars, args.Channel)

if __name__ == '__main__':
    main()

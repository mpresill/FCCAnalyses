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
        cnn.load_state_dict(torch.load('/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_'+ model_struct+cat + '_9325.pt'))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/stage2_R5_for_training_20250302/'
    vars_list = training_vars[cat]

    cat_sel = sig_filter[cat]


    df_sig = {}
    df_bkg = {}
    samples_sig = []
    samples_bkg = []

    for sig in sample_list['training']['sig']:
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{sig}:  0")
                continue
            samples_sig.append(sig)
            events = file['events']
            temp_df = events.arrays(expressions=liste, library="pd").query(f'{cat_sel} == 1')
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
            temp_df = events.arrays(expressions=liste, library="pd")
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
        pred = pred.detach().numpy()
        df_bkg[bkg]["CNN"] = pred
        print(pred)
        print (f"{bkg}: raw number {len(temp_df)}")

    plot_vars = training_vars[cat]
    plot_vars.append('CNN')
    binnings = {}
    for pv in plot_vars:
        binnings[pv] = {}
        binnings[pv]['max'] = 0
        binnings[pv]['min'] = 0
        for s in samples_sig:
            if df_sig[s][pv].max() > binnings[pv]['max']:
                binnings[pv]['max'] = df_sig[s][pv].max()
            if df_sig[s][pv].min() < binnings[pv]['min']:
                binnings[pv]['min'] = df_sig[s][pv].min()
        for b in samples_bkg:
            if df_bkg[b][pv].max() > binnings[pv]['max']:
                binnings[pv]['max'] = df_bkg[b][pv].max()
            if df_bkg[b][pv].min() < binnings[pv]['min']:
                binnings[pv]['min'] = df_bkg[b][pv].min()

    h_sig_all = {}
    h_sig_true = {}
    h_bkg = {}

    eff_sig_true = {}
    bin_edges = {}

    for sig in samples_sig:
        N_sel = len(df_sig[sig]) + 1e-3
        sig_label = "label"
        N_true = len(df_sig[sig].query(f"{sig_label}==1"))
        df_true = df_sig[sig].query(f"{sig_label}==1")
        for pv in plot_vars:
            h_sig_all[f'{cat}_{sig}_{pv}'], bin_edges[pv] = np.histogram(df_sig[sig][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],100), weights=df_sig[sig]['evt_wgt'])
            h_sig_true[f'{cat}_{sig}_{pv}'], bin_edges[pv] = np.histogram(df_true[pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],100), weights=df_true['evt_wgt'])
        print(f'{cat} of {sig}')
        print(N_sel)
        print(N_true / N_sel)
        eff_sig_true[f'{cat}_{sig}'] = N_true / N_sel

    for bkg in samples_bkg:
        for pv in plot_vars:
            h_bkg[f'{cat}_{bkg}_{pv}'], bin_edges[pv] = np.histogram(df_bkg[bkg][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],100), weights=df_bkg[bkg]['evt_wgt'])


    fig_dir = f'/web/awiedl/public_html/ML/DNN/topVts/yields_7325_R5_CNN_{cat}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for pv in plot_vars:
        fig, ax = plt.subplots(figsize=(8,8))
        for sig in samples_sig:
            plt.stairs(h_sig_all [f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv],  linewidth=4, linestyle='dashed', label=f'{sig} no match') #color=colors['sig'][sig],
            plt.stairs(h_sig_true[f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv],  linewidth=2, linestyle='solid', label=sig+' match %.2f'%eff_sig_true[f'{cat}_{sig}']) #color=colors['sig'][sig],
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_title( FCC_label, loc='right', fontsize=20)
        plt.xlabel(pv,fontsize=30)
        plt.ylabel('count',fontsize=30)
        #plt.yscale('log')
        ymin,ymax = plt.ylim()
#            if 'energy' in pv or 'mass' in pv or 'dR' in pv:
#                plt.yscale('log')
#                plt.ylim(0,10*ymax)
#            else:
        plt.ylim(0,1.4*ymax)

        plt.legend(fontsize=5, loc="upper right", ncol=2)
        plt.grid(alpha=0.4,which="both")
        ax.annotate(f'{cat} category', xy=(0.45,0.75), xycoords='axes fraction', fontsize=20)
        plt.tight_layout()
        save_name = f'sig_eff_{cat}_{pv}'
        fig.savefig(f"{fig_dir}/{save_name}.pdf")
        fig.savefig(f"{fig_dir}/{save_name}.png")
        plt.close()



    for pv in plot_vars:
        fig, ax = plt.subplots(figsize=(8,8))
        bkg_plot = h_bkg[f'{cat}_p8_ee_WW_ecm365_{pv}'] 
        for bkg in samples_bkg:
            if('WW' in bkg):
                continue
            bkg_plot += h_bkg[f'{cat}_{bkg}_{pv}']
        #print(bkg_plot)
        for bkg in samples_bkg:
            plot_label = f'WbWb {bkg}'
            if bkg in ["WW", "ZZ", "ZH"]: plot_label = bkg
            plt.stairs(bkg_plot, edges=bin_edges[pv],  fill=True,  alpha=0.8, label=plot_label) #
            bkg_plot = bkg_plot - h_bkg[f'{cat}_{bkg}_{pv}']
        for sig in samples_sig:
            ### could use only matched sig, or all sigs.
            ### without refined selection, the semilep cs and full had match rate is low
            ### use full sig for this round
            plt.stairs(h_sig_all[f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv],  linewidth=2, linestyle='solid', label=f'WsWb {sig}') #color=colors['sig'][sig],
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_title( FCC_label, loc='right', fontsize=20)
        plt.xlabel(pv,fontsize=30)
        plt.ylabel('count',fontsize=30)
        #plt.yscale('log')
        ymin,ymax = plt.ylim()
        if 'energy' in pv or 'mass' in pv or 'dR' in pv or 'CNN' in pv:
            plt.yscale('log')
            plt.ylim(0,10*ymax)
        else:
            plt.ylim(0,1.4*ymax)

        plt.legend(fontsize=5, loc="upper right", ncol=2)
        plt.grid(alpha=0.4,which="both")
        ax.annotate(f'{cat} category', xy=(0.45,0.75), xycoords='axes fraction', fontsize=20)
        plt.tight_layout()
        save_name = f'event_yield_{cat}_{pv}'
        fig.savefig(f"{fig_dir}/{save_name}.pdf")
        fig.savefig(f"{fig_dir}/{save_name}.png")
        plt.close()

    bkg_root = TFile( fig_dir + "/bkg_hists.root" , "RECREATE")
    sig_root = TFile( fig_dir + "/sig_hists.root" , "RECREATE")
    root_hists = {}

    for pv in plot_vars:
        bkg_root.cd()
        for bkg in samples_bkg:
            save_name = f"bkg_{cat}_{bkg}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 100, binnings[pv]['min'], binnings[pv]['max'])
            for ibin in range(99):
                root_hists[save_name].SetBinContent(ibin, h_bkg[f'{cat}_{bkg}_{pv}'][ibin] )
            root_hists[save_name].Write()
        sig_root.cd()
        for sig in samples_sig:
            save_name = f"sig_{cat}_{sig}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 100, binnings[pv]['min'], binnings[pv]['max'])
            for ibin in range(99):
                root_hists[save_name].SetBinContent(ibin, h_sig_true[f'{cat}_{sig}_{pv}'][ibin] )
            root_hists[save_name].Write()
    bkg_root.Close()
    sig_root.Close()



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

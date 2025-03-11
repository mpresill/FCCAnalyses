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
    liste = vars_list + [cat_sel] + ['Is_dihad_ud_only','Is_dihad_cs_only','Is_dihad_udcs','Is_dihad_CKMmix','Is_dilep_0tau','Is_dilep_1tau','Is_dilep_2tau','Is_semilep_1tau_cs','Is_semilep_1tau_cs','Is_semilep_0tau_ud','Is_semilep_0tau_ud']
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

    for sig in sample_list['fitting']['sig']:
        with uproot.open(f'{path}{cat}/{sig}/chunk0.root') as file:
            if file.keys() == ['eventsProcessed;1']:
                print (f"{sig}:  0")
                continue
            samples_sig.append(sig)
            events = file['events']
            temp_df = events.arrays(library="pd") #.query(f'{cat_sel} == 1')
            temp_df["label"] = 1
            temp_df["evt_wgt"] = evt_wgt['sigs'][sig]
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
        print (f"{bkg}: raw number {len(temp_df)}")


    df_true_sig = {}
    #df_false_sig = {}
    df_merged_bkg = {}

    modes = ['dihad', 'dilep', 'semilep_ud', 'semilep_cs', 'higgs', 'WW', 'ZZ']
    tt = ['dihad', 'dilep', 'semilep_ud', 'semilep_cs']

    modes_cut = {
        'dihad': 'Is_dihad_ud_only==1 or Is_dihad_cs_only ==1 or Is_dihad_udcs==1 or Is_dihad_CKMmix==1',
        'dilep': 'Is_dilep_0tau == 1 or Is_dilep_1tau == 1 or Is_dilep_2tau == 1',
        'semilep_cs': 'Is_semilep_1tau_cs==1 or Is_semilep_1tau_cs==1',
        'semilep_ud': 'Is_semilep_0tau_ud==1 or Is_semilep_0tau_ud==1'
    }

    for c in cats:
        df_true_sig[c] = pd.DataFrame()
        for s in samples_sig:
            df_true_sig[c] = pd.concat([df_true_sig[c],df_sig[s].query(f'{sig_filter[c]} == 1')])
            #df_false_sig[s] = df_sig[s].query(f'{cat_sel} == 0')

    df_merged_bkg['dihad'] = pd.DataFrame()
    df_merged_bkg['dilep'] = pd.DataFrame()
    df_merged_bkg['semilep_ud'] = pd.DataFrame()
    df_merged_bkg['semilep_cs'] = pd.DataFrame()
    df_merged_bkg['higgs'] = pd.DataFrame()
    df_merged_bkg['WW'] = df_bkg['p8_ee_WW_ecm365']
    df_merged_bkg['ZZ'] = df_bkg['p8_ee_ZZ_ecm365']

    for b in samples_bkg:
        if(b == 'wzp6_ee_bbH_ecm365' or b=='wzp6_ee_ccH_ecm365' or b=='wzp6_ee_ssH_ecm365' or b=='wzp6_ee_qqH_ecm365' or b=='wzp6_ee_tautauH_ecm365' or b=='wzp6_ee_mumuH_ecm365' or b=='wzp6_ee_eeH_ecm365' or b=='wzp6_ee_nunuH_ecm365'):
            df_merged_bkg['higgs'] = pd.concat([df_merged_bkg['higgs'], df_bkg[b]])
        #elif(b == 'wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ecm365' or b == 'wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365' or b == 'wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365' or b == 'wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365'):
        else:
            for t in tt:
                df_merged_bkg[t] = pd.concat([df_merged_bkg[t],df_bkg[b].query(modes_cut[t])]) 



    plot_vars = training_vars[cat]
    plot_vars.append('CNN')
    binnings = {}
    for pv in plot_vars:
        binnings[pv] = {}
        binnings[pv]['max'] = 0
        binnings[pv]['min'] = 0
        for c in cats:
            if df_true_sig[c][pv].max() > binnings[pv]['max']:
                binnings[pv]['max'] = df_true_sig[c][pv].max()
            if df_true_sig[c][pv].min() < binnings[pv]['min']:
                binnings[pv]['min'] = df_true_sig[c][pv].min()

        for m in modes:
            if df_merged_bkg[m][pv].max() > binnings[pv]['max']:
                binnings[pv]['max'] = df_merged_bkg[m][pv].max()
            if df_merged_bkg[m][pv].min() < binnings[pv]['min']:
                binnings[pv]['min'] = df_merged_bkg[m][pv].min()

    h_sig_false = {}
    h_sig_true = {}
    h_bkg = {}

    eff_sig_true = {}
    bin_edges = {}

    for c in cats:
        for pv in plot_vars:
            h_sig_true[f'{cat}_{c}_{pv}'], bin_edges[pv] = np.histogram(df_true_sig[c][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],100), weights=df_true_sig[c]['evt_wgt'])
        
    for m in modes:
        for pv in plot_vars:
            h_bkg[f'{cat}_{m}_{pv}'], bin_edges[pv] = np.histogram(df_merged_bkg[m][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],100), weights=df_merged_bkg[m]['evt_wgt'])


    fig_dir = f'/web/awiedl/public_html/ML/DNN/topVts/yields_7325_R5_CNN_{cat}_10325_plot'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    '''for pv in plot_vars:
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
        plt.close()'''



    for pv in plot_vars:
        fig, ax = plt.subplots(figsize=(8,8))
        bkg_plot = h_bkg[f'{cat}_WW_{pv}']
        for m in modes:
            if(m == 'WW'):
                continue
            bkg_plot += h_bkg[f'{cat}_{m}_{pv}']
        sig_plot = h_sig_true[f'{cat}_{cats[0]}_{pv}']
        
        for c in cats:
            if(c == cat or c == cats[0]):
                continue 
            sig_plot += h_sig_true[f'{cat}_{c}_{pv}']
        #sig_plot -= h_sig_true[f'{cat}_{cat}_{pv}']

        for m in modes:
            plot_label = f'WbWb {m}'
            if m in ["WW", "ZZ"]: plot_label = m
            elif m in "higgs": plot_label = 'ZH+VBF'
            plt.stairs(bkg_plot, edges=bin_edges[pv],  fill=True,  alpha=0.8, label=plot_label) #
            bkg_plot = bkg_plot - h_bkg[f'{cat}_{m}_{pv}']

            ### could use only matched sig, or all sigs.
            ### without refined selection, the semilep cs and full had match rate is low
            ### use full sig for this round
        #print(h_sig_true[f'{cat}_{cat}_{pv}'])
        plt.stairs(h_sig_true[f'{cat}_{cat}_{pv}'] , edges=bin_edges[pv],  linewidth=2, linestyle='solid', label=f'WsWb {cat}')
        plt.stairs(sig_plot , edges=bin_edges[pv],  linewidth=2, linestyle='solid', label=f'WsWb others')
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
        for m in modes:
            save_name = f"bkg_{cat}_{m}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 100, binnings[pv]['min'], binnings[pv]['max'])
            for ibin in range(99):
                root_hists[save_name].SetBinContent(ibin, h_bkg[f'{cat}_{m}_{pv}'][ibin] )
            root_hists[save_name].Write()
        sig_root.cd()
        for c in cats:
            save_name = f"true_sig_{cat}_{c}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 100, binnings[pv]['min'], binnings[pv]['max'])
            for ibin in range(99):
                root_hists[save_name].SetBinContent(ibin, h_sig_true[f'{cat}_{c}_{pv}'][ibin] )
            root_hists[save_name].Write()
    bkg_root.Close()
    sig_root.Close()



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

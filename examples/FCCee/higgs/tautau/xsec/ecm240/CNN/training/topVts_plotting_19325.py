import sys, os, argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
#import xgboost as xgb
#import joblib
from ROOT import *
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
            train_df, temp_df = train_test_split(temp_df, train_size=0.5, test_size=0.5, random_state=12)
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


    df_true_sig = {}
    #df_false_sig = {}
    df_merged_bkg = {}

    ##CHANGE0318
    ##Rearrange bkg order
    modes = ['dihad', 'dilep', 'semilep_ud', 'semilep_cs', 'Z', 'WW', 'ZZ', 'higgs', 'WWZ']
    tt = ['dihad', 'dilep', 'semilep_ud', 'semilep_cs']

    modes_cut = {
        'dihad': 'Is_dihad_ud_only==1 or Is_dihad_cs_only ==1 or Is_dihad_udcs==1 or Is_dihad_CKMmix==1',
        'dilep': 'Is_dilep_0tau == 1 or Is_dilep_1tau == 1 or Is_dilep_2tau == 1',
        'semilep_cs': 'Is_semilep_0tau_cs==1 or Is_semilep_1tau_cs==1',
        'semilep_ud': 'Is_semilep_0tau_ud==1 or Is_semilep_1tau_ud==1'
    }


    ##CHANGE0318
    ##bkg colors
    modes_color = {
        'dihad':'gray',
        'dilep':'green',
        'semilep_cs':'brown',
        'semilep_ud':'pink',
        'higgs':'orange',
        'ZZ':'cyan',
        'WW':'olive',
        'WWZ': 'blue',
        'Z': 'purple'
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
    df_merged_bkg['Z'] = pd.DataFrame()
    df_merged_bkg['higgs'] = pd.DataFrame()
    df_merged_bkg['WW'] = df_bkg['p8_ee_WW_ecm365']
    df_merged_bkg['ZZ'] = df_bkg['p8_ee_ZZ_ecm365']
    df_merged_bkg['WWZ'] = df_bkg['wzp6_ee_WWZ_Zbb_ecm365']

    for b in samples_bkg:
        if(b == 'wzp6_ee_bbH_ecm365' or b=='wzp6_ee_ccH_ecm365' or b=='wzp6_ee_ssH_ecm365' or b=='wzp6_ee_qqH_ecm365' or b=='wzp6_ee_tautauH_ecm365' or b=='wzp6_ee_mumuH_ecm365' or b=='wzp6_ee_eeH_ecm365' or b=='wzp6_ee_nunuH_ecm365'):
            df_merged_bkg['higgs'] = pd.concat([df_merged_bkg['higgs'], df_bkg[b]])
        elif(b == 'p8_ee_Zbb_ecm365' or b == 'p8_ee_Zcc_ecm365' or b == 'p8_ee_Zss_ecm365' or b == 'p8_ee_Zqq_ecm365' or b == 'wzp6_ee_tautau_ecm365'):
            df_merged_bkg['Z'] = pd.concat([df_merged_bkg['Z'], df_bkg[b]])
        else:
            for t in tt:
                df_merged_bkg[t] = pd.concat([df_merged_bkg[t],df_bkg[b].query(modes_cut[t])]) 



    plot_vars = training_vars[cat]
    plot_vars.append('DNN')
    binnings = {}
    N = {}
    for c in cats:
        N[c] = len(df_true_sig[c])
    for m in modes:
        N[m] = len(df_merged_bkg[m])


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

    ##CHANGE0318
    ##enforce range for DNN
    binnings['DNN']['max'] = 1.0
    binnings['DNN']['min'] = 0.0

    h_sig_false = {}
    h_sig_true = {}
    h_bkg = {}
    h_sig_sumw2 = {}
    h_bkg_sumw2 = {}

    eff_sig_true = {}
    bin_edges = {}

    ## CHANGE0318 fix bin number
    for c in cats:
        for pv in plot_vars:
            h_sig_true[f'{cat}_{c}_{pv}'], bin_edges[pv] = np.histogram(df_true_sig[c][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],51), weights=df_true_sig[c]['evt_wgt'])
            ##CHANGE0318 add nonzero placeholder
            h_sig_true[f'{cat}_{c}_{pv}'] += [1.0e-8]*50
            h_sig_sumw2[f'{cat}_{c}_{pv}'], bin_edges[pv] = np.histogram(df_true_sig[c][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],51), weights=df_true_sig[c]['evt_wgt']**2)        
    for m in modes:
        for pv in plot_vars:
            h_bkg[f'{cat}_{m}_{pv}'], bin_edges[pv] = np.histogram(df_merged_bkg[m][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],51), weights=df_merged_bkg[m]['evt_wgt'])
            ##CHANGE0318 add nonzero placeholder
            h_bkg[f'{cat}_{m}_{pv}'] += [1.0e-8]*50
            h_bkg_sumw2[f'{cat}_{m}_{pv}'], bin_edges[pv] = np.histogram(df_merged_bkg[m][pv], bins=np.linspace(binnings[pv]['min'],binnings[pv]['max'],51), weights=df_merged_bkg[m]['evt_wgt']**2)

    fig_dir = f'/web/awiedl/public_html/ML/DNN/topVts/yields_7325_R5_CNN_{cat}_17325_50bin_plot'
    ##CHANGE0318, output dir
    fig_dir = f'/web/xzuo/public_html/Vts_plots/20250319/{cat}'

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
        bkg_plot = h_bkg[f'{cat}_ZZ_{pv}'] + h_bkg[f'{cat}_Z_{pv}'] + h_bkg[f'{cat}_WWZ_{pv}'] + h_bkg[f'{cat}_WW_{pv}'] + h_bkg[f'{cat}_dihad_{pv}'] + h_bkg[f'{cat}_dilep_{pv}'] + h_bkg[f'{cat}_semilep_ud_{pv}'] + h_bkg[f'{cat}_semilep_cs_{pv}'] + h_bkg[f'{cat}_higgs_{pv}']

        ##CHANGE0318 fix Nbins 49 to 50
        sig_plot = [1e-8]*50 + h_sig_true[f'{cat}_dilep_0tau_{pv}'] + h_sig_true[f'{cat}_dilep_1tau_{pv}'] + h_sig_true[f'{cat}_dilep_2tau_{pv}'] + h_sig_true[f'{cat}_semilep_0tau_ud_{pv}'] + h_sig_true[f'{cat}_semilep_1tau_ud_{pv}'] + h_sig_true[f'{cat}_semilep_0tau_cs_{pv}'] + h_sig_true[f'{cat}_semilep_1tau_cs_{pv}'] + h_sig_true[f'{cat}_dihad_ud_only_{pv}'] + h_sig_true[f'{cat}_dihad_cs_only_{pv}'] + h_sig_true[f'{cat}_dihad_udcs_{pv}'] - h_sig_true[f'{cat}_{cat}_{pv}']

        for m in modes:
            plot_label = f'WbWb {m}'
            if m in ["WW", "ZZ", "Z", "WWZ"]: plot_label = m
            ##CHANGE0318, Higgs label
            elif m in "higgs": plot_label = 'Higgs(ZH+VBF)'
            plt.stairs(bkg_plot, edges=bin_edges[pv], color = modes_color[m], fill=True,  alpha=0.8, label=plot_label) #
            bkg_plot = bkg_plot - h_bkg[f'{cat}_{m}_{pv}']

            ### could use only matched sig, or all sigs.
            ### without refined selection, the semilep cs and full had match rate is low
            ### use full sig for this round
        #print(h_sig_true[f'{cat}_{cat}_{pv}'])
        #for c in cats:
        plt.stairs(h_sig_true[f'{cat}_{cat}_{pv}'] , edges=bin_edges[pv], color = 'red',  linewidth=2, linestyle='solid', label=f'WsWb {c}')
        plt.stairs(sig_plot , edges=bin_edges[pv],  color = 'black', linewidth=2, linestyle='solid', label=f'WsWb others')
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_title( FCC_label, loc='right', fontsize=20)
        plt.xlabel(pv,fontsize=30)
        plt.ylabel('count',fontsize=30)
        #plt.yscale('log')
        ymin,ymax = plt.ylim()
        if 'energy' in pv or 'mass' in pv or 'dR' in pv or 'DNN' in pv:
            plt.yscale('log')
            ##CHANGE0318, y range
            plt.ylim(1e-2,50*ymax)
        else:
            plt.ylim(0,1.4*ymax)

        plt.legend(fontsize=10, loc="upper right", ncol=2)
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
            root_hists[save_name] = TH1F(save_name,save_name, 50, binnings[pv]['min'], binnings[pv]['max'])
            ##CHANGE0318 fix bin number
            for ibin in range(50):
                root_hists[save_name].SetBinContent(ibin, h_bkg[f'{cat}_{m}_{pv}'][ibin] )
            root_hists[save_name].Write()
        sig_root.cd()
        for c in cats:
            save_name = f"true_sig_{cat}_{c}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 50, binnings[pv]['min'], binnings[pv]['max'])
            ## CHANGE0318 fix bin number
            for ibin in range(50):
                root_hists[save_name].SetBinContent(ibin, h_sig_true[f'{cat}_{c}_{pv}'][ibin] )
            root_hists[save_name].Write()
    bkg_root.Close()
    sig_root.Close()

    bkg_root = TFile( fig_dir + "/bkg_hists_sumW2.root" , "RECREATE")
    sig_root = TFile( fig_dir + "/sig_hists_sumW2.root" , "RECREATE")
    root_hists = {}

    for pv in plot_vars:
        bkg_root.cd()
        for m in modes:
            save_name = f"bkg_{cat}_{m}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 50, binnings[pv]['min'], binnings[pv]['max'])
            ## CHANGE0318 fix bin number
            for ibin in range(50):
                root_hists[save_name].SetBinContent(ibin, h_bkg[f'{cat}_{m}_{pv}'][ibin] )
                root_hists[save_name].SetBinError(ibin, np.sqrt(h_bkg_sumw2[f'{cat}_{m}_{pv}'][ibin] ))
            root_hists[save_name].SetEntries(N[m])
            root_hists[save_name].Write()
        sig_root.cd()
        for c in cats:
            save_name = f"true_sig_{cat}_{c}_{pv}"
            root_hists[save_name] = TH1F(save_name,save_name, 50, binnings[pv]['min'], binnings[pv]['max'])
            ## CHANGE0318 fix bin number
            for ibin in range(50):
                root_hists[save_name].SetBinContent(ibin, h_sig_true[f'{cat}_{c}_{pv}'][ibin] )
                root_hists[save_name].SetBinError(ibin, np.sqrt(h_sig_sumw2[f'{cat}_{c}_{pv}'][ibin] ))
            root_hists[save_name].SetEntries(N[c])
            root_hists[save_name].Write()
    bkg_root.Close()
    sig_root.Close()


def main():
    parser = argparse.ArgumentParser(description='Plot xgb model for Bc -> tau nu vs. Z -> qq, cc, bb')
    parser.add_argument("--Vars", choices=["normal","vtx"],required=False,help="Event-level vars (normal) or added vertex vars (vtx)",default="vtx")
    parser.add_argument("--Channel", choices=['dilep_0tau', 'dilep_1tau', 'dilep_2tau', 'semilep_0tau_ud', 'semilep_0tau_cs','semilep_1tau_ud', 'semilep_1tau_cs',
        'dihad_ud_only', 'dihad_udcs', 'dihad_cs_only'],required=False,help="Which event category to train",default='dilep_0tau')
    parser.add_argument("--doBDT", choices=[True, False],required=False,help="Whether to use BDT",default=True)
    args = parser.parse_args()

    run(args.Vars, args.Channel, args.doBDT)

if __name__ == '__main__':
    main()

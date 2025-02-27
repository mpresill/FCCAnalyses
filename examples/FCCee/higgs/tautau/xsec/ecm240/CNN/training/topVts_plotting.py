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

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

from topVts_config import *
model_struct = 'DNN'
def run(vars, cat, doBDT):

    # Load trained model
    if model_struct == 'CNN':
        cnn = CNN_Model(num_train_vars[cat])
        cnn.load_state_dict(torch.load(f"/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_{model_struct+cat}.pt"))
    else:
        cnn = DNN(num_train_vars[cat])
        cnn.load_state_dict(torch.load(f"/work/awiedl/FCCAnalyses/examples/FCCee/higgs/tautau/xsec/ecm240/CNN/models/topVts_model_{model_struct+cat}.pt"))
    cnn.eval()
    path = '/ceph/xzuo/FCC_ntuples/topVts/training_samples/'
    vars_list = train_vars[cat]

    cat_sel = cat_sel_cutsfinal
    if doBDT: cat_sel = cat_sel_base

    cats = [cat]
    sigs = { #"dilep" :  "teal",
             "semilep_light": "orange",
             #"semilep_heavy": "purple",
             #"dihad":  "red"
           }

    bkgs = { "dilep":   "wheat",
             "semilep": "skyblue",
             "dihad":   "darkseagreen",
             "WW":      "mediumpurple",
             "ZZ":      "coral",
             "ZH":      "pink"
           }


    df_sig = {}
    df_bkg = {}

    for cat in cats:
        df_sig[cat] = {}
        df_bkg[cat] = {}


    sig_test_samp_wgt_scale = 1/(0.3 * 0.3)
    bkg_test_samp_wgt_scale = 1

    path_train_sig = f"{path}sigs_{pkl_version}"
    path_train_bkg = f"{path}bkgs_{pkl_version}"
    for sig in sigs:
        df_temp = pd.read_pickle(f"{path_train_sig}/{sig}_test.pkl")
        df_temp = df_temp.query(cat_sel[cat])
        df_sig[cat][sig] = df_temp
        df_sig[cat][sig] = df_sig[cat][sig][vars_list]
        x = df_sig[cat][sig].to_numpy()
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        df_sig[cat][sig]["jet_leadS_isSig"] = df_temp["jet_leadS_isSig"]
        df_sig[cat][sig]["evt_wgt"] = df_temp["evt_wgt"] * sig_test_samp_wgt_scale
        if x_tensor.nelement()==0:
            df_sig[cat][sig]["CNN"] = []
            continue
        pred = cnn(x_tensor)
        pred = pred.detach().numpy()
        df_sig[cat][sig]["CNN"] = pred
        #df_sig[cat][sig]["BDT"] = df_sig[cat][sig]["BDT"].apply(lambda x: x[1])
        print (f"{sig}: raw number {len(df_temp)}")

    for bkg in bkgs:
        df_temp = pd.read_pickle(f"{path_train_bkg}/{bkg}_test.pkl")
        df_temp = df_temp.query(cat_sel[cat])
        df_bkg[cat][bkg] = df_temp
        df_bkg[cat][bkg] = df_bkg[cat][bkg][vars_list]
        x = df_bkg[cat][bkg].to_numpy() #.applymap(lambda x: float(x[0]) if isinstance(x, list) and len(x) == 1 else x)
        if model_struct == 'CNN':
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        df_bkg[cat][bkg]["evt_wgt"] = df_temp["evt_wgt"] * bkg_test_samp_wgt_scale
        if x_tensor.nelement()==0:
            df_bkg[cat][bkg]["CNN"] = []
            continue
        pred = cnn(x_tensor)
        pred = pred.detach().numpy()
        df_bkg[cat][bkg]["CNN"] = pred
          #df_bkg[cat][bkg]["BDT"] = df_bkg[cat][bkg]["BDT"].apply(lambda x: x[1])
        print (f"{bkg}: raw number {len(df_temp)}")


    plot_vars = ["jet_leadS_energy", "jet_leadS_mass",
                 "jet_leadS_isS",
                 "jet_leadB_energy",
                 "jet_leadB_mass",
                 "jet_leadB_isB",
                 "CNN"
                ]
    #if doBDT: plot_vars.append("CNN")


    h_sig_all = {}
    h_sig_true = {}
    h_bkg = {}

    eff_sig_true = {}
    bin_edges = {}
    for cat in cats:
        for sig in sigs:
            N_sel = len(df_sig[cat][sig]) + 1e-3
            sig_label = "jet_leadS_isSig"
            N_true = len(df_sig[cat][sig].query(f"{sig_label}==1"))
            df_true = df_sig[cat][sig].query(f"{sig_label}==1")
            for pv in plot_vars:
                h_sig_all[f'{cat}_{sig}_{pv}'], bin_edges[pv] = np.histogram(df_sig[cat][sig][pv], bins=binnings[pv][0], weights=df_sig[cat][sig]['evt_wgt'])
                h_sig_true[f'{cat}_{sig}_{pv}'], bin_edges[pv] = np.histogram(df_true[pv], bins=binnings[pv][0], weights=df_true['evt_wgt'])
            print(f'{cat} of {sig}')
            print(N_sel)
            print(N_true / N_sel)
            eff_sig_true[f'{cat}_{sig}'] = N_true / N_sel

        for bkg in bkgs:
            for pv in plot_vars:
                h_bkg[f'{cat}_{bkg}_{pv}'], bin_edges[pv] = np.histogram(df_bkg[cat][bkg][pv], bins=binnings[pv][0], weights=df_bkg[cat][bkg]['evt_wgt'])


    fig_dir = f'figs/event_sel_{pkl_version}_DNN_{cat}'
    fig_dir = f'/web/awiedl/public_html/ML/DNN/topVts/yields_{pkl_version}_R5_top_trio_tightBjet_{cat}'
    if doBDT: fig_dir = f'/web/awiedl/public_html/ML/DNN/topVts/yields_{pkl_version}_R5_CNN_{cat}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for cat in cats:
        for pv in plot_vars:
            fig, ax = plt.subplots(figsize=(8,8))
            for sig in sigs:
                plt.stairs(h_sig_all [f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv], color=sigs[sig], linewidth=4, linestyle='dashed', label=f'{sig} no match')
                plt.stairs(h_sig_true[f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv], color=sigs[sig], linewidth=2, linestyle='solid', label=sig+' match %.2f'%eff_sig_true[f'{cat}_{sig}'])
            ax.tick_params(axis='both', which='major', labelsize=20)
            #ax.set_title( FCC_label, loc='right', fontsize=20)
            plt.xlabel(binnings[pv][1],fontsize=30)
            plt.ylabel('count',fontsize=30)
            #plt.yscale('log')
            ymin,ymax = plt.ylim()
#            if 'energy' in pv or 'mass' in pv or 'dR' in pv:
#                plt.yscale('log')
#                plt.ylim(0,10*ymax)
#            else:
            plt.ylim(0,1.4*ymax)

            plt.legend(fontsize=13, loc="upper right", ncol=2)
            plt.grid(alpha=0.4,which="both")
            ax.annotate(f'{cat} category', xy=(0.45,0.75), xycoords='axes fraction', fontsize=20)
            plt.tight_layout()
            save_name = f'sig_eff_{cat}_{pv}'
            fig.savefig(f"{fig_dir}/{save_name}.pdf")
            fig.savefig(f"{fig_dir}/{save_name}.png")
            plt.close()


    for cat in cats:
        for pv in plot_vars:
            fig, ax = plt.subplots(figsize=(8,8))
            bkg_plot = h_bkg[f'{cat}_semilep_{pv}'] + h_bkg[f'{cat}_dilep_{pv}'] + h_bkg[f'{cat}_dihad_{pv}'] + h_bkg[f'{cat}_WW_{pv}'] + h_bkg[f'{cat}_ZZ_{pv}'] + h_bkg[f'{cat}_ZH_{pv}']
            for bkg in bkgs:
                plot_label = f'WbWb {bkg}'
                if bkg in ["WW", "ZZ", "ZH"]: plot_label = bkg
                plt.stairs(bkg_plot, edges=bin_edges[pv], color=bkgs[bkg], fill=True,  alpha=0.8, label=plot_label)
                bkg_plot = bkg_plot - h_bkg[f'{cat}_{bkg}_{pv}']
            for sig in sigs:
                ### could use only matched sig, or all sigs.
                ### without refined selection, the semilep cs and full had match rate is low
                ### use full sig for this round
                plt.stairs(h_sig_all[f'{cat}_{sig}_{pv}'] , edges=bin_edges[pv], color=sigs[sig], linewidth=2, linestyle='solid', label=f'WsWb {sig}')
            ax.tick_params(axis='both', which='major', labelsize=20)
            #ax.set_title( FCC_label, loc='right', fontsize=20)
            plt.xlabel(binnings[pv][1],fontsize=30)
            plt.ylabel('count',fontsize=30)
            #plt.yscale('log')
            ymin,ymax = plt.ylim()
            if 'energy' in pv or 'mass' in pv or 'dR' in pv or 'CNN' in pv:
                plt.yscale('log')
                plt.ylim(0,10*ymax)
            else:
                plt.ylim(0,1.4*ymax)

            plt.legend(fontsize=13, loc="upper right", ncol=2)
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
    for cat in cats:
        for pv in plot_vars:
            bkg_root.cd()
            for bkg in bkgs:
                save_name = f"bkg_{cat}_{bkg}_{pv}"
                root_hists[save_name] = TH1F(save_name,save_name, len(binnings[pv][0])-1, binnings[pv][0][0], binnings[pv][0][-1])
                for ibin in range(len(binnings[pv][0])-1):
                    root_hists[save_name].SetBinContent(ibin, h_bkg[f'{cat}_{bkg}_{pv}'][ibin] )
                root_hists[save_name].Write()
            sig_root.cd()
            for sig in sigs:
                save_name = f"sig_{cat}_{sig}_{pv}"
                root_hists[save_name] = TH1F(save_name,save_name, len(binnings[pv][0])-1, binnings[pv][0][0], binnings[pv][0][-1])
                for ibin in range(len(binnings[pv][0])-1):
                    root_hists[save_name].SetBinContent(ibin, h_sig_true[f'{cat}_{sig}_{pv}'][ibin] )
                root_hists[save_name].Write()
    bkg_root.Close()
    sig_root.Close()



def main():
    parser = argparse.ArgumentParser(description='Plot xgb model for Bc -> tau nu vs. Z -> qq, cc, bb')
    parser.add_argument("--Vars", choices=["normal","vtx"],required=False,help="Event-level vars (normal) or added vertex vars (vtx)",default="vtx")
    parser.add_argument("--Channel", choices=["dilep", "semilep_heavy", "semilep_light"],required=False,help="Which event category to train",default="semilep_light")
    parser.add_argument("--doBDT", choices=[True, False],required=False,help="Whether to use BDT",default=True)
    args = parser.parse_args()

    run(args.Vars, args.Channel, args.doBDT)

if __name__ == '__main__':
    main()

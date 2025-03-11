import numpy as np

train_vars = {}
num_train_vars = {}

num_train_vars["dilep"] = 24
train_vars["dilep"] = ["recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
                       #"jet_leadS_phi",
                       "jet_leadS_eta",  "jet_leadS_energy", "jet_leadS_mass",
                       "jet_leadS_isS",    "jet_leadS_isC",  "jet_leadS_isB",
                       #"jet_leadB_phi",
                       "jet_leadB_eta",  "jet_leadB_energy", "jet_leadB_mass",
                       "jet_leadB_isS",    "jet_leadB_isC",  "jet_leadB_isB",
                       #"muon_1_phi", "muon_1_eta",
                       "muon_1_energy", "muon_1_charge",
                       #"muon_2_phi", "muon_2_eta",
                       "muon_2_energy", "muon_2_charge",
                       #"electron_1_phi", "electron_1_eta",
                       "electron_1_energy", "electron_1_charge",
                       #"electron_2_phi", "electron_2_eta",
                       "electron_2_energy", "electron_2_charge"]

num_train_vars["semilep_heavy"] = 32
train_vars["semilep_heavy"] = ["recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
                               #"jet_leadS_phi",
                               "jet_leadS_eta",  "jet_leadS_energy", "jet_leadS_mass",
                               "jet_leadS_isS",    "jet_leadS_isC",  "jet_leadS_isB",
                               #"jet_leadB_phi",
                               "jet_leadB_eta",  "jet_leadB_energy", "jet_leadB_mass",
                               "jet_leadB_isS",    "jet_leadB_isC",  "jet_leadB_isB",
                               #"jet_subS_phi",
                               "jet_subS_eta", "jet_subS_energy",  "jet_subS_mass",
                               "jet_subS_isS",     "jet_subS_isC",   "jet_subS_isB",
                               #"jet_leadC_phi",
                               "jet_leadC_eta",  "jet_leadC_energy", "jet_leadC_mass",
                               "jet_leadC_isS",    "jet_leadC_isC",  "jet_leadC_isB",
                               #"muon_1_phi", "muon_1_eta",
                               "muon_1_energy", "muon_1_charge",
                               #"electron_1_phi", "electron_1_eta",
                               "electron_1_energy", "electron_1_charge"]
                               #"muon_1_phi", "muon_1_eta", "muon_1_energy", "muon_1_charge",
                               #"electron_1_phi", "electron_1_eta", "electron_1_energy", "electron_1_charge"]
                               #"dijet_cs_R5_energy", "dijet_cs_R5_mass", "trijet_Scs_R5_energy", "trijet_Scs_R5_mass", "trijet_Bcs_R5_energy", "trijet_Bcs_R5_mass"]

num_train_vars["semilep_light"] = 26
train_vars["semilep_light"] = ["recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
                               #"jet_leadS_phi",
                               "jet_leadS_eta",  "jet_leadS_energy", "jet_leadS_mass",
                               "jet_leadS_isS",    "jet_leadS_isC",  "jet_leadS_isB",
                               #"jet_leadB_phi",
                               "jet_leadB_eta",  "jet_leadB_energy", "jet_leadB_mass",
                               "jet_leadB_isS",    "jet_leadB_isC",  "jet_leadB_isB",
                               #"jet_subS_phi",
                               "jet_subS_eta", "jet_subS_energy",  "jet_subS_mass",
                               "jet_subS_isS",     "jet_subS_isC",   "jet_subS_isB",
                               #"muon_1_phi", "muon_1_eta",
                               "muon_1_energy", "muon_1_charge",
                               #"electron_1_phi", "electron_1_eta",
                               "electron_1_energy", "electron_1_charge"]

cat_sel_base = {'dilep':         'dilep_R5_cat == 1   ',
                'semilep_heavy': 'semilep_R5_cat == 1 and n_ctags == 1',
                'semilep_light': 'semilep_R5_cat == 1 and n_ctags == 0',
                'dihad':         'dihad_R5_cat == 1   '}


cat_sel_cutsfinal = {'dilep':      'dilep_R5_cat == 1   and recoEmiss_e > 80 and jet_leadS_energy > 45 and jet_leadB_energy > 25 and jet_leadB_isB > 0.9',
                     'semilep_heavy': 'semilep_R5_cat == 1 and recoEmiss_e > 30 and n_ctags == 1',
                     'semilep_light': 'semilep_R5_cat == 1 and recoEmiss_e > 30 and n_ctags == 0 and jet_leadB_energy > 40 and jet_leadS_energy > 60 and jet_leadB_isB > 0.9',
                     'dihad':      'dihad_R5_cat == 1   and recoEmiss_e < 10 and jet_leadB_energy > 40 and jet_leadS_energy > 60 and jet_leadB_isB > 0.9'}




bkg_list = ["dilep", "semilep", "dihad", "WW", "ZZ", "ZH"]
#path = "/afs/cern.ch/work/x/xzuo/FCC_studies/FCCeePhysicsPerformance/case-studies/top/Vts"
#pkl_version = "Oct29_R5_training"
pkl_version = "Nov16_R5_training"
bdt_suffix = 'T200D3'

plot_vars = ["jet_leadS_energy", "jet_leadS_mass", "jet_leadS_flavor",
                 "jet_leadS_dR_b",
                 "jet_leadS_dR_s",
                 "jet_leadS_isS",
                 "jet_subS_energy", "jet_subS_mass", "jet_subS_flavor",
                 "jet_subS_isS",
                 "jet_leadB_energy",
                 "jet_leadB_mass",
                 "jet_leadB_flavor",
                 "jet_leadB_dR_b",
                 "jet_leadB_isB",
                 "recoEmiss_e",
                ]
binnings = {'jet_leadS_phi'      : [np.linspace(-3.5,-3.5,51),  "$\\phi_{lead~s-tag}$"],
            'jet_leadS_eta'      : [np.linspace(-5,5,51),       "$\\eta_{lead~s-tag}$"],
            'jet_leadS_energy'   : [np.linspace(0,180,91),      "$E_{lead~s-tag}$ (GeV)"],
            'jet_leadS_mass'     : [np.linspace(0,50,51),       "$m_{lead~s-tag}$ (GeV)"],
            'jet_leadS_isS'      : [np.linspace(0.5,1,51),      "s-score(lead s-tag)"],
            'jet_leadS_isC'      : [np.linspace(0,1,101),       "c-score(lead s-tag)"],
            'jet_leadS_isB'      : [np.linspace(0,1,101),       "b-score(lead s-tag)"],
            'jet_leadS_flavor'   : [np.linspace(0,6,7),         "true flav(lead s-tag)"],
            'jet_leadS_dR_b'     : [np.linspace(0,5,51),        "$\\Delta$R(lead s-tag, true b)"],
            'jet_leadS_dR_s'     : [np.linspace(0,5,51),        "$\\Delta$R(lead s-tag, true s)"],
            'jet_leadS_isS'      : [np.linspace(0.5,1,51),      "s-score(lead s-tag)"],
            'jet_subS_phi'       : [np.linspace(-3.5,-3.5,51),  "$\\phi_{sub~s-tag}$"],
            'jet_subS_eta'       : [np.linspace(-5,5,51),       "$\\eta_{sub~s-tag}$"],
            'jet_subS_energy'    : [np.linspace(0,180,91),      "$E_{sub~s-tag}$ (GeV)"],
            'jet_subS_mass'      : [np.linspace(0,50,51),       "$m_{sub~s-tag}$ (GeV)"],
            'jet_subS_isS'       : [np.linspace(0.5,1,51),      "s-score(sub s-tag)"],
            'jet_subS_isC'       : [np.linspace(0,1,101),       "c-score(sub s-tag)"],
            'jet_subS_isB'       : [np.linspace(0,1,101),       "b-score(sub s-tag)"],
            'jet_subS_flavor'    : [np.linspace(0,6,7),         "true flav(sub s-tag)"],
            'jet_leadB_phi'      : [np.linspace(-3.5,-3.5,51),  "$\\phi_{lead~b-tag}$"],
            'jet_leadB_eta'      : [np.linspace(-5,5,51),       "$\\eta_{lead~b-tag}$"],
            'jet_leadB_energy'   : [np.linspace(0,180,91),      "$E_{lead~b-tag}$ (GeV)"],
            'jet_leadB_mass'     : [np.linspace(0,50,51),       "$m_{lead~b-tag}$ (GeV)"],
            'jet_leadB_isB'      : [np.linspace(0.5,1,51),      "b-score(lead b-tag)"],
            'jet_leadB_isC'      : [np.linspace(0,1,101),       "c-score(lead b-tag)"],
            'jet_leadB_isS'      : [np.linspace(0,1,101),       "s-score(lead b-tag)"],
            'jet_leadB_flavor'   : [np.linspace(0,6,7),         "true flav(lead b-tag)"],
            'jet_leadB_dR_b'     : [np.linspace(0,5,51),        "$\\Delta$R(lead b-tag, true b)"],
            'jet_leadC_phi'      : [np.linspace(-3.5,-3.5,51),  "$\\phi_{lead~c-tag}$"],
            'jet_leadC_eta'      : [np.linspace(-5,5,51),       "$\\eta_{lead~c-tag}$"],
            'jet_leadC_energy'   : [np.linspace(0,180,91),      "$E_{lead~c-tag}$ (GeV)"],
            'jet_leadC_mass'     : [np.linspace(0,50,51),       "$m_{lead~c-tag}$ (GeV)"],
            'jet_leadC_isC'      : [np.linspace(0.5,1,51),      "c-score(lead c-tag)"], 
            'jet_leadC_isB'      : [np.linspace(0,1,101),       "b-score(lead c-tag)"],
            'jet_leadC_isS'      : [np.linspace(0,1,101),       "s-score(lead c-tag)"],
            'jet_leadC_flavor'   : [np.linspace(0,6,7),         "true flav(lead c-tag)"],
            'recoEmiss_e'        : [np.linspace(0,200,101),     "$E^{miss}$ (GeV)"],
            'recoEmiss_px'       : [np.linspace(0,200,101),     "$E^{miss}_{x}$ (GeV)"],
            'recoEmiss_py'       : [np.linspace(0,200,101),     "$E^{miss}_{y}$ (GeV)"],
            'recoEmiss_pz'       : [np.linspace(0,200,101),     "$E^{miss}_{z}$ (GeV)"],
            'muon_1_energy'      : [np.linspace(0,180,91),      "$E_{lead~\\mu}$ (GeV)"],
            'muon_1_charge'      : [np.linspace(-2,2,5),        "$q_{lead~\\mu}$"],
            'muon_2_energy'      : [np.linspace(0,180,91),      "$E_{sub~\\mu}$ (GeV)"],
            'muon_2_charge'      : [np.linspace(-2,2,5),        "$q_{sub~\\mu}$"],
            'electron_1_energy'  : [np.linspace(0,180,91),      "$E_{lead~e}$ (GeV)"],
            'electron_1_charge'  : [np.linspace(-2,2,5),        "$q_{lead~e}$"],
            'electron_2_energy'  : [np.linspace(0,180,91),      "$E_{sub~e}$ (GeV)"],
            'electron_2_charge'  : [np.linspace(-2,2,5),        "$q_{sub~e}$"],
            'CNN'                : [np.linspace(0,1,101),       "CNN output"],
            }

FCC_label = '\\textbf{FCC-ee Simulation (IDEA Delphes)}'

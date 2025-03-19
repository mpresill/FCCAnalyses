import numpy as np
### sample list
sample_list = {}
sample_list['training'] = {}
sample_list['fitting'] = {}

colors = {}

colors['sig'] = {
        'wzp6_ee_SM_tt_tWsTWb_tlepTall_ecm365':'teal',
        'wzp6_ee_SM_tt_tWbTWs_tallTlep_ecm365':'black',
        'wzp6_ee_SM_tt_tWbTWs_tallTlight_ecm365':'green',
        'wzp6_ee_SM_tt_tWbTWs_tallTheavy_ecm365':'purple',
        'wzp6_ee_SM_tt_tWsTWb_tlightTall_ecm365':'orange',
        'wzp6_ee_SM_tt_tWsTWb_theavyTall_ecm365':'red',
}

colors['bkg'] = {
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365":'brown',
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365":'maroon',
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365":'darkred',
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365":'tomato',
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365":'mistyrose',
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365":'coral',

        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365":'forestgreen', 
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365":'darkseagreen',
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365":'honeydew',
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365":'chartreuse',
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365":'palegreen',
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365":'yellowgreen',

        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365":'hotpink',
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365":'mediumslateblue',
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365":'darkviolet',
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365":'thistle',
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365":'lightpink',
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365":'magenta',

        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365":'lightyellow',
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365":'gold',
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365":'wheat',
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365":'goldenrod',
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365":'beige',
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365":'darkgoldenrod',

        'p8_ee_WW_ecm365':'olive',
        'p8_ee_ZZ_ecm365':'cyan',

        # ZH samples, no inclusive ones
        'wzp6_ee_bbH_ecm365':'azure',
        'wzp6_ee_ccH_ecm365':'skyblue',
        'wzp6_ee_ssH_ecm365':'dodgerblue',
        'wzp6_ee_qqH_ecm365':'aliceblue',
        'wzp6_ee_tautauH_ecm365':'aquamarine',
        'wzp6_ee_mumuH_ecm365':'lightseagreen',
        'wzp6_ee_eeH_ecm365':'cadetblue',
        'wzp6_ee_nunuH_ecm365':'steelblue',
}

sample_list['training']['sig'] = [
        'wzp6_ee_SM_tt_tWsTWb_tlepTall_ecm365',
        'wzp6_ee_SM_tt_tWbTWs_tallTlep_ecm365',
        'wzp6_ee_SM_tt_tWbTWs_tallTlight_ecm365',
        'wzp6_ee_SM_tt_tWbTWs_tallTheavy_ecm365',
        'wzp6_ee_SM_tt_tWsTWb_tlightTall_ecm365',
        'wzp6_ee_SM_tt_tWsTWb_theavyTall_ecm365'
        ]

sample_list['training']['bkg'] = [
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365",
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365",
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365",
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365",
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365",
        "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365",

        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365", 
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365",
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365",
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365",
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365",
        "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365",

        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365",
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365",
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365",
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365",
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365",
        "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365",

        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365",
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365",
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365",
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365",
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365",
        "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365",

        'p8_ee_WW_ecm365',
        'p8_ee_ZZ_ecm365',

        'p8_ee_Zbb_ecm365'      ,
        'p8_ee_Zcc_ecm365'      ,
        'p8_ee_Zss_ecm365'      ,
        'p8_ee_Zqq_ecm365'      ,
        'wzp6_ee_tautau_ecm365' ,

        'wzp6_ee_WWZ_Zbb_ecm365',

        # ZH samples, no inclusive ones
        'wzp6_ee_bbH_ecm365',
        'wzp6_ee_ccH_ecm365',
        'wzp6_ee_ssH_ecm365',
        'wzp6_ee_qqH_ecm365',
        'wzp6_ee_tautauH_ecm365',
        'wzp6_ee_mumuH_ecm365',
        'wzp6_ee_eeH_ecm365',
        'wzp6_ee_nunuH_ecm365'
        ]
FCC_label = '\\textbf{FCC-ee Simulation (IDEA Delphes)}'
sample_list['fitting']['sig'] = sample_list['training']['sig']
ttbar = ['wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365',]
sample_list['fitting']['bkg'] = [
        'wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365',
        'wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365',

        'p8_ee_WW_ecm365',
        'p8_ee_ZZ_ecm365',

        'p8_ee_Zbb_ecm365'      ,
        'p8_ee_Zcc_ecm365'      ,
        'p8_ee_Zss_ecm365'      ,
        'p8_ee_Zqq_ecm365'      ,
        'wzp6_ee_tautau_ecm365' ,

        'wzp6_ee_WWZ_Zbb_ecm365',


        # ZH samples, no inclusive ones
        'wzp6_ee_bbH_ecm365',
        'wzp6_ee_ccH_ecm365',
        'wzp6_ee_ssH_ecm365',
        'wzp6_ee_qqH_ecm365',
        'wzp6_ee_tautauH_ecm365',
        'wzp6_ee_mumuH_ecm365',
        'wzp6_ee_eeH_ecm365',
        'wzp6_ee_nunuH_ecm365',
        ]


### categories
cats = ['dilep_0tau', 'dilep_1tau', 'dilep_2tau', 
        'semilep_0tau_ud', 'semilep_0tau_cs',
        'semilep_1tau_ud', 'semilep_1tau_cs',
        'dihad_ud_only', 'dihad_udcs', 'dihad_cs_only']
sig_filter = {}
for cat in cats:
    sig_filter[cat] = 'Is_' + cat


### training variable list
training_vars = {}
num_training_vars = {}
training_vars['dilep_0tau'] = [
        "lep_1_eta", "lep_1_energy", "lep_2_eta", "lep_2_energy",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB"
        ]

num_training_vars['dilep_0tau'] = 22

binning = {
            'recoEmiss_e'        : [np.linspace(0,200,101),     "$E^{miss}$ (GeV)"],
            'recoEmiss_px'       : [np.linspace(0,200,101),     "$E^{miss}_{x}$ (GeV)"],
            'recoEmiss_py'       : [np.linspace(0,200,101),     "$E^{miss}_{y}$ (GeV)"],
            'recoEmiss_pz'       : [np.linspace(0,200,101),     "$E^{miss}_{z}$ (GeV)"],
            'CNN'                : [np.linspace(0,1,101),       "CNN output"],
            }
training_vars['dilep_1tau'] = [
        "lep_1_eta", "lep_1_energy",
        "tau_1_eta", "tau_1_energy", "tau_1_mass", "tau_1_isTau",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB"
        ]

num_training_vars['dilep_1tau'] = 24


training_vars['dilep_2tau'] = [
        "tau_1_eta", "tau_1_energy", "tau_1_mass", "tau_1_isTau",
        "tau_2_eta", "tau_2_energy", "tau_2_mass", "tau_2_isTau",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB"
        ]

num_training_vars['dilep_2tau'] = 26


training_vars['semilep_0tau_ud'] = [
        "lep_1_eta", "lep_1_energy",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",

        "jet_ud_1_phi", "jet_ud_1_eta", "jet_ud_1_energy", "jet_ud_1_mass",
        "jet_ud_1_isU", "jet_ud_1_isD",
        "jet_ud_2_phi", "jet_ud_2_eta", "jet_ud_2_energy", "jet_ud_2_mass",
        "jet_ud_2_isU", "jet_ud_2_isD",
        ]

num_training_vars['semilep_0tau_ud'] = 32


training_vars['semilep_0tau_cs'] = [
        "lep_1_eta", "lep_1_energy",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_S_1_phi", "jet_S_1_eta", "jet_S_1_energy", "jet_S_1_mass",
        "jet_S_1_isS", "jet_S_1_isC", "jet_S_1_isB",
        "jet_S_2_phi", "jet_S_2_eta", "jet_S_2_energy", "jet_S_2_mass",
        "jet_S_2_isS", "jet_S_2_isC", "jet_S_2_isB",
        "jet_candC_phi", "jet_candC_eta", "jet_candC_energy", "jet_candC_mass",
        "jet_candC_isS", "jet_candC_isC", "jet_candC_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",
        ]

num_training_vars['semilep_0tau_cs'] = 34


training_vars['semilep_1tau_ud'] = [
        "tau_1_eta", "tau_1_energy", "tau_1_mass", "tau_1_isTau",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",

        "jet_ud_1_phi", "jet_ud_1_eta", "jet_ud_1_energy", "jet_ud_1_mass",
        "jet_ud_1_isU", "jet_ud_1_isD",
        "jet_ud_2_phi", "jet_ud_2_eta", "jet_ud_2_energy", "jet_ud_2_mass",
        "jet_ud_2_isU", "jet_ud_2_isD"
        ]

num_training_vars['semilep_1tau_ud'] = 34


training_vars['semilep_1tau_cs'] = [
        "tau_1_eta", "tau_1_energy", "tau_1_mass", "tau_1_isTau",
        "recoEmiss_px", "recoEmiss_py", "recoEmiss_pz", "recoEmiss_e",
        "jet_S_1_phi", "jet_S_1_eta", "jet_S_1_energy", "jet_S_1_mass",
        "jet_S_1_isS", "jet_S_1_isC", "jet_S_1_isB",
        "jet_S_2_phi", "jet_S_2_eta", "jet_S_2_energy", "jet_S_2_mass",
        "jet_S_2_isS", "jet_S_2_isC", "jet_S_2_isB",
        "jet_candC_phi", "jet_candC_eta", "jet_candC_energy", "jet_candC_mass",
        "jet_candC_isS", "jet_candC_isC", "jet_candC_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB"
        ]

num_training_vars['semilep_1tau_cs'] = 36


training_vars['dihad_ud_only'] = [
        "recoEmiss_e",
        "jet_candS_phi", "jet_candS_eta", "jet_candS_energy", "jet_candS_mass",
        "jet_candS_isS", "jet_candS_isC", "jet_candS_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",

        "jet_ud_1_phi", "jet_ud_1_eta", "jet_ud_1_energy", "jet_ud_1_mass",
        "jet_ud_1_isU", "jet_ud_1_isD",
        "jet_ud_2_phi", "jet_ud_2_eta", "jet_ud_2_energy", "jet_ud_2_mass",
        "jet_ud_2_isU", "jet_ud_2_isD",
        "jet_ud_3_phi", "jet_ud_3_eta", "jet_ud_3_energy", "jet_ud_3_mass",
        "jet_ud_3_isU", "jet_ud_3_isD",
        "jet_ud_4_phi", "jet_ud_4_eta", "jet_ud_4_energy", "jet_ud_4_mass",
        "jet_ud_4_isU", "jet_ud_4_isD"
        ]

num_training_vars['dihad_ud_only'] = 39


training_vars['dihad_cs_only'] = [
        "recoEmiss_e",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",

        "jet_C_1_phi", "jet_C_1_eta", "jet_C_1_energy", "jet_C_1_mass",
        "jet_C_1_isS", "jet_C_1_isC", "jet_C_1_isB",
        "jet_C_2_phi", "jet_C_2_eta", "jet_C_2_energy", "jet_C_2_mass",
        "jet_C_2_isS", "jet_C_2_isC", "jet_C_2_isB",

        "jet_S_1_phi", "jet_S_1_eta", "jet_S_1_energy", "jet_S_1_mass",
        "jet_S_1_isS", "jet_S_1_isC", "jet_S_1_isB",
        "jet_S_2_phi", "jet_S_2_eta", "jet_S_2_energy", "jet_S_2_mass",
        "jet_S_2_isS", "jet_S_2_isC", "jet_S_2_isB",
        "jet_S_3_phi", "jet_S_3_eta", "jet_S_3_energy", "jet_S_3_mass",
        "jet_S_3_isS", "jet_S_3_isC", "jet_S_3_isB"
        ]

num_training_vars['dihad_cs_only'] = 43


training_vars['dihad_udcs'] = [
        "recoEmiss_e",
        "jet_S_1_phi", "jet_S_1_eta", "jet_S_1_energy", "jet_S_1_mass",
        "jet_S_1_isS", "jet_S_1_isC", "jet_S_1_isB",
        "jet_S_2_phi", "jet_S_2_eta", "jet_S_2_energy", "jet_S_2_mass",
        "jet_S_2_isS", "jet_S_2_isC", "jet_S_2_isB",
        "jet_candC_phi", "jet_candC_eta", "jet_candC_energy", "jet_candC_mass",
        "jet_candC_isS", "jet_candC_isC", "jet_candC_isB",
        "jet_candB_phi", "jet_candB_eta", "jet_candB_energy", "jet_candB_mass",
        "jet_candB_isS", "jet_candB_isC", "jet_candB_isB",

        "jet_ud_1_phi", "jet_ud_1_eta", "jet_ud_1_energy", "jet_ud_1_mass",
        "jet_ud_1_isU", "jet_ud_1_isD",
        "jet_ud_2_phi", "jet_ud_2_eta", "jet_ud_2_energy", "jet_ud_2_mass",
        "jet_ud_2_isU", "jet_ud_2_isD"
        ]

num_training_vars['dihad_udcs'] = 41
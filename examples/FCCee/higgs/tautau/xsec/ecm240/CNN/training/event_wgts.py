
## CKM elements in SM
## from https://pdg.lbl.gov/2024/reviews/rpp2024-rev-ckm-matrix.pdf
Vtb = 0.999118
Vts = 0.04111
Vtd = 0.00858

## assuming Vtb^2 + Vts^2 + Vtd^2 = 1
BR_tb = Vtb * Vtb
BR_ts = Vts * Vts

## W branching ratio
## from https://pdglive.lbl.gov/Particle.action?node=S043&init=0
Wlnu = 0.3258
Wud  = 0.6472/2
Wcs  = 0.6472/2

## 2.65 ab-1 should correspond to 2M tt events
## exact tt xsec is unclear
## normalize tt events with 2M, and other events with 2.65 and their xsec
lumi = 2.65 #ab-1
N_tt = 2E6  #total expected number of tt events

## event weight calculated with N_exp / N_MC
evt_wgt = {}
evt_wgt['sigs'] = {"wzp6_ee_SM_tt_tWsTWb_tlepTall_ecm365"   : N_tt * BR_tb * BR_ts * Wlnu / 4730233,
                   "wzp6_ee_SM_tt_tWsTWb_tlightTall_ecm365" : N_tt * BR_tb * BR_ts * Wud  / 4564328,
                   "wzp6_ee_SM_tt_tWsTWb_theavyTall_ecm365" : N_tt * BR_tb * BR_ts * Wcs  / 4978856,
                   "wzp6_ee_SM_tt_tWbTWs_tallTlep_ecm365"   : N_tt * BR_tb * BR_ts * Wlnu / 4894704,
                   "wzp6_ee_SM_tt_tWbTWs_tallTlight_ecm365" : N_tt * BR_tb * BR_ts * Wud  / 4396175,
                   "wzp6_ee_SM_tt_tWbTWs_tallTheavy_ecm365" : N_tt * BR_tb * BR_ts * Wcs  / 4813736
                  }

evt_wgt['bkgs'] = {"wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ecm365": N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5330291,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365": N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5412033,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5329449,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5329622,
                     
                     "p8_ee_WW_ecm365"        :  lumi * 10.7165E6   / 100754213,
                         "p8_ee_ZZ_ecm365"        :  lumi *  0.6428E6   /  61470944,

                         'p8_ee_Zbb_ecm365'       :  lumi *  4.0560E6   /  55932700,
                         'p8_ee_Zcc_ecm365'       :  lumi *  4.5060E6   /  55879156,
                         'p8_ee_Zss_ecm365'       :  lumi *  4.0840E6   /  55848129,
                         'p8_ee_Zqq_ecm365'       :  lumi *  8.6790E6   /  55742194,
                         'wzp6_ee_tautau_ecm365'  :  lumi *  2.0170E6   /  12800000,

                         'wzp6_ee_WWZ_Zbb_ecm365' :  lumi *  0.002018E6 /   1200000,

                         "wzp6_ee_bbH_ecm365"     :  lumi *  0.018389E6 /   1100000,
                         "wzp6_ee_ccH_ecm365"     :  lumi *  0.014436E6 /    900000,
                         "wzp6_ee_ssH_ecm365"     :  lumi *  0.018538E6 /    900000,
                         "wzp6_ee_qqH_ecm365"     :  lumi *  0.032997E6 /   2400000,
                         "wzp6_ee_tautauH_ecm365" :  lumi *  0.004172E6 /   1100000,
                         "wzp6_ee_mumuH_ecm365"   :  lumi *  0.004185E6 /   1200000,
                         "wzp6_ee_eeH_ecm365"     :  lumi *  0.00739E6  /   1000000,
                         "wzp6_ee_nunuH_ecm365"   :  lumi *  0.05394E6  /   2200000,

                    
                    "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * Wlnu         / 5000000 / 6,

                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365"  : N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,
                      "wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365": N_tt * BR_tb * BR_tb * Wlnu * (1-Wlnu)     / 5000000 / 6,

                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ta_ttAdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_tv_ttAdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_vr_ttZdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * Wlnu     / 5000000 / 6,

                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ta_ttAdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_tv_ttAdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZup_ecm365"  : N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                      "wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_vr_ttZdown_ecm365": N_tt * BR_tb * BR_tb * (1-Wlnu) * (1-Wlnu) / 5000000 / 6,
                        }

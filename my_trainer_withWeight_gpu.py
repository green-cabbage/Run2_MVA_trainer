import xgboost as xgb
import argparse
#from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import awkward as ak
import dask_awkward as dak
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import tqdm
from distributed import LocalCluster, Client, progress
import os
import coffea.util as util
import time
from xgboost import plot_importance, plot_tree
import copy
import json
# import cmsstyle as CMS
import pickle
import glob
from modules.utils import auc_from_eff, PairNAnnhilateNegWgt, addErrByQuadrature, GetAucStdErrHanleyMcNeil, fullROC_operations
import seaborn as sb

def getGOF_KS_bdt(valid_hist, train_hist, weight_val, bin_edges, save_path:str, fold_idx):
    """
    Get KS value for specific value
    """
    print(f"valid_hist: {valid_hist}")
    print(f"train_hist: {train_hist}")
    print(f"valid_hist: {np.sum(valid_hist)}")
    print(f"train_hist: {np.sum(train_hist)}")
    
    data_counts = valid_hist
    pdf_counts = train_hist


    plot_line_width = 0.5

    # -------------------------------------------
    # Do KS test
    # -------------------------------------------


    
    data_cdf = np.cumsum(data_counts) / np.sum(data_counts)
    pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
    ks_statistic = np.max(np.abs(data_cdf - pdf_cdf))
    print(f"ks_statistic: {ks_statistic}")
    nevents = weight_val.size
    print(f"nevents: {nevents}")
    
    
    alpha1 = 0.1
    pass_threshold1 = 1.22385 / (nevents**(0.5))
    alpha2 = 0.001
    pass_threshold2 = 1.94947 / (nevents**(0.5))

    df_dict= {
        "ks_statistic": [ks_statistic],
        "nevents" : [nevents],
        "alpha1":[ alpha1],
        "alpha1 pass threshold": [pass_threshold1],
        "alpha1 test pass": [ks_statistic<pass_threshold1],
        "alpha2":[ alpha2],
        "alpha2 pass threshold": [pass_threshold2],
        "alpha2 test pass": [ks_statistic<pass_threshold2],
        
   }
    gof_df = pd.DataFrame(df_dict)
    gof_df.to_csv(f"{save_path}/KS_stats_{fold_idx}.csv")
    

    # Draw the cdf histogram
    
    # bin_centers = np.array(bin_centers)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, data_cdf, label='Validation CDF', linewidth=plot_line_width)
    plt.plot(bin_centers, pdf_cdf, label='Train CDF', linewidth=plot_line_width)
    plt.xlabel('BDT SCORE')
    plt.ylabel('')
    plt.title('BDT distribution of signal sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_cdfs_{fold_idx}.pdf")
    plt.clf()

    # Draw the normalized pdf histogram:
    plt.plot(bin_centers, data_counts/np.sum(data_counts), label='Validation PDF', linewidth=plot_line_width)
    plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='Train PDF', linewidth=plot_line_width)
    plt.xlabel('BDT SCORE')
    plt.ylabel('')
    plt.title('BDT distribution of signal sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_pdfs_{fold_idx}.pdf")
    plt.clf()
        

# def auc_from_eff(eff_sig, eff_bkg):
#     fpr = 1.0 - np.asarray(eff_bkg)
#     tpr = np.asarray(eff_sig)
#     # sort by FPR ascending before integrating
#     order = np.argsort(fpr)
#     return np.trapezoid(tpr[order], fpr[order])


def prepare_features(events, features, variation="nominal"):
    plt.style.use(hep.style.CMS)
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation
        
        if f"{trf}_{variation_current}" in events.fields:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in events.fields:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var

def deactivateWgts(wgt_arr, events, wgts2deactivate):
    """
    """
    wgt_arr_out = copy.deepcopy(wgt_arr) # make a copy bc you never know
    for wgt in wgts2deactivate:
        wgt_arr_out = wgt_arr_out/events[wgt]
    return wgt_arr_out

def generate_normalized_histogram(values, weights, bins):
    """
    Generates a normalized histogram using values and weights.

    Parameters:
    - values: array-like, the data values.
    - weights: array-like, the weights for each data point.
    - bins: int or array-like, defines the bin edges.

    Returns:
    - hist: array, the values of the normalized histogram.
    - bin_edges: array, the edges of the bins.
    """
    # Compute the histogram with weights and density normalization
    hist, bin_edges = np.histogram(values, bins=bins, weights=weights, density=True)
    return hist, bin_edges


def transformBdtOut(pred):
    """
    change the BDT output range from [0,1] to [-1,1]
    """
    pred = pred *2 -1
    return pred

def get6_5(label, pred, weight, save_path:str, name: str):
    # seperate signal and background
    with open("my_data.pkl", "wb") as f:
        pickle.dump((label, pred, weight), f)  #
        
    # binning = np.linspace(start=0,stop=1, num=60) 
    binning = np.linspace(start=-1,stop=1, num=41) 
    bkg_filter = (label ==0)
    bkg_pred = transformBdtOut(pred[bkg_filter])
    bkg_wgt = weight[bkg_filter]
    bkg_wgt = bkg_wgt / np.sum(bkg_wgt) # normalize
    bkg_hist, edges = np.histogram(bkg_pred, bins=binning, weights=bkg_wgt)
    sig_filter = (label ==1)
    sig_pred = transformBdtOut(pred[sig_filter])
    sig_wgt = weight[sig_filter]
    sig_wgt = sig_wgt / np.sum(sig_wgt) # normalize
    sig_hist, _ = np.histogram(sig_pred, bins=binning, weights=sig_wgt)
    # plot
    fig, ax_main = plt.subplots(figsize=(10, 8.6))
    ax_main.stairs(bkg_hist, edges, label = "background", color="Red")
    ax_main.stairs(sig_hist, edges, label = "signal", color="Blue")
    
    
        
    # Add legend and axis labels
    ax_main.set_xlabel('BDT Score')
    ax_main.set_ylabel("a.u.")
    ax_main.legend()
    
    # Set Range
    # ax_main.set_xlim(-0.9, 0.9)
    ax_main.set_xlim(-1.0, 1.0)
    ax_main.set_xticks([ -0.8, -0.6, -0.4, -0.2 , 0. ,  0.2 , 0.4 , 0.6,  0.8])
    ax_main.set_ylim(0, 0.09)
    
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    hep.cms.label(data=False, ax=ax_main)
    
    plt.savefig(f"{save_path}/6_5_{name}.png")
    plt.savefig(f"{save_path}/6_5_{name}.pdf")
    plt.clf()

    # save hist in pd df
    df_6p5 = pd.DataFrame({
        'bin_left': binning[:-1],
        'bin_right': binning[1:],
        'bin_center': 0.5 * (binning[:-1] + binning[1:]),
        "bkg_hist" : bkg_hist,
        "sig_hist" : sig_hist,
    })
    df_6p5.to_csv(f"{save_path}/6_5_{name}.csv")
    

# def customROC_curve_AN(label, pred, weight, doClassBalance = False):
#     """
#     generates signal and background efficiency consistent with the AN,
#     as described by Fig 4.6 of Dmitry's PhD thesis
#     """
#     # we assume sigmoid output with labels 0 = background, 1 = signal
#     thresholds = np.linspace(start=0,stop=1, num=500) 
#     effBkg_total = -99*np.ones_like(thresholds) # effBkg = false positive rate
#     effSig_total = -99*np.ones_like(thresholds) # effSig = true positive rate
#     FP_total = -99*np.ones_like(thresholds) 
#     TP_total = -99*np.ones_like(thresholds) 
#     TN_total = -99*np.ones_like(thresholds) 
#     FN_total = -99*np.ones_like(thresholds) 
#     FP_err_total = -99*np.ones_like(thresholds) 
#     TP_err_total = -99*np.ones_like(thresholds) 
#     TN_err_total = -99*np.ones_like(thresholds) 
#     FN_err_total = -99*np.ones_like(thresholds) 

#     # apply class balance if requested
#     if doClassBalance:
#         is_bkg = label == 0
#         is_sig = ~is_bkg
#         bkg_weight = weight[is_bkg]
#         weight = np.where(is_bkg, weight/np.sum(bkg_weight), weight)
#         sig_weight = weight[is_sig]
#         weight = np.where(is_sig, weight/np.sum(sig_weight), weight)
#         print(f"bkg wgt sum: {np.sum(weight[is_bkg])}")
#         print(f"sig wgt sum: {np.sum(weight[is_sig])}")
#     for ix in range(len(thresholds)):
#         threshold = thresholds[ix]
#         # get FP and TP
#         positive_filter = (pred > threshold)
#         falsePositive_filter = positive_filter & (label == 0)
#         FP = np.sum(weight[falsePositive_filter])#  FP = false positive
#         truePositive_filter = positive_filter & (label == 1)
#         TP = np.sum(weight[truePositive_filter])#  TP = true positive
        

#         # get TN and FN
#         negative_filter = (pred <= threshold) # just picked negative to be <=
#         trueNegative_filter = negative_filter & (label == 0)
#         TN = np.sum(weight[trueNegative_filter])#  TN = true negative
#         falseNegative_filter = negative_filter & (label == 1)
#         FN = np.sum(weight[falseNegative_filter])#  FN = false negative

#         # obtain the err of FP, TP, TN, FN
#         FP_err = np.sqrt(np.sum(weight[falsePositive_filter]**2))
#         TP_err = np.sqrt(np.sum(weight[truePositive_filter]**2))
#         TN_err = np.sqrt(np.sum(weight[trueNegative_filter]**2))
#         FN_err = np.sqrt(np.sum(weight[falseNegative_filter]**2))
        


#         # effBkg = TN / (TN + FP) # Dmitry PhD thesis definition
#         # effSig = FN / (FN + TP) # Dmitry PhD thesis definition
#         effBkg = FP / (TN + FP) # AN-19-124 ggH Cat definition
#         effSig = TP / (FN + TP) # AN-19-124 ggH Cat definition
#         effBkg_total[ix] = effBkg
#         effSig_total[ix] = effSig

#         # print(f"ix: {ix}") 
#         # print(f"threshold: {threshold}")
#         # print(f"effBkg: {effBkg}")
#         # print(f"effSig: {effSig}")

#         FP_total[ix] = FP
#         TP_total[ix] = TP
#         TN_total[ix] = TN
#         FN_total[ix] = FN
#         FP_err_total[ix] = FP_err
#         TP_err_total[ix] = TP_err
#         TN_err_total[ix] = TN_err
#         FN_err_total[ix] = FN_err
        
        
#         # sanity check
#         assert ((np.sum(positive_filter) + np.sum(negative_filter)) == len(pred))
#         total_yield = FP + TP + FN + TN
#         assert(np.isclose(total_yield, np.sum(weight)))
#         # print(f"total_yield: {total_yield}")
#         # print(f"np.sum(weight): {np.sum(weight)}")
    
#     # print(f"np.sum(effBkg_total ==-99) : {np.sum(effBkg_total ==-99)}")
#     # print(f"np.sum(effSig_total ==-99) : {np.sum(effSig_total ==-99)}")
#     # neither_zeroNorOne = ~((label == 0) | (label == 1))
#     # print(f"np.sum(neither_zeroNorOne) : {np.sum(neither_zeroNorOne)}")
#     effBkg_total[np.isnan(effBkg_total)] = 1
#     effSig_total[np.isnan(effSig_total)] = 1
#     # print(f"effBkg_total: {effBkg_total}")
#     # print(f"effSig_total: {effSig_total}")
#     # print(f"thresholds: {thresholds}")
#     # raise ValueError
#     effBkgSig_df = pd.DataFrame({
#         "FP" : FP_total,
#         "TP" : TP_total,
#         "TN" : TN_total,
#         "FN" : FN_total,
#         "FP_err" : FP_err_total,
#         "TP_err" : TP_err_total,
#         "TN_err" : TN_err_total,
#         "FN_err" : FN_err_total,
#     })
    
#     return (effBkg_total, effSig_total, thresholds, effBkgSig_df)




#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel', 'dimuon_eta', 'dimuon_mass', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta nominal', 'jet1_phi nominal', 'jet1_pt nominal', 'jet1_qgl nominal', 'jet2_eta nominal', 'jet2_phi nominal', 'jet2_pt nominal', 'jet2_qgl nominal', 'jj_dEta nominal', 'jj_dPhi nominal', 'jj_eta nominal', 'jj_mass nominal', 'jj_mass_log nominal', 'jj_phi nominal', 'jj_pt nominal', 'll_zstar_log nominal', 'mmj1_dEta nominal', 'mmj1_dPhi nominal', 'mmj2_dEta nominal', 'mmj2_dPhi nominal', 'mmj_min_dEta nominal', 'mmj_min_dPhi nominal', 'mmjj_eta nominal', 'mmjj_mass nominal', 'mmjj_phi nominal', 'mmjj_pt nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt', 'mu2_pt_over_mass', 'zeppenfeld nominal']
#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet1_qgl_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal', 'jet2_qgl_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']

# training_features = ['dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld', 'njets']  # AN 19-124


# training_features = ['dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld', 'njets', "jet1_qgl", "jet2_qgl"] 

# features exactly matching BDT from AN
# training_features = [
#     'dimuon_pt', 
#     'dimuon_rapidity',
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'mu1_pt_over_mass', 
#     'mu1_eta', 
#     'mu2_pt_over_mass', 
#     'mu2_eta', 
#     'jet1_eta', 
#     'jet1_pt', 
#     'jet2_pt', 
#     'mmj1_dEta', 
#     'mmj1_dPhi',  
#     'jj_dEta', 
#     'jj_dPhi', 
#     'jj_mass', 
#     'zeppenfeld', 
#     'mmj_min_dEta', 
#     'mmj_min_dPhi', 
#     'njets'
# ]  # WgtON_original_AN_BDT_Sept27

# training_features = [
#     'dimuon_pt', 
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'mu1_pt_over_mass', 
#     'mu1_eta', 
#     'mu2_pt_over_mass', 
#     'mu2_eta', 
#     'jet1_eta', 
#     'jet1_pt', 
#     'jet2_pt', 
#     'mmj1_dEta', 
#     'mmj1_dPhi',  
#     'jj_dEta', 
#     'jj_dPhi', 
#     'jj_mass', 
#     'zeppenfeld', 
#     'mmj_min_dEta', 
#     'mmj_min_dPhi', 
#     'njets'
# ]  # WgtON_original_AN_BDT_noDimuRap_Sept27



# training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal',  'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal'] 


# training_features = [
#     'dimuon_cos_theta_cs_pisa', 
#     'dimuon_eta', 
#     'dimuon_phi_cs_pisa', 
#     'dimuon_pt', 
#     'jet1_eta_nominal', 
#     'jet1_pt_nominal', 
#     'jet2_pt_nominal', 
#     'jj_dEta_nominal', 
#     'jj_dPhi_nominal', 
#     'jj_mass_nominal', 
#     'mmj1_dEta_nominal', 
#     'mmj1_dPhi_nominal',  
#     'mmj_min_dEta_nominal', 
#     'mmj_min_dPhi_nominal', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld_nominal',
#     'njets_nominal'
# ]
# PhiFixed_rereco_yun


# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     'jet1_pt', 
#     # 'jet2_pt', 
#     'jj_dEta', 
#     'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     # 'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     'njets'
# ]
# # V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeJetVar

# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     # 'jet1_pt', 
#     # 'jet2_pt', 
#     # 'jj_dEta', 
#     # 'jj_dPhi', 
#     # 'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     # 'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     # 'mu2_eta', 
#     # 'mu2_pt_over_mass', 
#     # 'zeppenfeld',
#     # 'njets'
# ]
# # #V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeAllJetVar

# training_features = [
#     'dimuon_pt', 
# ]
# #V2_UL_Mar30_2025_DyMiNNLOGghVbf_justDimuPt


# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     # 'jet1_pt', 
#     # 'jet2_pt', 
#     # 'jj_dEta', 
#     # 'jj_dPhi', 
#     # 'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     # 'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     # 'zeppenfeld',
#     # 'njets'
# ]
# # #V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar

# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     # 'jet1_pt', 
#     # 'jet2_pt', 
#     # 'jj_dEta', 
#     # 'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     # 'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     # 'njets'
# ]
# # V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass



# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     # 'jet1_pt', 
#     # 'jet2_pt', 
#     # 'jj_dEta', 
#     # 'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     'mmj_min_dEta', 
#     'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     'njets'
# ]
# # V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_DeltaVars

# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     # 'jet1_eta', 
#     # 'jet1_pt', 
#     # 'jet2_pt', 
#     # 'jj_dEta', 
#     # 'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     # 'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     'njets'
# ]
# # V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_Njets

# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     # "dimuon_cos_theta_eta",
#     # "dimuon_phi_eta",
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     'jet1_eta', 
#     'jet1_pt', 
#     'jet2_pt', 
#     'jj_dEta', 
#     'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     'mmj_min_dEta', 
#     'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     'njets'
# ]
# # V2 Jan 18 UL

# training_features = [
#     'dimuon_cos_theta_cs', 
#     'dimuon_phi_cs', 
#     'dimuon_rapidity', 
#     'dimuon_pt', 
#     'jet1_eta', 
#     'jet2_eta', 
#     'jet1_pt', 
#     'jet2_pt', 
#     'jj_dEta', 
#     'jj_dPhi', 
#     'jj_mass', 
#     # 'mmj1_dEta', 
#     # 'mmj1_dPhi',  
#     # 'mmj_min_dEta', 
#     'mmj_min_dPhi', 
#     'mu1_eta', 
#     'mu1_pt_over_mass', 
#     'mu2_eta', 
#     'mu2_pt_over_mass', 
#     'zeppenfeld',
#     'njets'
# ]
# # V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass


training_features = [
    'dimuon_cos_theta_cs', 
    'dimuon_phi_cs', 
    'dimuon_rapidity', 
    'dimuon_pt', 
    'jet1_eta', 
    'jet2_eta', 
    'jet1_pt', 
    'jet2_pt', 
    'jj_dEta', 
    'jj_dPhi', 
    'jj_mass', 
    # 'mmj1_dEta', 
    # 'mmj1_dPhi',  
    'mmj_min_dEta', 
    'mmj_min_dPhi', 
    'mu1_eta', 
    'mu1_pt_over_mass', 
    'mu2_eta', 
    'mu2_pt_over_mass', 
    'zeppenfeld',
    'njets',
    'year',
]
# V2_UL_Apr09_2025_DyTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt0_75


#---------------------------------------------------------------------------

training_samples = {
        "background": [
            "dy_M-100To200", 
            "dy_M-100To200_MiNNLO",
            # "dy_m105_160_amc",
            # "dy_m100_200_UL",
            "ttjets_dl",
            "ttjets_sl",
            "st_tw_top",
            "st_tw_antitop",
            # "st_t_top",
            # "st_t_antitop",
            "ww_2l2nu",
            "wz_1l1nu2q",
            "wz_2l2q",
            "wz_3lnu",
            "zz",
            # "www",
            # "wwz",
            # "wzz",
            # "zzz",
            "ewk_lljj_mll50_mjj120",
        ],
        "signal": [
            "ggh_powhegPS", 
            "vbf_powheg_dipole", # adding vbf only makes BDT concentrate on vbf for some reason
        ], # copperheadV2
        # ],
        
        #"ignore": [
        #    "tttj",
        #    "tttt",
        #    "tttw",
        #    "ttwj",
        #    "ttww",
        #   "ttz",
        #    "st_s",
        #    "st_t_antitop",
        #    "st_tw_top",
        #    "st_tw_antitop",
        #    "zz_2l2q",
        #],
    }

def convert2df(dak_zip, dataset: str, is_vbf=False, is_UL=False):
    """
    small wrapper that takes delayed dask awkward zip and converts them to pandas dataframe
    with zip's keys as columns with extra column "dataset" to be named the string value given
    Note: we also filter out regions not in h-peak region first, and apply cuts for
    ggH production mode
    """
    # filter out arrays not in h_peak
    train_region = (dak_zip.dimuon_mass > 115.0) & (dak_zip.dimuon_mass < 135.0) # line 1169 of the AN: when training, we apply a tigher cut
    train_region = ak.fill_none(train_region, value=False)
    # print(f"is_vbf: {is_vbf}")
    # print(f"convert2df train_region:{train_region}")
    # not entirely sure if this is what we use for ROC curve, however

    vbf_cut = ak.fill_none(dak_zip.jj_mass_nominal > 400, value=False) & ak.fill_none(dak_zip.jj_dEta_nominal > 2.5, value=False) # for ggH $ VBF
    jet1_cut =  ak.fill_none((dak_zip.jet1_pt_nominal > 35), value=False) 
    
    if is_vbf: # VBF
        prod_cat_cut =  vbf_cut & jet1_cut
    else: # ggH
        prod_cat_cut =  ~(vbf_cut & jet1_cut)
        print("ggH cat!")

    # btag_cut = ak.fill_none((dak_zip.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((dak_zip.nBtagMedium_nominal >= 1), value=False)
    btagLoose_filter = ak.fill_none((dak_zip.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((dak_zip.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((dak_zip.njets_nominal >= 2), value=False)
    btag_cut = btagLoose_filter | btagMedium_filter
   
    category_selection = (
        prod_cat_cut & 
        train_region &
        ~btag_cut # btag cut is for VH and ttH categories
    )
    print(f"category_selection sum: {ak.sum(category_selection)}")
    # computed_zip = dak_zip[category_selection].compute() # original
    computed_zip = dak_zip[category_selection]

    # recalculate BDT variables that you're not certain is up to date from stage 1 start -----------------
    # if is_UL:
    #     # copperheadV2
    #     min_dEta_filter  = ak.fill_none((computed_zip.mmj1_dEta < computed_zip.mmj2_dEta), value=True)
    #     computed_zip["mmj_min_dEta"]  = ak.where(
    #         min_dEta_filter,
    #         computed_zip.mmj1_dEta,
    #         computed_zip.mmj2_dEta,
    #     )
    #     min_dPhi_filter = ak.fill_none((computed_zip.mmj1_dPhi < computed_zip.mmj2_dPhi), value=True)
    #     computed_zip["mmj_min_dPhi"] = ak.where(
    #         min_dPhi_filter,
    #         computed_zip.mmj1_dPhi,
    #         computed_zip.mmj2_dPhi,
    #     )
    # else:
    #     # copperheadV1
    #     min_dEta_filter  = ak.fill_none((computed_zip.mmj1_dEta_nominal < computed_zip.mmj2_dEta_nominal), value=True)
    #     computed_zip["mmj_min_dEta_nominal"]  = ak.where(
    #         min_dEta_filter,
    #         computed_zip.mmj1_dEta_nominal,
    #         computed_zip.mmj2_dEta_nominal,
    #     )
    #     min_dPhi_filter = ak.fill_none((computed_zip.mmj1_dPhi_nominal < computed_zip.mmj2_dPhi_nominal), value=True)
    #     computed_zip["mmj_min_dPhi_nominal"] = ak.where(
    #         min_dPhi_filter,
    #         computed_zip.mmj1_dPhi_nominal,
    #         computed_zip.mmj2_dPhi_nominal,
    #     )
    # recalculate BDT variables that you're not certain is up to date from stage 1 end -----------------
    
    print(f"computed_zip : {computed_zip}")
    # for copperheadV1, you gotta fill none b4 and store them in a dictionary b4 converting to dataframe
    computed_dict = {}
    nan_val = -999.0
    for field in computed_zip.fields:
        # print(f"field: {field}")
        # print(f"computed_dict[{field}] b4 fill none: {ak.to_dataframe(computed_zip[field]) }")
        
        if "dPhi" in field:
            computed_dict[field] = ak.fill_none(computed_zip[field], value=nan_val)
        else:
            computed_dict[field] = ak.fill_none(computed_zip[field], value=nan_val)
        # print(f"computed_dict[{field}] : {computed_dict[field]}")
        
    # # recalculate pt over masses. They're all inf and zeros for copperheadV1
    # computed_dict["mu1_pt_over_mass"] = computed_dict["mu1_pt"] / computed_dict["dimuon_mass"]
    # computed_dict["mu2_pt_over_mass"] = computed_dict["mu2_pt"] / computed_dict["dimuon_mass"]
    
    # mu1_pt_over_mass = computed_dict["mu1_pt_over_mass"]
    # mu2_pt_over_mass = computed_dict["mu2_pt_over_mass"]
    # df = ak.to_dataframe(computed_zip) 
    df = pd.DataFrame(computed_dict)
    print(f"df : {df.head()}")

    # make sure to replace nans with zeros,  unless it's delta phis, in which case it's -1, as specified in line 1117 of the AN
    dPhis = [] # collect all dPhi features
    for field in df.columns:
        if "dPhi" in field:
            dPhis.append(field)
    df.fillna({field: nan_val for field in dPhis},inplace=True)
    df.fillna(nan_val,inplace=True)
    # add columns
    df["dataset"] = dataset 
    df["cls_avg_wgt"] = -1.0
    # if is_UL:
    #     df["wgt_nominal"] = np.abs(df["wgt_nominal_total"])
    # else:
    df["wgt_nominal_orig"] = copy.deepcopy(df["wgt_nominal"])
    df["wgt_nominal"] = np.abs(df["wgt_nominal"])
    # df["wgt_nominal_total"] = np.abs(df["wgt_nominal_total"]) # enforce poisitive weights OR:
    # # drop negative values
    # if "wgt_nominal" in df.columns:
    #     df["wgt_nominal_total"] = df["wgt_nominal"] 
    # positive_wgts = df["wgt_nominal_total"] > 0 
    # df = df.loc[positive_wgts]
    print(f"df.dataset.unique(): {df.dataset.unique()}")
    return df

def normalizeBdtWgt(df, sig_datasets):
    cols = ["dataset", "bdt_wgt", "dimuon_ebe_mass_res"]#debug
    # print(f"df b4: {df[cols]}")
    # Overwrite bdt_wgt for signals with wgt_nominal_orig
    df.loc[df["dataset"].isin(sig_datasets), "bdt_wgt"] = df["wgt_nominal_orig"]

    # Normalize only signal rows so to match the sum of bkg events
    mask = df["dataset"].isin(sig_datasets)
    print(f"df sig b4: {df.loc[mask, cols]}")
    print(f"df bkg b4: {df.loc[~mask, cols]}")
    sig_wgt_sum = df.loc[mask, "bdt_wgt"].sum()
    bkg_wgt_sum = df.loc[~mask, "wgt_nominal_orig"].sum()
    # normalize signal
    df.loc[mask, "bdt_wgt"] = df.loc[mask, "bdt_wgt"] * 1/sig_wgt_sum
    # normalize bkg
    df.loc[~mask, "bdt_wgt"] = df.loc[~mask, "bdt_wgt"] * 1/bkg_wgt_sum
    print(f"sig_wgt_sum: {sig_wgt_sum}")
    print(f"bkg_wgt_sum: {bkg_wgt_sum}")
    # print(f"df after: {df[cols]}")
    print(f"df sig after: {df.loc[mask, cols]}")
    print(f"df bkg after: {df.loc[~mask, cols]}")
    # raise ValueError
    return df

def prepare_dataset(df, ds_dict):
    # Convert dictionary of datasets to a more useful dataframe
    df_info = pd.DataFrame()
    train_samples = []
    for icls, (cls, ds_list) in enumerate(ds_dict.items()):
        for ds in ds_list:
            df_info.loc[ds, "dataset"] = ds
            df_info.loc[ds, "iclass"] = -1
            if cls != "ignore":
                train_samples.append(ds)
                df_info.loc[ds, "class_name"] = cls
                df_info.loc[ds, "iclass"] = icls
    nan_val = -999.0
    df_info["iclass"] = df_info["iclass"].fillna(nan_val).astype(int)
    df = df[df.dataset.isin(df_info.dataset.unique())]
    
    # Assign numerical classes to each event
    cls_map = dict(df_info[["dataset", "iclass"]].values)
    print(f"cls_map: {cls_map}")
    cls_name_map = dict(df_info[["dataset", "class_name"]].values)
    df["class"] = df.dataset.map(cls_map)
    df["class_name"] = df.dataset.map(cls_name_map)
    # df.loc[:,'mu1_pt_over_mass'] = np.divide(df['mu1_pt'], df['dimuon_mass'])
    # df.loc[:,'mu2_pt_over_mass'] = np.divide(df['mu2_pt'], df['dimuon_mass'])
    # df[df['njets']<2]['jj_dPhi'] = -1
    #df[df['dataset']=="ggh_amcPS"].loc[:,'wgt_nominal_total'] = np.divide(df[df['dataset']=="ggh_amcPS"]['wgt_nominal_total'], df[df['dataset']=="ggh_amcPS"]['dimuon_ebe_mass_res'])

    
    # --------------------------------------------------------
    # df["wgt_nominal_orig"] = copy.deepcopy(df["wgt_nominal"])
    # multiply by dimuon mass resolutions if signal
    # sig_datasets = training_samples["signal"]
    # sig_datasets = ["ggh_amcPS"]
    sig_datasets = ["ggh_powhegPS", "vbf_powheg_dipole"]
    bkg_datasets = ["dy_M-100To200_MiNNLO",]
    print(f"df.dataset.unique(): {df.dataset.unique()}")
    df['bdt_wgt'] = (df['wgt_nominal_orig'])
    df = normalizeBdtWgt(df, sig_datasets)

    print(f"df any neg wgt: {np.any(df['wgt_nominal_orig']<0)}")

    # # debugging 
    cols = ['dataset', 'bdt_wgt', 'dimuon_ebe_mass_res',]
    print(f"df[cols] b4: {df[cols]}")
    # sig
    for dataset in sig_datasets:
        ebe_factor = 1
        df.loc[df['dataset']==dataset,'bdt_wgt'] = df.loc[df['dataset']==dataset,'bdt_wgt'] * ebe_factor*(1 / df[df['dataset']==dataset]['dimuon_ebe_mass_res'])
    # bkg
    # for dataset in bkg_datasets:
    #     ebe_factor = 1
    #     df.loc[df['dataset']==dataset,'bdt_wgt'] = df.loc[df['dataset']==dataset,'bdt_wgt'] * ebe_factor*(1 / df[df['dataset']==dataset]['dimuon_ebe_mass_res']) # FIXME
    # original end -----------------------------------------------
    print(f"df[cols] after: {df[cols]}")

    # -------------------------------------------------
    # normalize sig dataset again to one
    # -------------------------------------------------
    mask = df["dataset"].isin(sig_datasets)
    sig_wgt_sum = np.sum(df.loc[mask, "bdt_wgt"])
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {sig_wgt_sum}')
    df.loc[mask, "bdt_wgt"] = df.loc[mask, "bdt_wgt"] / sig_wgt_sum

    print(f"df[cols] after normalization: {df[cols]}")
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {sig_wgt_sum}')
    print(f'new np.sum(df.loc[mask, "bdt_wgt"]): {np.sum(df.loc[mask, "bdt_wgt"])}')


    # -------------------------------------------------
    # normalize bkg dataset again to one
    # -------------------------------------------------
    mask = ~df["dataset"].isin(sig_datasets)
    bkg_wgt_sum = np.sum(df.loc[mask, "bdt_wgt"])
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {bkg_wgt_sum}')
    df.loc[mask, "bdt_wgt"] = df.loc[mask, "bdt_wgt"] / bkg_wgt_sum

    print(f"df[cols] after bkg normalization: {df[cols]}")
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {bkg_wgt_sum}')
    print(f'new np.sum(df.loc[mask, "bdt_wgt"]): {np.sum(df.loc[mask, "bdt_wgt"])}')

    # -------------------------------------------------
    # increase bdt wgts for bdt to actually learn
    # -------------------------------------------------
    # df['bdt_wgt'] = df['bdt_wgt'] * 10_000
    df['bdt_wgt'] = df['bdt_wgt'] * 100_000
    print(f"df[cols] after increase in value: {df[cols]}")
    #print(df.head)
    columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta',]
    columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta','jet2_pt']
    columns2 = ['mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 'mmj_min_dEta', 'mmj_min_dPhi']
    # with open("df.txt", "w") as f:
    #     print(df[columns_print], file=f)
    # with open("df2.txt", "w") as f:
    #     print(df[columns2], file=f)
    # print(df[df['dataset']=="ggh_powheg"].head)
    # print(f"prepare_dataset df: {df["dataset","class"]}")
    return df

def scale_data_withweight(inputs, x_train, x_val, x_eval, df_train, fold_label):
    """
    NOTE: Scaling is not used any more since BDTs don't need it
    """
    # scale data, save the mean and std. This has to be done b4 mixup

    wgt_train = df_train["wgt_nominal_orig"].values
    x_mean = np.average(x_train,axis=0, weights=wgt_train)
    x_std = weighted_std(x_train, wgt_train)

    print(f"x_mean: {x_mean}")
    print(f"x_std: {x_std}")
    
    training_data = (x_train[inputs]-x_mean)/x_std
    validation_data = (x_val[inputs]-x_mean)/x_std
    evaluation_data = (x_eval[inputs]-x_mean)/x_std
    output_path = args["output_path"]
    print(f"scalar output_path: {output_path}/scalers_{name}_{fold_label}")
    print(f"name: {name}")
    save_path = f'{output_path}/bdt_{name}_{year}'
    print(f"scalar save_path: {save_path}/scalers_{name}_{fold_label}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + f"/scalers_{name}_{fold_label}", [x_mean, x_std]) 
    return training_data, validation_data, evaluation_data

def removeStrInColumn(df, str2remove):
    """
    helper function that removes str2remove in df columns if they exist
    """
    new_df = df.rename(
        columns=lambda c: c.replace(str2remove, "") if str2remove in c else c
    )
    return new_df

def getCorrMatrix(df, training_features, save_path=""):
    plt.style.use('default')
    corr_features = training_features + ["dimuon_mass"]
    corr_df = df[corr_features]
    corr_df = removeStrInColumn(corr_df, "_nominal")
    corr_matrix = corr_df.corr() 

    if save_path != "": # save as csv and heatmap
        corr_matrix.to_csv(f"{save_path}/correlation_matrix.csv")
        # corr_matrix = corr_matrix.round(2) # round to 2 d.p.
        
        corr = corr_matrix # NOTE: do this instead when loading from csv: corr = corr_matrix.set_index(corr_matrix.columns[0]).astype(float) 
        heatmap = sb.heatmap(corr, fmt=".2f", cmap="coolwarm", annot=True, annot_kws={"fontsize": 12})
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 12)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 45, fontsize = 12)
        
        plt.title("BDT input features + dimuon mass correlation matrix")
        # plt.tight_layout()
        fig = heatmap.get_figure()
        fig.set_size_inches(16,12)
        fig.savefig(f"{save_path}/correlation_matrix.pdf", bbox_inches='tight', pad_inches=0)

    plt.style.use(hep.style.CMS)
    # raise ValueError
    return corr_matrix

def classifier_train(df, args, training_samples, random_seed_val: int):
    print(f"random_seed_val: {random_seed_val}")
    if args['dnn']:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
        from tensorflow.keras import backend as K
    if args['bdt']:
        import xgboost as xgb
        from xgboost import XGBClassifier
        import pickle

    nfolds = 4
    # classes = df.dataset.unique()
    #print(df["class"])
    #cls_idx_map = {dataset:idx for idx,dataset in enumerate(classes)}
    add_year = (args['year']=='')
    #df = prepare_features(df, args, add_year)
    #df['cls_idx'] = df['dataset'].map(cls_idx_map)
    print("Training features: ", training_features)

    save_path = f"output/bdt_{name}_{year}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save training features as json for readability
    with open(f'{save_path}/training_features.json', 'w') as file:
        json.dump(training_features, file)
    # get the overal correlation matrix
    corr_matrix = getCorrMatrix(df, training_features, save_path=save_path)

    
    
    for i in range(nfolds):
        if args['year']=='':
            label = f"allyears_{args['label']}_{i}"
        else:
            label = f"{args['year']}_{args['label']}{i}"
        
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Train classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print(f"Samples used: ",df.dataset.unique())
        
        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        
        other_columns = ['event']
        
        df_train = df[train_filter]
        df_val = df[val_filter]
        df_eval = df[eval_filter]

        # # annhilate neg wgts
        df_train = PairNAnnhilateNegWgt(df_train)
        
        x_train = df_train[training_features]
        #y_train = df_train['cls_idx']
        y_train = df_train['class']
        x_val = df_val[training_features]
        x_eval = df_eval[training_features]
        #y_val = df_val['cls_idx']
        y_val = df_val['class']
        y_eval = df_eval['class']

        # print(f"y_train: {y_train}")
        # print(f"y_val: {y_val}")
        # print(f"y_eval: {y_eval}")
        # print(f"df_train: {df_train.head()}")
        
        # original start -------------------------------------------------------
        classes = {
            0 : 'background',
            1 : 'signal',
        }
        print(f"classes: {classes}")
        # for icls, cls in enumerate(classes):
        
        for icls, cls in classes.items():
            print(f"icls: {icls}")
            train_evts = len(y_train[y_train==icls])
            df_train.loc[y_train==icls,'cls_avg_wgt'] = df_train.loc[y_train==icls,'wgt_nominal'].values.mean()
            df_val.loc[y_val==icls,'cls_avg_wgt'] = df_val.loc[y_val==icls,'wgt_nominal'].values.mean()
            df_eval.loc[y_eval==icls,'cls_avg_wgt'] = df_eval.loc[y_eval==icls,'wgt_nominal'].values.mean()
            # df_train.loc[y_train==icls,'cls_avg_wgt'] = df_train.loc[y_train==icls,'wgt_nominal'].values.sum()
            # df_val.loc[y_val==icls,'cls_avg_wgt'] = df_val.loc[y_val==icls,'wgt_nominal'].values.sum()
            # df_eval.loc[y_eval==icls,'cls_avg_wgt'] = df_eval.loc[y_eval==icls,'wgt_nominal'].values.sum()
            print(f"{train_evts} training events in class {cls}")
        # original end -------------------------------------------------------
        # test start -------------------------------------------------------
        # bkg_l = training_samples["background"]
        # sig_l = training_samples["signal"]
        
        # df_train.loc[filter, "training_wgt"] = df_train.loc[filter,"training_wgt"]/df_train.loc[filter,"cls_avg_wgt"]
        # test end -------------------------------------------------------
        
        # df_train['training_wgt'] = df_train['wgt_nominal_total']/df_train['cls_avg_wgt']
        # df_val['training_wgt'] = df_val['wgt_nominal_total']/df_val['cls_avg_wgt']
        # df_eval['training_wgt'] = df_eval['wgt_nominal_total']/df_eval['cls_avg_wgt']
        # df_train.loc[:,'training_wgt'] = df_train['training_wgt']/df_train['cls_avg_wgt']
        # df_val.loc[:,'training_wgt'] = df_val['training_wgt']/df_val['cls_avg_wgt']
        # df_eval.loc[:,'training_wgt'] = df_eval['training_wgt']/df_eval['cls_avg_wgt']


        # df_train['training_wgt'] = df_train['wgt_nominal']/df_train['cls_avg_wgt']
        # df_val['training_wgt'] = df_val['wgt_nominal']/df_val['cls_avg_wgt']
        # df_eval['training_wgt'] = df_eval['wgt_nominal']/df_eval['cls_avg_wgt']

        
        # df_train['training_wgt'] = np.ones_like(df_train['wgt_nominal'])
        # df_val['training_wgt'] = np.ones_like(df_val['wgt_nominal'])
        # df_eval['training_wgt'] = np.ones_like(df_eval['wgt_nominal'])

        # V2_UL_Mar24_2025_DyTtStVvEwkGghVbf
        # df_train['training_wgt'] = np.ones_like(df_train['wgt_nominal']) / df_train['dimuon_ebe_mass_res']
        # df_val['training_wgt'] = np.ones_like(df_val['wgt_nominal']) / df_val['dimuon_ebe_mass_res']
        # df_eval['training_wgt'] = np.ones_like(df_eval['wgt_nominal']) / df_eval['dimuon_ebe_mass_res']

        # # V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight or V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_allOtherParamsOn
        # AN-19-124 line 1156: "the final BDTs have been trained by flipping the sign of negative weighted events"
        df_val['training_wgt'] = np.abs(df_val['wgt_nominal'])
        df_eval['training_wgt'] = np.abs(df_eval['wgt_nominal'])
        
        
        # scale data
        #x_train, x_val = scale_data(training_features, x_train, x_val, df_train, label)#Last used
        # x_train, x_val, x_eval = scale_data_withweight(training_features, x_train, x_val, x_eval, df_train, label)

        
        # print(f"x_train.shape: {x_train.shape}")
        # print(f"x_val.shape: {x_val.shape}")
        # print(f"x_train: {x_train}")
        # print(f"x_val: {x_val}")
        # print(f"x_train[training_features]: {x_train[training_features]}")
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]
        x_eval[other_columns] = df_eval[other_columns]

        # load model
        if args['dnn']:
            input_dim = len(training_features)
            inputs = Input(shape=(input_dim,), name = label+'_input')
            x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
            model.summary()

            #history = model.fit(x_train[training_features], y_train, epochs=100, batch_size=1024,\
            #                    sample_weight=df_train['training_wgt'].values, verbose=1,\
            #                    validation_data=(x_val[training_features], y_val, df_val['training_wgt'].values), shuffle=True)
            history = model.fit(x_train[training_features], y_train, epochs=10, batch_size=1024,
                                verbose=1,
                                validation_data=(x_val[training_features], y_val), shuffle=True)
            # util.save(history.history, f"output/trained_models/history_{label}_dnn.coffea")
            
            y_pred = model.predict(x_val[training_features]).ravel()
            nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_val, y_pred)
            auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
            #plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
            # roc_auc_gus = auc(nn_fpr_keras,nn_tpr_keras)
            fig, ax = plt.subplots(1,1)
            ax.plot(nn_fpr_keras, nn_tpr_keras, label='Raw ROC curve (area = %0.2f)' % auc_keras)
            #ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic example')
            ax.legend(loc="lower right")
            output_path = args["output_path"]
            print(f"output_path: {output_path}")
            print(f"name: {name}")
            # save_path = f'{output_path}/{name}_{year}/scalers_{name}_{label}'
            local_save_path = f"ouput/dnn_{name}_{year}"
            if not os.path.exists(local_save_path):
                os.makedirs(local_save_path)
            
            fig.savefig(f"{local_save_path}/test_{label}.png")

            
          
            model_save_path = f"{output_path}/dnn_{name}_{year}/"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model.save(f'{model_save_path}/{name}_{eval_label}.h5')        
        if args['bdt']:
            seed = 7
            xp_train = x_train[training_features].values
            xp_val = x_val[training_features].values
            xp_eval = x_eval[training_features].values
            y_train = y_train.values
            y_val = y_val.values
            y_eval = y_eval.values

            print(f"xp_train.shape: {xp_train.shape}")
            print(f"xp_val.shape: {xp_val.shape}")
            print(f"xp_eval.shape: {xp_eval.shape}")

            w_train = df_train['bdt_wgt'].values
            w_val = df_val['training_wgt'].values
            w_eval = df_eval['training_wgt'].values

            weight_nom_train = df_train['wgt_nominal_orig'].values
            weight_nom_val = df_val['wgt_nominal_orig'].values
            weight_nom_eval = df_eval['wgt_nominal_orig'].values

            # random_seed_val= 125 # M of Higgs as random seed
            
            np.random.seed(random_seed_val)
            
            shuf_ind_tr = np.arange(len(xp_train))
            np.random.shuffle(shuf_ind_tr)
            shuf_ind_val = np.arange(len(xp_val))
            np.random.shuffle(shuf_ind_val)
            shuf_ind_eval = np.arange(len(xp_eval))
            np.random.shuffle(shuf_ind_eval)
            xp_train = xp_train[shuf_ind_tr]
            xp_val = xp_val[shuf_ind_val]
            y_train = y_train[shuf_ind_tr]
            y_val = y_val[shuf_ind_val]

            
            xp_eval = xp_eval[shuf_ind_eval]
            y_eval = y_eval[shuf_ind_eval]

            weight_nom_train = weight_nom_train[shuf_ind_tr]
            weight_nom_val = weight_nom_val[shuf_ind_val]
            weight_nom_eval = weight_nom_eval[shuf_ind_eval]
            #print(np.isnan(xp_train).any())
            #print(np.isnan(y_train).any())
            #print(np.isinf(xp_train).any())
            #print(np.isinf(y_train).any())
            #print(np.isfinite(x_train).all())
            #print(np.isfinite(y_train).all())
            
            w_train = w_train[shuf_ind_tr]
            w_val = w_val[shuf_ind_val]

            # AN Model new start ---------------------------------------------------------------   
            verbosity=2
            
            # AN-19-124 p 45: "a correction factor is introduced to ensure that the same amount of background events are expected when either negative weighted events are discarded or they are considered with a positive weight"
            scale_pos_weight = float(np.sum(np.abs(weight_nom_train[y_train == 0]))) / np.sum(np.abs(weight_nom_train[y_train == 1])) 
            
            
            # V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_allOtherParamsOn
            # print(f"len(x_train): {len(x_train)}")
            print(f"len(x_train): {len(x_train)}")
            bdt_wgt = df_train["bdt_wgt"]
            scale_pos_weight = 0.7 # FIXME
            print(f"(scale_pos_weight): {(scale_pos_weight)}")
            model = XGBClassifier(
                n_estimators=1000,           # Number of trees
                max_depth=4,                 # Max depth
                learning_rate=0.10,          # Shrinkage
                subsample=0.5,               # Bagged sample fraction
                min_child_weight=0.03 ,  # NOTE: this causes AUC == 0.5
                tree_method='hist',          # Needed for max_bin
                max_bin=30,                  # Number of cuts
                # objective='binary:logistic', # CrossEntropy (logloss)
                # use_label_encoder=False,     # Optional: suppress warning
                eval_metric='logloss',       # Ensures logloss used during training
                n_jobs=30,                   # Use all CPU cores
                # scale_pos_weight=scale_pos_weight*0.005,
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=15,#15
                verbosity=verbosity,
                random_state=random_seed_val,
            )
            # AN Model new end ---------------------------------------------------------------
            
            print(model)
            print(f"negative w_train: {w_train[w_train <0]}")

            eval_set = [(xp_train, y_train), (xp_val, y_val)]#Last used
            
            model.fit(xp_train, y_train, sample_weight = w_train, eval_set=eval_set, verbose=False)

            y_pred_signal_val = model.predict_proba(xp_val)[:, 1].ravel()
            y_pred_signal_train = model.predict_proba(xp_train)[:, 1]
            y_pred_bkg_val = model.predict_proba(xp_val)[ :,0 ].ravel()
            y_pred_bkg_train = model.predict_proba(xp_train)[:,0]
            fig1, ax1 = plt.subplots(1,1)
            plt.hist(y_pred_signal_val, bins=50, alpha=0.5, color='blue', label='Validation Sig')
            plt.hist(y_pred_signal_train, bins=50, alpha=0.5, color='deepskyblue', label='Training Sig')
            plt.hist(y_pred_bkg_val, bins=50, alpha=0.5, color='red', label='Validation BKG')
            plt.hist(y_pred_bkg_train, bins=50, alpha=0.5, color='firebrick', label='Training BKG')

            ax1.legend(loc="upper right")
            
            fig1.savefig(f"{save_path}/Validation_{label}.png")
            
            y_pred = model.predict_proba(xp_val)[:, 1].ravel()
            y_pred_train = model.predict_proba(xp_train)[:, 1].ravel()
            y_eval_pred = model.predict_proba(xp_eval)[:, 1].ravel()
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            # print(f"y_pred: {y_pred}")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            # print(f"y_val: {y_val}")
            # original start ------------------------------------------------------------------------------
            nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_val.ravel(), y_pred, sample_weight=w_val) 
            # original end ------------------------------------------------------------------------------

            # test start ------------------------------------------------------------------------------
            # nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_val.ravel(), y_pred) # test
            # test end ------------------------------------------------------------------------------
            
            # np.save(f"test_roc_curve", [y_val.ravel(), y_pred]) 
            print(nn_fpr_xgb)
            print(nn_tpr_xgb)
            print(nn_thresholds_xgb)
            """
            for i in range(len(nn_fpr_xgb)-1):
                if(nn_fpr_xgb[i]>nn_fpr_xgb[i+1]):
                    print(i,nn_fpr_xgb[i])
                    print(i+1,nn_fpr_xgb[i+1])
                if(nn_tpr_xgb[i]>nn_tpr_xgb[i+1]):
                    print(i,nn_tpr_xgb[i])
                    print(i+1,nn_tpr_xgb[i+1])
            """
            sorted_index = np.argsort(nn_fpr_xgb)
            fpr_sorted =  np.array(nn_fpr_xgb)[sorted_index]
            tpr_sorted = np.array(nn_tpr_xgb)[sorted_index]
            #auc_xgb = auc(nn_fpr_xgb[:-2], nn_tpr_xgb[:-2])
            auc_xgb = auc(fpr_sorted, tpr_sorted)
            #auc_xgb = roc_auc_score(y_val, y_pred, sample_weight=w_val)
            print("The AUC score is:", auc_xgb)
            #plt.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='Neural Network (auc = %0.3f)' % auc_xgb)
            #roc_auc_gus = auc(nn_fpr_xgb,nn_tpr_xgb)
            fig, ax = plt.subplots(1,1)
            ax.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='BDT (auc = %0.3f)' % auc_xgb)
            #ax.plot(nn_fpr_xgb, nn_tpr_xgb, label='Raw ROC curve (area = %0.2f)' % roc_auc)
            #ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic')
            ax.legend(loc="lower right")
            fig.savefig(f"output/bdt_{name}_{year}/auc_{label}.png")
            plt.clf()

            # output shape dist start --------------------------------------------------------------------------
            fig, ax_main = plt.subplots()
            # Define custom bins from 0 to 1
            binning = np.linspace(0, 1, 31)  # 30 bins between 0 and 1

            # get the distributions
            is_bkg_train = y_train.ravel() == 0
            y_pred_train_bkg = y_pred_train[is_bkg_train]
            weight_nom_train_bkg = weight_nom_train[is_bkg_train]
            hist_train_bkg, _ = generate_normalized_histogram(y_pred_train_bkg, weight_nom_train_bkg, binning)

            is_sig_train = y_train.ravel() == 1
            y_pred_train_sig = y_pred_train[is_sig_train]
            weight_nom_train_sig = weight_nom_train[is_sig_train]
            hist_train_sig, _ = generate_normalized_histogram(y_pred_train_sig, weight_nom_train_sig, binning)
            
            is_bkg_val = y_val.ravel() == 0
            y_pred_val_bkg = y_pred[is_bkg_val]
            weight_nom_val_bkg = weight_nom_val[is_bkg_val]
            hist_val_bkg, _ = generate_normalized_histogram(y_pred_val_bkg, weight_nom_val_bkg, binning)

            is_sig_val = y_val.ravel() == 1
            y_pred_val_sig = y_pred[is_sig_val]
            weight_nom_val_sig = weight_nom_val[is_sig_val]
            hist_val_sig, _ = generate_normalized_histogram(y_pred_val_sig, weight_nom_val_sig, binning)

            is_bkg_eval = y_eval.ravel() == 0
            y_pred_eval_bkg = y_eval_pred[is_bkg_eval]
            weight_nom_eval_bkg = weight_nom_eval[is_bkg_eval]
            hist_eval_bkg, _ = generate_normalized_histogram(y_pred_eval_bkg, weight_nom_eval_bkg, binning)

            is_sig_eval = y_eval.ravel() == 1
            y_pred_eval_sig = y_eval_pred[is_sig_eval]
            weight_nom_eval_sig = weight_nom_eval[is_sig_eval]
            hist_eval_sig, _ = generate_normalized_histogram(y_pred_eval_sig, weight_nom_eval_sig, binning)
            
           
            hep.histplot(
                hist_train_bkg, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='blue', 
                label='Background train', 
                ax=ax_main,
            )
            hep.histplot(
                hist_val_bkg, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='green', 
                label='Background Validation', 
                ax=ax_main,
            )
            hep.histplot(
                hist_eval_bkg, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='blue', 
                label='Background Eval', 
                ax=ax_main,
            )
            hep.histplot(
                hist_train_sig, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='red', 
                label='Signal train', 
                ax=ax_main,
            )
            hep.histplot(
                hist_val_sig, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='blue', 
                label='Signal Validation', 
                ax=ax_main,
            )
            hep.histplot(
                hist_eval_sig, 
                bins=binning, 
                stack=False, 
                histtype='step', 
                # color='blue', 
                label='Signal Eval', 
                ax=ax_main,
            )
            
            
            # Add labels, title, and legend
            ax_main.set_xlabel('BDT Score')
            ax_main.set_ylabel('A.U.')
            ax_main.legend()
            ax_main.set_title('BDT Score distribution')
            
            fig.savefig(f"output/bdt_{name}_{year}/BDT_Score_{label}.png")
            ax_main.clear()
            plt.cla()
            plt.clf()

            # output shape dist end --------------------------------------------------------------------------

            
            # -------------------------------------------
            # GoF test
            # -------------------------------------------
            gof_save_path = f"output/bdt_{name}_{year}/"
            # print(f"weight_nom_val_sig: {weight_nom_val_sig}")
            # print(f"weight_nom_train_sig: {weight_nom_train_sig}")
            print(f"weight_nom_val_sig: {type(weight_nom_val_sig)}")
            print(f"weight_nom_train_sig: {type(weight_nom_train_sig)}")
            print(f"weight_nom_val_sig: {len(weight_nom_val_sig)}")
            print(f"weight_nom_train_sig: {len(weight_nom_train_sig)}")
            
            # we compare validation distribution with evaluation distribution to see if there's any over-training
            getGOF_KS_bdt(hist_eval_sig, hist_val_sig, weight_nom_val_sig, binning, gof_save_path, label)

            # -------------------------------------------
            # Log scale ROC curve
            # -------------------------------------------
            roc_data_dict = {
                "y_train": y_train.ravel(),
                "y_pred_train": y_pred_train,
                "weight_nom_train": weight_nom_train,
            
                "y_val": y_val.ravel(),
                "y_pred": y_pred,
                "weight_nom_val": weight_nom_val,
            
                "y_eval": y_eval.ravel(),
                "y_eval_pred": y_eval_pred,
                "weight_nom_eval": weight_nom_eval,
            }
            fullROC_operations(fig, roc_data_dict, name, year, label, doClassBalance=False)
            fullROC_operations(fig, roc_data_dict, name, year, label, doClassBalance=True)
            
            # # eff_bkg_train, eff_sig_train, thresholds_train, TpFpTnFn_df_train = customROC_curve_AN(y_train.ravel(), y_pred_train, weight_nom_train)
            # # eff_bkg_val, eff_sig_val, thresholds_val, TpFpTnFn_df_val = customROC_curve_AN(y_val.ravel(), y_pred, weight_nom_val)
            # # eff_bkg_eval, eff_sig_eval, thresholds_eval, TpFpTnFn_df_eval = customROC_curve_AN(y_eval.ravel(), y_eval_pred, weight_nom_eval)

            # eff_bkg_train, eff_sig_train, thresholds_train, _ = customROC_curve_AN(y_train.ravel(), y_pred_train, weight_nom_train, doClassBalance=True)
            # eff_bkg_val, eff_sig_val, thresholds_val, _ = customROC_curve_AN(y_val.ravel(), y_pred, weight_nom_val, doClassBalance=True)
            # eff_bkg_eval, eff_sig_eval, thresholds_eval, _ = customROC_curve_AN(y_eval.ravel(), y_eval_pred, weight_nom_eval, doClassBalance=True)

            # # deprecated -----------------------
            # # cols2merge = ["TP","FP","TN","FN"]
            # # TpFpTnFn_df_train = addErrByQuadrature(TpFpTnFn_df_train, columns=cols2merge)
            # # TpFpTnFn_df_val = addErrByQuadrature(TpFpTnFn_df_val, columns=cols2merge)
            # # TpFpTnFn_df_eval = addErrByQuadrature(TpFpTnFn_df_eval, columns=cols2merge)
            # # deprecated -----------------------
            

            # # -------------------------------------
            # # save ROC curve
            # # -------------------------------------
            # csv_savepath = f"output/bdt_{name}_{year}/rocEffs_{label}.csv"
            # roc_df = pd.DataFrame({
            #     "eff_sig_eval" : eff_sig_eval,
            #     "eff_bkg_eval" : eff_bkg_eval,
            #     "eff_sig_train" : eff_sig_train,
            #     "eff_bkg_train" : eff_bkg_train,
            #     "eff_sig_val" : eff_sig_val,
            #     "eff_bkg_val" : eff_bkg_val,
            # })
            # # roc_df = pd.concat([roc_df, TpFpTnFn_df_train, TpFpTnFn_df_val, TpFpTnFn_df_eval], axis=1)
            # roc_df.to_csv(csv_savepath)

            
            # auc_eval  = auc_from_eff(eff_sig_eval,  eff_bkg_eval)
            # auc_train = auc_from_eff(eff_sig_train, eff_bkg_train)
            # auc_val   = auc_from_eff(eff_sig_val,   eff_bkg_val)

            # # Calculate auc err using HM method
            # n_pos_eval = np.sum(y_eval.ravel() ==1)
            # n_neg_eval = np.sum(y_eval.ravel() !=1)
            # auc_err_eval = GetAucStdErrHanleyMcNeil(auc_eval, n_pos_eval, n_neg_eval)
            # n_pos_train = np.sum(y_train.ravel() ==1)
            # n_neg_train = np.sum(y_train.ravel() !=1)
            # auc_err_train = GetAucStdErrHanleyMcNeil(auc_train, n_pos_train, n_neg_train)
            # n_pos_val = np.sum(y_val.ravel() ==1)
            # n_neg_val = np.sum(y_val.ravel() !=1)
            # auc_err_val = GetAucStdErrHanleyMcNeil(auc_val, n_pos_val, n_neg_val)
            
            # print(f"auc_err_train: {auc_err_train}")

            # # -------------------------------------
            # # save auc and auc err
            # # -------------------------------------
            # csv_savepath = f"output/bdt_{name}_{year}/aucInfo_{label}.csv"
            # auc_df = pd.DataFrame({
            #     "auc_eval" : [auc_eval],
            #     "auc_err_eval" : [auc_err_eval],
            #     "auc_train" : [auc_train],
            #     "auc_err_train" : [auc_err_train],
            #     "auc_val" : [auc_val],
            #     "auc_err_val" : [auc_err_val],
            # })
            # # roc_df = pd.concat([roc_df, TpFpTnFn_df_train, TpFpTnFn_df_val, TpFpTnFn_df_eval], axis=1)
            # auc_df.to_csv(csv_savepath)

            
            # plt.plot(eff_sig_eval, eff_bkg_eval, label=f"ROC (Eval)  — AUC={auc_eval:.4f}+/-{auc_err_eval:.4f}")
            # plt.plot(eff_sig_val, eff_bkg_val, label=f"ROC (Val)   — AUC={auc_val:.4f}+/-{auc_err_val:.4f}")
            
            # # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
            # plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            # plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            # # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
            # plt.xlim([0.0, 1.0])
            # # plt.ylim([0.0, 1.0])
            # plt.xlabel('Signal eff')
            # plt.ylabel('Background eff')
            # plt.yscale("log")
            # plt.ylim([0.0001, 1.0])
            
            # plt.legend(loc="lower right")
            # plt.title(f'ROC curve for ggH BDT {year}')
            # fig.savefig(f"output/bdt_{name}_{year}/log_auc_{label}.pdf")

            
            # plt.plot(eff_sig_train, eff_bkg_train, label=f"ROC (Train) — AUC={auc_train:.4f}+/-{auc_err_train:.4f}")
            # plt.legend(loc="lower right")
            # fig.savefig(f"output/bdt_{name}_{year}/log_auc_{label}_w_train.pdf")
            
            # plt.clf()
            # superimposed log ROC end --------------------------------------------------------------------------

            # # superimposed flipped log ROC start --------------------------------------------------------------------------
            # plt.plot(1-eff_sig_eval,  1-eff_bkg_eval,  label=f"Stage2 ROC (Eval)  — AUC={auc_eval:.4f}+/-{auc_err_eval:.4f}")
            # plt.plot(1-eff_sig_val,   1-eff_bkg_val,   label=f"Stage2 ROC (Val)   — AUC={auc_val:.4f}+/-{auc_err_val:.4f}")

            
            # # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
            # plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            # plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            # # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
            # plt.xlim([0.0, 1.0])
            # # plt.ylim([0.0, 1.0])
            # plt.xlabel('1 - Signal eff')
            # plt.ylabel('1- Background eff')
            # plt.yscale("log")
            # plt.ylim([0.0001, 1.0])
            
            # plt.legend(loc="lower right")
            # plt.title(f'ROC curve for ggH BDT {year}')
            # fig.savefig(f"output/bdt_{name}_{year}/logFlip_auc_{label}.pdf")

            # plt.plot(1-eff_sig_train, 1-eff_bkg_train, label=f"Stage2 ROC (Train) — AUC={auc_train:.4f}+/-{auc_err_train:.4f}")
            # plt.legend(loc="lower right")
            # fig.savefig(f"output/bdt_{name}_{year}/logFlip_auc_{label}_w_train.pdf")
            
            # plt.clf()
            # superimposed flipped log ROC end --------------------------------------------------------------------------
            
            
            # do fig 6.5 start --------------------------------------------------------------
            save_path = f"output/bdt_{name}_{year}" 
            get6_5(y_eval.ravel(), y_eval_pred, weight_nom_eval, save_path, f"eval_{label}")
            get6_5(y_val.ravel(), y_pred, weight_nom_val, save_path, f"val_{label}")
            # do fig 6.5 end --------------------------------------------------------------
            
            # Also save ROC curve for evaluation just in case start --------------
            # shuf_ind_eval = np.arange(len(xp_eval))
            # xp_eval = xp_eval[shuf_ind_eval]
            # y_eval = y_eval[shuf_ind_eval]
            # y_eval_pred = model.predict_proba(xp_eval)[:, 1].ravel()
            nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_eval.ravel(), y_eval_pred)
            sorted_index = np.argsort(nn_fpr_xgb)
            fpr_sorted =  np.array(nn_fpr_xgb)[sorted_index]
            tpr_sorted = np.array(nn_tpr_xgb)[sorted_index]
            #auc_xgb = auc(nn_fpr_xgb[:-2], nn_tpr_xgb[:-2])
            auc_xgb = auc(fpr_sorted, tpr_sorted)
            #auc_xgb = roc_auc_score(y_val, y_pred, sample_weight=w_val)
            print("The AUC score is:", auc_xgb)
            #plt.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='Neural Network (auc = %0.3f)' % auc_xgb)
            #roc_auc_gus = auc(nn_fpr_xgb,nn_tpr_xgb)
            fig, ax = plt.subplots(1,1)
            ax.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='eval data BDT (auc = %0.3f)' % auc_xgb)
            #ax.plot(nn_fpr_xgb, nn_tpr_xgb, label='Raw ROC curve (area = %0.2f)' % roc_auc)
            #ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic')
            ax.legend(loc="lower right")
            fig.savefig(f"output/bdt_{name}_{year}/eval_auc_{label}.png")
            plt.clf()
            # Also save ROC curve for evaluation just in case end --------------

            
            # results = model.evals_result()
            # print(results.keys())
            # plt.plot(results['validation_0']['logloss'], label='train')
            # plt.plot(results['validation_1']['logloss'], label='test')
            # # show the legend
            # plt.legend()
            # plt.savefig(f"output/bdt_{name}_{year}/Loss_{label}.png")

            # -----------------------------
            # Retrieve evaluation results
            # -----------------------------
            results = model.evals_result()
            epochs = len(results["validation_0"]["logloss"])
            plot_x_axis = range(0, epochs)
            
            # -----------------------------
            # Plot training vs. validation loss
            # -----------------------------
            plt.clf()
            train_loss = results["validation_0"]["logloss"]
            val_loss = results["validation_1"]["logloss"]
            plt.plot(plot_x_axis, train_loss, label="Train Loss")
            plt.plot(plot_x_axis, val_loss, label="Validation Loss")

            plt.xlabel("Boosting Round")
            plt.ylabel("Log Loss")
            plt.title("XGBoost Training vs. Validation Loss")
            perf_text = plt.Line2D([], [], color='none', label=f'Best iteration: {model.best_iteration} \n Best validation loss: {model.best_score:.5f}')
            plt.legend(handles=[perf_text])
            plt.savefig(f"output/bdt_{name}_{year}/loss_{label}.png")
            
            # -----------------------------
            # 6. Check the best iteration
            # -----------------------------
            print(f"Best iteration: {model.best_iteration}")
            print(f"Best validation loss: {model.best_score:.5f}")
            
            csv_savepath = f"output/bdt_{name}_{year}/loss_{label}.csv"
            loss_df = pd.DataFrame({
                "epoch" : plot_x_axis,
                "train_loss" : train_loss,
                "val_loss" : val_loss,
            })
            loss_df.to_csv(csv_savepath)
            

            labels = [feat.replace("_nominal","") for feat in training_features]
            model.get_booster().feature_names = labels # set my training features as feature names
            
            # plot trees
            plot_tree(model)
            plt.savefig(f"output/bdt_{name}_{year}/{name}_{year}_TreePlot_{i}.png",dpi=400)
            
            # plot importance
            # plot_importance(model, importance_type='weight', xlabel="Score by weight",show_values=False)
            # plt.savefig(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byWeight.png")
            # plt.clf()
            # plot_importance(model, importance_type='gain', xlabel="Score by gain",show_values=False)
            # plt.savefig(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byGain.png")
            # plt.clf()
            
            # feature_important = model.get_booster().get_score(importance_type='gain')
            # feature_important = model.get_booster().get_score(importance_type='weight')
            # keys = list(feature_important.keys())
            # values = list(feature_important.values())
            # print(f"feat importance keys b4 sorting: {keys}")
            # print(f"feat importance value b4 sorting: {values}")
            # print(f"feat importance training_features b4 sorting: {training_features}")

            # # data = pd.DataFrame(data=values, index=training_features, columns=["score"]).sort_values(by = "score", ascending=True)
            # data = pd.DataFrame(data=values, index=training_features[:-2], columns=["score"]).sort_values(by = "score", ascending=True)
            # data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10))
            # plt.savefig(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}.png")
            # plt.clf()

            feature_important = model.get_booster().get_score(importance_type='weight')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            score_name = "Weight Score"
            data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
            data["Normalized Score"] = data[score_name] / data[score_name].sum()
            data.to_csv(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byWeight.csv")
            data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
            data.plot(kind='barh', figsize = (20,10))
            plt.savefig(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byWeight.png")
    
            feature_important = model.get_booster().get_score(importance_type='gain')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            score_name = "Gain Score"
            data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
            data["Normalized Score"] = data[score_name] / data[score_name].sum()
            data.to_csv(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byGain.csv")
            data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
            data.plot(kind='barh', figsize = (20,10))
            plt.savefig(f"output/bdt_{name}_{year}/BDT_FeatureImportance_{label}_byGain.png")

            
            #save('x_val_{label}.npy', x_val[training_features])
            #save('y_val_{label}.npy', y_val)
            #save('weight_val_{label}.npy', df_val['training_wgt'].values)
            output_path = args["output_path"]
            #util.save(history.history, f"output/trained_models/history_{label}_bdt.coffea")            
            save_path = f"{output_path}/bdt_{name}_{year}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_fname = (f"{save_path}/{name}_{label}.pkl")
            pickle.dump(model, open(model_fname, "wb"))
            print ("wrote model to",model_fname)
            

def evaluation(df, args):
    if df.shape[0]==0: return df
    if args['dnn']:
        import keras.backend as K
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count = {'CPU': 1})
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        sess = tf.compat.v1.Session(config=config)
    
        if args['do_massscan']:
            mass_shift = args['mass']-125.0
        add_year = args['evaluate_allyears_dnn']
        #df = prepare_features(df, args, add_year)
        df['dnn_score'] = 0
        with sess:
            nfolds = 4
            for i in range(nfolds):
                if args['evaluate_allyears_dnn']:
                    label = f"allyears_{args['label']}_{i}"
                else:
                    label = f"{args['year']}_{args['label']}{i}"

                train_folds = [(i+f)%nfolds for f in [0,1]]
                val_folds = [(i+f)%nfolds for f in [2]]
                eval_folds = [(i+f)%nfolds for f in [3]]
                
                eval_filter = df.event.mod(nfolds).isin(eval_folds)
                eval_label = f"{args['year']}_{args['label']}{eval_folds[0]}"
                print(f"eval_label: {eval_label}")
                # scalers_path = f'output/trained_models/scalers_{label}.npy'
                output_path = args["output_path"]
                print(f"output_path: {output_path}")
                # scalers_path = f"{output_path}/dnn_{name}_{year}/scalers_{name}_{eval_label}.npy"
                # scalers = np.load(scalers_path)
                # model_path = f'output/trained_models/test_{label}.h5'
                model_path = f'{output_path}/dnn_{name}_{year}/{name}_{eval_label}.h5'
                dnn_model = load_model(model_path)
                df_i = df[eval_filter]
                #if args['r']!='h-peak':
                #    df_i['dimuon_mass'] = 125.
                #if args['do_massscan']:
                #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift

                # df_i = (df_i[training_features]-scalers[0])/scalers[1]
                print(f"DNN evaluation df_i: {df_i}")
                prediction = np.array(dnn_model.predict(df_i[training_features])).ravel()
                df.loc[eval_filter,'dnn_score'] = np.arctanh((prediction))
    if args['bdt']:
        import xgboost as xgb
        import pickle
        if args['do_massscan']:
            mass_shift = args['mass']-125.0
        add_year = args['evaluate_allyears_dnn']
        #df = prepare_features(df, args, add_year)
        df['bdt_score'] = 0
        nfolds = 4
        for i in range(nfolds):    
            if args['evaluate_allyears_dnn']:
                label = f"allyears_{args['label']}_{i}"
            else:
                label = f"{args['year']}_{args['label']}{i}"
            
            
            train_folds = [(i+f)%nfolds for f in [0,1]]
            val_folds = [(i+f)%nfolds for f in [2]]
            eval_folds = [(i+f)%nfolds for f in [3]]
            
            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            # print(f"train_folds: {train_folds}")
            # print(f"val_folds: {val_folds}")
            # print(f"eval_folds: {eval_folds}")

            # eval_label = f"{args['year']}_{args['label']}{eval_folds[0]}"
            eval_label = f"{args['year']}_{args['label']}{i}"
            print(f"eval_label: {eval_label}")
            
            # scalers_path = f"{output_path}/{name}_{year}/scalers_{name}_{eval_label}.npy"
            # start_path = "/depot/cms/hmm/copperhead/trained_models/"
            output_path = args["output_path"]
            print(f"output_path: {output_path}")
            # scalers_path = f"{output_path}/bdt_{name}_{year}/scalers_{name}_{eval_label}.npy"

            # print(f"scalers_path: {scalers_path}")
            #scalers_path = f'output/trained_models_nest10000/scalers_{label}.npy'
            # scalers = np.load(scalers_path)

            model_path = f"{output_path}/bdt_{name}_{year}/{name}_{label}.pkl"
            #model_path = f'output/trained_models_nest10000/BDT_model_earlystop50_{label}.pkl'
            bdt_model = pickle.load(open(model_path, "rb"))
            print(f"bdt_model.classes_: {bdt_model.classes_}")
            df_i = df[eval_filter]
            #if args['r']!='h-peak':
            #    df_i['dimuon_mass'] = 125.
            #if args['do_massscan']:
            #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift
            #print("Scaler 0 ",scalers[0])
            #print("Scaler 1 ",scalers[1])
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",df_i.head())
            # df_i = (df_i[training_features]-scalers[0])/scalers[1]
            #print("************************************************************",df_i.head())
            prediction_sig = np.array(bdt_model.predict_proba(df_i.values)[:, 1]).ravel()
            prediction_bkg = np.array(bdt_model.predict_proba(df_i.values)[:, 0]).ravel()
            fig1, ax1 = plt.subplots(1,1)
            plt.hist(prediction_sig, bins=50, alpha=0.5, color='blue', label='Validation Sig')
            plt.hist(prediction_bkg, bins=50, alpha=0.5, color='red', label='Validation BKG')


            ax1.legend(loc="upper right")
            save_path = f"output/bdt_{name}_{year}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}/Validation_{label}.png")
            

            # df.loc[eval_filter,'bdt_score'] = prediction
    return df
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        default="test",
        action="store",
        help="Name of classifier",
    )
    parser.add_argument(
        "--vbf",
        dest="is_vbf",
        default=False, 
        action=argparse.BooleanOptionalAction,
        help="If true we filter out for VBF production mode category, if not, cut for ggH category",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/DNN_test2/",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    sysargs = parser.parse_args()
    year = sysargs.year
    name = sysargs.name
    args = {
        "dnn": False,
        "bdt": True,
        # "dnn": True,
        # "bdt": False,
        "year": year,
        "name": name,
        "do_massscan": False,
        "evaluate_allyears_dnn": False,
        "output_path": "/depot/cms/users/yun79/hmm/trained_MVAs",
        "label": ""
    }
    start_time = time.time()
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 

    
    if year == "2016":
        load_path = f"{sysargs.load_path}/{year}*/f1_0" # copperheadV2
    elif year == "all":
        load_path = f"{sysargs.load_path}/*/f1_0" # copperheadV2
    else:
        load_path = f"{sysargs.load_path}/{year}/f1_0" # copperheadV2
        # load_path = f"{sysargs.load_path}/{year}/" # copperheadV1
    print(f"load_path: {load_path}")
    sample_l = training_samples["background"] + training_samples["signal"]
    is_UL = True
    
    
    # if is_UL:
    #     fields2load = [ # copperheadV2
    #         "dimuon_mass",
    #         "jj_mass",
    #         "jj_dEta",
    #         "jet1_pt",
    #         "nBtagLoose",
    #         "nBtagMedium",
    #         "mmj1_dEta",
    #         "mmj2_dEta",
    #         "mmj1_dPhi",
    #         "mmj2_dPhi",
    #         "wgt_nominal_total",
    #         "dimuon_ebe_mass_res",
    #         "event",
    #         "mu1_pt",
    #         "mu2_pt",
    #     ]
    # else:
    #     fields2load = [ # copperheadV1
    #         "dimuon_mass",
    #         "jj_mass_nominal",
    #         "jj_dEta_nominal",
    #         "jet1_pt_nominal",
    #         "nBtagLoose_nominal",
    #         "nBtagMedium_nominal",
    #         "mmj1_dEta_nominal",
    #         "mmj2_dEta_nominal",
    #         "mmj1_dPhi_nominal",
    #         "mmj2_dPhi_nominal",
    #         "wgt_nominal",
    #         "dimuon_ebe_mass_res",
    #         "event",
    #         "mu1_pt",
    #         "mu2_pt",
    #     ]

    fields2load = [ 
            "dimuon_mass",
            "jj_mass_nominal",
            "jj_dEta_nominal",
            "jet1_pt_nominal",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            # "mmj1_dEta_nominal",
            # "mmj2_dEta_nominal",
            # "mmj1_dPhi_nominal",
            # "mmj2_dPhi_nominal",
            "wgt_nominal",
            "dimuon_ebe_mass_res",
            "event",
            "mu1_pt",
            "mu2_pt",
            "year",
        ]

    

    
    fields2load = list(set(fields2load + training_features)) # remove redundancies
    # load data to memory using compute()
    
    
    
    # df_total = pd.concat([df_ggh,df_dy],ignore_index=True)
    # old code end --------------------------------------------------------------------------------------------
    # new code start --------------------------------------------------------------------------------------------
    df_l = []
    print(f"sample_l: {sample_l}")
    print(f"training_features: {training_features}")
    print(f"load_path: {load_path}")
    for sample in sample_l:
        print(f"running sample {sample}")
        parquet_path = load_path+f"/{sample}/*/*.parquet"
        try:
            # zip_sample = dak.from_parquet(parquet_path) 
            filelist = glob.glob(parquet_path)
            print(f"filelist len: {len(filelist)}")
            if year == "all":
                # year_paths = {
                #     2015: f"{sysargs.load_path}/2016preVFP/f1_0/{sample}/*/*.parquet",
                #     2016: f"{sysargs.load_path}/2016postVFP/f1_0/{sample}/*/*.parquet",
                #     2017: f"{sysargs.load_path}/2017/f1_0/{sample}/*/*.parquet",
                #     2018: f"{sysargs.load_path}/2018/f1_0/{sample}/*/*.parquet",
                # }
                # print(parquet_path)
                # zip_sample =  ReadNMergeParquet_dak(year_paths, fields2load)
                zip_sample = dak.from_parquet(filelist)
                # print(f"zip_sample.fields: {zip_sample.fields}")
                # zip_sample["bdt_year"] = int(year)
                # fields2load = fields2load + ["bdt_year_nominal"]
                print(f"fields2load b4: {fields2load}")
                fields2load_prepared = prepare_features(zip_sample, fields2load) # add variation to the name
                print(f"fields2load after: {fields2load_prepared}")
                zip_sample = ak.zip({
                    field : zip_sample[field] for field in fields2load_prepared
                })
                zip_sample = zip_sample.compute()
            else:
                zip_sample = dak.from_parquet(filelist)
                # print(f"zip_sample.fields: {zip_sample.fields}")
                # zip_sample["bdt_year"] = int(year)
                # fields2load = fields2load + ["bdt_year_nominal"]
                print(f"fields2load b4: {fields2load}")
                fields2load_prepared = prepare_features(zip_sample, fields2load) # add variation to the name
                print(f"fields2load after: {fields2load_prepared}")
                zip_sample = ak.zip({
                    field : zip_sample[field] for field in fields2load_prepared
                })
                zip_sample = zip_sample.compute()
        except Exception as error:
            print(f"Parquet for {sample} not found with error {error}. skipping!")
            continue
        # zip_sample = dak.from_parquet(load_path+f"/{sample}/*.parquet") # copperheadV1
        # temporary introduction of mu_pt_over_mass variables. Some tt and top samples don't have them
        zip_sample["mu1_pt_over_mass"] = zip_sample["mu1_pt"] / zip_sample["dimuon_mass"]
        zip_sample["mu2_pt_over_mass"] = zip_sample["mu2_pt"] / zip_sample["dimuon_mass"]

        if "dy" in sample: # NOTE: not sure what this if statement is for
            wgts2load = []
            # for field in zip_sample.fields:
            #     if "wgt" in field:
            #         wgts2load.append(field)
            # print(f"wgts2load: {wgts2load}")
            fields2load = list(set(fields2load + wgts2load))
            fields2load = prepare_features(zip_sample, fields2load) # add variation to the name
            training_features = prepare_features(zip_sample, training_features) # do the same thing to training features
            zip_sample = ak.zip({
                field : zip_sample[field] for field in fields2load
            })#.compute()
            # wgts2deactivate = [
            #     # 'wgt_nominal_btag_wgt',
            #     # 'wgt_nominal_pu',
            #     'wgt_nominal_zpt_wgt',
            #     # 'wgt_nominal_muID',
            #     # 'wgt_nominal_muIso',
            #     # 'wgt_nominal_muTrig',
            #     # 'wgt_nominal_LHERen',
            #     # 'wgt_nominal_LHEFac',
            #     # 'wgt_nominal_pdf_2rms',
            #     # 'wgt_nominal_jetpuid_wgt',
            #     # 'wgt_nominal_qgl'
            # ]
            # wgt_nominal = zip_sample["wgt_nominal_total"]
            # print(f"wgt_nominal: {wgt_nominal}")
            # zip_sample["wgt_nominal_total"] = deactivateWgts(wgt_nominal, zip_sample, wgts2deactivate)
        else:
            fields2load = prepare_features(zip_sample, fields2load) # add variation to the name
            training_features = prepare_features(zip_sample, training_features) # do the same thing to training features
            zip_sample = ak.zip({
                field : zip_sample[field] for field in fields2load
            })#.compute()
        is_vbf = sysargs.is_vbf
        df_sample = convert2df(zip_sample, sample, is_vbf=is_vbf, is_UL=is_UL)
        df_l.append(df_sample)
    df_total = pd.concat(df_l,ignore_index=True)   
    # new code end --------------------------------------------------------------------------------------------
    # apply random shuffle, so that signal and bkg samples get mixed up well
    random_seed_val = 125
    df_total = df_total.sample(frac=1, random_state=random_seed_val)
    # print(f"df_total after shuffle: {df_total}")
    

    del df_l # delete redundant df to save memory. Not sure if this is necessary
    # print(f"df_total: {df_total}")
    print("starting prepare_dataset")
    df_total = prepare_dataset(df_total, training_samples)
    print("prepare_dataset done")
    # raise ValueError
    # print(f"len(df_total): {len(df_total)}")
    print(f"df_total.columns: {df_total.columns}")

    classifier_train(df_total, args, training_samples, random_seed_val)
    # evaluation(df_total, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    #print(df)
    runtime = int(time.time()-start_time)
    print(f"Success! run time is {runtime} seconds")
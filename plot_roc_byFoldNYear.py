"""
Plot ROC curves by fold for given model name and years.

Usage:
    python plot_roc_by_fold.py --model <model_base_name> --years 2016 2017 2018 all
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from modules.utils import plot6_5FromHist

plt.style.use(hep.style.CMS)


def auc_from_eff(eff_sig, eff_bkg):
    """
    Compute the area under the ROC curve from signal and background efficiencies.
    """
    fpr = 1.0 - np.asarray(eff_bkg)
    tpr = np.asarray(eff_sig)
    order = np.argsort(fpr)
    return np.trapezoid(tpr[order], fpr[order])


def PlotRocByFold(model_name, year, nfolds=4 ):
    trainValEval_l = ["train", "val", "eval"]
    for mode in trainValEval_l:
        for nfold in range(nfolds):
            csv_savepath = f"output/{model_name}_{year}/rocEffs_{year}_{nfold}.csv"
            roc_df = pd.read_csv(csv_savepath)
            eff_sig = roc_df[f"eff_sig_{mode}"]
            eff_bkg = roc_df[f"eff_bkg_{mode}"]
            auc  = auc_from_eff(eff_sig,  eff_bkg)
            csv_savepath = f"output/{model_name}_{year}/aucInfo_{year}_{nfold}.csv"
            auc_df = pd.read_csv(csv_savepath)
            assert(np.isclose(auc,auc_df[f"auc_{mode}"][0]))
            auc_err = auc_df[f"auc_err_{mode}"][0]
            plt.plot(eff_sig, eff_bkg, label=f"fold{nfold} ROC ({mode})   — AUC={auc:.4f}+/-{auc_err:.4f}")
            
            # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
            plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
            plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.0])
            plt.xlabel('Signal eff')
            plt.ylabel('Background eff')
            plt.yscale("log")
            plt.ylim([0.0001, 1.0])
            
            plt.legend(loc="lower right")
            plt.title(f'ROC curve for ggH BDT {year}')
        fig_savepath = f"output/{model_name}_{year}/RocByFold_{year}_{mode}.pdf"
        plt.savefig(fig_savepath)
        plt.clf()
    

def PlotRocByYear(base_model: str, years: list[str], nfolds: int = 4):
    """
    Combine ROC curves across years into one plot per fold.
    Searches for CSVs like:
      <base_model>_<year>/roc_data_<year>_fold*.csv
    and merges all years onto a single ROC per fold.
    """

    # Extract unique fold numbers
    folds = list(range(nfolds))

    trainValEval_l = ["train", "val", "eval"]
    
    # --- Loop over folds ---
    for mode in trainValEval_l:
        for fold_num in folds:
            for year in years:
                model_name = f"{base_model}_{year}"
                csv_path = f"output/{model_name}/rocEffs_{year}_{fold_num}.csv"
    
                roc_df = pd.read_csv(csv_path)
    
                eff_sig = roc_df[f"eff_sig_{mode}"]
                eff_bkg = roc_df[f"eff_bkg_{mode}"]
                
                auc  = auc_from_eff(eff_sig,  eff_bkg)
                csv_savepath = f"output/{model_name}/aucInfo_{year}_{fold_num}.csv"
                auc_df = pd.read_csv(csv_savepath)
                assert(np.isclose(auc,auc_df[f"auc_{mode}"][0]))
                auc_err = auc_df[f"auc_err_{mode}"][0]
                plt.plot(eff_sig, eff_bkg, label=f"YEAR {year} ROC ({mode})   — AUC={auc:.4f}+/-{auc_err:.4f}")

            # --- finalize plot ---
            
            plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            plt.xlim([0.0, 1.0])
            plt.xlabel('Signal eff')
            plt.ylabel('Background eff')
            plt.yscale("log")
            plt.ylim([0.0001, 1.0])
            
            plt.legend(loc="lower right")
            plt.title(f'ROC curve for ggH BDT fold {fold_num}')
            # hep.cms.label("Preliminary", data=True, lumi=138)
    
            # save_path = os.path.join(save_dir, f"roc_fold{fold_num}_allYears.pdf")
            # save the plot on each year's bdt save path
            for year in years:
                model_name = f"{base_model}_{year}"
                save_path = f"output/{model_name}/RocByYear_{mode}_{fold_num}.pdf"
                # save_path = f"test_fold{fold_num}.pdf"
                plt.savefig(save_path, bbox_inches="tight")
            plt.clf()

        print(f"[INFO] Saved combined ROC for fold {fold_num}: {save_path}")

def Plot6_5ByYear(base_model: str, years: list[str], nfolds: int = 4):
    """
    Combine ROC curves across years into one plot per fold.
    Searches for CSVs like:
      <base_model>_<year>/roc_data_<year>_fold*.csv
    and merges all years onto a single ROC per fold.
    """

    # Extract unique fold numbers
    folds = list(range(nfolds))

    trainValEval_l = ["val", "eval"]
    sigBkg_l = ["sig", "bkg", "both"]
    # --- Loop over folds ---
    for sigOrBkg in sigBkg_l:
        for mode in trainValEval_l:
            for fold_num in folds:
                hist_dict = {}
                for year in years:
                    model_name = f"{base_model}_{year}"
                    csv_path = f"output/{model_name}/6_5_{mode}_{year}_{fold_num}.csv"
        # 6_5_val_all_3.csv
                    df_6_5 = pd.read_csv(csv_path)
                    if sigOrBkg == "both":
                        hist_dict_year = {
                            f"Bkg {year} {mode} fold {fold_num}" : df_6_5["bkg_hist"],
                            f"Sig {year} {mode} fold {fold_num}" : df_6_5["sig_hist"],
                        }
                        save_fname = f"6_5ByYear_{mode}_{fold_num}"
                    else:
                        hist_dict_year = {
                            f"Bkg {year} {mode} fold {fold_num}" : df_6_5[f"{sigOrBkg}_hist"],
                        }
                        save_fname = f"6_5ByYear_{sigOrBkg}_{mode}_{fold_num}"
                    hist_dict.update(hist_dict_year)
                    bin_edges = np.concatenate([df_6_5['bin_left'], [df_6_5['bin_right'].iloc[-1]]]) # this is same among different years and folds
                    
    
                # plot and save the plot on each year's bdt save path
                for year in years:
                    model_name = f"{base_model}_{year}"
                    save_path = f"output/{model_name}"
                    if sigOrBkg == "both":
                        plot6_5FromHist(hist_dict, bin_edges, save_path, save_fname, unifiedColorScheme=True)
                    else:
                        plot6_5FromHist(hist_dict, bin_edges, save_path, save_fname)
                    
                plt.clf()
    
            print(f"[INFO] Saved fig 6.5 for {mode} fold {fold_num}: {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ROC curves by fold for given model name and list of years."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name (without year suffix)."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        help="List of years to process, e.g. 2016 2017 2018 all"
    )
    args = parser.parse_args()

    base_model = args.model
    model_name = f"bdt_{base_model}"
    years = args.years
    
    for year in years:
        print(f"[INFO] Processing model: {model_name}")
        PlotRocByFold(model_name, year)

    PlotRocByYear(model_name, years)
    Plot6_5ByYear(model_name, years)
    
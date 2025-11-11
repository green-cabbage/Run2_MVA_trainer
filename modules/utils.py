import pandas as pd
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def dimuMassPlot(df, save_path):
    # --- Define regions ---
    barrel_cut = 0.9  # threshold for barrel 
    endcap_cut_low = 1.8  # threshold for endcap
    endcap_cut_high = 2.4  # threshold for endcap
    
    barrel_barrel = df[(df["mu1_eta"].abs() < barrel_cut) & (df["mu2_eta"].abs() < barrel_cut)]
    endcap_endcap = df[
        (endcap_cut_low < df["mu1_eta"].abs()) &  
        (df["mu1_eta"].abs()  < endcap_cut_high) & 
        (endcap_cut_low < df["mu2_eta"].abs()) & 
        (df["mu2_eta"].abs() < endcap_cut_high)
         ]
    
    # --- Plot histograms ---
    plt.hist(barrel_barrel["dimuon_ebe_mass_res"]/barrel_barrel["dimuon_mass"], bins=40, alpha=0.6, label="Barrel-Barrel", histtype="step", density=True)
    plt.hist(endcap_endcap["dimuon_ebe_mass_res"]/endcap_endcap["dimuon_mass"], bins=40, alpha=0.6, label="Endcap-Endcap", histtype="step", density=True)

    plt.xlim(0, 0.05)
    plt.xlabel("EBE Dimuon mass resolution / dimuon mass")
    plt.ylabel("A.U.")
    plt.legend()
    plt.title("Dimuon mass resolution in BB and EE regions")
    plt.savefig(f"{save_path}/dimuMassBB_EE.pdf")


def dimuMassResScatterPlot(df, save_path):
    # Scatter plot
    # plt.figure(figsize=(6,4))
    plt.ylim(0, 10)
    plt.scatter(df["mu1_eta"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("mu1_eta")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs mu1_eta")
    plt.grid(True)
    plt.savefig(f"{save_path}/dimuResMu1EtaScatter.png")
    plt.clf()


    # mu2 eta
    plt.ylim(0, 10)
    plt.scatter(df["mu2_eta"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("mu2_eta")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs mu2_eta")
    plt.savefig(f"{save_path}/dimuResMu2EtaScatter.png")
    plt.clf()
    
    
    # dimuon rap
    plt.ylim(0, 10)
    plt.scatter(df["dimuon_rapidity"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("dimuon_rapidity")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs dimuon_rapidity")
    plt.savefig(f"{save_path}/dimuResDimuRapScatter.png")
    plt.clf()

    # bdt wgt
    # plt.ylim(0, 10)
    dataset_l = df["dataset"].unique()
    for dataset in dataset_l:
        subset = df[df["dataset"] == dataset]
        plt.scatter(subset["dimuon_mass"], subset["bdt_wgt"], alpha=0.05, label=dataset)
        plt.legend()
        plt.xlabel("dimuon_mass")
        plt.ylabel("bdt_wgt")
        plt.title("Scatter plot: dimuon_mass_res vs dimuon_rapidity")
        plt.savefig(f"{save_path}/dimuMBdtWgtScatter_{dataset}.png")
        plt.clf()
    
    
def plotBdtWgt(df, save_path):
    # --- Mass window cut ---
    mass_bins_edges = [115, 120, 125, 130, 135]
    dataset_l = df["dataset"].unique()
    for low, high in zip(mass_bins_edges[:-1], mass_bins_edges[1:]):
        mass_mask = (df["dimuon_mass"] > low) & (df["dimuon_mass"] < high)
        df_sel = df[mass_mask]
        # wgt_name = 'wgt_nominal_orig'
        wgt_name = 'bdt_wgt'
        print(f"any neg wgt: {np.any((df['bdt_wgt'] <0))}")
        print(f"any neg wgt wgt_nominal: {np.any((df['wgt_nominal_orig'] <0))}")
        # ===================================================
        # 1) Overlay all datasets in a single plot
        # ===================================================
        # plt.figure(figsize=(6,4))
        
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            plt.hist(
                subset[wgt_name],
                bins=30,
                histtype="step",
                label=dataset
            )

        plt.xlim(-7e-6, 7e-6)
        
        plt.xlabel(wgt_name)
        plt.ylabel("Counts")
        plt.title(f"wgt_name distribution per dataset\n{low} $ < mMuMu < $ {high}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.show()
        plt.savefig(f"{save_path}/bdtWgts{low}to{high}.pdf")
        plt.clf()

        # --------------------------------------------------------------
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            plt.hist(
                subset[wgt_name]/subset["dimuon_ebe_mass_res"],
                bins=30,
                histtype="step",
                label=dataset
            )
        
        plt.xlabel(wgt_name)
        plt.ylabel("Counts")
        plt.title(f"wgt_name distribution per dataset\n{low} $ < mMuMu < $ {high}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.show()
        plt.savefig(f"{save_path}/bdtWgtsDivDimuonMassRes{low}to{high}.pdf")
        plt.clf()
        
        
        # ===================================================
        # 2) One separate figure per dataset
        # ===================================================
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            # plt.figure(figsize=(6,4))
            plt.hist(
                subset[wgt_name],
                bins=30,
                histtype="step",
                color="blue",
                label=dataset
            )
            plt.xlabel(wgt_name)
            plt.ylabel("Counts")
            plt.title(f"{wgt_name} distribution for {dataset}\n{low} $ < mMuMu < $ {high}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            # plt.show()
            plt.savefig(f"{save_path}/{dataset}_bdtWgts{low}to{high}.pdf")
            plt.clf()

def pair_and_remove(df, cols=("mu1_eta","mu2_eta"), wgt_col="wgt_nominal"):
    """
    Pair each negative-weight row with at most one positive-weight row
    using Hungarian algorithm (minimizing L1 distance).
    Remove the paired rows from the original df.

    Returns:
        matches_df: DataFrame with match info (neg_idx, pos_idx, dist, ...)
        remaining_df: df with matched rows removed
    """    
    # Split
    neg = df[df[wgt_col] < 0].copy()
    pos = df[df[wgt_col] > 0].copy()

    if len(neg) == 0 or len(pos) == 0:
        return pd.DataFrame(), df.copy()

    # Arrays
    X = neg.loc[:, cols].to_numpy(dtype=float)
    Y = pos.loc[:, cols].to_numpy(dtype=float)
    # print(f"X: {X}")
    # print(f"y: {Y}")
    # Cost matrix (L1 distance)
    cost = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
    # print(f"cost: {cost}")
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # print(f"row_ind: {row_ind}")
    # print(f"col_ind: {col_ind}")
    
    # Map back to indices
    neg_idx = neg.index.to_numpy()[row_ind]
    pos_idx = pos.index.to_numpy()[col_ind]
    dists   = cost[row_ind, col_ind]

    # Matches dataframe
    data = {
        "neg_idx": neg_idx,
        "pos_idx": pos_idx,
        "dist": dists,
        "neg_wgt": df.loc[neg_idx, wgt_col].to_numpy(),
        "pos_wgt": df.loc[pos_idx, wgt_col].to_numpy(),
    }
    for c in cols:
        data[f"neg_{c}"] = df.loc[neg_idx, c].to_numpy()
        data[f"pos_{c}"] = df.loc[pos_idx, c].to_numpy()

    matches_df = pd.DataFrame(data).sort_values("dist").reset_index(drop=True)

    # Drop matched rows from original df
    matched_indices = np.concatenate([neg_idx, pos_idx])
    remaining_df = df.drop(index=matched_indices).reset_index(drop=True)

    return matches_df, remaining_df

def PairNAnnhilateNegWgt(df):
    print(f"df len {len(df)}")
    
    datasets = df["dataset"].unique()
    # year_param_name = "bdt_year"
    year_param_name = "year"
    years = df[year_param_name].unique()
    # Make an empty copy of df (same columns, no rows)
    df_out = df.iloc[0:0].copy()
    
    # Iterate over unique datasets
    for year in years:
        for ds in datasets:
            # Select rows matching this dataset
            subset = df[(df["dataset"] == ds) & (df[year_param_name] == year)].copy()
            if np.all(subset["wgt_nominal_orig"] >=0):
                print(f"no neg wgts in {ds} {year}")
            else:
                print(f"neg wgts in {ds} {year}")
                # --- Preprocess here ---
                print(f"subset len b4 {year} {len(subset)}")
                
                colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
                _, subset = pair_and_remove(subset, cols=colsOfInterest, wgt_col="wgt_nominal_orig")
                print(f"subset len after {year} {len(subset)}")
    
                # # sanity check:
                # assert np.all(subset["wgt_nominal_orig"] >=0)
        
            # Append processed rows into the output df
            df_out = pd.concat([df_out, subset], ignore_index=True)

    print(f"final df_out len {len(df_out)}")
    # print(df_out)
    # raise ValueError
    df_out = df_out[df_out["wgt_nominal_orig"] >=0] # FIXME. we see two entries (so very few) that still have negative events, so temp solution. The two entries are from one of the none DY bkg events.
    return df_out

def fillNanJetvariables(df, forward_filter, jet_variables):
    dijet_variables = [ 
        # 'jet1_eta', 
        # 'jet2_eta', 
        # 'jet1_pt', 
        # 'jet2_pt', 
        'jj_dEta', 
        'jj_dPhi', 
        'jj_mass', 
        'mmj_min_dEta', 
        'mmj_min_dPhi', 
    ]
    jet_variables = list(set(jet_variables + dijet_variables))
    jet_variables  = [var+"_nominal" for var in jet_variables] 
    # print(f"df.loc[forward_filter, jet_variables] b4: {df.loc[forward_filter, jet_variables]}")
    # df.loc[forward_filter, jet_variables].to_csv("dfb4.csv")
    # print(f"forward_filter: {forward_filter}")
    # print(f"jet_variables: {jet_variables}")
    
    for jet_var in jet_variables:
        if jet_var in df.columns:
            if "dPhi" in jet_var:
                df.loc[forward_filter, jet_var] = -999.0
            else:
                df.loc[forward_filter, jet_var] = -999.0

    # print(f"df.loc[forward_filter, jet_variables] after: {df.loc[forward_filter, jet_variables]}")
    # df.loc[forward_filter, jet_variables].to_csv("dfafter.csv")

    # remove njets
    df.loc[forward_filter, "njets_nominal"] = df.loc[forward_filter, "njets_nominal"] - 1
    print(f'np.all(df["njets_nominal"]): {np.all(df["njets_nominal"])}')
    assert(np.all(df["njets_nominal"]>=0))
    return df
    
def removeForwardJets(df):
    """
    remove jet variables that are in the forward region abs(jet eta)> 2.5 with with fill nan
    values consistent with the rest of the framework for ggH BDT.
    """
    df_new = copy.deepcopy(df)
    
    # leading jet  --------------------------
    forward_filter = (abs(df["jet1_eta_nominal"]) > 2.5) & (abs(df["jet1_eta_nominal"]) < 5.0)
    jet_variables = [
        "jet1_eta",
        "jet1_pt",
        # "jet1_phi",
        # "jet1_mass",
        "mmj1_dEta",
        # "mmj1_dPhi",
    ]
    df_new = fillNanJetvariables(df_new, forward_filter, jet_variables)
    # raise ValueError
    
    # sub-leading jet  --------------------------
    forward_filter = (abs(df["jet2_eta_nominal"]) > 2.5) & (abs(df["jet2_eta_nominal"]) < 5.0)
    jet_variables = [
        "jet2_eta",
        "jet2_pt",
        # "jet2_phi",
        # "jet2_mass",
    ]
    df_new = fillNanJetvariables(df_new, forward_filter, jet_variables)
    return df_new

def auc_from_eff(eff_sig, eff_bkg):
    fpr = 1.0 - np.asarray(eff_bkg)
    tpr = np.asarray(eff_sig)
    # sort by FPR ascending before integrating
    order = np.argsort(fpr)
    return np.trapezoid(tpr[order], fpr[order])


def addErrByQuadrature(df, columns=[]):
    rel_errByCol = []
    for col in columns:
        val = df[col]
        val_err = df[f"{col}_err"]
        rel_err = val_err/val
        rel_err = np.nan_to_num(rel_err, nan=0.0)
        rel_errByCol.append(rel_err)
    print(f"rel_errByCol: {len(rel_errByCol)}")
    rel_errByCol = np.array(rel_errByCol).T
    print(f"rel_errByCol: {(rel_errByCol)}")
    print(f"rel_errByCol: {(rel_errByCol.shape)}")
    rel_errByQuad = np.sqrt(np.sum(rel_errByCol**2, axis=1))
    print(f"rel_errByQuad len: {(rel_errByQuad.shape)}")
    print(f"rel_errByQuad: {rel_errByQuad}")
    print(f"rel_errByQuad: {np.mean(rel_errByQuad)}")
    df["relErrTotal"] = rel_errByQuad
    return df

def GetAucStdErrHanleyMcNeil(auc, n_positive, n_negative):
    """
    Source: https://jhanley.biostat.mcgill.ca/software/Hanley_McNeil_Radiology_82.pdf
    
    """
    print(f"auc: {auc}")
    print(f"n_positive: {n_positive}")
    print(f"n_negative: {n_negative}")
    Q1 = auc/(2-auc)
    Q2 = 2*auc**2/(1+auc)
    var_auc = auc*(1-auc)
    var_auc += (n_positive-1)*(Q1-auc**2)
    var_auc += (n_negative-1)*(Q2-auc**2)
    var_auc = var_auc / (n_positive*n_negative)
    return np.sqrt(var_auc)
    
# def processYearCol(df):
#     df_new = copy.deepcopy(df)
#     print(df_new["year"].unique())
#     raise ValueError
    
#     # Mapping dictionary
#     replace_map = {
#         "2016preVFP": 2015,
#         "2016postVFP": 2016
#     }
    
#     df_new["year"] = df_new["year"].replace(replace_map)
#     return df_new
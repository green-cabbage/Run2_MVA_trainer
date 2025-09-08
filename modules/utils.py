import pandas as pd
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment


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
    year_param_name = "bdt_year"
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
    
                # sanity check:
                assert np.all(subset["wgt_nominal_orig"] >=0)
        
            # Append processed rows into the output df
            df_out = pd.concat([df_out, subset], ignore_index=True)

    print(f"final df_out len {len(df_out)}")
    # print(df_out)
    # raise ValueError
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
                df.loc[forward_filter, jet_var] = -1.0
            else:
                df.loc[forward_filter, jet_var] = 0.0

    # print(f"df.loc[forward_filter, jet_variables] after: {df.loc[forward_filter, jet_variables]}")
    # df.loc[forward_filter, jet_variables].to_csv("dfafter.csv")
    return df
    
def removeForwardJets(df):
    """
    remove jet variables that are in the forward region abs(jet eta)> 2.5 with with fill nan
    values consistent with the rest of the framework for ggH BDT.
    """
    df_new = copy.deepcopy(df)
    
    # leading jet  --------------------------
    forward_filter = (abs(df["jet1_eta_nominal"]) > 2.5)
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
    forward_filter = (abs(df["jet2_eta_nominal"]) > 2.5)
    jet_variables = [
        "jet2_eta",
        "jet2_pt",
        # "jet2_phi",
        # "jet2_mass",
    ]
    df_new = fillNanJetvariables(df_new, forward_filter, jet_variables)
    return df_new


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
import pandas as pd
import numpy as np
import copy

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
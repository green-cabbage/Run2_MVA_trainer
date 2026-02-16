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
from modules.workflow import prepare_features, prepare_dataset, classifier_train, convert2df
from modules.variables import training_features, training_samples
from modules.utils import split_into_n_parts, apply_gghChannelSelection, reweightMassToFlat

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
    # client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='12 GiB') 
    client =  Client(n_workers=10,  threads_per_worker=1, processes=True, memory_limit='25 GiB') 

    
    if year == "2016":
        load_path = f"{sysargs.load_path}/{year}*/f1_0" # copperheadV2
    elif year == "all":
        load_path = f"{sysargs.load_path}/*/f1_0" # copperheadV2
    else:
        load_path = f"{sysargs.load_path}/{year}/f1_0" # copperheadV2
        # load_path = f"{sysargs.load_path}/{year}/compacted" # copperheadV2
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
            "njets_nominal",
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
        print(f"parquet_path: {parquet_path}")
        filelist_big = glob.glob(parquet_path)
        if "dy" in sample:
            n_parts = 4
        else:
            n_parts = 1
        filelist_l = split_into_n_parts(filelist_big, n_parts)
        # print(get_subdirs(load_path+f"/{sample}/"))
        # if "dy" in sample:
        #     # subdirectory_names = get_subdirs(load_path+f"/{sample}/")
        #     subdirectory_names = ["*"]
        # else:
        #     subdirectory_names = ["*"]
        # print(f"subdirectory_names: {subdirectory_names}")
        # for subdirectory_name in subdirectory_names:
        #     parquet_path = load_path+f"/{sample}/{subdirectory_name}/*.parquet"
        for filelist in filelist_l:
            try:
                # zip_sample = dak.from_parquet(parquet_path) 
                # filelist = glob.glob(parquet_path)
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
                    # zip_sample = zip_sample.compute()
                    zip_sample = apply_gghChannelSelection(zip_sample)
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
                    # zip_sample = zip_sample.compute()
                    zip_sample = apply_gghChannelSelection(zip_sample)
            except Exception as error:
                print(f"Parquet for {sample} not found with error {error}. skipping!")
                continue
            # zip_sample = dak.from_parquet(load_path+f"/{sample}/*.parquet") # copperheadV1
            # temporary introduction of mu_pt_over_mass variables. Some tt and top samples don't have them
            # zip_sample["mu1_pt_over_mass"] = zip_sample["mu1_pt"] / zip_sample["dimuon_mass"]
            # zip_sample["mu2_pt_over_mass"] = zip_sample["mu2_pt"] / zip_sample["dimuon_mass"]
    
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
            max_num_rows = 80_000
            # max_num_rows = 8_000
            # max_num_rows = 800_000_000
            # df_sample = PairNAnnhilateNegWgt_inChunks(df_sample, max_num_rows=max_num_rows) # FIXME
            # df_sample = PairNAnnhilateNegWgt(df_sample, max_num_rows=max_num_rows) # FIXME
            
            df_l.append(df_sample)
            # print(f"df_sample: {df_sample.head()}")
    # print(f"df_l: {df_l}")
    df_total = pd.concat(df_l,ignore_index=True)   
    del df_l # delete redundant df to save memory. Not sure if this is necessary
    print(f"df_total.dataset.unique(): {df_total.dataset.unique()}")
    sig_datasets = ["ggh_powhegPS", "vbf_powheg_dipole", "vbf_powheg", "vbf_aMCatNLO"]
    save_path = f"output/bdt_{name}_{year}"
    os.makedirs(save_path, exist_ok=True)
    df_total = reweightMassToFlat(df_total, sig_datasets, save_path)
    # new code end --------------------------------------------------------------------------------------------

    # one hot-encode start ----------------------------------------------------
    # One-hot encode the 'year' column
    # year_col_name = "year"
    # if year_col_name in df_total.columns:
    #     print(f"df_total b4: {df_total[year_col_name]}")
    #     one_hot_df = pd.get_dummies(df_total[year_col_name], prefix=year_col_name, dtype=int)
    #     # one_hot_df.columns = one_hot_df.columns.str.replace('.', '_') # replace "." with "_" in year columns

    #     # Concatenate the new dummy columns with the original DataFrame
    #     df_total = pd.concat([df_total, one_hot_df], axis=1)
        
    #     # Drop the original 'Segment' column
    #     df_total.drop(year_col_name, axis=1, inplace=True)
    #     print(df_total.columns)
    #     # print(df_total)
    #     filtered_df = df_total.filter(like=year_col_name, axis=1) # axis=1 specifies filtering columns
    #     training_features.remove(year_col_name)
    #     training_features = training_features + list(one_hot_df.columns)
    #     print(f"training_features after: {training_features}")
    #     print(f"df_total after: {filtered_df}")
    #     print(f"one_hot_df.columns: {one_hot_df.columns}")
    # one hot-encode end ----------------------------------------------------

    
    # apply random shuffle, so that signal and bkg samples get mixed up well
    random_seed_val = 125
    df_total = df_total.sample(frac=1, random_state=random_seed_val)
    # print(f"df_total after shuffle: {df_total}")
    

    
    # print(f"df_total: {df_total}")
    print("starting prepare_dataset")
    df_total = prepare_dataset(df_total, training_samples)
    print("prepare_dataset done")
    # raise ValueError
    # print(f"len(df_total): {len(df_total)}")
    print(f"df_total.columns: {df_total.columns}")
    
    classifier_train(df_total, args, training_samples, training_features, random_seed_val, save_path)
    # evaluation(df_total, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    #print(df)
    runtime = int(time.time()-start_time)
    print(f"Success! run time is {runtime} seconds")
import argparse
import glob
import os
import time

import awkward as ak
import dask_awkward as dak
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from distributed import Client
from modules.variables import training_features, training_samples
from modules.workflow import (
    classifier_train,
    convert2df,
    prepare_dataset,
    prepare_features,
)

from modules.utils import (
    apply_gghChannelSelection,
    reweightMassToFlat,
    reweightMassToTargetDist_workflow,
    split_into_n_parts,
)
from modules.utils_logger import logger

plt.style.use(hep.style.CMS)

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
    parser.add_argument(
    "-param_search",
    "--do_hyperparam_search",
    dest="do_hyperparam_search",
    type=int,
    default=0,
    # action="store",
    help="integer flag to do hyperparam search. If zero, then do not do it",
    )
    parser.add_argument(
    "--massDeCorrStrat",
    dest="mass_decorrelation_strat",
    default="default",
    action="store",
    help="Dimuon mass decorrelation method for training. Available options are: default (do nothing), peking, targetZpeakMass.",
    )
    sysargs = parser.parse_args()
    year = sysargs.year
    name = sysargs.name
    # mass_decorrelation_strat = sysargs.mass_decorrelation_strat
    args = {
        "year": year,
        "name": name,
        # "mass_decorrelation_strat": mass_decorrelation_strat,
        "do_massscan": False,
        "evaluate_allyears_dnn": False,
        "output_path": "/depot/cms/users/yun79/hmm/trained_MVAs",
        "label": "",
        # "do_hyperparam_search": !(sysargs.do_hyperparam_search==0), # if zero, then do not do hyperparam search
    }
    do_hyperparam_search = (sysargs.do_hyperparam_search!=0) # if zero, then do not do hyperparam search
    print(f"do_hyperparam_search: {do_hyperparam_search}")
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


        # FIXME --------------------
        # print(f"===> year: {year}")
        # print(f"===> sysargs.load_path: {sysargs.load_path}")
        # if (year == "all") or (year == "2024"):
        #     if sample=="dyTo2L_M-50_incl": 
        #         load_path_exception = f"{sysargs.load_path}/2024/f1_0" 
        #         parquet_path = load_path_exception+f"/dyTo2Mu_M-50_aMCatNLO/*/*.parquet"
        #         filelist_big += glob.glob(parquet_path)
                
        if "dy" in sample:
            n_parts = 4
        else:
            n_parts = 1
        filelist_l = split_into_n_parts(filelist_big, n_parts)

        for filelist in filelist_l:
            try:
                if len(filelist) == 0:
                    print(f"[WARN] No parquet files found for {sample} at: {parquet_path}. Skipping.")
                    continue  # skip this sample                
                zip_sample = dak.from_parquet(filelist)

                fields2load_prepared = prepare_features(zip_sample, fields2load) # add variation to the name
                training_features_prepared = prepare_features(zip_sample, training_features) # do the same thing to training features
                
                # print(f"fields2load after: {fields2load_prepared}")
                zip_sample = ak.zip({
                    field : zip_sample[field] for field in fields2load_prepared
                })
                zip_sample = apply_gghChannelSelection(zip_sample)
            except Exception as error:
                raise FileNotFoundError(f"Parquet for {sample} not found with error {error}. skipping!")

            df_sample = convert2df(zip_sample, sample)
            max_num_rows = 80_000
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
    
    if sysargs.mass_decorrelation_strat == "peking":
        print("Peking decorrlation method!")
        df_total = reweightMassToFlat(df_total, sig_datasets, save_path)
    elif sysargs.mass_decorrelation_strat == "targetZpeakMass":
        print("targetZpeakMass decorrlation method!")
        df_total = reweightMassToTargetDist_workflow(df_total, sig_datasets, save_path)
    elif sysargs.mass_decorrelation_strat == "targetHpeakMass":
        print("targetHpeakMass decorrlation method!")
        dy_target_mass_centre = 125
        # nbins=40
        nbins=20
        df_total = reweightMassToTargetDist_workflow(df_total, sig_datasets, save_path, nbins=nbins, target_mass_centre=dy_target_mass_centre)
    elif sysargs.mass_decorrelation_strat == "targetHsidebandMass":
        print("targetHpeakMass decorrlation method!")
        dy_target_mass_centre = 105
        nbins=40
        df_total = reweightMassToTargetDist_workflow(df_total, sig_datasets, save_path, nbins=nbins, target_mass_centre=dy_target_mass_centre)
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
    #     training_features_prepared.remove(year_col_name)
    #     training_features_prepared = training_features_prepared + list(one_hot_df.columns)
    #     print(f"training_features_prepared after: {training_features_prepared}")
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
    # print(f"len(df_total): {len(df_total)}")
    print(f"df_total.columns: {df_total.columns}")

    classifier_train(df_total, args, training_samples, training_features_prepared, random_seed_val, save_path, do_hyperparam_search=do_hyperparam_search)
    # evaluation(df_total, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    #print(df)
    runtime = int(time.time()-start_time)
    print(f"Success! run time is {runtime} seconds")
import glob
import tqdm
import time
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd
from functools import partial

import pandas as pd
import pickle

from python.io import load_dataframe
from stage2.categorizer import split_into_channels
from stage2.quick_train import train_dnn

__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.149.133"
node_ip = "172.18.36.39"

if use_local_cluster:
    ncpus_local = 4
    dashboard_address = f"{node_ip}:34875"
    slurm_cluster_ip = ""
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

#model_name = "pytorch_may24_pisa"
model_name = "pytorch_jun27"

# global parameters
parameters = {
    "global_path": "/depot/cms/hmm/copperhead/",
    "years": args.years,
    "label": "2022jun1",
    #"datasets": ["dy_m105_160_amc", "dy_m105_160_vbf_amc", "ttjets_dl", "ggh_amcPS", "vbf_powheg_dipole", "ewk_lljj_mll105_160_py_dipole",]
    "datasets": ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
}

if __name__ == "__main__":
    tick = time.time()

    load_from_disk = True
    save_to_disk = not load_from_disk
    categorize = save_to_disk
    df_path = "/depot/cms/hmm/coffea/training_dataset_jun9_vbf.pickle"

    if load_from_disk:
        #df = df_path
        df = pd.read_pickle(df_path)
        #df = pickle.load(df_path)
    else:
        
        client = Client(
            processes=True,
            n_workers=1,
            threads_per_worker=1,
            memory_limit="192GB",
        )
        
        # prepare lists of paths to parquet files (stage1 output) for each year and dataset
        all_paths = {}
        for year in parameters["years"]:
            all_paths[year] = {}
            for dataset in parameters["datasets"]:
                paths = glob.glob(
                    f"{parameters['global_path']}/"
                    f"{parameters['label']}/stage1_output/{year}/"
                    f"{dataset}/*.parquet"
                )
                all_paths[year][dataset] = paths

        # run postprocessing
        dfs = []
        for year in parameters["years"]:
            print(f"Processing {year}")
            for dataset, path in tqdm.tqdm(all_paths[year].items()):
                if len(path) == 0:
                    continue

                # read stage1 outputs
                df = load_dataframe(client, parameters, inputs=[path])
                if not isinstance(df, dd.DataFrame):
                    continue
                dfs.append(df)

        print("Merging Dask dataframes...")
        ignore_columns = []
        ignore_columns += [c for c in df.columns if "pdf_" in c]
        ignore_columns += [c for c in df.columns if ("wgt_" in c) and ("nominal" not in c)]

        df = pd.DataFrame()
        for df_ in tqdm.tqdm(dfs):
            df_ = df_[[c for c in df_.columns if c not in ignore_columns]]
            df_ = df_.compute()
            if categorize:
                vbf_filter = (
                    (df_[f"nBtagLoose_nominal"] < 2) &
                    (df_[f"nBtagMedium_nominal"] < 1) &
                    (df_[f"jj_mass_nominal"] > 400) &
                    (df_[f"jj_dEta_nominal"] > 2.5) &
                    (df_[f"jet1_pt_nominal"] > 35)
                )
                df_ = df_[vbf_filter]
            df = pd.concat([df, df_])

        with open(df_path, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        import sys
        sys.exit()
    
    if categorize:
        print("Categorizing events")
        #df = pd.read_pickle(df_path)
        vbf_filter = (
            (df[f"nBtagLoose_nominal"] < 2) &
            (df[f"nBtagMedium_nominal"] < 1) &
            (df[f"jj_mass_nominal"] > 400) &
            (df[f"jj_dEta_nominal"] > 2.5) &
            (df[f"jet1_pt_nominal"] > 35)
        )
        df = df[vbf_filter]

    for step in [0, 1, 2, 3]:#range(4):
        train_dnn(step, df, model_name, "pytorch_dnn")

    #train_dnn(0, df, model_name, "pytorch_dnn")
    #print(df)
    #for step in [1, 2, 3]:
    #    train_dnn(step, df, model_name, "pytorch_pisa")


    elapsed = round(time.time()-tick, 2)
    print(f"Done in {elapsed} s.")



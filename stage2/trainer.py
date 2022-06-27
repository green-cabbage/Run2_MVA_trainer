import os

import pickle
import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep

from python.workflow import parallelize
from python.io import mkdir
from python.variable import Variable


style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)



class Trainer(object):
    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", pd.DataFrame())
        self.channel = kwargs.pop("channel", "")
        self.ds_dict = kwargs.pop("ds_dict", {})
        self.features = kwargs.pop("features", [])
        self.out_path = kwargs.pop("out_path", "./")
        self.training_cut = kwargs.pop("training_cut", None)
        self.models = {}
        self.trained_models = {}
        self.scalers = {}

        self.fix_variables()
        self.prepare_dataset()
        self.prepare_features()
        print()
        print("*" * 60)
        print("In channel", self.channel)
        print("Event counts in classes:")
        print(self.df["class_name"].value_counts())
        print("Training features:")
        print(self.features)

        # The part below is needed for cross-validation.
        # Training will be done in 4 steps and at each step
        # the dataset will be split as shown below:
        #
        # T = training (50% of data)
        # V = validation (25% of data)
        # E = evaluation (25% of data)
        # ----------------
        # step 0: T T V E
        # step 1: E T T V
        # step 2: V E T T
        # step 3: T V E T
        # ----------------
        # (splitting is based on event number mod 4)
        #
        # This ensures that all data is used for training
        # and for a given model evaluation is never done
        # on the same data as training.

        self.nfolds = 4
        folds_def = {"train": [0, 1], "val": [2], "eval": [3]}
        self.fold_filters_list = []
        for step in range(self.nfolds):
            fold_filters = {}
            fold_filters["step"] = step
            for fname, folds in folds_def.items():
                folds_shifted = [(step + f) % self.nfolds for f in folds]
                fold_filters[f"{fname}_filter"] = self.df.event.mod(self.nfolds).isin(
                    folds_shifted
                )
            self.fold_filters_list.append(fold_filters)

    def fix_variables(self):
        self.df.loc[:, "mu1_pt_over_mass"] = self.df.mu1_pt / self.df.dimuon_mass
        self.df.loc[:, "mu2_pt_over_mass"] = self.df.mu2_pt / self.df.dimuon_mass

    def prepare_features(self, variation='nominal'):
        features = []
        for trf in self.features:
            if f'{trf} {variation}' in self.df.columns:
                features.append(f'{trf} {variation}')
            elif trf in self.df.columns:
                features.append(trf)
            else:
                print(f'Variable {trf} not found in training dataframe!')
        self.features = features

    def prepare_dataset(self):
        # Convert dictionary of datasets to a more useful dataframe
        df_info = pd.DataFrame()
        self.train_samples = []
        for icls, (cls, ds_list) in enumerate(self.ds_dict.items()):
            for ds in ds_list:
                df_info.loc[ds, "dataset"] = ds
                df_info.loc[ds, "iclass"] = -1
                if cls != "ignore":
                    self.train_samples.append(ds)
                    df_info.loc[ds, "class_name"] = cls
                    df_info.loc[ds, "iclass"] = icls
        df_info["iclass"] = df_info["iclass"].fillna(-1).astype(int)
        self.df = self.df[self.df.dataset.isin(df_info.dataset.unique())]

        # Assign numerical classes to each event
        cls_map = dict(df_info[["dataset", "iclass"]].values)
        cls_name_map = dict(df_info[["dataset", "class_name"]].values)
        self.df["class"] = self.df.dataset.map(cls_map)
        self.df["class_name"] = self.df.dataset.map(cls_name_map)

    def add_models(self, model_dict):
        if self.channel in model_dict.keys():
            self.models = model_dict[self.channel]
        self.trained_models = {n: {} for n in self.models.keys()}
        self.scalers = {n: {} for n in self.models.keys()}

    def add_saved_models(self, model_dict):
        for model_name, model_props in model_dict.items():
            model_path = model_props["path"]
            self.models[model_name] = {"type": model_props["type"]}
            print(f"Loading model {model_name} from {model_path}")
            self.trained_models[model_name] = {}
            self.scalers[model_name] = {}
            for step in range(self.nfolds):
                self.trained_models[model_name][
                    step
                ] = f"{model_path}/models/model_{model_name}_{step}.h5"
                self.scalers[model_name][
                    step
                ] = f"{model_path}/scalers/scalers_{model_name}_{step}"


    def normalize_data(self, reference, features, to_normalize_dict, model_name, step):
        mean = np.mean(reference[features].values, axis=0)
        std = np.std(reference[features].values, axis=0)
        out_path = f"{self.out_path}/scalers/"
        mkdir(out_path)
        save_path = f"{out_path}/scalers_{model_name}_{step}"
        np.save(save_path, [mean, std])

        normalized = {}
        for key, item in to_normalize_dict.items():
            item_normalized = (item[features] - mean) / std
            normalized[key] = item_normalized
        return normalized, save_path


    def plot_roc_curves(self):
        roc_curves = {}
        fig = plt.figure()
        fig, ax = plt.subplots()
        df = self.df[self.df.dataset.isin(self.train_samples)]
        for model_name, model in self.models.items():
            score_name = f"{model_name}_score"
            roc_curves[score_name] = roc_curve(
                y_true=df["class"],
                y_score=df[score_name],
                sample_weight=df["wgt_nominal"],
            )
            ax.plot(
                roc_curves[score_name][0], roc_curves[score_name][1], label=score_name
            )
        ax.legend(prop={"size": "x-small"})
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        out_name = f"{self.out_path}/rocs.png"
        fig.savefig(out_name)


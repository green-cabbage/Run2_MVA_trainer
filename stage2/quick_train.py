import os
import tqdm
import copy 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)

from stage2.trainer import Trainer
from stage2.categorizer import split_into_channels
from stage2.mva_models import Net, NetPisaRun2, NetPisaRun2Combination


training_features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_pisa_mass_res",
    "dimuon_pisa_mass_res_rel",
    "dimuon_cos_theta_cs_pisa",
    "dimuon_phi_cs_pisa",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_qgl",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_qgl",
    "jj_mass",
    "jj_mass_log",
    "jj_dEta",
    "rpt",
    "ll_zstar_log",
    "mmj_min_dEta",
    "nsoftjets5",
    "htsoft2",
]

training_features_mass = [
    "dimuon_mass",
    "dimuon_pisa_mass_res",
    "dimuon_pisa_mass_res_rel",
]

training_features_nomass = [
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_cos_theta_cs_pisa",
    "dimuon_phi_cs_pisa",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet1_qgl_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jet2_qgl_nominal",
    "jj_mass_nominal",
    "jj_mass_log_nominal",
    "jj_dEta_nominal",
    "rpt_nominal",
    "ll_zstar_log_nominal",
    "mmj_min_dEta_nominal",
    "nsoftjets5_nominal",
    "htsoft2_nominal",
    "year"
]

#training_datasets = {
#    "background": ["dy_m105_160_amc", "dy_m105_160_vbf_amc", "ttjets_dl", "ewk_lljj_mll105_160_py_dipole",],
#    "signal": ["ggh_amcPS", "vbf_powheg_dipole"],
#}
training_datasets = {
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],
    'signal': ['vbf_powhegPS','vbf_powheg_herwig', 'vbf_powheg_dipole', 'ggh_amcPS'],
}

def prepare_features(df, variation='nominal', add_year=False):
    features = []
    for trf in training_features:
        if f'{trf}_{variation}' in df.columns:
            features.append(f'{trf}_{variation}')
        elif trf in df.columns:
            features.append(trf)
        else:
            print(f'Variable {trf} not found in training dataframe!')
    if add_year:
        features.append("year")
    return features

def save_model(model, model_name, step):
    model_path = f"data/trained_models/vbf/models/{model_name}_{step}.pt"
    torch.save(model.state_dict(), model_path)


def train_dnn(step, df, model_name, model_type):
    if isinstance(df, str):
        df = pd.read_pickle(df)
    add_year = True
    features = prepare_features(df, add_year=add_year)
    if add_year:
        assert len(features) == len(training_features)+1
    else:
        assert len(features) == len(training_features)
    
    trainer = Trainer(
        df = df,
        channel="vbf",
        ds_dict=training_datasets,
        features=features,
        out_path="data/trained_models/vbf/",
        #training_cut="(dimuon_mass > 110) & (dimuon_mass < 150)",
    )

    df = trainer.df[trainer.df.dataset.isin(trainer.train_samples)]

    #df["class"] = None
    #for cls, ds in training_datasets.items():
    #    df.loc[df.dataset.isin(ds), "class"] = cls
    
    mean_cls_wgts = df.groupby("class").wgt_nominal.mean()
    df["mean_cls_wgt"] = df["class"].map(mean_cls_wgts)
    df["wgt_aux"] = 1.0
    df.loc[df.dataset=="ttjets_dl", "wgt_aux"] = 0.5

    folds_def = {"train": [0, 1], "val": [2], "eval": [3]}
    folds_shifted = {}
    for fname, folds in folds_def.items():
        folds_shifted[fname] = [(step + f) % 4 for f in folds]

    train_filter = df.event.mod(4).isin(folds_shifted["train"])
    val_filter = df.event.mod(4).isin(folds_shifted["val"])
    eval_filter = df.event.mod(4).isin(folds_shifted["eval"])

    df_train = df.loc[train_filter, :]
    df_val = df.loc[val_filter, :]
    
    normalized, scalers_save_path = trainer.normalize_data(
        reference=df_train,
        features=trainer.features,
        to_normalize_dict={"x_train": df_train.loc[:, trainer.features], "x_val": df_val.loc[:, trainer.features]},
        model_name=model_name,
        step=step,
    )
    df_train.loc[:, trainer.features] = normalized["x_train"]
    df_val.loc[:, trainer.features] = normalized["x_val"]
    
    best_loss = 1000
    train_histories = {}
    train_history = {}

    if model_type == "pytorch_dnn":
        """
        configurations = {
            #"sigLoss_atanh": {"loss": "sigLoss", "out_transform": lambda x: torch.arctanh(x)},
            "x_entr": {"loss": "x_entr", "out_transform": lambda x: x},
            "x_entr_1vbf": {"loss": "x_entr", "out_transform": lambda x: x,},
            #"sigLoss": {"loss": "sigLoss", "out_transform": lambda x: x},
            #"likelihoodLoss": {"loss": "likelihoodLoss", "out_transform": lambda x: x}
        }
        for key, config in configurations.items():
            train_history, best_loss, best_model = train_pytorch_simple(model_name, step, df_train, df_val, trainer, **config)
            train_histories[key] = train_history
        """
        # regular training
        train_history, best_loss, last_model = train_pytorch_simple(
            model_name, step, df_train, df_val, trainer,
            **{"loss": "x_entr", "out_transform": lambda x: x, "nepochs": 5, "niterations": 10000, "save_every": 100}
        )
        train_histories["baseline"] = train_history
        

        for drop_factor in [0]:
            train_history, best_loss, last_model = train_pytorch_simple(
                model_name, step, df_train, df_val, trainer,
                **{"loss": "x_entr", "out_transform": lambda x: x, "nepochs": 10, "save_every": 100,
                   "niterations": 10000,
                   #"exclude_features": ["dimuon_pisa_mass_res", "dimuon_pisa_mass_res_rel",]
                   "exclude_datasets": ['vbf_powhegPS','vbf_powheg_herwig'],
                   #"exclude_datasets": ['vbf_powhegPS','vbf_powheg_dipole'],
                   #"exclude_datasets": ['vbf_powheg_herwig','vbf_powheg_dipole'],
                   "vbf_drop_factor": drop_factor
                  }
            )
            keep = round((1-drop_factor)*100)
            train_histories[f"only vbf_dipole"] = train_history

        
        """
        for lam in [0.5, 0.7, 0.9]:
            # first regular, then fine-tune
            train_history, best_loss, last_model = train_pytorch_simple(
                model_name, step, df_train, df_val, trainer,
                **{"loss": "x_entr", "out_transform": lambda x: x, "nepochs": 10, "save_every": 5, "finetune": True, "lambda": lam}
            )
            train_histories[f"x_entr_modified_{lam}"] = train_history
        """

        
    elif model_type == "pytorch_pisa":
        train_pytorch_pisa(model_name, step, df_train, df_val, trainer)


    """
    if ("batch_n" in train_history) and ("val_losses" in train_history):
        fig = plt.figure()
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        opts = {"linewidth": 2}
        #ax.plot(train_history["batch_n"], train_history["train_losses"], label="training loss", **opts)
        ax.plot(train_history["batch_n"], train_history["val_losses"], label="validation loss", **opts)
        ax.legend(prop={"size": "x-small"})
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(f"plots/pytorch/losses_{model_name}_{step}.png")
    """

    fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    opts = {"linewidth": 2}

    save = False    

    for key, train_history in train_histories.items():
        if ("batch_n_sign" in train_history) and ("significance" in train_history):
            save = True
            sign = round(np.array(train_history["significance"])[20:].mean(), 5)
            ax.plot(train_history["batch_n_sign"], train_history["significance"], label=f"{key}: {sign}", **opts)
            #ax.plot(train_history["batch_n_sign"], train_history["asimov_sign"], label=f"asimov sign.: {key}", **opts)
            #ax.plot(train_history["batch_n_sign"], train_history["val_losses"], label=f"val. loss ({key})", **opts)
    ax.legend(prop={"size": "x-small"})
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Significance")
    fig.tight_layout()
    if save:
        fig.savefig(f"plots/pytorch/significance_{model_name}_{step}.png")
    
    return best_loss

def significanceLossInvert(expectedSignal,expectedBkgd):
    def sigLossInvert(y_pred, y_true):
        signalWeight=expectedSignal/torch.sum(y_true)
        bkgdWeight=expectedBkgd/torch.sum(1-y_true)
        s = signalWeight*torch.sum(y_pred*y_true)
        b = bkgdWeight*torch.sum(y_pred*(1-y_true))
        return (s+b)/(s*s+1e-7) #Add the epsilon to avoid dividing by 0
    return sigLossInvert

def likelihoodLoss(expectedSignal, expectedBkgd):
    def likLoss(y_pred, y_true):
        signalWeight=expectedSignal/torch.sum(y_true)
        bkgdWeight=expectedBkgd/torch.sum(1-y_true)
        L = bkgdWeight*torch.sum((1-y_true)*(torch.exp(y_pred)-1)) - torch.sum(y_pred*y_true)
        return L
    return likLoss


def asimovLossInvert(expectedSignal,expectedBkgd,systematic):
    def asimovLossInvert_(y_pred,y_true):
        signalWeight=expectedSignal/torch.sum(y_true)
        bkgdWeight=expectedBkgd/torch.sum(1-y_true)
        s = signalWeight*torch.sum(y_pred*y_true)
        b = bkgdWeight*torch.sum(y_pred*(1-y_true))
        sigB=systematic*b
        eps = 1e-7
        return 1./(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+eps)+eps)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+eps))/(sigB*sigB+eps))) 
    return asimovLossInvert_


def train_pytorch_simple(model_name, step, df_train, df_val, trainer, **kwargs):
    weighted = kwargs.get("weighted", False)
    loss_f = kwargs.get("loss_f", "x_entr")
    out_transform = kwargs.get("out_transform", lambda x : x)
    pretrained_model = kwargs.get("pretrained_model", None)
    train_history = kwargs.get("train_history", None)
    do_finetune = kwargs.get("finetune", False)
    lambda_ = kwargs.get("lambda", 0)
    exclude_features = kwargs.get("exclude_features", [])
    exclude_datasets = kwargs.get("exclude_datasets", [])
    vbf_drop_factor = kwargs.get("vbf_drop_factor", 0)
    
    features = [f for f in trainer.features if f not in exclude_features]

    df_train = copy.deepcopy(df_train).reset_index(drop=True)
    df_val = copy.deepcopy(df_val).reset_index(drop=True)

    df_train = df_train[~df_train.dataset.isin(exclude_datasets)]
    #df_val = df_val[~df_val.dataset.isin(exclude_datasets)]

    
    signals = ["vbf_powheg_dipole", 'vbf_powhegPS','vbf_powheg_herwig']
    df_train = df_train.drop(df_train[df_train.dataset.isin(signals)].sample(frac=vbf_drop_factor).index)
    print(df_train.dataset.value_counts())
    #df_val = df_val.drop(df_val[df_val.dataset.isin(signals)].sample(frac=vbf_drop_factor).index)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    if pretrained_model is None:
        model = Net(input_shape = len(features))
        model.to(device)
    else:
        model = pretrained_model
        
    reduction = 'mean'
    if weighted:
        reduction = 'none'

    exp_s = 48.7982068771 #vbf+ggH
    exp_b = 15786.6000629724 #DY+EWK

    if loss_f == "x_entr":
        criterion = nn.BCELoss(reduction=reduction)
    elif loss_f == "mse":
        criterion = nn.MSELoss()
    elif loss_f == "x_entr_logits":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_f == "likelihoodLoss":
        criterion = likelihoodLoss(exp_s, exp_b)
    elif loss_f == "sigLoss":
        criterion = significanceLossInvert(exp_s, exp_b)
    elif loss_f == "asimov0.1":
        criterion = asimovLossInvert(exp_s, exp_b, 0.1)     
    else:
        print(f"Incorrect loss specified: {loss_f}")
        return

    optimizer = optim.Adam(model.parameters(), eps=1e-07, lr=1e-2)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    if train_history is None:
        train_history = {
            "train_losses": [],
            "val_losses": [],
            "batch_n": [],
            "significance": [],
            "asimov_sign": [],
            "batch_n_sign": []
        }
        batch_n_sign = 0
    else:
        batch_n_sign = max(train_history["batch_n_sign"])


    #batch_size = 2048
    batch_size = 65536
    n_training_batches = int(df_train.shape[0] / batch_size)

    epochs = kwargs.get("nepochs", 1)
    iterations = kwargs.get("niterations", None)
    if iterations is not None:
        #overrides nepochs
        epochs = math.ceil(iterations / n_training_batches)
    
    print(f"Training dataset size: {df_train.shape[0]}.")
    print(f"Number of batches of size {batch_size} is {n_training_batches}.")
    print(f"Will train for {epochs} epochs.")

    best_loss_so_far = 1000
    best_significance_so_far = 0
    best_model = model
    lr = 0.2*1e-1
    #for epoch in tqdm.tqdm(range(epochs)):
    for epoch in range(epochs):

        #finetune = False
        #if (epoch>=5) and do_finetune:
        #    finetune = True
        #if finetune:
            #lr = lr*5
            #optimizer = optim.Adam(model.parameters(), eps=1e-07, lr=lr)
            #optimizer = optim.SGD(model.parameters(), lr=lr)
            #to_remove = ['vbf_powhegPS','vbf_powheg_herwig']
            #df_train = df_train[~df_train.dataset.isin(to_remove)]
            #df_val = df_val[~df_val.dataset.isin(to_remove)]

        for batch in tqdm.tqdm(range(n_training_batches)):
            batch_n_sign += 1
            if batch_n_sign > iterations:
                continue


            
            train_batch = df_train.sample(batch_size)
            val_batch = df_val.sample(batch_size)

            x_train_batch = train_batch.loc[:, features]
            y_train_batch = train_batch.loc[:, "class"]
            x_train_batch = torch.tensor(x_train_batch.values).float().to(device)
            y_train_batch = torch.tensor(y_train_batch.values).float().view(-1, 1).to(device)

            x_val_batch = val_batch.loc[:, features]
            y_val_batch = val_batch.loc[:, "class"]
            x_val_batch = torch.tensor(x_val_batch.values).float().to(device)
            y_val_batch = torch.tensor(y_val_batch.values).float().view(-1, 1).to(device)

            model.train()

            optimizer.zero_grad(set_to_none=True)
            output = model(x_train_batch)
            output_best = best_model(x_train_batch)
            
            loss = criterion(output, y_train_batch)
            loss_wrt_best = criterion(output, output_best)

            if do_finetune:
                #print(loss.item(), loss_wrt_best.item())
                #total_iterations = n_training_batches*epochs
                #factor = (batch_n_sign - 1) / total_iterations
                #factor = epoch / (epochs-1)
                loss = (1-lambda_)*loss + lambda_*loss_wrt_best
            
            if weighted:
                batch_weights = abs(train_batch.wgt_nominal) * train_batch.wgt_aux / train_batch.mean_cls_wgt
                batch_weights = torch.tensor(batch_weights.values).float().to(device)
                loss = loss.t() * batch_weights
                loss = torch.mean(loss)
                train_loss = loss.to(device).item()
            else:
                train_loss = loss.item()
            loss.backward()
            optimizer.step()   

            model.eval()

            #best_model_mode = "loss"
            best_model_mode = "significance"
            with torch.no_grad():
                if best_model_mode == "loss":
                    output = model(x_val_batch)
                    loss = criterion(output, y_val_batch)
                    if weighted:
                        batch_weights = abs(val_batch.wgt_nominal) * val_batch.wgt_aux / val_batch.mean_cls_wgt
                        batch_weights = torch.tensor(batch_weights.values).float().to(device)
                        loss = loss.t() * batch_weights
                        loss = torch.mean(loss)
                        val_loss = loss.to(device).item()
                    else:
                        val_loss = loss.item()
                    if val_loss < best_loss_so_far:
                        best_loss_so_far = val_loss
                        best_model = copy.deepcopy(model)
                        save_model(model, model_name, step)

                elif best_model_mode == "significance":   
                    if batch_n_sign % kwargs.get("save_every", 1) != 1:
                        continue
                    df_val_aux = df_val.copy()
                    score_name = "score"
                    val_input = torch.tensor(df_val_aux.loc[:, features].values).float()
                    df_val_aux[score_name] = np.arctanh(model(val_input.to(device)).cpu())
                    #df_val_aux[score_name] = model(val_input.to(device)).cpu()
                    
                    """
                    #######
                    #######
                    
                    from sklearn.isotonic import IsotonicRegression
                    iso_reg = IsotonicRegression().fit(df_val_aux[score_name], df_val_aux["class"])
                    df_val_aux["calibrated"] = iso_reg.predict(df_val_aux[score_name])

                    entries = {
                        "dy": ['dy_m105_160_amc','dy_m105_160_vbf_amc'],
                        "vbf": ['vbf_powheg_dipole'],
                        "ewk": ['ewk_lljj_mll105_160_ptj0']
                    }
                    hists = {}
                    nbins = 50
                    centers = np.array(list(range(nbins))) + 1/(2*nbins)
                    centers = centers / nbins
                    for lbl, datasets in entries.items():
                        hists[lbl] = np.histogram(
                            df_val_aux.loc[df_val_aux.dataset.isin(datasets), "calibrated"],
                            nbins, (0,1),
                            weights =  df_val_aux.loc[df_val_aux.dataset.isin(datasets), "wgt_nominal"]
                        )

                                        
                    fig = plt.figure()
                    fig, ax = plt.subplots()
                    fig.set_size_inches(8, 6)

                    opts = {"linewidth": 2}
                    for lbl, hist in hists.items():
                        ax.step(centers, hist[0], label=lbl, **opts)

                    ax.plot(iso_reg.y_thresholds_, iso_reg.X_thresholds_, "C1.", markersize=12)
 
                    ax.legend(prop={"size": "x-small"})
                    ax.set_xlabel("bin")
                    ax.set_ylabel("calibrated score")
                    ax.set_yscale('log')
                    fig.tight_layout()
                    fig.savefig(f"plots/pytorch/isotonic.png")
                    
                    #######
                    #######
                    """
                    #df_val_aux.loc[df_val_aux.dataset=='vbf_powheg_dipole', "wgt_nominal"] = df_val_aux.loc[df_val_aux.dataset=='vbf_powheg_dipole', "wgt_nominal"]/df_val_aux.loc[df_val_aux.dataset=='vbf_powheg_dipole', "wgt_nominal"].sum() * 6.63989928647691
                    
                    vbf_signal = df_val_aux[df_val_aux.dataset=='vbf_powheg_dipole']
                    vbf_signal_yield = vbf_signal.wgt_nominal.sum()

                    nbins = 13
                    #nbins = 128

                    #bins_mode = "fixed"
                    bins_mode = "quantiles_unwgted"
                    #bins_mode = "manual"

                    if bins_mode == "fixed":
                        bins = [0.0, 0.029673779383301735, 0.23851346969604492, 0.4396916627883911, 0.6212533712387085, 0.7922319173812866, 0.9605764746665955, 1.1178971529006958, 1.268202781677246, 1.4059489965438843, 1.5321850776672363, 1.6467801332473755, 1.75693941116333, 1.9684618711471558]
                        nbins = len(bins) - 1

                    elif bins_mode == "quantiles_unwgted":
                        grid = [(i+1)/nbins for i in range(nbins)]
                        bins = vbf_signal["score"].quantile(grid).values

                    elif bins_mode == "manual":
                        vbf_signal = vbf_signal.sort_values(by=score_name, ascending=True).reset_index(drop=True)
                        vbf_signal["wgt_cumsum"] = vbf_signal.wgt_nominal.cumsum()
                        bins = [vbf_signal[score_name].max()]
                        tot_yield = 0
                        target_yields = [vbf_signal_yield/nbins for i in range(nbins)]
                        for yi in target_yields:
                            tot_yield += yi
                            for i in range(vbf_signal.shape[0] - 1):
                                value = vbf_signal.loc[i, "wgt_cumsum"]
                                value1 = vbf_signal.loc[i+1, "wgt_cumsum"]
                                if (value < tot_yield) & (value1 > tot_yield):
                                    bins.append(vbf_signal.loc[i, score_name])

                        bins.append(0.0)
                        bins = sorted(bins)

                    df_val_aux["bin_number"] = 0
                    for i in range(nbins - 1):
                        lo = bins[i]
                        hi = bins[i + 1]
                        cut = (df_val_aux[score_name] > lo) & (
                            df_val_aux[score_name] <= hi
                        )
                        df_val_aux.loc[cut, "bin_number"] = i
                    
                    signal = ['vbf_powheg_dipole', 'ggh_amcPS']
                    background = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0']
                    df_sig = df_val_aux[df_val_aux.dataset.isin(signal)]
                    df_bkg = df_val_aux[df_val_aux.dataset.isin(background)]
                    yields_sig = df_sig.groupby("bin_number")["wgt_nominal"].sum()
                    yields_bkg = df_bkg.groupby("bin_number")["wgt_nominal"].sum()
                    num_bkg = df_bkg.groupby("bin_number")["wgt_nominal"].count()
                    mask = (num_bkg>100)

                    significance = np.sqrt((yields_sig*yields_sig / yields_bkg)[mask].sum())
                    #asimov_sign = np.sqrt((2*(
                    #        (yields_sig+yields_bkg)*np.log(1+yields_sig/yields_bkg)-yields_sig
                    #)).sum())
                    train_history["significance"].append(significance)
                    #train_history["asimov_sign"].append(asimov_sign)
                    #print(significance, asimov_sign)
                    train_history["batch_n_sign"].append(batch_n_sign)
                    
                    if significance > best_significance_so_far:
                        best_significance_so_far = significance
                        best_model = copy.deepcopy(model)
                        save_model(model, model_name, step)
                        print(f"Update best significance: {significance}")
                    
                    val_loss = criterion(out_transform(model(x_val_batch)), y_val_batch).item()
                    train_history["val_losses"].append(val_loss)




        if best_model_mode == "loss":
            print(f"Fold #{step}    Epoch #{epoch}    Best val loss: {best_loss_so_far}")
        if best_model_mode == "significance":
            print(f"Fold #{step}    Epoch #{epoch}    Best significance: {best_significance_so_far}")

        train_history["train_losses"].append(train_loss)
        #train_history["val_losses"].append(val_loss)
        train_history["batch_n"].append(epoch)

    return train_history, best_loss_so_far, model




def train_pytorch_pisa(model_name, step, df_train_, df_val_, trainer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_setup = {
        "sig_vs_ewk": {
            "datasets": ['ewk_lljj_mll105_160_ptj0', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
            "features": training_features_mass + training_features_nomass,
        },
        "sig_vs_dy": {
            "datasets": ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
            "features": training_features_mass + training_features_nomass,
        },
        "no_mass": {
            "datasets": ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
            "features": training_features_nomass,
        },
        "mass": {
            "datasets": ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
            "features": training_features_mass,
        }, 
        "combination": {
            "datasets": ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0', 'vbf_powheg_dipole', 'vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
        },
    }
    
    def train_subnetwork(name, epochs, batch_size, setup):
        df_train = df_train_[df_train_.dataset.isin(setup[name]["datasets"])]
        df_val = df_val_[df_val_.dataset.isin(setup[name]["datasets"])]
        n_training_batches = int(df_train.shape[0] / batch_size)
        print(f"Training subnetwork called {name}")
        print(f"Training dataset size: {df_train.shape[0]}.")
        print(f"Number of batches of size {batch_size} is {n_training_batches}.")
        print(f"Will train for {epochs} epochs.")
        
        nlayers = 3
        nnodes = [64, 32, 16]
        model = NetPisaRun2(name, len(setup[name]["features"]), nlayers, nnodes)
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), eps=1e-07)

        best_loss_so_far = 1000
        best_model = model

        for epoch in range(epochs):
            for batch in tqdm.tqdm(range(n_training_batches)):
                train_batch = df_train.sample(batch_size)
                val_batch = df_val.sample(batch_size)

                x_train_batch = train_batch.loc[:, setup[name]["features"]]
                y_train_batch = train_batch.loc[:, "class"]
                x_train_batch = torch.tensor(x_train_batch.values).float().to(device)
                y_train_batch = torch.tensor(y_train_batch.values).float().view(-1, 1).to(device)

                x_val_batch = val_batch.loc[:, setup[name]["features"]]
                y_val_batch = val_batch.loc[:, "class"]
                x_val_batch = torch.tensor(x_val_batch.values).float().to(device)
                y_val_batch = torch.tensor(y_val_batch.values).float().view(-1, 1).to(device)

                model.train()

                optimizer.zero_grad(set_to_none=True)
                output = model(x_train_batch)
                loss = criterion(output, y_train_batch)
                train_loss = loss.item()
                loss.backward()
                optimizer.step()   

                model.eval()

                with torch.no_grad():
                    output = model(x_val_batch)
                    loss = criterion(output, y_val_batch)
                    val_loss = loss.item()
                    if val_loss < best_loss_so_far:
                        best_loss_so_far = val_loss
                        best_model = copy.deepcopy(model)
                        save_model(model, f"{model_name}_{name}", step)

            print(f"Fold #{step}    Epoch #{epoch}    Best val loss: {best_loss_so_far}")
        return best_model

    def train_combination(subnetworks, epochs, batch_size, freeze, from_saved_model = False):
        weighted = False
        n_training_batches = int(df_train_.shape[0] / batch_size)
        print(f"Training combined subnetwork")
        print(f"Training dataset size: {df_train_.shape[0]}.")
        print(f"Number of batches of size {batch_size} is {n_training_batches}.")
        print(f"Will train for {epochs} epochs.")

        train_history = {
            "train_losses": [],
            "val_losses": [],
            "batch_n": [],
            "significance": [],
            "batch_n_sign": []
        }
        
        nlayers = 3
        nnodes = [64, 32, 16]
        model = NetPisaRun2Combination("combination", nlayers, nnodes, subnetworks, freeze)

        if from_saved_model:
            model_path = f"data/trained_models/vbf/models/{model_name}_combination_{step}.pt"
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), eps=1e-07)

        best_loss_so_far = 1000
        best_significance_so_far = 0
        best_model = model
        batch_n_sign  = 0

        for epoch in range(epochs):
            for batch in tqdm.tqdm(range(n_training_batches)):
                batch_n_sign += 1
                train_batch = df_train_.sample(batch_size)
                val_batch = df_val_.sample(batch_size)

                x_train_batch_nomass = train_batch.loc[:, training_features_nomass]
                x_train_batch_nomass = torch.tensor(x_train_batch_nomass.values).float().to(device)

                x_val_batch_nomass = val_batch.loc[:, training_features_nomass]
                x_val_batch_nomass = torch.tensor(x_val_batch_nomass.values).float().to(device)

                x_train_batch_mass = train_batch.loc[:, training_features_mass]
                x_train_batch_mass = torch.tensor(x_train_batch_mass.values).float().to(device)

                x_val_batch_mass = val_batch.loc[:, training_features_mass]
                x_val_batch_mass = torch.tensor(x_val_batch_mass.values).float().to(device)
                
                y_train_batch = train_batch.loc[:, "class"]
                y_train_batch = torch.tensor(y_train_batch.values).float().view(-1, 1).to(device)
                y_val_batch = val_batch.loc[:, "class"]
                y_val_batch = torch.tensor(y_val_batch.values).float().view(-1, 1).to(device)

                
                model.train()

                optimizer.zero_grad(set_to_none=True)
                output = model(x_train_batch_nomass, x_train_batch_mass)
                loss = criterion(output, y_train_batch)
                train_loss = loss.item()
                loss.backward()
                optimizer.step()   

                model.eval()

                with torch.no_grad():

                    best_model_mode = "loss"
                    #best_model_mode = "significance"

                    if best_model_mode == "loss":
                        output = model(x_val_batch_nomass, x_val_batch_mass)
                        loss = criterion(output, y_val_batch)
                        val_loss = loss.item()

                        if val_loss < best_loss_so_far:
                            best_loss_so_far = val_loss
                            best_model = copy.deepcopy(model)
                            save_model(model, f"{model_name}_combination", step)

                    elif best_model_mode == "significance":                    
                        df_val_aux = df_val_.copy()
                        score_name = "score"
                        #val_input = torch.tensor(df_val_aux.loc[:, trainer.features].values).float()
                        val_input_nomass = torch.tensor(df_val_aux.loc[:, training_features_nomass].values).float()
                        val_input_mass = torch.tensor(df_val_aux.loc[:, training_features_mass].values).float()

                        df_val_aux[score_name] = np.arctanh(
                            model(
                                val_input_nomass.to(device),
                                val_input_mass.to(device)
                            ).cpu()
                        )
                        vbf_signal = df_val_aux[df_val_aux.dataset=='vbf_powheg_dipole']
                        vbf_signal_yield = vbf_signal.wgt_nominal.sum()
                        nbins = 13

                        grid = [(i+1)/nbins for i in range(nbins)]
                        bins = vbf_signal["score"].quantile(grid).values

                        for i in range(nbins - 1):
                            lo = bins[i]
                            hi = bins[i + 1]
                            cut = (df_val_aux[score_name] > lo) & (
                                df_val_aux[score_name] <= hi
                            )
                            df_val_aux.loc[cut, "bin_number"] = i

                        signal = ['vbf_powheg_dipole', 'ggh_amcPS']
                        background = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0']
                        df_sig = df_val_aux[df_val_aux.dataset.isin(signal)]
                        df_bkg = df_val_aux[df_val_aux.dataset.isin(background)]
                        yields_sig = df_sig.groupby("bin_number")["wgt_nominal"].sum()
                        yields_bkg = df_bkg.groupby("bin_number")["wgt_nominal"].sum()

                        significance = np.sqrt((yields_sig*yields_sig / yields_bkg).sum())

                        train_history["significance"].append(significance)
                        train_history["batch_n_sign"].append(batch_n_sign)

                        if significance > best_significance_so_far:
                            best_significance_so_far = significance
                            best_model = copy.deepcopy(model)
                            save_model(model, f"{model_name}_combination", step)
                            print(f"Update best significance: {significance}")

                        val_loss = criterion(model(x_val_batch_nomass, x_val_batch_mass), y_val_batch).item()
                        train_history["val_losses"].append(val_loss)

            if best_model_mode == "loss":
                print(f"Fold #{step}    Epoch #{epoch}    Best val loss: {best_loss_so_far}")
            if best_model_mode == "significance":
                print(f"Fold #{step}    Epoch #{epoch}    Best significance: {best_significance_so_far}")
        return train_history

    subnetworks = {}
    retrain = True
    if retrain:
        for name in ["sig_vs_ewk", "sig_vs_dy", "no_mass", "mass"]:
            subnetworks[name] = train_subnetwork(name, 10, 65536, training_setup)
    else:
        nlayers = 3
        nnodes = [64, 32, 16]
        for name in ["sig_vs_ewk", "sig_vs_dy", "no_mass", "mass"]:
            subnetworks[name] = NetPisaRun2(name, len(training_setup[name]["features"]), nlayers, nnodes)
            subnetworks[name].to(device)
            model_path = (
                f"data/trained_models/vbf/models/{model_name}_{name}_{step}.pt"
            )
            subnetworks[name].load_state_dict(torch.load(model_path, map_location=device))

    def plot_history(key, train_history):
        if ("batch_n_sign" in train_history) and ("significance" in train_history):
            fig = plt.figure()
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 6)

            opts = {"linewidth": 2}
            ax.plot(train_history["batch_n_sign"], train_history["significance"], label="significance (1/4 data)", **opts)
            ax.plot(train_history["batch_n_sign"], train_history["val_losses"], label="validation loss", **opts)
            ax.legend(prop={"size": "x-small"})
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Significance")
            fig.tight_layout()
            fig.savefig(f"plots/pytorch/significance_{model_name}_combination_{key}_{step}.png")


    ths = {}
    ths["first"] = train_combination(subnetworks, 10, 65536, freeze=["sig_vs_ewk", "sig_vs_dy", "no_mass"])
    plot_history("first", ths["first"])
    ths["second"] = train_combination(subnetworks, 10, 65536, freeze=["sig_vs_ewk", "sig_vs_dy"], from_saved_model=True)
    plot_history("second", ths["second"])


    #train_combined(model, 5, 1024, training_setup)
    # then train_combined: load weights from sig_vs_ewk and sig_vs_dy
    


    
    """
    train_history = {
        "train_losses": [],
        "val_losses": [],
        "batch_n": []
    }
    epochs = 100
    
    batch_size = 1024
    n_training_batches = int(df_train.shape[0] / batch_size)
    print(f"Training dataset size: {df_train.shape[0]}.")
    print(f"Number of batches of size {batch_size} is {n_training_batches}.")
    print(f"Will train for {epochs} epochs.")
    best_loss_so_far = 1000
    best_model = model
    for epoch in range(epochs):
        for batch in tqdm.tqdm(range(n_training_batches)):
            train_batch = df_train.sample(batch_size)
            val_batch = df_val.sample(batch_size)

            x_train_batch = train_batch.loc[:, trainer.features]
            y_train_batch = train_batch.loc[:, "class"]
            x_train_batch = torch.tensor(x_train_batch.values).float().to(device)
            y_train_batch = torch.tensor(y_train_batch.values).float().view(-1, 1).to(device)

            x_val_batch = val_batch.loc[:, trainer.features]
            y_val_batch = val_batch.loc[:, "class"]
            x_val_batch = torch.tensor(x_val_batch.values).float().to(device)
            y_val_batch = torch.tensor(y_val_batch.values).float().view(-1, 1).to(device)

            model.train()

            optimizer.zero_grad(set_to_none=True)
            output = model(x_train_batch)
            loss = criterion(output, y_train_batch)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()   

            model.eval()

            with torch.no_grad():
                output = model(x_val_batch)
                loss = criterion(output, y_val_batch)
                val_loss = loss.item()
                if val_loss < best_loss_so_far:
                    best_loss_so_far = val_loss
                    best_model = copy.deepcopy(model)
                    save_model(model, model_name, step)

        print(f"Fold #{step}    Epoch #{epoch}    Best val loss: {best_loss_so_far}")

        train_history["train_losses"].append(train_loss)
        train_history["val_losses"].append(val_loss)
        train_history["batch_n"].append(epoch)

    return train_history
    """




    
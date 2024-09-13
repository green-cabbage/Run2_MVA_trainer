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
#import mplhep as hep
import tqdm
from distributed import LocalCluster, Client, progress
import os
import coffea.util as util
import time

#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel', 'dimuon_eta', 'dimuon_mass', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta nominal', 'jet1_phi nominal', 'jet1_pt nominal', 'jet1_qgl nominal', 'jet2_eta nominal', 'jet2_phi nominal', 'jet2_pt nominal', 'jet2_qgl nominal', 'jj_dEta nominal', 'jj_dPhi nominal', 'jj_eta nominal', 'jj_mass nominal', 'jj_mass_log nominal', 'jj_phi nominal', 'jj_pt nominal', 'll_zstar_log nominal', 'mmj1_dEta nominal', 'mmj1_dPhi nominal', 'mmj2_dEta nominal', 'mmj2_dPhi nominal', 'mmj_min_dEta nominal', 'mmj_min_dPhi nominal', 'mmjj_eta nominal', 'mmjj_mass nominal', 'mmjj_phi nominal', 'mmjj_pt nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt', 'mu2_pt_over_mass', 'zeppenfeld nominal']
#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet1_qgl_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal', 'jet2_qgl_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']

# training_features = ['dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld', 'njets']  # AN 19-124


training_features = ['dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld', 'njets', "jet1_qgl", "jet2_qgl"] 


#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal',  'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']

training_samples = {
        "background": [
            "dy_M-100To200", 
            "ttjets_dl",
            "ttjets_sl",
            "st_tw_top",
            "st_tw_antitop",
            "ww_2l2nu",
            "wz_1l1nu2q",
            "wz_2l2q",
            "wz_3lnu",
            "zz",
            "ewk_lljj_mll50_mjj120",
        ],
        "signal": ["ggh_powheg", "vbf_powheg"],
        
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

def convert2df(dak_zip, dataset: str, is_vbf=False):
    """
    small wrapper that takes delayed dask awkward zip and converts them to pandas dataframe
    with zip's keys as columns with extra column "dataset" to be named the string value given
    Note: we also filter out regions not in h-peak region first, and apply cuts for
    ggH production mode
    """
    # filter out arrays not in h_peak
    is_hpeak = dak_zip.h_peak
    vbf_cut = ak.fill_none(dak_zip.vbf_cut, value=False)
    if is_vbf: # VBF
        prod_cat_cut =  vbf_cut
    else: # ggH
        prod_cat_cut =  ~vbf_cut
    btag_cut = ak.fill_none((dak_zip.nBtagLoose >= 2), value=False) | ak.fill_none((dak_zip.nBtagMedium >= 1), value=False)
    
    category_selection = (
        prod_cat_cut & 
        is_hpeak &
        ~btag_cut # btag cut is for VH and ttH categories
    )
    computed_zip = dak_zip[category_selection].compute()
    df = ak.to_dataframe(computed_zip)
    df.fillna(-99,inplace=True)
    # add columns
    df["dataset"] = dataset 
    df["cls_avg_wgt"] = -1.0
    # df["wgt_nominal_total"] = np.abs(df["wgt_nominal_total"]) # enforce poisitive weights
    # drop negative values
    positive_wgts = df["wgt_nominal_total"] > 0 
    df = df.loc[positive_wgts]
    
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
    df_info["iclass"] = df_info["iclass"].fillna(-1).astype(int)
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

    # apply dimuon_ebe_mass_res to the weights
    # original start -----------------------------------------------
    df.loc[df['dataset']=="ggh_powheg",'wgt_nominal_total'] = np.divide(df[df['dataset']=="ggh_powheg"]['wgt_nominal_total'], df[df['dataset']=="ggh_powheg"]['dimuon_ebe_mass_res'])
    df.loc[df['dataset']=="vbf_powheg",'wgt_nominal_total'] = np.divide(df[df['dataset']=="vbf_powheg"]['wgt_nominal_total'], df[df['dataset']=="vbf_powheg"]['dimuon_ebe_mass_res'])
    # original end -----------------------------------------------

    # test start -----------------------------------------------
    # is_signal = df['dataset']=="ggh_powheg"
    # df.loc[is_signal,'wgt_nominal_total'] = np.divide(1, df[df['dataset']=="ggh_powheg"]['dimuon_ebe_mass_res'])
    # df.loc[~is_signal,'wgt_nominal_total'] = 1
    # # print(f"df[wgt_nominal_total]: {df['wgt_nominal_total']}")
    # test end -----------------------------------------------

    
    df.fillna(-99,inplace=True)
    #print(df.head)
    columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta',]
    columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta','jet2_pt']
    columns2 = ['mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 'mmj_min_dEta', 'mmj_min_dPhi']
    # with open("df.txt", "w") as f:
    #     print(df[columns_print], file=f)
    # with open("df2.txt", "w") as f:
    #     print(df[columns2], file=f)
    # print(df[df['dataset']=="ggh_powheg"].head)
    print(f"prepare_dataset df: {df["dataset","class_name", "iclass"]}")
    return df

# def scale_data(inputs, x_train, x_val, df_train, label):
#     x_mean = np.mean(x_train[inputs].values,axis=0)
#     x_std = np.std(x_train[inputs].values,axis=0)
#     training_data = (x_train[inputs]-x_mean)/x_std
#     validation_data = (x_val[inputs]-x_mean)/x_std
#     np.save(f"{output_path}/{name}_{year}/scalers_{name}_{year}_{label}", [x_mean, x_std])
#     return training_data, validation_data
def scale_data_withweight(inputs, x_train, x_val, df_train, label):
    masked_x_train = np.ma.masked_array(x_train[x_train[inputs]!=-99][inputs], np.isnan(x_train[x_train[inputs]!=-99][inputs]))
    #x_mean = np.average(x_train[inputs].values,axis=0, weights=df_train['training_wgt'].values)
    x_mean = np.average(masked_x_train,axis=0, weights=df_train['training_wgt'].values).filled(np.nan)
    #x_std = np.std(x_train[inputs].values,axis=0)
    #masked_x_std = np.ma.masked_array(x_train[x_train[inputs]!=-99][inputs], np.isnan(x_train[x_train[inputs]!=-99][inputs]))
    #x_std = np.average((x_train[inputs].values-x_mean)**2,axis=0, weights=df_train['training_wgt'].values)
    x_std = np.average((masked_x_train-x_mean)**2,axis=0, weights=df_train['training_wgt'].values).filled(np.nan)
    sumw2 = (df_train['training_wgt']**2).sum()
    sumw = df_train['training_wgt'].sum()
    
    x_std = np.sqrt(x_std/(1-sumw2/sumw**2))
    training_data = (x_train[inputs]-x_mean)/x_std
    validation_data = (x_val[inputs]-x_mean)/x_std
    output_path = args["output_path"]
    print(f"output_path: {output_path}")
    print(f"name: {name}")
    save_path = f'{output_path}/bdt_{name}_{year}'
    print(f"scalar save_path: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + f"/scalers_{name}_{label}", [x_mean, x_std]) #label contains year
    return training_data, validation_data

def classifier_train(df, args):
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
        classes = {
            0 : 'dy_M-100To200',
            1 : 'ggh_powheg',
        }
        print(f"classes: {classes}")
        # for icls, cls in enumerate(classes):
        # original start -------------------------------------------------------
        for icls, cls in classes.items():
            print(f"icls: {icls}")
            train_evts = len(y_train[y_train==icls])
            df_train.loc[y_train==icls,'cls_avg_wgt'] = df_train.loc[y_train==icls,'wgt_nominal_total'].values.mean()
            df_val.loc[y_val==icls,'cls_avg_wgt'] = df_val.loc[y_val==icls,'wgt_nominal_total'].values.mean()
            df_eval.loc[y_eval==icls,'cls_avg_wgt'] = df_eval.loc[y_eval==icls,'wgt_nominal_total'].values.mean()
            print(f"{train_evts} training events in class {cls}")

        # df_train['training_wgt'] = df_train['wgt_nominal_total']/df_train['cls_avg_wgt']
        # df_val['training_wgt'] = df_val['wgt_nominal_total']/df_val['cls_avg_wgt']
        # df_eval['training_wgt'] = df_eval['wgt_nominal_total']/df_eval['cls_avg_wgt']
        df_train.loc[:,'training_wgt'] = df_train['wgt_nominal_total']/df_train['cls_avg_wgt']
        df_val.loc[:,'training_wgt'] = df_val['wgt_nominal_total']/df_val['cls_avg_wgt']
        df_eval.loc[:,'training_wgt'] = df_eval['wgt_nominal_total']/df_eval['cls_avg_wgt']
        # original end -------------------------------------------------------

        # test start -------------------------------------------------------
        # # just take the wgt_nominal_total values 
        # df_train.loc[:,'training_wgt'] = df_train['wgt_nominal_total']
        # df_val.loc[:,'training_wgt'] = df_val['wgt_nominal_total']
        # df_eval.loc[:,'training_wgt'] = df_eval['wgt_nominal_total']
        # test end -------------------------------------------------------

        
        # scale data
        #x_train, x_val = scale_data(training_features, x_train, x_val, df_train, label)#Last used
        x_train, x_val = scale_data_withweight(training_features, x_train, x_val, df_train, label)
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
            #print("I am here")
            w_train = df_train['training_wgt'].values
            w_val = df_val['training_wgt'].values
            w_eval = df_eval['training_wgt'].values
            shuf_ind_tr = np.arange(len(xp_train))
            np.random.shuffle(shuf_ind_tr)
            shuf_ind_val = np.arange(len(xp_val))
            np.random.shuffle(shuf_ind_val)
            xp_train = xp_train[shuf_ind_tr]
            xp_val = xp_val[shuf_ind_val]
            y_train = y_train[shuf_ind_tr]
            y_val = y_val[shuf_ind_val]
            #print(np.isnan(xp_train).any())
            #print(np.isnan(y_train).any())
            #print(np.isinf(xp_train).any())
            #print(np.isinf(y_train).any())
            #print(np.isfinite(x_train).all())
            #print(np.isfinite(y_train).all())
            
            w_train = w_train[shuf_ind_tr]
            w_val = w_val[shuf_ind_val]
            #data_dmatrix = xgb.DMatrix(data=X,label=y)
            # # original start ---------------------------------------------------------------   
            model = xgb.XGBClassifier(max_depth=4,#for 2018
                                      #max_depth=6,previous value
                                      n_estimators=100000,
                                      #n_estimators=100,
                                      early_stopping_rounds=80, 
                                      eval_metric="logloss",
                                      #learning_rate=0.001,#for 2018
                                      learning_rate=0.1,#previous value
                                      #reg_alpha=0.680159426755822,
                                      #colsample_bytree=0.47892268305051233,
                                      colsample_bytree=0.5,
                                      min_child_weight=3,
                                      subsample=0.5,
                                      #reg_lambda=16.6,
                                      #gamma=24.505,
                                      #n_jobs=35,
                                      tree_method='hist')
                                      #tree_method='hist')
            # # original end ---------------------------------------------------------------
            
            # AN Model start ---------------------------------------------------------------   
            # model = xgb.XGBClassifier(max_depth=4,
            #                           n_estimators=1000,
            #                           early_stopping_rounds=80, 
            #                           eval_metric="logloss",
            #                           #learning_rate=0.001,#for 2018
            #                           learning_rate=0.1,#previous value
            #                           #reg_alpha=0.680159426755822,
            #                           #colsample_bytree=0.47892268305051233,
            #                           colsample_bytree=0.5,
            #                           min_child_weight=3,
            #                           subsample=0.5,
            #                           #reg_lambda=16.6,
            #                           #gamma=24.505,
            #                           #n_jobs=35,
            #                           tree_method='hist')
            # AN Model end ---------------------------------------------------------------
            
            print(model)
            print(f"negative w_train: {w_train[w_train <0]}")

            eval_set = [(xp_train, y_train), (xp_val, y_val)]#Last used

            model.fit(xp_train, y_train, sample_weight = w_train, eval_set=eval_set, verbose=True)


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
            save_path = f"output/bdt_{name}_{year}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}/Validation_{label}.png")
            
            y_pred = model.predict_proba(xp_val)[:, 1].ravel()
            y_pred_train = model.predict_proba(xp_train)[:, 1].ravel()
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print(y_pred)
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print(y_val)
            # original start ------------------------------------------------------------------------------
            nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_val.ravel(), y_pred, sample_weight=w_val) 
            # original end ------------------------------------------------------------------------------

            # test start ------------------------------------------------------------------------------
            # nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_val.ravel(), y_pred) # test
            # test end ------------------------------------------------------------------------------
            
            np.save(f"test_roc_curve", [y_val.ravel(), y_pred]) 
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

            
            # Also save ROC curve for evaluation just in case start --------------
            shuf_ind_eval = np.arange(len(xp_eval))
            xp_eval = xp_eval[shuf_ind_eval]
            y_eval = y_eval[shuf_ind_eval]
            y_eval_pred = model.predict_proba(xp_eval)[:, 1].ravel()
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

            
            results = model.evals_result()
            print(results.keys())
            plt.plot(results['validation_0']['logloss'], label='train')
            plt.plot(results['validation_1']['logloss'], label='test')
            # show the legend
            plt.legend()
            plt.savefig(f"output/bdt_{name}_{year}/Loss_{label}.png")

            feature_important = model.get_booster().get_score(importance_type='gain')
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=training_features, columns=["score"]).sort_values(by = "score", ascending=True)
            data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10))
            plt.savefig(f"test.png")
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
                scalers_path = f"{output_path}/dnn_{name}_{year}/scalers_{name}_{eval_label}.npy"
                scalers = np.load(scalers_path)
                # model_path = f'output/trained_models/test_{label}.h5'
                model_path = f'{output_path}/dnn_{name}_{year}/{name}_{eval_label}.h5'
                dnn_model = load_model(model_path)
                df_i = df[eval_filter]
                #if args['r']!='h-peak':
                #    df_i['dimuon_mass'] = 125.
                #if args['do_massscan']:
                #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift

                df_i = (df_i[training_features]-scalers[0])/scalers[1]
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

            eval_label = f"{args['year']}_{args['label']}{eval_folds[0]}"
            print(f"eval_label: {eval_label}")
            
            # scalers_path = f"{output_path}/{name}_{year}/scalers_{name}_{eval_label}.npy"
            # start_path = "/depot/cms/hmm/copperhead/trained_models/"
            output_path = args["output_path"]
            print(f"output_path: {output_path}")
            scalers_path = f"{output_path}/bdt_{name}_{year}/scalers_{name}_{eval_label}.npy"

            print(f"scalers_path: {scalers_path}")
            #scalers_path = f'output/trained_models_nest10000/scalers_{label}.npy'
            scalers = np.load(scalers_path)

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
            df_i = (df_i[training_features]-scalers[0])/scalers[1]
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


    
    load_path = "/depot/cms/users/yun79/results/stage1/DNN_test2/2018/f1_0"
    sample_l = training_samples["background"] + training_samples["signal"]
    # old code start --------------------------------------------------------------------------------------------
    # zip_ggh = dak.from_parquet(load_path+"/ggh_powheg/*/*.parquet")
    # is_vbf = sysargs.is_vbf
    # df_ggh = convert2df(zip_ggh, "ggh_powheg", is_vbf=is_vbf)
    # zip_dy = dak.from_parquet(load_path+"/dy_M-100To200/*/*.parquet")

    
    # # df_dy = convert2df(zip_dy, "dy_M-100To200", is_vbf=is_vbf) # for full run
    # df_dy = convert2df(zip_ggh, "dy_M-100To200", is_vbf=is_vbf) # for quick test

    
    
    
    # df_total = pd.concat([df_ggh,df_dy],ignore_index=True)
    # old code end --------------------------------------------------------------------------------------------
    # new code start --------------------------------------------------------------------------------------------
    df_l = []
    print(f"sample_l: {sample_l}")
    for sample in sample_l:
        zip_sample = dak.from_parquet(load_path+f"/{sample}/*/*.parquet")
        is_vbf = sysargs.is_vbf
        df_sample = convert2df(zip_sample, sample, is_vbf=is_vbf)
        df_l.append(df_sample)
    df_total = pd.concat(df_l,ignore_index=True)   
    # new code end --------------------------------------------------------------------------------------------
    # apply random shuffle, so that signal and bkg samples get mixed up well
    df_total = df_total.sample(frac=1, random_state=123)
    # print(f"df_total after shuffle: {df_total}")
    
    print(f"len(df_dy.index): {len(df_dy.index)}")
    print(f"len(df_ggh.index): {len(df_ggh.index)}")
    del df_ggh # delete redundant df to save memory. Not sure if this is necessary
    del df_dy # delete redundant df to save memory. Not sure if this is necessary
    # print(f"df_total: {df_total}")
    print("starting prepare_dataset")
    df_total = prepare_dataset(df_total, training_samples)
    df_total.to_csv("test.csv")
    raise ValueError
    print("prepare_dataset done")
    # raise ValueError
    
    #print(df["class"])
    #print([ x for x in df.columns if "nominal" in x])
    # print(df[df['dataset']=="dy_M-100To200"])
    # print(df[df['dataset']=="ggh_powheg"])

    classifier_train(df_total, args)
    evaluation(df_total, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    #print(df)
    runtime = int(time.time()-start_time)
    print(f"run time is {runtime} seconds")



import optuna
from xgboost import XGBClassifier
from modules.utils import customROC_curve_AN, auc_from_eff


def objective(trial, xp_train, xp_val, y_train, y_val, w_train, w_val, weight_nom_val, random_seed) -> float:
    params = {
        # "objective": "binary:logistic",
        # "eval_metric": "auc",
        "eval_metric": "logloss",
        # "tree_method": "hist",            # change to "gpu_hist" if you have a GPU
        "random_state": random_seed,
        # "n_estimators": trial.suggest_int("n_estimators", 500, 2100),
        # "max_depth": trial.suggest_int("max_depth", 3, 10),
        # "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        # "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 20.0),
        # "max_bin": trial.suggest_int("max_bin", 10, 80),
        "n_jobs": 30,
    }
    verbosity=2
    clf = XGBClassifier(
        n_estimators= trial.suggest_int("n_estimators", 500, 2100),           # Number of trees
        max_depth=trial.suggest_int("max_depth", 3, 10),                 # Max depth
        learning_rate=trial.suggest_float("learning_rate", 0.05, 0.3),          # Shrinkage
        subsample=trial.suggest_float("subsample", 0.2, 1.0),               # Bagged sample fraction
        min_child_weight=trial.suggest_float("min_child_weight", 0.001, 10.0),
        tree_method='hist',          # Needed for max_bin
        max_bin=trial.suggest_int("max_bin", 10, 80),                  # Number of cuts
        # objective='binary:logistic', # CrossEntropy (logloss)
        # use_label_encoder=False,     # Optional: suppress warning
        eval_metric='logloss',       # Ensures logloss used during training
        n_jobs=30,                   # Use all CPU cores
        # scale_pos_weight=scale_pos_weight*0.005,
        # scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=15,#15
        verbosity=verbosity,
        random_state=random_seed,
    )

    # clf = XGBClassifier(**params)
    eval_set = [(xp_train, y_train), (xp_val, y_val)]
    
    clf.fit(
        xp_train, y_train,
        # eval_set=[(xp_val, y_val)],
        eval_set=eval_set,
        sample_weight=w_train,
        verbose=False
    )

    proba = clf.predict_proba(xp_val)[:, 1]
    # auc = roc_auc_score(y_val, proba, sample_weight=w_val)
    eff_bkg_val, eff_sig_val, thresholds_val, TpFpTnFn_df_val = customROC_curve_AN(y_val, proba, weight_nom_val, doClassBalance=False)
    
    auc  = auc_from_eff(eff_sig_val, eff_bkg_val)
    

    trial.report(auc, step=getattr(clf, "best_iteration", 0) or 0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return auc
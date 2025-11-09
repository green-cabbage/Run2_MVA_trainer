"""
Plot ROC curves by fold for given model name and years.

Usage:
    python plot_roc_by_fold.py --model <model_base_name> --years 2016 2017 2018 all
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from modules.utils import PlotRocByFold

plt.style.use(hep.style.CMS)


def auc_from_eff(eff_sig, eff_bkg):
    """
    Compute the area under the ROC curve from signal and background efficiencies.
    """
    fpr = 1.0 - np.asarray(eff_bkg)
    tpr = np.asarray(eff_sig)
    order = np.argsort(fpr)
    return np.trapezoid(tpr[order], fpr[order])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ROC curves by fold for given model name and list of years."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name (without year suffix)."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        help="List of years to process, e.g. 2016 2017 2018 all"
    )
    args = parser.parse_args()

    base_model = args.model
    years = args.years

    for year in years:
        model_name = f"bdt_{base_model}_{year}"
        print(f"[INFO] Processing model: {model_name}")
        PlotRocByFold(model_name, year)
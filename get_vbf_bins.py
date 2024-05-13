import glob
from stage2.categorizer import split_into_channels, categorize_dnn_output
from stage2.mva_evaluators import evaluate_pytorch_dnn, evaluate_pytorch_dnn_pisa
from python.io import load_dataframe

parameters = {
    "global_path": "/depot/cms/hmm/vscheure/",
    "label": "UL16_stitchingtest5",
    "channels": ["vbf"],
    "custom_npartitions": {
        "vbf_powheg": 1,
    },
    "models_path": "data/trained_models/",
}

dataset = "vbf_powheg"
#model_name = "pytorch_may18"
#model_name = "pytorch_may24_pisa"
model_name = "ValerieDNNtest2"
score_name = f"score_{model_name}_nominal"
channel = "vbf"
region = "h-peak"

#for year in ["2016", "2017", "2018"]:
for year in ["2016"]:
    print(year)
    #paths = glob.glob(f"/depot/cms/hmm/coffea/{year}_2022apr6/{dataset}/*.parquet")
    paths = glob.glob(f"/depot/cms/hmm/vscheure/UL16_stitchingtest5/stage1_output/{year}/{dataset}/*.parquet")

    df = load_dataframe(None, parameters, inputs=paths, dataset=dataset)
    df = df.compute()

    split_into_channels(df, v="nominal")

    df.loc[df[f"channel_nominal"] == channel, score_name] = evaluate_pytorch_dnn(
        df[df[f"channel_nominal"] == channel],
        "nominal",
        model_name,
        parameters,
        score_name,
        channel,
    )
    #print(df.loc[df[f"channel_nominal"] == channel, score_name].describe())
    #import numpy as np
    #df["new"] = df[score_name].apply(np.tanh)
    #for i in range(4):
    #    print(df.loc[(df[f"channel_nominal"] == channel)&(df.event.mod(4)==i), score_name].value_counts())
    #print(df.loc[(df[f"channel_nominal"] == channel)&(df[score_name]>2.0421700477600098), "wgt_nominal"].sum())
    #print(df.loc[(df[f"channel_nominal"] == channel)&(df[score_name]>2.1076695919036865), "wgt_nominal"].sum())

    """
    df.loc[df[f"channel_nominal"] == channel, score_name] = evaluate_pytorch_dnn_pisa(
        df[df[f"channel_nominal"] == channel],
        "nominal",
        model_name,
        parameters,
        score_name,
        channel,
    )
    """

    categorize_dnn_output(df, score_name, channel, region, str(year))



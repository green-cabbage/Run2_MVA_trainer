#!/bin/bash
set -e

# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_DavideEbeSigOnly_HyperParamed"
# model_name="Run3PrelimResultsJan25_2026"
# model_name="Run3PrelimResultsJan25_2026_NoAnnhilateWgts"
# model_name="Run3PrelimResultsJan25_2026_DataCorrected"
# model_name="Run3PrelimResultsJan29_2026_reducedInput"
# model_name="Run3PrelimResultsJan29_2026_reducedInput2"
# model_name="Run3PrelimResultsFeb3_2026_jecjer"
# model_name="Run3PrelimResultsFeb4_2026_jecjer"
# model_name="Run3PrelimResultsFeb09_2026_jecjer"
# model_name="Run3PrelimResultsFeb09_2026_jecjer_onehotencode"
# model_name="Run3PrelimResultsFeb10_2026_jecjer_flatDimuMass"
model_name="test"


# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
# label="DYMiNNLO_11Apr2025"
# label="fullRun_May30_2025"
# label="fullRun_Jun21_2025"
# label="fullRun_Jun23_2025_1n2Revised"
# label="BSC_off_Aug26_2025"
# label="Run3_nanoAODv15_24Jan2025"
# label="Run3_nanoAODv12_01Feb_JecJer"
# label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"
# label="Run3_nanoAODv12_09Feb_FilterJetsHorn30GeV"
label="Run3_nanoAODv12_15Feb_FilterJetsHorn30GeV"

# year="2018"
# year="2017"
# year="2016postVFP"
# year="2016preVFP"
# year="2016"
# year="all"
year="2022postEE"
python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/${label}/stage1_output"

# year="2024"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/${label}/stage1_output"


# year="2023BPix"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output"
# year="2023"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output"
# year="2022postEE"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output"
# year="2022preEE"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run3_nanoAODv12_23Jan_JVMFilterJets/stage1_output"

# year="2017"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"

# year="2018"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"

# year="all"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"

# # add extra plots
# years="2016 2017 2018 all"
# python plot_roc_byFoldNYear.py --model $model_name --years $years
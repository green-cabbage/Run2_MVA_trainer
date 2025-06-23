#!/bin/bash
set -e

# model_name="V2_UL_Mar20_2025_DyTtStVvEwkGghVbf"
# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf"
# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight"
# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_allOtherParamsOn"
# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight"
# model_name="V2_UL_Mar25_2025_DyTtStVvEwkGghVbf_scale_pos_weight_newZpt"
# model_name="V2_UL_Mar26_2025_DyGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar26_2025_DyTtStVvEwkGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar26_2025_DyTtStVvEwkGgh_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeAllJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_Njets"
# model_name="V2_UL_Mar30_2025_DyMinnloTtStVvEwkGghVbf"
# model_name="V2_UL_Mar30_2025_DyMinnloTtStVvEwkGghVbf_allOtherParamsOn"
# model_name="V2_UL_Mar30_2025_DyMinnloTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt5"
# model_name="V2_UL_Mar30_2025_DyTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt0_75"
# model_name="V2_UL_Mar30_2025_DyTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt1_5_maxDep8"
# model_name="V2_UL_Mar30_2025_DyTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt1_25"

# model_name="V2_UL_Apr09_2025_DyTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt0_75"
# model_name="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_allOtherParamsOn_ScaleWgt0_75"
# model_name="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOff"
# model_name="V2_UL_Apr09_2025_DyTtStVvEwkGghVbf_hyperParamOff"
# model_name="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"
# model_name="V2_UL_Apr11_2025_DyTtStVvEwkGghVbf"
# model_name="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf"
# model_name="V2_UL_Apr11_2025_DyMinnloTtStVvEwkGghVbf_bTagMediumFix"
# model_name="V2_UL_Jun09_2025"
# model_name="V2_fullRun_Jun21_2025_allYear"
model_name="V2_fullRun_Jun21_2025_1n2Revised"



# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_30Mar2025"
# label="DYamcNLO_11Apr2025"
# label="DYMiNNLO_11Apr2025"
# label="fullRun_May30_2025"
# label="fullRun_Jun21_2025"
label="fullRun_Jun23_2025_1n2Revised"

# year="2018"
# year="2016postVFP"
# year="2016preVFP"
# year="2016"
year="all"
python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"



# year="2017"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"


# year="2016"
# python my_trainer_withWeight_gpu.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"

#!/bin/bash
set -e

model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_Variation1"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_Variation2"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_addEbeMassRel"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_var1"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_test"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_Davide"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_StdClsWgt_Peking"
# model_name="V2_fullRun_Jun21_2025_1n2Revised_ReProduction_Run3_ClsWgtC1000"


label="fullRun_Jun23_2025_1n2Revised"


year="all"
# year="2016postVFP"

python BDT_BayesianOptim.py --name $model_name --year $year -load  "/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output"

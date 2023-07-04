//
// Created by SXT on 2022/10/5.
//

#ifndef PIMCOMP_PREPARATION_H
#define PIMCOMP_PREPARATION_H

#include "../common.h"
#include "../configure.h"

void EliminatePaddingOperator(std::string model_name, Json::Value & DNNInfo);
void EliminateBatchNormOperator(std::string model_name, Json::Value & DNNInfo);
void PreProcess(Json::Value & DNNInfo);
void GetStructNodeListFromJson(Json::Value DNNInfo);
void CopyFromOriginNodeList();
void ShowModelInfo();
void GetConvPoolInputOutputInfo();
void GetTopologyInformation();
void GetConvPoolInputOutputInfoForInputPreparation();
void SaveIntermediateInfo(Json::Value DNNInfo);
void EP_delay_for_conv_and_pool();
void GetPriorInfoForElementPipeline();
#endif //PIMCOMP_PREPARATION_H

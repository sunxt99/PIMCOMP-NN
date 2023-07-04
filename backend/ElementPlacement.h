//
// Created by SXT on 2022/8/21.
//

#ifndef PIMCOMP_ELEMENTPLACEMENT_H
#define PIMCOMP_ELEMENTPLACEMENT_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class ElementPlacement
{
public:
    ElementPlacement(int mode);
    void PlaceElement();
    void SaveSimulationInfo();
    void SaveVerificationInfo(Json::Value DNNInfo);
private:
    ////////////////////////// For Simulation //////////////////////////
    Json::Value PIMCOMP_SIMULATION_INFO;
    int effective_instruction_group_num;
    int placement_mode;
    void DetermineCoreMap();
    void PlaceCore();
    void PlaceCrossbar();
    void Clear();
};




#endif //PIMCOMP_ELEMENTPLACEMENT_H

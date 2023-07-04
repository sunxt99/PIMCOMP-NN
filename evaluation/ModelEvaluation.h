//
// Created by SXT on 2022/9/20.
//

#ifndef PIMCOMP_MODELEVALUATION_H
#define PIMCOMP_MODELEVALUATION_H

#include "../common.h"
#include "../configure.h"
#include "vector"
#include "../backend/PIMCOMPVariable.h"
#include "EvaluationConfig.h"

class ModelEvaluation
{
public:
    ModelEvaluation(int index, double given_interval);
    ~ ModelEvaluation();
    void EvaluateCompute();
    void SaveEvaluation();
private:
    double EVA_MVMUL_start_interval;
    int EvaluationIndex;
    int instruction_group_num;
    bool ** COMM_cycle_flag;
    bool ** MEM_cycle_flag;
    bool ** WB_cycle_flag;
    void Clear();
    ////////////////////////////////////////////// Evaluate Computation //////////////////////////////////////////////
    // Get the instruction index of the SEND/RECV instruction in the core according to comm_index
    // 根据comm_index得到该SEND/RECV指令在core中的位置
    std::map<int, int> comm_index_2_index_in_core;
    // Get the core index of the SEND/RECV instruction in the core according to comm_index
    // 根据comm_index得到该SEND/RECV指令所在的core
    std::map<int, int> comm_index_2_core_index;
    void EvaluateRecursionSingleInstructionGroup(int instruction_group_index, int core_index, int index_in_core);
    void ShowEvaluationResultSingleInstructionGroup();
    void ResetSingleInstructionGroup(bool clear_other_info);
    int CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle);
    int CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle); // LOAD
    int CheckGlobalMemoryBandwidthWB(int chip_index, long long current_time, int global_memory_needed_cycle); // WB
    void ShowVisualResultOfCore();
};


#endif //PIMCOMP_MODELEVALUATION_H

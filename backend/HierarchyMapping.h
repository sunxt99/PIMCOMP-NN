//
// Created by SXT on 2022/8/19.
//

#ifndef PIMCOMP_HIERARCHYMAPPING_H
#define PIMCOMP_HIERARCHYMAPPING_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class HierarchyMapping
{
public:
    HierarchyMapping();
    void MapHierarchy(std::string replicating_method);
    void ShowOriginalInfo();
    void ShowMappingInfo();
    void SaveMappingResult();
private:
    int node_num;
    int * ResourceList;
    void MapBaseline();
    void MapBaseline2();
    void MapDistributed();
    void GatherForAccelerate();
    int MapDistributedTry();
    void AllocateMapInfo();
    int GetInputElementNumFromAG(int node_index, int index_in_replication);
    void Check();
    void Clear();
    // For GA method
    void LoadGAMappingResult(int candidate_index);
    // Fast Evaluation For Element Pipeline
    void FastEvaluationForElement();
    void FastSingleInstructionGroupEvaluation(int instruction_group_index, int core_index, int index_in_core);
    int CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle);
    void FastEvaluationInstructionGroupNum();
    bool ** COMM_cycle_flag;
    // Fast Evaluation For Batch Pipeline
    void FastEvaluationForBatch();
    int CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle);
    bool ** MEM_cycle_flag;
};


#endif //PIMCOMP_HIERARCHYMAPPING_H

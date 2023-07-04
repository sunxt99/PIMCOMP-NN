//
// Created by SXT on 2022/9/16.
//

#ifndef PIMCOMP_BATCHPIPELINESCHEDULE_H
#define PIMCOMP_BATCHPIPELINESCHEDULE_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class BatchPipelineSchedule
{
public:
    BatchPipelineSchedule(std::string model_name_);
    void ScheduleExecution();

private:
    std::string model_name;
    int node_num;
    int AG_num_total;
    std::vector<int> post_start_address;
    std::vector<int> post_end_address;
    // Pipeline Design
    void PipelineDesign();
    void ClassifyNodes(int node_index, int level_index);
    // Dataflow Schedule
    void ResetPostStartAndEndAddress(int origin_length, int assumed_core_num);
    void SchedulePreparation();
    void ScheduleMain();
    void ScheduleStage0(int instruction_group_index, bool append_instruction);
    void ScheduleStage1(int instruction_group_index, bool append_instruction);
    void ScheduleStage2(int instruction_group_index, bool append_instruction);
    void ScheduleStage3(int instruction_group_index, bool append_instruction);
    void ScheduleStageAct(int instruction_group_index, bool append_instruction);
    void SchedulePickOnePostOperation(int start_instruction_group_index);
    void ScheduleScheduleOnePostOperation(int instruction_group_index, int post_node_index);
    int GetEffectiveInstructionGroupNum();
    void ScheduleMerge();
    void Clear();
};

#endif //PIMCOMP_BATCHPIPELINESCHEDULE_H

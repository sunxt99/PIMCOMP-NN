//
// Created by SXT on 2022/10/11.
//

#ifndef PIMCOMP_ELEMENTPIPELINESCHEDULE_H
#define PIMCOMP_ELEMENTPIPELINESCHEDULE_H
#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"
#include "MemoryManager.h"

class ElementPipelineSchedule
{
public:
    ElementPipelineSchedule(std::string model_name);
    void ScheduleExecution();
    void SaveInstruction();
private:
    MemoryManager MM;
    std::string model_name;
    int first_node_index;
    int last_node_index;
    int last_node_output_channel_num;
    int node_num;
    int effective_instruction_group_num;
    void SchedulePreparation();
    void SavePreparation();
    void Check();
    void Clear();

    bool CheckInputPrepared(int node_index, int replication_index, int input_cycle_index);
    int GetInputChannelFromOutputIndex( int node_index, int output_index, bool is_last);
    void GetInputChannelFromOutputIndex(int first_last[2], int node_index, int output_index);
    ////////////////////////// Instruction //////////////////////////
    void ScheduleSplitInstruction();
    void ScheduleSplitInstructionMain(int instruction_group_index);
    void ScheduleSplitInstructionPost(int instruction_group_index, int this_node_index, int next_node_index, int input_channel_index, struct Comm_struct Comm);
    void ScheduleSplitInstructionStage0LoadBias(int instruction_group_index);
    void ScheduleSplitInstructionStage1MVMUL(int instruction_group_index, int start_AG_index_in_total, int AG_num_this_replication, int input_cycle_index);
    void ScheduleSplitInstructionStage3ACC(int instruction_group_index, int start_AG_index_in_total, int AG_num_this_replication);
    void ScheduleSplitInstructionStage3ACT(int instruction_group_index, int start_AG_index_in_total, int input_cycle_index);
    void ScheduleSplitInstructionStage3VER(int instruction_group_index, int AG_index_in_total, int output_channel_index);
    void ScheduleSplitInstructionStage3CLIP(int instruction_group_index, int start_AG_index_in_total, int input_cycle_index);
    void ScheduleSplitInstructionStage4Pool(int instruction_group_index, int pool_node_index, int input_cycle_index);
    void ScheduleSplitInstructionStage4Eltwise(int instruction_group_index, int vec_node_index, int input_cycle_index);
    void ScheduleSplitInstructionStage4Activate(int instruction_group_index, int vec_node_index, int input_cycle_index);
    void ScheduleSplitInstructionStage4Concat(int instruction_group_index, int vec_node_index, int input_cycle_index);
    void ScheduleSplitInstructionStage4Shuffle(int instruction_group_index, int vec_node_index, int input_cycle_index);
    void ScheduleSplitInstructionStage4Verify(int instruction_group_index, int node_index, int core_index, long long source_address, int source_offset, int element_num, int input_cycle_index);
    void ScheduleSplitInstructionCOMM(int instruction_group_index, int send_node_index, int from_core, long long source_address, int recv_node_index, int to_core, long long destination_address, int element_num);
    void ScheduleSplitInstructionCommForMain(int instruction_group_index, struct Main_Comm_struct Main_Comm);
    void ScheduleSplitInstructionCOMMForPost(int instruction_group_index, struct Post_Comm_struct Post_Comm);
    void ScheduleSplitInstructionWriteBack(int instruction_group_index, int input_channel_index, struct Comm_struct Comm);
    ////////////////////////// Memory //////////////////////////
    void MemoryPreparation();
    void PrepareMemoryINFO();
    void PrepareDependency();
    void PreparePadding();
    void MemoryAllocationForFirstLayer(int instruction_group_index, int input_cycle_index);
    void MemoryAllocationForAG(int AG_index_in_total, int core_index, bool need_input);
    void MemoryAllocationForPool(int instruction_group_index, int node_index, int input_channel_index, struct Comm_struct Comm);
    void MemoryAllocationForVec(int instruction_group_index, int node_index, int index_in_all_providers, int input_channel_index, int input_channel_element_num, struct Comm_struct Comm);
    void MemoryFreeForPost();
    void MemoryFreeForMain();
    void MemoryAllocationForInputOrganization(int instruction_group_index, int start_AG_index_in_total, int AG_num_this_replication, int output_channel_index);
};


#endif //PIMCOMP_ELEMENTPIPELINESCHEDULE_H

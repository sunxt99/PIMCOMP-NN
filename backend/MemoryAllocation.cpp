//
// Created by SXT on 2023/2/23.
//

#include "MemoryAllocation.h"

void MemoryAllocation::AllocateMemory()
{
    GetAGBaseInfo();
    GetAGReuseInfo();
    BaseAllocate();
    GenerateAddress();
    Clear();
}

void MemoryAllocation::GetAGBaseInfo()
{
    // AG_info_list includes various information of all AGs
    int AG_num = PIMCOMP_2_resource_info.AGs;
    for (int i = 0; i < AG_num; ++i)
    {
        int core_index = PIMCOMP_3_hierarchy_map.whole[i][0].vcore;
        int rep_index = PIMCOMP_3_hierarchy_map.whole[i][0].replication_index;
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        int AG_index_in_total = PIMCOMP_3_hierarchy_map.whole[i][0].array_group_total;
        int AG_index_in_replication = PIMCOMP_3_hierarchy_map.whole[i][0].array_group_in_weight;
        int AG_num_per_replication = PIMCOMP_3_hierarchy_map.whole[i][0].AG_num_per_replication;
        int replication_index = PIMCOMP_3_hierarchy_map.whole[i][0].replication_index;

        struct AG_base_info AGInfo;
        AGInfo.AG_index = AG_index_in_total;
        AGInfo.AG_index_in_replication = AG_index_in_replication;
        AGInfo.AG_num_per_replication = AG_num_per_replication;
        AGInfo.replication_index = rep_index;
        AGInfo.replication_num = PIMCOMP_node_list[node_index].replication_num;
        AGInfo.core_index = core_index;
        AGInfo.node_index = node_index;

        int effective_node_index = PIMCOMP_node_list[node_index].effective_node_index;
        int crossbar_num_AG = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list.size();
        int crossbar_start_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[0];
        int crossbar_end_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[crossbar_num_AG - 1];
        int input_element_num = PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_start + 1;
        int output_element_num = PIMCOMP_2_virtual_crossbar[crossbar_end_index].width_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].width_start + 1;

        AGInfo.input_element_num = input_element_num;
        AGInfo.output_element_num = output_element_num;
        PIMCOMP_5_AG_base_info.push_back(AGInfo);
    }
}

struct AG_reuse_info
{
    int AG_index;
    bool reuse_another_address;
    int reuse_AG_index;
};
std::vector<struct AG_reuse_info> PIMCOMP_5_AG_reuse_info;
void MemoryAllocation::GetAGReuseInfo()
{
    // The first AG of each rep of each core needs to be used to save the final result and cannot be used for AG-reuse.
    // 添加这个是因为每个核每个rep的第一个AG需要用来保存最终结果，不能用于AG-reuse。
    std::set<int> recv_AG_index_set;
    int node_num = PIMCOMP_node_list.size();
    for (int i = 0; i < node_num; ++i)
    {
        int recv_AG_index = PIMCOMP_4_recv_info.node_list[i].size();
        for (int j = 0; j < recv_AG_index; ++j)
        {
            recv_AG_index_set.insert(PIMCOMP_4_recv_info.node_list[i][j].AG_index);
        }
    }

    int reuse_period = 3;
    int AG_num = PIMCOMP_2_resource_info.AGs;
    PIMCOMP_5_AG_reuse_info.resize(AG_num);
    int core_num = PIMCOMP_4_virtual_core_AG_map.core_list.size();
    for (int i = 0; i < core_num; ++i)
    {
        int AG_num_in_core = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        for (int j = 0; j < AG_num_in_core; ++j)
        {
            int AG_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
            int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].node_index;
            int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
            // The reuse_candidate_index here refers to the order in the core, not the AG index number
            // 这里的的reuse_candidate_index指的是在core中的顺序，而不是AG序号
            int reuse_candidate_index = j - reuse_period;
            if (reuse_candidate_index >= 0)
            {
                int reuse_candidate_AG_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[reuse_candidate_index].AG_index_in_total;
                int reuse_candidate_node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[reuse_candidate_index].node_index;
                int reuse_candidate_replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[reuse_candidate_index].replication_index;
                if (recv_AG_index_set.count(reuse_candidate_AG_index) == 0 && reuse_candidate_node_index == node_index && reuse_candidate_replication_index == replication_index)
                {
                    // If there is no the following loop, then suppose node_x has 6 AGs (0-5) that need to be accumulated, reuse_period=3, then their AG_reuse_index is 0, 1, 2, 3, 4, 5.
                    // If there is the following loop, then their AG_reuse_index is 0, 1, 2, 0, 1, 2.
                    // 如果没有下面这个循环，那么假设node_x有6个AG（0-5）需要累加，reuse_period=3，则它们的AG_reuse_index为0、1、2、3、4、5。
                    // 如果有下面这个循环，那么它们的AG_reuse_index为0、1、2、0、1、2。
                    while(PIMCOMP_5_AG_reuse_info[reuse_candidate_AG_index].reuse_another_address)
                    {
                        reuse_candidate_AG_index = PIMCOMP_5_AG_reuse_info[reuse_candidate_AG_index].reuse_AG_index;
                    }
                    struct AG_reuse_info AGReuseInfo;
                    AGReuseInfo.AG_index = AG_index;
                    AGReuseInfo.reuse_another_address = true;
                    AGReuseInfo.reuse_AG_index = reuse_candidate_AG_index;
                    PIMCOMP_5_AG_reuse_info[AGReuseInfo.AG_index] = AGReuseInfo;
                }
                else
                {
                    struct AG_reuse_info AGReuseInfo;
                    AGReuseInfo.AG_index = AG_index;
                    AGReuseInfo.reuse_another_address = false;
                    AGReuseInfo.reuse_AG_index = AG_index;
                    PIMCOMP_5_AG_reuse_info[AGReuseInfo.AG_index] = AGReuseInfo;
                }
            }
            else
            {
                struct AG_reuse_info AGReuseInfo;
                AGReuseInfo.AG_index = AG_index;
                AGReuseInfo.reuse_another_address = false;
                AGReuseInfo.reuse_AG_index = AG_index;
                PIMCOMP_5_AG_reuse_info[AGReuseInfo.AG_index] = AGReuseInfo;
            }
        }
    }
//    for (int i = 0; i < AG_num; ++i)
//    {
//        std::cout << PIMCOMP_5_AG_reuse_info[i].AG_index << "  " << PIMCOMP_5_AG_reuse_info[i].reuse_another_address << "  " << PIMCOMP_5_AG_reuse_info[i].reuse_AG_index <<  std::endl;
//    }
}

std::vector<int> core_max_memory;
void MemoryAllocation::BaseAllocate()
{
    // usage for PIMCOMP_5_AG_memory_info:
    // PIMCOMP_5_AG_memory_info[0] stands for 0th-instruction_group
    // PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[2] stands for 2nd-core in 0th-instruction_group
    // PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[2].AG_memory_info_of_one_core[3] stands for 3rd-AG in 2nd-core in 0th-instruction_group

    core_max_memory.resize(ChipH * ChipW);
    int instruction_group_num = PIMCOMP_4_base_instruction_ir.size();
//    PIMCOMP_5_AG_memory_info.resize(instruction_group_num);
    // in Batch-pipeline, all other instruction groups can adopt the same memory allocation as the first instruction group.
    PIMCOMP_5_AG_memory_info.resize(1);
    int AG_total_num = PIMCOMP_2_resource_info.AGs;
    std::vector<int> AG_appearance_num;
    std::vector<int> AG_recv_element_num;
    PIMCOMP_GUI_memory_usage_every_instruction_group.resize(ChipW * ChipH);
//    for (int i = 0; i < instruction_group_num; ++i)
    for (int i = 0; i < 1; ++i)
    {
        PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG.resize(ChipH * ChipW);
        for (int j = 0; j < ChipH * ChipW; ++j)
        {
            AG_appearance_num.resize(AG_total_num);
            AG_recv_element_num.resize(AG_total_num);

            PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core.resize(AG_total_num);
//            long long memory_address_pointer = 0;
            long long memory_address_pointer = PIMCOMP_5_memory_start_address[j];

            if (PIMCOMP_4_base_instruction_ir[i].core_list.size() == 0)
                continue;
            int instruction_ir_num = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list.size();
            for (int r = 0; r < instruction_ir_num; ++r)
            {
                struct INST tmpInstruction = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list[r];
                if (tmpInstruction.operation == "MVMUL")
                {
                    int AG_index = tmpInstruction.AG_index_in_total;
                    AG_appearance_num[AG_index] ++;
                }
                else if (tmpInstruction.operation == "RECV")
                {
                    int AG_index = tmpInstruction.destination;
                    int element_num = tmpInstruction.element_num;
                    AG_recv_element_num[AG_index] = element_num;
                    AG_appearance_num[AG_index] = -1; // Usually only appears once here
                }
            }
            // These two variables are set here to increase memory reuse.
            // The situation considered is that when the same replication of the same node is distributed in multiple cores, the memory of recv can be reused.
            // 这里设置这两个变量是为了增加内存复用。考虑的情况就是当同一个节点的同一个rep分布在多个核时，能够复用recv的内存。
            int last_node_index = -1, last_replication_index = -1;
            int last_memory_output_start_address = 0, last_memory_element_length = 0;
            for (int k = 0; k < AG_total_num; ++k)
            {
                if (AG_appearance_num[k] > 0)
                {
                    // Old Version
//                    struct AG_memory_info thisAG;
//                    thisAG.AG_index = k;
//                    thisAG.input_element_num = PIMCOMP_5_AG_base_info[k].input_element_num;
//                    thisAG.output_element_num = PIMCOMP_5_AG_base_info[k].output_element_num;
//                    thisAG.appearance_num_in_one_core = AG_appearance_num[k];
//                    thisAG.is_double_buffer = thisAG.appearance_num_in_one_core > 1;
//                    thisAG.total_input_element_num = thisAG.input_element_num * (thisAG.is_double_buffer ? 2 : 1);
//                    thisAG.memory_element_length = thisAG.output_element_num * thisAG.appearance_num_in_one_core + thisAG.total_input_element_num;
//                    thisAG.memory_start_address = memory_address_pointer;
//                    thisAG.memory_output_start_address = thisAG.memory_start_address + thisAG.total_input_element_num;
//                    memory_address_pointer += thisAG.memory_element_length;
//                    PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[k] = thisAG;

                    // New Version, support AG-reuse
                    if (!PIMCOMP_5_AG_reuse_info[k].reuse_another_address)
                    {
                        struct AG_memory_info thisAG;
                        thisAG.AG_index = k;
                        thisAG.input_element_num = PIMCOMP_5_AG_base_info[k].input_element_num;
                        thisAG.output_element_num = PIMCOMP_5_AG_base_info[k].output_element_num;
                        thisAG.appearance_num_in_one_core = AG_appearance_num[k];
                        thisAG.is_double_buffer = thisAG.appearance_num_in_one_core > 1;
                        thisAG.total_input_element_num = thisAG.input_element_num * (thisAG.is_double_buffer ? 2 : 1);
                        thisAG.memory_element_length = thisAG.output_element_num * thisAG.appearance_num_in_one_core + thisAG.total_input_element_num;
                        thisAG.memory_start_address = memory_address_pointer;
                        thisAG.memory_output_start_address = thisAG.memory_start_address + thisAG.total_input_element_num;
                        memory_address_pointer += thisAG.memory_element_length;
                        PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[k] = thisAG;
                    }
                    else
                    {
                        struct AG_memory_info thisAG;
                        thisAG.AG_index = k;
                        thisAG.input_element_num = PIMCOMP_5_AG_base_info[k].input_element_num;
                        thisAG.output_element_num = PIMCOMP_5_AG_base_info[k].output_element_num;
                        thisAG.appearance_num_in_one_core = AG_appearance_num[k];
                        thisAG.is_double_buffer = thisAG.appearance_num_in_one_core > 1;
                        thisAG.total_input_element_num = thisAG.input_element_num * (thisAG.is_double_buffer ? 2 : 1);
                        thisAG.memory_element_length = thisAG.total_input_element_num;
                        thisAG.memory_start_address = memory_address_pointer;
                        // Only the output location is reused. So locate the memory_output_start_address to a previous AG.
                        // 重用的只有输出位置。所以就将memory_output_start_address定位到之前某个AG处。
                        int reuse_AG_index = PIMCOMP_5_AG_reuse_info[k].reuse_AG_index;
                        thisAG.memory_output_start_address = PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[reuse_AG_index].memory_output_start_address;
                        memory_address_pointer += thisAG.memory_element_length;
                        PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[k] = thisAG;
                    }
                }
                else if (AG_appearance_num[k] < 0) // RECV
                {
                    struct AG_memory_info thisAG;
                    thisAG.AG_index = k;
                    int node_index = PIMCOMP_5_AG_base_info[k].node_index;
                    int replication_index = PIMCOMP_5_AG_base_info[k].replication_index;
                    if (last_node_index == node_index && last_replication_index == replication_index)
                    {
                        thisAG.memory_output_start_address = last_memory_output_start_address;
                        thisAG.memory_element_length = last_memory_element_length;
                    }
                    else
                    {
                        thisAG.memory_output_start_address = memory_address_pointer;
//                        thisAG.memory_element_length = PIMCOMP_5_AG_base_info[k].output_element_num;
                        thisAG.memory_element_length = AG_recv_element_num[thisAG.AG_index];
                        memory_address_pointer += thisAG.memory_element_length;
                    }
                    PIMCOMP_5_AG_memory_info[i].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[k] = thisAG;
                    // Increase the chances of RECV reuse.
                    // 增加RECV的复用机会。
                    last_node_index = node_index;
                    last_replication_index = replication_index;
                    last_memory_output_start_address = thisAG.memory_output_start_address;
                    last_memory_element_length = thisAG.memory_element_length;
                }
            }
            AG_appearance_num.resize(0);
            AG_recv_element_num.resize(0);
            PIMCOMP_GUI_memory_usage_every_instruction_group[j].push_back(static_cast<double>(memory_address_pointer) * ArithmeticPrecision / 8 / 1024);
            if (core_max_memory[j] < memory_address_pointer)
                core_max_memory[j] = memory_address_pointer;
        }
    }

//    int IG_index = 0;
//    for (int i = 0; i < ChipW * ChipH; ++i)
//    {
//        for (int j = 0; j < AG_total_num; ++j)
//        {
//            if(PIMCOMP_5_AG_memory_info[IG_index].AG_memory_info_of_one_IG[i].AG_memory_info_of_one_core[j].appearance_num_in_one_core != 0)
//                std::cout << "core:" << i
//                << "  " << PIMCOMP_5_AG_memory_info[IG_index].AG_memory_info_of_one_IG[i].AG_memory_info_of_one_core[j].AG_index
//                << "  " << PIMCOMP_5_AG_memory_info[IG_index].AG_memory_info_of_one_IG[i].AG_memory_info_of_one_core[j].memory_start_address
//                << "  " << PIMCOMP_5_AG_memory_info[IG_index].AG_memory_info_of_one_IG[i].AG_memory_info_of_one_core[j].memory_element_length << std::endl;
//        }
//    }
}



void MemoryAllocation::GenerateAddress()
{
    int instruction_group_num = PIMCOMP_4_base_instruction_ir.size();
    PIMCOMP_5_base_instruction_with_address.resize(instruction_group_num);
    for (int i = 0; i < instruction_group_num; ++i)
    {
        PIMCOMP_5_base_instruction_with_address[i].core_list.resize(ChipH * ChipW);
        for (int j = 0; j < ChipH * ChipW; ++j)
        {
            if (PIMCOMP_4_base_instruction_ir[i].core_list.size() == 0)
                continue;
            int instruction_ir_num = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list.size();
            for (int k = 0; k < instruction_ir_num; ++k)
            {
                struct INST tmpInstruction = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list[k];
                if (tmpInstruction.operation == "MVMUL")
                {
                    int AG_index = tmpInstruction.destination;
                    tmpInstruction.source_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                    tmpInstruction.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_output_start_address;
                }
                else if (tmpInstruction.operation == "VVADD")
                {
                    if (tmpInstruction.stage == "MAIN-C" || tmpInstruction.stage == "MAIN-A")
                    {
                        int rs_1 = tmpInstruction.source_1;
                        int rs_2 = tmpInstruction.source_2;
                        int rd = tmpInstruction.destination;
                        tmpInstruction.source_1_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rs_1].memory_output_start_address;
                        tmpInstruction.source_2_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rs_2].memory_output_start_address;
                        tmpInstruction.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rd].memory_output_start_address;
                    }
                    else if (tmpInstruction.stage == "MAIN-B")
                    {
                        int rs_1 = tmpInstruction.source_1;
                        int rd = tmpInstruction.destination;
                        tmpInstruction.source_1_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rs_1].memory_output_start_address;
                        tmpInstruction.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rd].memory_output_start_address;
                    }
                }
                else if (tmpInstruction.operation == "VRELU" && tmpInstruction.stage == "MAIN")
                {
                    int rs = tmpInstruction.source;
                    int rd = tmpInstruction.destination;
                    tmpInstruction.source_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rs].memory_output_start_address;
                    tmpInstruction.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[rd].memory_output_start_address;
                }
                PIMCOMP_5_base_instruction_with_address[i].core_list[j].instruction_ir_list.push_back(tmpInstruction);
            }
        }
    }
}


void MemoryAllocation::Clear()
{
    PIMCOMP_5_AG_reuse_info.resize(0);
}
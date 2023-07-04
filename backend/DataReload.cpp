//
// Created by SXT on 2023/2/22.
//

#include "DataReload.h"

void DataReload::ReloadData()
{
    instruction_with_reload = 1;
    GetAGInputInfo();
    ReloadInput();
    StoreOutput();
    Clear();
}


void DataReload::GetAGInputInfo()
{
    int AG_num = PIMCOMP_2_resource_info.AGs;
    for (int i = 0; i < AG_num; ++i)
    {
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        int index_in_replication = PIMCOMP_3_hierarchy_map.whole[i][0].array_group_in_weight ;
        int replication_index = PIMCOMP_3_hierarchy_map.whole[i][0].replication_index;
        // construct AG_input_info
        struct AG_input_info thisAGInputInfo;
        thisAGInputInfo.node_index = node_index;
        thisAGInputInfo.AG_index = i;
        thisAGInputInfo.AG_index_in_replication = index_in_replication;
        if (PIMCOMP_node_list[node_index].operation == "OP_CONV")
        {
            int kernel_w = PIMCOMP_node_list[node_index].param.kernel_w;
            int kernel_h = PIMCOMP_node_list[node_index].param.kernel_h;
            int input_channel_length = PIMCOMP_node_list[node_index].param.input_channel;
            int total_input_element_num = kernel_w * kernel_h * input_channel_length;
            int start_position = index_in_replication * CrossbarH;
            int end_position = ((index_in_replication+1) * CrossbarH -1) > (total_input_element_num - 1) ? (total_input_element_num - 1) : ((index_in_replication+1) * CrossbarH -1);

            int start_channel_index = start_position / input_channel_length;
            int end_channel_index = end_position / input_channel_length;
            int start_channel_start_to_end = (start_channel_index + 1) * input_channel_length - start_position;
            int end_channel_start_to_end = end_position - end_channel_index * input_channel_length + 1;

            thisAGInputInfo.is_CONV = true;
            thisAGInputInfo.is_FC = false;
            thisAGInputInfo.start_position = start_position;
            thisAGInputInfo.end_position = end_position;
            thisAGInputInfo.start_channel_index = start_channel_index;
            thisAGInputInfo.end_channel_index = end_channel_index;
            thisAGInputInfo.start_channel_start_to_end = start_channel_start_to_end;
            thisAGInputInfo.end_channel_start_to_end = end_channel_start_to_end;
//            std::cout << "  AG:" << i << "  node:" << node_index << "  rep:" << replication_index << "  start:" << start_position << "  end:" << end_position << std::endl;
        }
        else if (PIMCOMP_node_list[node_index].operation == "OP_FC")
        {
            int num_input = PIMCOMP_node_list[node_index].param.num_input;

            int FC_start_position = index_in_replication * CrossbarH;
            int FC_end_position = ((index_in_replication+1) * CrossbarH -1) > (num_input - 1) ? (num_input - 1) : ((index_in_replication+1) * CrossbarH -1);
            thisAGInputInfo.is_CONV = false;
            thisAGInputInfo.is_FC = true;
            thisAGInputInfo.FC_start_position = FC_start_position;
            thisAGInputInfo.FC_end_position = FC_end_position;
//            std::cout << "  AG:" << i << "  node:" << node_index << "  rep:" << replication_index << "  FC_start:" << FC_start_position << "  FC_end:" << FC_end_position << std::endl;
        }
        PIMCOMP_6_AG_input_info.push_back(thisAGInputInfo);
    }
}


void DataReload::ReloadInput()
{
    int instruction_group_num = PIMCOMP_5_base_instruction_with_address.size();
    std::cout << "Reload Input Stage" << std::endl;
    PIMCOMP_6_base_instruction_with_input.resize(instruction_group_num);

    for (int i = 0; i < instruction_group_num; ++i)
        PIMCOMP_6_base_instruction_with_input[i].core_list.resize(ChipH * ChipW);

    // Set a record table for each instruction_group for each core to record the reusing of input data.
    // 为每个核在每个instruction_group设置一个记录表，记录输入数据复用情况
    // node_index (max 999) * 50000 + load_source_addr % 50000
    const int node_interval = 50000;
    const int total_interval = 1000 * node_interval;
    std::vector<int> node_input_reuse_record_for_element_num; // node_input_reuse_record[node_index][rs_offset] records the element_num of an LD instruction
    std::vector<int> node_input_reuse_record_for_local_address; // node_input_reuse_record[node_index][rs_offset] records the dst_address of an LOAD instruction
    node_input_reuse_record_for_element_num.resize(total_interval);
    node_input_reuse_record_for_local_address.resize(total_interval);

    for (int j = 0; j < ChipH * ChipW; ++j)
    {
        for (int i = 0; i < instruction_group_num; ++i)
        {
            std::set<int> node_load_info;
            if (PIMCOMP_5_base_instruction_with_address[i].core_list.size() == 0)
                continue;
            int instruction_ir_num = PIMCOMP_5_base_instruction_with_address[i].core_list[j].instruction_ir_list.size();
            for (int k = 0; k < instruction_ir_num; ++k)
            {
                struct INST tmpInstruction = PIMCOMP_5_base_instruction_with_address[i].core_list[j].instruction_ir_list[k];
                if (tmpInstruction.operation == "MVMUL") //// LOAD
                {
                    int node_index = tmpInstruction.node_index;
                    int AG_index = tmpInstruction.AG_index_in_total;
                    if (PIMCOMP_node_list[node_index].operation == "OP_CONV")
                    {
                        int start_channel_index_in_window = PIMCOMP_6_AG_input_info[AG_index].start_channel_index;
                        int end_channel_index_in_window = PIMCOMP_6_AG_input_info[AG_index].end_channel_index;
                        int start_position = PIMCOMP_6_AG_input_info[AG_index].start_position;
                        int end_position = PIMCOMP_6_AG_input_info[AG_index].end_position;
                        int start_channel_start_to_end = PIMCOMP_6_AG_input_info[AG_index].start_channel_start_to_end; //// start_channel 从start_position到ending
                        int end_channel_start_to_end = PIMCOMP_6_AG_input_info[AG_index].end_channel_start_to_end;     //// end_channel 从beginning到end_position
                        int output_index = tmpInstruction.input_cycle_index;
                        int input_channel_length = PIMCOMP_node_list[node_index].param.input_channel;
                        if (start_channel_index_in_window == end_channel_index_in_window)
                        {
                            int real_channel_index_in_input_feature = PIMCOMP_conv_pool_full_output_info[node_index].output_index[output_index][start_channel_index_in_window];
                            if (real_channel_index_in_input_feature == -1) // padding
                            {
                                struct INST Instruction_lldi;
                                Instruction_lldi.type = LLDI;
                                Instruction_lldi.level_index = PIMCOMP_node_list[node_index].level_index;
                                Instruction_lldi.level_diff = 0;
                                Instruction_lldi.operation = "LLDI";
                                Instruction_lldi.node_index = node_index;
                                Instruction_lldi.stage = "MAIN";
                                Instruction_lldi.destination = AG_index;
                                Instruction_lldi.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                Instruction_lldi.destination_offset = tmpInstruction.source_offset;
                                Instruction_lldi.element_num = end_position - start_position + 1;
                                Instruction_lldi.imm_value = 0;
                                Instruction_lldi.instruction_group_index = i;
                                PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_lldi);
                            }
                            else // load in dram
                            {
                                int load_address_in_dram = real_channel_index_in_input_feature * input_channel_length + start_position % input_channel_length;
                                int load_element_num = end_position - start_position + 1;
//                                if (node_load_info.count(node_index * node_interval + load_address_in_dram % node_interval) == 0
//                                    || node_input_reuse_record_for_element_num[node_index * node_interval + load_address_in_dram % node_interval] != load_element_num)
                                int hash_value = load_address_in_dram / 5 +
                                                 pow(double(load_address_in_dram), 1.0/2) +
                                                 pow(double(load_address_in_dram), 1.0/3) +
                                                 pow(double(load_address_in_dram), 1.0/5) +
                                                 pow(double(load_address_in_dram), 1.0/7);
                                if (node_load_info.count(node_index * node_interval + hash_value % node_interval) == 0
                                    || node_input_reuse_record_for_element_num[node_index * node_interval + hash_value % node_interval] != load_element_num)
                                {
                                    struct INST Instruction_ld;
                                    Instruction_ld.type = MEM;
                                    Instruction_ld.level_index = PIMCOMP_node_list[node_index].level_index;
                                    Instruction_ld.level_diff = 0;
                                    Instruction_ld.operation = "LD";
                                    Instruction_ld.node_index = node_index;
                                    Instruction_ld.stage = "INPUT";
                                    int provider_node_index = PIMCOMP_node_list[node_index].provider_index[0];
                                    Instruction_ld.source = -1 * provider_node_index; // Regulation: The position in DRAM is replaced by -1*node_index.
                                    Instruction_ld.source_address = -1 * provider_node_index;
                                    Instruction_ld.source_offset = real_channel_index_in_input_feature * input_channel_length + start_position % input_channel_length; // load_address_in_dram
                                    Instruction_ld.destination = AG_index;
                                    Instruction_ld.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                    Instruction_ld.destination_offset = tmpInstruction.source_offset;
                                    Instruction_ld.element_num = end_position - start_position + 1;
                                    Instruction_ld.instruction_group_index = i;
                                    PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_ld);
                                    int interval_index = hash_value % node_interval + node_index * node_interval;
                                    node_load_info.insert(interval_index);
                                    node_input_reuse_record_for_element_num[interval_index] = Instruction_ld.element_num;
                                    node_input_reuse_record_for_local_address[interval_index] = Instruction_ld.destination_address + Instruction_ld.destination_offset;
                                }
                                else // Reuse
                                {
                                    struct INST Instruction_vm;
                                    Instruction_vm.type = VEC1OP;
                                    Instruction_vm.operation = "LMV";
                                    Instruction_vm.level_index = PIMCOMP_node_list[node_index].level_index;
                                    Instruction_vm.node_index = node_index;
                                    Instruction_vm.stage = "MAIN";
//                                    Instruction_vm.source_address = node_input_reuse_record_for_local_address[node_index * node_interval + load_address_in_dram % node_interval];
                                    Instruction_vm.source_address = node_input_reuse_record_for_local_address[node_index * node_interval + hash_value % node_interval];
                                    Instruction_vm.source_offset = 0;
                                    Instruction_vm.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                    Instruction_vm.destination_offset = tmpInstruction.source_offset;
                                    Instruction_vm.element_num = end_position - start_position + 1;
                                    Instruction_vm.instruction_group_index = i;
                                    PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_vm);
                                }
                            }
                        }
                        else
                        {
                            int rd_offset = tmpInstruction.source_offset;
                            for (int c = start_channel_index_in_window; c <= end_channel_index_in_window; ++c)
                            {
                                int element_num = 0;
                                int rs_offset_offset = 0; //// offset of rs_offset
                                if (c == start_channel_index_in_window)
                                {
                                    element_num = start_channel_start_to_end;
                                    rs_offset_offset = start_position % input_channel_length;
                                }
                                else if (c == end_channel_index_in_window)
                                {
                                    element_num = end_channel_start_to_end;
                                }
                                else
                                {
                                    element_num = input_channel_length;
                                }

                                int real_channel_index_in_input_feature = PIMCOMP_conv_pool_full_output_info[node_index].output_index[output_index][c];
                                if (real_channel_index_in_input_feature == -1) // padding
                                {
                                    struct INST Instruction_lldi;
                                    Instruction_lldi.type = LLDI;
                                    Instruction_lldi.level_index = PIMCOMP_node_list[node_index].level_index;
                                    Instruction_lldi.level_diff = 0;
                                    Instruction_lldi.operation = "LLDI";
                                    Instruction_lldi.node_index = node_index;
                                    Instruction_lldi.stage = "MAIN";
                                    Instruction_lldi.destination = AG_index;
                                    Instruction_lldi.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                    Instruction_lldi.destination_offset = rd_offset;
                                    Instruction_lldi.element_num = element_num;
                                    Instruction_lldi.imm_value = 0;
                                    Instruction_lldi.instruction_group_index = i;
                                    PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_lldi);
                                }
                                else // load from dram
                                {
                                    int load_address_in_dram = real_channel_index_in_input_feature * input_channel_length + rs_offset_offset;
                                    int load_element_num = element_num;
//                                    if (node_load_info.count(node_index * node_interval + load_address_in_dram % node_interval) == 0
//                                        || node_input_reuse_record_for_element_num[node_index * node_interval + load_address_in_dram % node_interval] != load_element_num)
                                    int hash_value = load_address_in_dram / 5 +
                                            pow(double(load_address_in_dram), 1.0/2) +
                                            pow(double(load_address_in_dram), 1.0/3) +
                                            pow(double(load_address_in_dram), 1.0/5) +
                                            pow(double(load_address_in_dram), 1.0/7);
                                    if (node_load_info.count(node_index * node_interval + hash_value % node_interval) == 0
                                        || node_input_reuse_record_for_element_num[node_index * node_interval + hash_value % node_interval] != load_element_num)
                                    {
                                        struct INST Instruction_ld;
                                        Instruction_ld.type = MEM;
                                        Instruction_ld.level_index = PIMCOMP_node_list[node_index].level_index;
                                        Instruction_ld.level_diff = 0;
                                        Instruction_ld.operation = "LD";
                                        Instruction_ld.node_index = node_index;
                                        Instruction_ld.stage = "INPUT";
                                        int provider_node_index = PIMCOMP_node_list[node_index].provider_index[0];
                                        Instruction_ld.source = -1 * provider_node_index; // Regulation: The position in DRAM is replaced by -1*node_index.
                                        Instruction_ld.source_address = -1 * provider_node_index;
                                        Instruction_ld.source_offset = real_channel_index_in_input_feature * input_channel_length + rs_offset_offset;
                                        Instruction_ld.destination = AG_index;
                                        Instruction_ld.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                        Instruction_ld.destination_offset = rd_offset;
                                        Instruction_ld.element_num = element_num;
                                        Instruction_ld.instruction_group_index = i;
                                        PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_ld);
                                        int interval_index = hash_value % node_interval + node_index * node_interval;
                                        node_load_info.insert(interval_index);
                                        node_input_reuse_record_for_element_num[interval_index] = Instruction_ld.element_num;
                                        node_input_reuse_record_for_local_address[interval_index] = Instruction_ld.destination_address + Instruction_ld.destination_offset;
                                    }
                                    else  // reuse
                                    {
                                        struct INST Instruction_vm;
                                        Instruction_vm.type = VEC1OP;
                                        Instruction_vm.operation = "LMV";
                                        Instruction_vm.level_index = PIMCOMP_node_list[node_index].level_index;
                                        Instruction_vm.node_index = node_index;
                                        Instruction_vm.stage = "MAIN";
//                                        Instruction_vm.source_address = node_input_reuse_record_for_local_address[node_index * node_interval + load_address_in_dram % node_interval];
                                        Instruction_vm.source_address = node_input_reuse_record_for_local_address[node_index * node_interval + hash_value % node_interval];
                                        Instruction_vm.source_offset = 0;
                                        Instruction_vm.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                                        Instruction_vm.destination_offset = rd_offset;
                                        Instruction_vm.element_num = element_num;
                                        Instruction_vm.instruction_group_index = i;
                                        PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_vm);
                                    }
                                }
                                rd_offset += element_num;
                            }
                        }
                    }
                    else if (PIMCOMP_node_list[node_index].operation == "OP_FC")
                    {
                        int FC_start_position = PIMCOMP_6_AG_input_info[AG_index].FC_start_position;
                        int FC_end_position = PIMCOMP_6_AG_input_info[AG_index].FC_end_position;

                        struct INST Instruction_ld;
                        Instruction_ld.type = MEM;
                        Instruction_ld.level_index = PIMCOMP_node_list[node_index].level_index;
                        Instruction_ld.level_diff = 0;
                        Instruction_ld.operation = "LD";
                        Instruction_ld.node_index = node_index;
                        Instruction_ld.stage = "INPUT";
                        int provider_node_index = PIMCOMP_node_list[node_index].provider_index[0];
                        Instruction_ld.source = -1 * provider_node_index;
                        Instruction_ld.source_address = -1 * provider_node_index;  // Regulation: The position in DRAM is replaced by -1*node_index.
                        Instruction_ld.destination = AG_index;
                        Instruction_ld.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_start_address;
                        Instruction_ld.source_offset = FC_start_position;
                        Instruction_ld.destination_offset = 0;
                        Instruction_ld.element_num = FC_end_position - FC_start_position + 1;
                        Instruction_ld.instruction_group_index = i;
                        PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_ld);
                    }
                }
                if (tmpInstruction.operation == "RECV" || tmpInstruction.operation == "SEND")
                {
                    if (tmpInstruction.operation == "RECV")
                    {
                        int destination_AG_index = tmpInstruction.destination;
                        tmpInstruction.destination_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[destination_AG_index].memory_output_start_address;
                    }
                    else
                    {
                        int source_AG_index = tmpInstruction.source;
                        tmpInstruction.source_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[source_AG_index].memory_output_start_address;
                    }
                    tmpInstruction.instruction_index_in_core = PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.size();
                }
                PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(tmpInstruction);
            }
//            for (auto k = node_load_info.begin(); k != node_load_info.end(); k++)
//            {
//                node_input_reuse_record_for_element_num[*k] = 0;
//                node_input_reuse_record_for_local_address[*k] = 0;
//            }
        }
    }
}


void DataReload::StoreOutput()
{
    std::cout << "Store Output Stage" << std::endl;
    int instruction_group_num = PIMCOMP_6_base_instruction_with_input.size();
    for (int i = 0; i < instruction_group_num; ++i)
    {
        for (int j = 0; j < ChipH * ChipW; ++j)
        {
            if (PIMCOMP_6_base_instruction_with_input[i].core_list.size() == 0)
                continue;
            PIMCOMP_6_base_instruction_with_input[i].core_list.resize(ChipH * ChipW);
            int instruction_ir_num = PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.size();
            for (int k = 0; k < instruction_ir_num; ++k)
            {
                struct INST tmpInstruction = PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list[k];
                if (tmpInstruction.operation == "MVMUL") //// LOAD
                {
                    int node_index = tmpInstruction.node_index;
                    int AG_index = tmpInstruction.AG_index_in_total;
                    int index_in_replication = tmpInstruction.AG_index_in_replication;
                    int element_num = tmpInstruction.output_element_num;
                    int rs_offset = 0;
                    int rd_offset = 0;
                    if (index_in_replication == 0)
                    {
                        if (PIMCOMP_node_list[node_index].operation == "OP_CONV")
                        {
                            int input_cycle_index = tmpInstruction.input_cycle_index;
                            rd_offset = element_num * input_cycle_index;
                        }
                        else if (PIMCOMP_node_list[node_index].operation == "OP_FC")
                        {
                            rd_offset = element_num * index_in_replication;
                        }
                        struct INST Instruction_st;
                        Instruction_st.type = MEM;
                        Instruction_st.level_index = PIMCOMP_node_list[node_index].level_index;
                        Instruction_st.level_diff = 0;
                        Instruction_st.operation = "ST";
                        Instruction_st.node_index = node_index;
                        Instruction_st.stage = "OUTPUT";
                        Instruction_st.source = AG_index;
                        Instruction_st.source_address = PIMCOMP_5_AG_memory_info[0].AG_memory_info_of_one_IG[j].AG_memory_info_of_one_core[AG_index].memory_output_start_address;
                        Instruction_st.source_offset = rs_offset + tmpInstruction.destination_offset;
                        Instruction_st.destination = -1 * node_index;
                        Instruction_st.destination_address = -1 * node_index; // Regulation: The position in DRAM is replaced by -1*node_index.
                        Instruction_st.destination_offset = rd_offset ;
                        Instruction_st.element_num = element_num;
                        Instruction_st.instruction_group_index = i;
                        PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.push_back(Instruction_st);
                    }
                }
            }
        }
    }
}


void DataReload::BatchInstruction()
{
    int batch_size = batch_end - batch_start + 1;
    int original_instruction_group_num = PIMCOMP_6_base_instruction_with_input.size();
    PIMCOMP_6_base_instruction_with_input_batch.resize(batch_size * original_instruction_group_num);
    for (int batch_idx = batch_start; batch_idx <= batch_end; ++batch_idx)
    {
        int instruction_group_num = PIMCOMP_6_base_instruction_with_input.size();
        for (int i = 0; i < instruction_group_num; ++i)
        {
            PIMCOMP_6_base_instruction_with_input_batch[(batch_idx-batch_start)*instruction_group_num+i].core_list.resize(ChipH * ChipW);
            for (int j = 0; j < ChipW * ChipH; ++j)
            {
                int instruction_num = PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list.size();
                if (instruction_num == 0)
                    continue;
                for (int k = 0; k < instruction_num; ++k)
                {
                    struct INST Instruction = PIMCOMP_6_base_instruction_with_input[i].core_list[j].instruction_ir_list[k];
                    int level_index = Instruction.level_index;
                    if (level_index <= batch_idx)
                    {
                        std::string node_operation = PIMCOMP_node_list[i].operation;
                        if (Instruction.operation == "LD")
                        {
                            if (Instruction.stage == "POST")
                            {
                                int provider_node_index = Instruction.provider_node_index;
                                int provider_level_index = PIMCOMP_node_list[provider_node_index].level_index;
                                if (provider_level_index < level_index)
                                {
                                    int max_level_gap = PIMCOMP_node_list[provider_node_index].max_level_gap;
                                    Instruction.source_offset_between_batch = (batch_idx - provider_level_index) % (max_level_gap+1) + 1;
                                }
                            }
                        }
                        else if (Instruction.operation == "ST")
                        {
                            int node_index = Instruction.node_index;
                            if (PIMCOMP_node_list[node_index].max_level_gap > 0) // this node has POST consumer with higher level
                            {
                                int max_level_gap = PIMCOMP_node_list[node_index].max_level_gap;
                                Instruction.source_offset_between_batch = 0;
                                struct INST Instruction_copy_store = Instruction;
                                Instruction_copy_store.source_offset_between_batch = (batch_idx - level_index) % (max_level_gap+1) + 1;
                                PIMCOMP_6_base_instruction_with_input_batch[(batch_idx-batch_start)*instruction_group_num+i].core_list[j].instruction_ir_list.push_back(Instruction_copy_store);
                            }
                        }
                        else if (Instruction.operation == "SEND" || Instruction.operation == "RECV")
                        {

                        }
                        PIMCOMP_6_base_instruction_with_input_batch[(batch_idx-batch_start)*instruction_group_num+i].core_list[j].instruction_ir_list.push_back(Instruction);
                    }
                }
            }
        }
    }
}

void DataReload::SaveInstruction()
{
    std::ofstream OutFile("../output/DataReload.inst", std::ios::out | std::ios::trunc);
    {
        int instruction_group_num = PIMCOMP_6_base_instruction_with_input_batch.size();
        for (int i = 0; i < instruction_group_num; ++i)
        {
            OutFile << "========================================= base instruction_group " << i << " =========================================" << std::endl;
            for (int j = 0; j < ChipW * ChipH; ++j)
            {
                int instruction_num = PIMCOMP_6_base_instruction_with_input_batch[i].core_list[j].instruction_ir_list.size();
                if (instruction_num == 0)
                    continue;
                OutFile << "core " << j << std::endl;
                for (int k = 0; k < instruction_num; ++k)
                {
                    struct INST Instruction = PIMCOMP_6_base_instruction_with_input_batch[i].core_list[j].instruction_ir_list[k];
                    SaveSingleInstructionWithAddress(OutFile, Instruction, i, j);
                }
            }
        }
    }
    OutFile.close();
}



void DataReload::Clear()
{

}
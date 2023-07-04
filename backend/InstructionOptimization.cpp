//
// Created by SXT on 2023/6/28.
//

#include "InstructionOptimization.h"

InstructionOptimization::InstructionOptimization()
{
    if (instruction_with_reload == 1) // Batch-Pipeline
        intermediate_instruction_ir = PIMCOMP_6_base_instruction_with_input;
    else  // Element-Pipeline
        intermediate_instruction_ir = PIMCOMP_4_base_instruction_ir;
}

void InstructionOptimization::OptimizeInstruction()
{
    MergePreparation();
    MergeLoadOperation();
    PIMCOMP_7_base_instruction_ir_with_optimization = intermediate_instruction_ir;
}

void InstructionOptimization::MergePreparation()
{
    std::vector<struct PIMCOMP_4_instruction_ir> tmp_instruction_ir = intermediate_instruction_ir;
    intermediate_instruction_ir.resize(0);
    intermediate_instruction_ir.resize(tmp_instruction_ir.size());
    for (int i = 0; i < tmp_instruction_ir.size(); ++i)
    {
        if (tmp_instruction_ir[i].core_list.size() > 0)
            intermediate_instruction_ir[i].core_list.resize(ChipW * ChipH);
        else
            continue;
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            int instruction_ir_num = tmp_instruction_ir[i].core_list[j].instruction_ir_list.size();
            for (int k = 0; k < instruction_ir_num; ++k)
            {
                struct INST tmpInstruction = tmp_instruction_ir[i].core_list[j].instruction_ir_list[k];
                bool append = true;
                if (k != 0)
                {
                    if (tmpInstruction.operation == "LD" && tmpInstruction.stage != "BIAS")
                    {
                        struct INST tmpLastInstruction = intermediate_instruction_ir[i].core_list[j].instruction_ir_list.back();
                        if (tmpLastInstruction.operation == "LD")
                        {
                            long long last_rs_addr = tmpLastInstruction.source_address;
                            long long last_rs_offset = tmpLastInstruction.source_offset;
                            long long last_rd_addr = tmpLastInstruction.destination_address;
                            long long last_rd_offset = tmpLastInstruction.destination_offset;
                            int last_element_num = tmpLastInstruction.element_num;

                            long long this_rs_addr = tmpInstruction.source_address;
                            long long this_rs_offset = tmpInstruction.source_offset;
                            long long this_rd_addr = tmpInstruction.destination_address;
                            long long this_rd_offset = tmpInstruction.destination_offset;
                            int this_element_num = tmpInstruction.element_num;
                            if (instruction_with_reload == 1) // Batch-Pipeline
                            {
                                if (last_rs_addr == this_rs_addr &&
                                    last_rs_offset + last_element_num == this_rs_offset &&
                                    last_rd_addr == this_rd_addr &&
                                    last_rd_offset + last_element_num == this_rd_offset)
                                {
                                    intermediate_instruction_ir[i].core_list[j].instruction_ir_list.back().element_num += this_element_num;
                                    append = false;
                                }
                            }
                            else    // Element-Pipeline
                            {
                                if (last_rs_addr == this_rs_addr &&
                                    last_rs_offset + last_element_num == this_rs_offset &&
                                    last_rd_addr + last_element_num == this_rd_addr &&
                                    last_rd_offset == this_rd_offset)
                                {
                                    intermediate_instruction_ir[i].core_list[j].instruction_ir_list.back().element_num += this_element_num;
                                    append = false;
                                }
                            }
                        }
                    }
                    else if (tmpInstruction.operation == "LMV")
                    {
                        struct INST tmpLastInstruction = tmp_instruction_ir[i].core_list[j].instruction_ir_list[k-1];
                        if (tmpLastInstruction.operation == tmpInstruction.operation)
                        {
                            long long last_rs_addr = tmpLastInstruction.source_address;
                            long long last_rs_offset = tmpLastInstruction.source_offset;
                            long long last_rd_addr = tmpLastInstruction.destination_address;
                            long long last_rd_offset = tmpLastInstruction.destination_offset;
                            int last_element_num = tmpLastInstruction.element_num;

                            long long this_rs_addr = tmpInstruction.source_address;
                            long long this_rs_offset = tmpInstruction.source_offset;
                            long long this_rd_addr = tmpInstruction.destination_address;
                            long long this_rd_offset = tmpInstruction.destination_offset;
                            int this_element_num = tmpInstruction.element_num;

                            if (last_rs_addr + last_element_num == this_rs_addr &&
                                last_rs_offset == this_rs_offset &&
                                last_rd_addr == this_rd_addr &&
                                last_rd_offset + last_element_num == this_rd_offset)
                            {
                                intermediate_instruction_ir[i].core_list[j].instruction_ir_list.back().element_num += this_element_num;
                                append = false;
                            }
                        }
                    }
                    else if (tmpInstruction.operation == "LLDI")
                    {
                        struct INST tmpLastInstruction = tmp_instruction_ir[i].core_list[j].instruction_ir_list[k-1];
                        if (tmpLastInstruction.operation == tmpInstruction.operation)
                        {
                            long long last_rd_addr = tmpLastInstruction.destination_address;
                            long long last_rd_offset = tmpLastInstruction.destination_offset;
                            int last_element_num = tmpLastInstruction.element_num;

                            long long this_rd_addr = tmpInstruction.destination_address;
                            long long this_rd_offset = tmpInstruction.destination_offset;
                            int this_element_num = tmpInstruction.element_num;

                            if (last_rd_addr == this_rd_addr &&
                                last_rd_offset + last_element_num == this_rd_offset)
                            {
                                intermediate_instruction_ir[i].core_list[j].instruction_ir_list.back().element_num += this_element_num;
                                append = false;
                            }
                        }
                    }
                }
                if(append)
                {
                    tmpInstruction.instruction_index_in_core = intermediate_instruction_ir[i].core_list[j].instruction_ir_list.size();
                    intermediate_instruction_ir[i].core_list[j].instruction_ir_list.push_back(tmpInstruction);
                }
            }
        }
    }
}

struct SplitLoadInfo
{
    int node_index;
    long long rs_address;
    long long rs_offset;
    long long rd_address;
    long long rd_offset;
    int element_num;
    int instruction_index_in_core;
    SplitLoadInfo(int node_index_, long long rs_address_, long long rs_offset_, long long rd_address_, long long rd_offset_, int element_num_, int instruction_index_in_core_)
    {
        node_index = node_index_;
        rs_address = rs_address_;
        rs_offset = rs_offset_;
        rd_address = rd_address_;
        rd_offset = rd_offset_;
        element_num = element_num_;
        instruction_index_in_core = instruction_index_in_core_;
    }
    friend bool operator < (const SplitLoadInfo &a, const SplitLoadInfo &b)
    {
        return a.rs_offset > b.rs_offset;
    }
};

struct FullLoadInfo
{
    int node_index;
    long long rs_address;
    long long rs_offset;
    long long rd_address;
    long long rd_offset;
    int element_num;
    int exec_load_index_in_core = 100000000;
};

extern std::vector<int> core_max_memory;
void InstructionOptimization::MergeLoadOperation()
{
    if (instruction_with_reload != 1) // Batch-Pipeline
        return;

    std::vector<struct PIMCOMP_4_instruction_ir> tmp_instruction_ir = intermediate_instruction_ir;
    intermediate_instruction_ir.resize(0);
    intermediate_instruction_ir.resize(tmp_instruction_ir.size());
    std::map<int,int> load_element_num;
    for (int i = 0; i < tmp_instruction_ir.size(); ++i)
    {
        if (tmp_instruction_ir[i].core_list.size() == 0)
            continue;
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            std::vector<std::priority_queue<struct SplitLoadInfo>> node_load_info;
            node_load_info.resize(PIMCOMP_node_list.size());
            int instruction_ir_num = tmp_instruction_ir[i].core_list[j].instruction_ir_list.size();
            for (int k = 0; k < instruction_ir_num; ++k)
            {
                struct INST tmpInstruction = tmp_instruction_ir[i].core_list[j].instruction_ir_list[k];
                if (tmpInstruction.operation == "LD" && tmpInstruction.stage != "BIAS")
                {
                    int node_index = tmpInstruction.node_index;
                    long long rs_address = tmpInstruction.source_address;
                    long long rs_offset = tmpInstruction.source_offset;
                    long long rd_address = tmpInstruction.destination_address;
                    long long rd_offset = tmpInstruction.destination_offset;
                    int element_num = tmpInstruction.element_num;
                    int instruction_index_in_core = tmpInstruction.instruction_index_in_core;
                    if (load_element_num.count(element_num) == 0)
                        load_element_num[element_num] = 1;
                    else
                        load_element_num[element_num] ++;
                    if (element_num < 64)
                        node_load_info[node_index].push(SplitLoadInfo(node_index, rs_address, rs_offset, rd_address, rd_offset, element_num, instruction_index_in_core));
                }
            }

            std::map<int,int> instruction_index_to_load_index; // from split load to whole load
            std::vector<struct FullLoadInfo> full_load_info_list;
//            if (i == 0 && j == 1)
            {
                for (int n = 0; n < node_load_info.size(); ++n)
                {
                    long long start_addr = -1;
                    while (!node_load_info[n].empty())
                    {
//                        std::cout << "node:" << n << "  " << node_load_info[n].top().rs_offset << " " << node_load_info[n].top().rs_offset + node_load_info[n].top().element_num-1;
                        if (start_addr != -1 && node_load_info[n].top().rs_offset + node_load_info[n].top().element_num - start_addr <= 64)
                        {
                            long long end_addr = start_addr + 63;
                            if (full_load_info_list.back().exec_load_index_in_core > node_load_info[n].top().instruction_index_in_core)
                                full_load_info_list.back().exec_load_index_in_core = node_load_info[n].top().instruction_index_in_core;
                            instruction_index_to_load_index[node_load_info[n].top().instruction_index_in_core] = full_load_info_list.size()-1;
//                            std::cout << "      Begin:" << start_addr << "  End:" << end_addr;
//                            int instruction_index_in_core = node_load_info[n].top().instruction_index_in_core;
//                            int full_load_index = instruction_index_to_load_index[instruction_index_in_core];
//                            std::cout << "  " << full_load_index << "  " << full_load_info_list[full_load_index].rd_address << std::endl;
                        }
                        else
                        {
                            start_addr = node_load_info[n].top().rs_offset;
                            long long end_addr = start_addr + 63;
                            struct FullLoadInfo full_load_info;
                            full_load_info.element_num = 64;
                            full_load_info.rs_address = node_load_info[n].top().rs_address;
                            full_load_info.rs_offset = start_addr;
                            full_load_info.rd_address = core_max_memory[j] + (full_load_info_list.size() * 64);
                            full_load_info.rd_offset = 0;
                            full_load_info.node_index = node_load_info[n].top().node_index;
                            full_load_info.exec_load_index_in_core = node_load_info[n].top().instruction_index_in_core;
                            full_load_info_list.push_back(full_load_info);
                            instruction_index_to_load_index[node_load_info[n].top().instruction_index_in_core] = full_load_info_list.size()-1;
//                            std::cout << "      Begin:" << start_addr << "  End:" << end_addr;
//                            int instruction_index_in_core = node_load_info[n].top().instruction_index_in_core;
//                            int full_load_index = instruction_index_to_load_index[instruction_index_in_core];
//                            std::cout << "  " << full_load_index << "  " << full_load_info_list[full_load_index].rd_address << std::endl;
                        }
                        node_load_info[n].pop();
                    }
                }
            }


            for (int k = 0; k < instruction_ir_num; ++k)
            {
                if (tmp_instruction_ir[i].core_list.size() > 0)
                    intermediate_instruction_ir[i].core_list.resize(ChipW * ChipH);
                else
                    continue;
                struct INST tmpInstruction = tmp_instruction_ir[i].core_list[j].instruction_ir_list[k];
                int node_index = tmpInstruction.node_index;
                int instruction_index_in_core = tmpInstruction.instruction_index_in_core;
                if (tmpInstruction.operation == "LD" && tmpInstruction.stage != "BIAS" && instruction_index_to_load_index.count(instruction_index_in_core))
                {
                    int full_load_index = instruction_index_to_load_index[instruction_index_in_core];
                    long long full_load_rs_address = full_load_info_list[full_load_index].rs_address;
                    long long full_load_rs_offset = full_load_info_list[full_load_index].rs_offset;
                    long long full_load_rd_address = full_load_info_list[full_load_index].rd_address;
                    long long full_load_rd_offset = full_load_info_list[full_load_index].rd_offset;
                    int full_load_element_num = full_load_info_list[full_load_index].element_num;
                    int input_dim_num = PIMCOMP_node_list[node_index].input_dim_num;
                    int input_element_num = 1;
                    for (int l = 0; l < input_dim_num; ++l)
                        input_element_num *= PIMCOMP_node_list[node_index].input_dim[l];
                    if (full_load_rs_offset + full_load_element_num >= input_element_num)
                        full_load_element_num = input_element_num - full_load_rs_offset;

                    if (full_load_info_list[full_load_index].exec_load_index_in_core == instruction_index_in_core)
                    {
                        struct INST Instruction_ld;
                        Instruction_ld.type = MEM;
                        Instruction_ld.level_index = PIMCOMP_node_list[node_index].level_index;
                        Instruction_ld.level_diff = 0;
                        Instruction_ld.operation = "LD";
                        Instruction_ld.node_index = node_index;
                        Instruction_ld.stage = "INPUT";
                        Instruction_ld.source = tmpInstruction.source;
                        Instruction_ld.source_address = full_load_rs_address;
                        Instruction_ld.source_offset = full_load_rs_offset;
                        Instruction_ld.destination = tmpInstruction.destination;
                        Instruction_ld.destination_address = full_load_rd_address;
                        Instruction_ld.destination_offset = full_load_rd_offset;
                        Instruction_ld.element_num = full_load_element_num;
                        Instruction_ld.instruction_group_index = i;
                        intermediate_instruction_ir[i].core_list[j].instruction_ir_list.push_back(Instruction_ld);

                        struct INST Instruction_vm;
                        Instruction_vm.type = VEC1OP;
                        Instruction_vm.operation = "LMV";
                        Instruction_vm.level_index = PIMCOMP_node_list[node_index].level_index;
                        Instruction_vm.node_index = node_index;
                        Instruction_vm.stage = "MAIN";
                        Instruction_vm.source_address = full_load_rd_address + full_load_rd_offset;
                        Instruction_vm.source_offset = tmpInstruction.source_offset - full_load_rs_offset;
                        Instruction_vm.destination_address = tmpInstruction.destination_address;
                        Instruction_vm.destination_offset = tmpInstruction.destination_offset;
                        Instruction_vm.element_num = tmpInstruction.element_num;
                        Instruction_vm.instruction_group_index = i;
                        intermediate_instruction_ir[i].core_list[j].instruction_ir_list.push_back(Instruction_vm);
                    }
                    else
                    {
                        struct INST Instruction_vm;
                        Instruction_vm.type = VEC1OP;
                        Instruction_vm.operation = "LMV";
                        Instruction_vm.level_index = PIMCOMP_node_list[node_index].level_index;
                        Instruction_vm.node_index = node_index;
                        Instruction_vm.stage = "MAIN";
                        Instruction_vm.source_address = full_load_rd_address + full_load_rd_offset;
                        Instruction_vm.source_offset = tmpInstruction.source_offset - full_load_rs_offset;
                        Instruction_vm.destination_address = tmpInstruction.destination_address;
                        Instruction_vm.destination_offset = tmpInstruction.destination_offset;
                        Instruction_vm.element_num = tmpInstruction.element_num;
                        Instruction_vm.instruction_group_index = i;
                        intermediate_instruction_ir[i].core_list[j].instruction_ir_list.push_back(Instruction_vm);
                    }
                }
                else
                {
                    tmpInstruction.instruction_index_in_core = intermediate_instruction_ir[i].core_list[j].instruction_ir_list.size();
                    intermediate_instruction_ir[i].core_list[j].instruction_ir_list.push_back(tmpInstruction);
                }
            }
        }
    }

//    int load_total_num = 0;
//    for (auto iter = load_element_num.begin(); iter != load_element_num.end(); ++iter)
//    {
//        load_total_num += iter->second;
//    }
//    int merge_load_total_num = 0;
//    for (auto iter = load_element_num.begin(); iter != load_element_num.end(); ++iter)
//    {
//        float element_num = iter->first;
//        float load_time = iter->second;
//        int merge_num = element_num < 50 ? load_time / std::ceil(50.0/element_num) : load_time;
//        merge_load_total_num += merge_num;
//        std::cout << "element_num:" << iter->first
//                  << "    time:" << iter->second
//                  << "    ratio:" << float(iter->second) / float(load_total_num)*100 << "%"
//                  << "    merge:" << merge_num
//                  << std::endl;
//    }
//    std::cout << "total num:" << load_total_num << std::endl;
//    std::cout << "merge num:" << merge_load_total_num << std::endl;
}
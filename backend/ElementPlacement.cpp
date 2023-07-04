//
// Created by SXT on 2022/8/21.
//

#include "ElementPlacement.h"
#include "./zstr/zstr.hpp"

ElementPlacement::ElementPlacement(int mode)
{
    placement_mode = mode;
}

void ElementPlacement::PlaceElement()
{
    PIMCOMP_8_virtual_core_to_physical_core_map.resize(ChipH * ChipW);
    DetermineCoreMap();
    PlaceCore();
    Clear();
}

// Record the data exchange volume between each core and other cores,
// long long data records the exchange volume, and int data records the serial number of the corresponding core
// 记录每个核和其他各核的数据交流量,long long记录交流量，int记录对应核的序号
static std::vector<std::vector<std::pair<long long, int>>> core_comm_volume_detail;
// Records the total amount of data exchanged per core
// 记录每个核的数据交流总量
static std::vector<std::pair<long long, int>> each_core_comm_volume;


void GetFourNeighbours(int core_index, std::set<int> already_mapped_core, std::vector<int> & four_neighbours)
{
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (core_comm_volume_detail[core_index][i].second != core_index && already_mapped_core.count(core_comm_volume_detail[core_index][i].second) == 0)
        {
            four_neighbours.push_back(core_comm_volume_detail[core_index][i].second);
            if (four_neighbours.size() == 4)
                return;
        }
    }
}


int cmp(const std::pair<long long, int> &x, const std::pair<long long, int> &y) { return x.first > y.first; }

void ElementPlacement::DetermineCoreMap()
{
    srand((unsigned)time(NULL));
    each_core_comm_volume.resize(ChipH * ChipW);
    core_comm_volume_detail.resize(ChipH * ChipW);
    for (int i = 0; i < ChipH * ChipW; ++i)
        core_comm_volume_detail[i].resize(ChipH * ChipW);

    // First concentrate the data into the upper right half and then copy it to the lower left half.
    // Thus, each row (or column) represents all data communication between a core and other cores
    // 首先把数据集中到右上半区，然后复制到左下半区。这样每一行（或每一列）都代表一个核与其他核的全部数据通信
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        for (int j = 0; j <  i; ++j)
        {
            if(PIMCOMP_6_inter_core_communication[i][j] > 0)
            {
                PIMCOMP_6_inter_core_communication[j][i] += PIMCOMP_6_inter_core_communication[i][j];
                PIMCOMP_6_inter_core_communication[i][j] = PIMCOMP_6_inter_core_communication[j][i];
            }
            core_comm_volume_detail[j][i].first = PIMCOMP_6_inter_core_communication[j][i];
            core_comm_volume_detail[j][i].second = i;
            core_comm_volume_detail[i][j].first = PIMCOMP_6_inter_core_communication[j][i];
            core_comm_volume_detail[i][j].second = j;
        }
    }

    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        std::sort(core_comm_volume_detail[i].begin(), core_comm_volume_detail[i].end(), cmp);
    }

    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        long long core_comm = 0;
        for (int j = 0; j < ChipH * ChipW; ++j)
        {
//            std::cout << core_comm_volume_detail[i][j].first << " ";
            core_comm += core_comm_volume_detail[i][j].first;
        }
//        std::cout << std::endl;
        each_core_comm_volume[i].first = core_comm;
        each_core_comm_volume[i].second = i;
    }
    std::sort(each_core_comm_volume.begin(), each_core_comm_volume.end(), cmp);



    std::queue<std::pair<int, std::pair<int,int>>> core_queue;
    std::vector<std::vector<int>> core_map_2D;
    core_map_2D.resize(ChipH);
    for (int i = 0; i < ChipH; ++i)
        core_map_2D[i].resize(ChipW);
    std::set<int> already_mapped_core;
    int move[4][2] = {{-1,0},{0,1},{1,0},{0,-1}};

//    int first_place_core = each_core_comm_volume[0].second;
//    core_queue.push(std::make_pair(first_place_core, std::make_pair(ChipH/2,ChipW/2)));
//    core_map_2D[ChipH/2][ChipW/2] = first_place_core;
//    already_mapped_core.insert(first_place_core);

    for (int i = 0; i < ChipW * ChipH / 3; ++i)
    {
        int pre_place_core = each_core_comm_volume[i].second;
        int H_position, W_position;
        H_position = rand() % ChipH;
        W_position = rand() % ChipW;
        while (H_position < 0 || H_position >= ChipH || W_position < 0 || W_position >= ChipW || core_map_2D[H_position][W_position] != 0)
        {
            H_position = rand() % ChipH;
            W_position = rand() % ChipW;
        }
        core_queue.push(std::make_pair(pre_place_core, std::make_pair(H_position,W_position)));
        core_map_2D[H_position][W_position] = pre_place_core;
        already_mapped_core.insert(pre_place_core);
    }

    while (!core_queue.empty())
    {
        int core_index = core_queue.front().first;
        std::vector<int> four_neighbours;
        GetFourNeighbours(core_index, already_mapped_core, four_neighbours);
        int H_position = core_queue.front().second.first;
        int W_position = core_queue.front().second.second;
        if (four_neighbours.size() != 0)
        {
            int start_direction = rand() % 4;
            int neighbours_num = 0;
            for (int i = start_direction; i < start_direction+4; ++i)
            {
                int index = i % 4;
                int new_H_position = H_position + move[index][0];
                int new_W_position = W_position + move[index][1];
                if (new_H_position >= 0 && new_H_position < ChipH && new_W_position >= 0 && new_W_position < ChipW && core_map_2D[new_H_position][new_W_position] == 0)
                {
                    core_map_2D[new_H_position][new_W_position] = four_neighbours[neighbours_num];
                    already_mapped_core.insert(four_neighbours[neighbours_num]);
                    core_queue.push(std::make_pair(four_neighbours[neighbours_num], std::make_pair(new_H_position,new_W_position)));
                    neighbours_num++;
                }
            }
        }
        core_queue.pop();
    }

    // Check
    std::set<int> check_core;
    for (int i = 0; i < ChipH; ++i)
    {
        for (int j = 0; j < ChipW; ++j)
        {
//            std::cout << std::setw(4) << core_map_2D[i][j] << "  ";
            check_core.insert(core_map_2D[i][j]);
        }
//        std::cout << std::endl;
    }

    if (check_core.size() != ChipH * ChipW)
        fprintf(stderr, "Virtual Core To Physical Core Failed.\n");

    if (placement_mode == 0)
    {
        // Original
        for (int i = 0; i < ChipW * ChipH; ++i)
        {
            PIMCOMP_8_virtual_core_to_physical_core_map[i] = i;
        }
    }
    else
    {
        // Zig Zag
        for (int i = 0; i < ChipH; ++i)
        {
            for (int j = 0; j < ChipW; ++j)
            {
                if (i % 2 == 0)
                    PIMCOMP_8_virtual_core_to_physical_core_map[i * ChipW + j] = i * ChipW + j;
                else
                    PIMCOMP_8_virtual_core_to_physical_core_map[i * ChipW + ChipW - j - 1] = i * ChipW + j;
            }
        }

        // BFS + Greedy
//        for (int i = 0; i < ChipH; ++i)
//        {
//            for (int j = 0; j < ChipW; ++j)
//            {
//                PIMCOMP_8_virtual_core_to_physical_core_map[i * ChipW + j] = core_map_2D[i][j];
//            }
//        }
    }
}

void ElementPlacement::PlaceCore()
{
    std::vector<struct PIMCOMP_4_instruction_ir> * base_instruction_ir;
//    if (instruction_with_reload == 1) // Batch-Pipeline
//        base_instruction_ir = & PIMCOMP_6_base_instruction_with_input;
//    else  // Element-Pipeline
//        base_instruction_ir = & PIMCOMP_4_base_instruction_ir;
    base_instruction_ir = & PIMCOMP_7_base_instruction_ir_with_optimization;
    effective_instruction_group_num = base_instruction_ir->size();
    PIMCOMP_8_base_instruction_ir_with_placement.resize(base_instruction_ir->size());
    for (int i = 0; i < base_instruction_ir->size(); ++i)
    {
        if (base_instruction_ir->at(i).core_list.size() > 0)
        {
            PIMCOMP_8_base_instruction_ir_with_placement[i].core_list.resize(ChipW * ChipH);
            for (int j = 0; j < ChipH * ChipW; ++j)
            {
                int physical_core = PIMCOMP_8_virtual_core_to_physical_core_map[j];
                PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[physical_core].instruction_ir_list.assign(
                    base_instruction_ir->at(i).core_list[j].instruction_ir_list.begin(),
                    base_instruction_ir->at(i).core_list[j].instruction_ir_list.end());
            }
        }
    }
}



void ElementPlacement::SaveVerificationInfo(Json::Value DNNInfo)
{
    int AG_num = PIMCOMP_2_resource_info.AGs;
    PIMCOMP_VERIFICATION_INFO["AG_info"].resize(AG_num);
    for (int i = 0; i < AG_num; ++i)
    {
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["AG_index"] = i;
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["node_name"] = PIMCOMP_node_list[node_index].name;
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["height_start"] = PIMCOMP_3_hierarchy_map.whole[i][0].height_start;
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["height_end"] = PIMCOMP_3_hierarchy_map.whole[i][0].height_end;
        int crossbar_num = PIMCOMP_3_hierarchy_map.whole[i].size();
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["crossbar_num"] = crossbar_num;
        PIMCOMP_VERIFICATION_INFO["AG_info"][i]["crossbar"].resize(crossbar_num);
        for (int j = 0; j < crossbar_num; ++j)
        {
            PIMCOMP_VERIFICATION_INFO["AG_info"][i]["crossbar"][j]["width_start"] = PIMCOMP_3_hierarchy_map.whole[i][j].width_start;
            PIMCOMP_VERIFICATION_INFO["AG_info"][i]["crossbar"][j]["width_end"] = PIMCOMP_3_hierarchy_map.whole[i][j].width_end;
        }
    }

    PIMCOMP_VERIFICATION_INFO["node_list"] = DNNInfo["node_list"];
    PIMCOMP_VERIFICATION_INFO["reshape_info"] = DNNInfo["reshape_info"];

    PIMCOMP_VERIFICATION_INFO["instruction"]["core_list"].resize(ChipH * ChipW);
    for (int i = 0; i < effective_instruction_group_num; ++i)
    {
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            int instruction_num = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list.size();
            if (instruction_num == 0)
                continue;
            for (int k = 0; k < instruction_num; ++k)
            {
                struct INST Instruction = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list[k];
                Json::Value JsonInstruction = SaveInstructionWithAddressInJSON(Instruction); // core_index is j
                PIMCOMP_VERIFICATION_INFO["instruction"]["core_list"][j].append(JsonInstruction);
            }
        }
    }

    PIMCOMP_VERIFICATION_INFO["comm_pair_total_num"] = comm_pair_total_num;

    std::string strJson = PIMCOMP_VERIFICATION_INFO.toStyledString();
    std::ofstream fob("../output/VerificationInfo.json", std::ios::trunc | std::ios::out);
    if (fob.is_open())
    {
        fob.write(strJson.c_str(), strJson.length());
        fob.close();
    }
}

void ElementPlacement::SaveSimulationInfo()
{
    // Record the number of crossbars included in each AG
    std::vector<int> AG_crossbar_num;
    int AG_num = PIMCOMP_2_resource_info.AGs;
    AG_crossbar_num.resize(AG_num);
    for (int i = 0; i < AG_num; ++i)
        AG_crossbar_num[i] = PIMCOMP_3_hierarchy_map.whole[i].size();

    Json::Value offset;
    offset["offset_value"] = 0;
    offset["offset_select"] = 0;
    for (int i = 0; i < effective_instruction_group_num; ++i)
    {
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            std::string core_name = "core" + std::to_string(j);
            int instruction_num = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list.size();
            if (instruction_num == 0)
                continue;
            if (i == 0)
            {
                Json::Value SetbwInstruction;
                SetbwInstruction["op"] = "setbw";
                SetbwInstruction["ibiw"] = 8;
                SetbwInstruction["obiw"] = 8;
                PIMCOMP_SIMULATION_INFO[core_name].append(SetbwInstruction);
            }
            long long byw = 1;
            for (int k = 0; k < instruction_num; ++k)
            {
                struct INST Instruction = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list[k];
                std::string Operation = Instruction.operation;
                for (auto s = Operation.begin(); s != Operation.end(); s++) { *s = tolower(*s);} // TODO: cost too much time
                switch (Instruction.type)
                {
                    case MVMUL:
                    {
                        Json::Value JsonScalarInstruction_rd;
                        JsonScalarInstruction_rd["op"] = "sldi";
                        JsonScalarInstruction_rd["rd"] = 0;
                        Json::Int64 imm_dst = (Instruction.destination_address + Instruction.destination_offset) * byw;
                        JsonScalarInstruction_rd["imm"] = imm_dst;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rd);

                        Json::Value JsonScalarInstruction_rs1;
                        JsonScalarInstruction_rs1["op"] = "sldi";
                        JsonScalarInstruction_rs1["rd"] = 1;
                        Json::Int64 imm_src = (Instruction.source_address + Instruction.source_offset) * byw;
                        JsonScalarInstruction_rs1["imm"] = imm_src;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rs1);

                        Json::Value JsonInstruction;
                        JsonInstruction["op"] = Operation;
                        int AG_index = Instruction.source;
                        JsonInstruction["group"] = AG_crossbar_num[AG_index];
                        JsonInstruction["relu"] = 0;
                        JsonInstruction["rd"] = 0;
                        JsonInstruction["rs1"] = 1;
                        JsonInstruction["mbiw"] = 8;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        break;
                    }
                    case VEC1OP:
                    {
                        Json::Value JsonScalarInstruction_rd;
                        JsonScalarInstruction_rd["op"] = "sldi";
                        JsonScalarInstruction_rd["rd"] = 0;
                        Json::Int64 imm_dst = (Instruction.destination_address + Instruction.destination_offset) * byw;
                        JsonScalarInstruction_rd["imm"] = imm_dst;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rd);

                        Json::Value JsonScalarInstruction_rs1;
                        JsonScalarInstruction_rs1["op"] = "sldi";
                        JsonScalarInstruction_rs1["rd"] = 1;
                        Json::Int64 imm_src = (Instruction.source_address + Instruction.source_offset) * byw;
                        JsonScalarInstruction_rs1["imm"] = imm_src;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rs1);

                        Json::Value JsonInstruction;
                        JsonInstruction["op"] = Operation;
                        JsonInstruction["rd"] = 0;
                        JsonInstruction["rs1"] = 1;
                        JsonInstruction["len"] = Instruction.element_num;
                        JsonInstruction["offset"] = offset;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);

                        break;
                    }
                    case VEC2OP:
                    {
                        Json::Value JsonScalarInstruction_rd;
                        JsonScalarInstruction_rd["op"] = "sldi";
                        JsonScalarInstruction_rd["rd"] = 0;
                        Json::Int64 imm_dst = (Instruction.destination_address + Instruction.destination_offset) * byw;
                        JsonScalarInstruction_rd["imm"] = imm_dst;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rd);

                        Json::Value JsonScalarInstruction_rs1;
                        JsonScalarInstruction_rs1["op"] = "sldi";
                        JsonScalarInstruction_rs1["rd"] = 1;
                        Json::Int64 imm_rs1 = (Instruction.source_1_address + Instruction.source_1_offset) * byw;
                        JsonScalarInstruction_rs1["imm"] = imm_rs1;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rs1);

                        Json::Value JsonScalarInstruction_rs2;
                        JsonScalarInstruction_rs2["op"] = "sldi";
                        JsonScalarInstruction_rs2["rd"] = 2;
                        Json::Int64 imm_rs2 =  (Instruction.source_2_address + Instruction.source_2_offset) * byw;
                        JsonScalarInstruction_rs2["imm"] = imm_rs2;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rs2);

                        Json::Value JsonInstruction;
                        JsonInstruction["op"] = Operation;
                        JsonInstruction["rd"] = 0;
                        JsonInstruction["rs1"] = 1;
                        JsonInstruction["rs2"] = 2;
                        JsonInstruction["len"] = Instruction.element_num;
                        JsonInstruction["offset"] = offset;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        break;
                    }
                    case COMM:
                    {
                        if (Operation == "send")
                        {
                            Json::Value JsonScalarInstruction0;
                            JsonScalarInstruction0["op"] = "sldi";
                            JsonScalarInstruction0["rd"] = 0;
                            Json::Int64 imm = Instruction.source_address * byw;
                            JsonScalarInstruction0["imm"] = imm;
                            PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction0);

                            Json::Value JsonInstruction;
                            JsonInstruction["op"] = Operation;
                            JsonInstruction["rd"] = 0;
                            JsonInstruction["core"] = Instruction.to_core;
                            JsonInstruction["size"] = Instruction.element_num;
                            JsonInstruction["offset"] = offset;
                            PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        }
                        else
                        {
                            Json::Value JsonScalarInstruction0;
                            JsonScalarInstruction0["op"] = "sldi";
                            JsonScalarInstruction0["rd"] = 0;
                            Json::Int64 imm = Instruction.destination_address * byw;
                            JsonScalarInstruction0["imm"] = imm;
                            PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction0);

                            Json::Value JsonInstruction;
                            JsonInstruction["op"] = Operation;
                            JsonInstruction["rd"] = 0;
                            JsonInstruction["core"] = Instruction.from_core;
                            JsonInstruction["size"] = Instruction.element_num;
                            JsonInstruction["offset"] = offset;
                            PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        }
                        break;
                    }
                    case MEM:
                    {
                        if (Instruction.destination_address < 0)
                            Instruction.destination_address = 0;
                        if (Instruction.source_address < 0)
                            Instruction.source_address = 0;

                        Json::Value JsonScalarInstruction_rd;
                        JsonScalarInstruction_rd["op"] = "sldi";
                        JsonScalarInstruction_rd["rd"] = 0;
                        Json::Int64 imm_dst = (Instruction.destination_address + Instruction.destination_offset) * byw;
                        JsonScalarInstruction_rd["imm"] = imm_dst;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rd);

                        Json::Value JsonScalarInstruction_rs1;
                        JsonScalarInstruction_rs1["op"] = "sldi";
                        JsonScalarInstruction_rs1["rd"] = 1;
                        Json::Int64 imm_src = (Instruction.source_address + Instruction.source_offset) * byw;
                        JsonScalarInstruction_rs1["imm"] = imm_src;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rs1);

                        Json::Value JsonInstruction;
                        JsonInstruction["op"] = Operation;
                        JsonInstruction["rd"] = 0;
                        JsonInstruction["rs1"] = 1;
                        JsonInstruction["size"] = Instruction.element_num;
                        JsonInstruction["offset"] = offset;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        break;
                    }
                    case LLDI:
                    {
                        Json::Value JsonScalarInstruction_rd;
                        JsonScalarInstruction_rd["op"] = "sldi";
                        JsonScalarInstruction_rd["rd"] = 0;
                        Json::Int64 imm = (Instruction.destination_address + Instruction.destination_offset) * byw;
                        JsonScalarInstruction_rd["imm"] = imm;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonScalarInstruction_rd);

                        Json::Value JsonInstruction;
                        JsonInstruction["op"] = Operation;
                        JsonInstruction["rd"] = 0;
                        JsonInstruction["imm"] = Instruction.imm_value;
                        JsonInstruction["len"] = Instruction.element_num;
                        JsonInstruction["offset"] = offset;
                        PIMCOMP_SIMULATION_INFO[core_name].append(JsonInstruction);
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    }

    int core_cnt = PIMCOMP_SIMULATION_INFO.size();
    PIMCOMP_SIMULATION_INFO["config"]["core_cnt"] = core_cnt;
    PIMCOMP_SIMULATION_INFO["config"]["xbar_size"][0] = CrossbarH;
    PIMCOMP_SIMULATION_INFO["config"]["xbar_size"][1] = CrossbarW;
    PIMCOMP_SIMULATION_INFO["config"]["xbar_array_count"] = CoreH * CoreW;
    PIMCOMP_SIMULATION_INFO["config"]["cell_precision"] = CellPrecision;
    PIMCOMP_SIMULATION_INFO["config"]["adc_count"] = ArithmeticPrecision;

    std::string strJson = PIMCOMP_SIMULATION_INFO.toStyledString();
//    std::ofstream fob("../output/SimulationResult.json", std::ios::trunc | std::ios::out);
//    if (fob.is_open())
//    {
//        fob.write(strJson.c_str(), strJson.length());
//        fob.close();
//    }

    zstr::ofstream fob("../output/SimulationInfo.gz", std::ios::binary);
    if (fob.is_open())
    {
        fob.write(strJson.c_str(), strJson.length());
        fob.close();
    }
}

void ElementPlacement::Clear()
{
    core_comm_volume_detail.clear();
}

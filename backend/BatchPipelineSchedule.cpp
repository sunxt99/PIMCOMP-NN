//
// Created by SXT on 2022/9/16.
//

#include "BatchPipelineSchedule.h"

static int AG_accumulated_num[MAX_AG] = {0};
static int AG_output_element_size[MAX_AG] = {0};

static int activate_flag[MAX_AG] = {0};
static int add_flag[MAX_AG] = {0};
static int comm_flag[MAX_AG] = {0};
static int wb_flag[MAX_AG] = {0};

static int node_offset_instruction_group[MAX_AG] = {0};
static int node_offset_inference[MAX_AG] = {0};

std::vector<struct PIMCOMP_4_instruction_ir_body> PIMCOMP_4_base_instruction_ir_body;
std::vector<struct PIMCOMP_4_instruction_ir_vvadd> PIMCOMP_4_base_instruction_ir_vvadd;
std::vector<struct PIMCOMP_4_instruction_ir_vrelu> PIMCOMP_4_base_instruction_ir_vrelu;
std::vector<std::vector<int>> PIMCOMP_4_AG_input_cycle_index;

static std::vector<std::vector<long long>> bias_address_map; // bias_address_map[core_index][node_index]就得到了该核该节点的bias存储位置

BatchPipelineSchedule::BatchPipelineSchedule(std::string model_name_)
{
    model_name = model_name_;
    element_pipeline = 0;
    node_num = PIMCOMP_node_list.size();
    AG_num_total = PIMCOMP_3_hierarchy_map.whole.size();
    PIMCOMP_4_core_instruction_group_num.resize(ChipH * ChipW);
    PIMCOMP_6_inter_core_communication.resize(ChipW * ChipH);
    for (int i = 0; i < ChipH * ChipW; ++i)
        PIMCOMP_6_inter_core_communication[i].resize(ChipW * ChipH);
}

static int comm_index = 0;
void BatchPipelineSchedule::ScheduleExecution()
{
    PipelineDesign();
    SchedulePreparation();
    ScheduleMain();
    comm_pair_total_num = comm_index;
    Clear();
}

static int concat_rest_num[MAX_AG] = {0};
static int concat_max_level[MAX_AG] = {0};
static int eltwise_rest_num[MAX_AG] = {0};
static int eltwise_max_level[MAX_AG] = {0};
void BatchPipelineSchedule::PipelineDesign()
{
    // accelerate googlenet structure
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_CONCAT")
            concat_rest_num[i] = PIMCOMP_node_list[i].provider_num;
        if (PIMCOMP_node_list[i].operation == "OP_ELTWISE")
            eltwise_rest_num[i] = PIMCOMP_node_list[i].provider_num;
    }

    ClassifyNodes(0, -1);
    for (int i = 0; i < node_num; ++i)
    {
        int consumer_num = PIMCOMP_topology_provider_consumer_relation[i].size();
        int max_level_gap = 0;
        if (consumer_num != 0) // ALL nodes exclude ACT OP
        {
            int node_level_index = PIMCOMP_node_list[i].level_index;
            for (int j = 0; j < consumer_num; ++j)
            {
                int consumer_index = PIMCOMP_topology_provider_consumer_relation[i][j];
                if (consumer_index < node_num && consumer_index > 0)
                {
                    if (PIMCOMP_node_list[consumer_index].operation != "OP_CONV" && PIMCOMP_node_list[consumer_index].operation != "OP_FC")
                    {
                        int consumer_level_index = PIMCOMP_node_list[consumer_index].level_index;
                        if (consumer_level_index - node_level_index > max_level_gap)
                            max_level_gap = consumer_level_index - node_level_index;
                    }
                }
            }
            PIMCOMP_node_list[i].max_level_gap = max_level_gap;
//            if (max_level_gap > 0)
//                std::cout << PIMCOMP_node_list[i].index << "  " << PIMCOMP_node_list[i].name << "  " <<  PIMCOMP_node_list[i].operation << "  " << PIMCOMP_node_list[i].max_level_gap << std::endl;
        }
    }
}

void BatchPipelineSchedule::ClassifyNodes(int node_index, int level_index)
{
    if (node_index == 0)
    {
        PIMCOMP_node_list[node_index].level_index = level_index;
    }
    int previous_level = PIMCOMP_node_list[node_index].level_index;
    if (previous_level < level_index)
    {
        PIMCOMP_node_list[node_index].level_index = level_index;
    }
    if (PIMCOMP_node_list[node_index].consumer_num == 0)
    {
        return;
    }
    else
    {
        int consumer_num = PIMCOMP_node_list[node_index].consumer_num;
        for (int i = 0; i < consumer_num; ++i)
        {
            int consumer_index = PIMCOMP_node_list[node_index].consumer_index[i];
            std::string consumer_op = PIMCOMP_node_list[consumer_index].operation;
            if ( consumer_op == "OP_CONV" || consumer_op == "OP_FC")
            {
                ClassifyNodes(consumer_index, level_index+1);
            }
            // 对于inception结构进行优化，加速分级
            else if (consumer_op == "OP_CONCAT")
            {
                if (level_index > concat_max_level[consumer_index])
                    concat_max_level[consumer_index] = level_index;
                if (concat_rest_num[consumer_index] != 1)
                {
                    concat_rest_num[consumer_index]--;
                }
                else
                    ClassifyNodes(consumer_index, concat_max_level[consumer_index]);
            }
            else if (consumer_op == "OP_ELTWISE")
            {
                if (level_index > eltwise_max_level[consumer_index])
                    eltwise_max_level[consumer_index] = level_index;
                if (eltwise_rest_num[consumer_index] != 1)
                {
                    eltwise_rest_num[consumer_index]--;
                }
                else
                    ClassifyNodes(consumer_index, eltwise_max_level[consumer_index]);
            }
            else
                ClassifyNodes(consumer_index, level_index);
        }
        return;
    }
}


void BatchPipelineSchedule::SchedulePreparation()
{
    //// 其他操作都放到了Hierarchy Mapping阶段完成
    //// 得到两个调度过程中需要的值（每个结点Rep0 AG0所在的核、以及每个结点每个Rep的AG0所在的核）
    PIMCOMP_4_first_AG_info.node_list.resize(node_num);
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        for (int j = 0; j < AG_num; ++j)
        {
            int AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication;
            if (AG_index_in_replication == 0)
            {
                int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
                int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
                int replication_num = PIMCOMP_node_list[node_index].replication_num;
                if (PIMCOMP_4_first_AG_info.node_list[node_index].replication_list.size() != replication_num)
                    PIMCOMP_4_first_AG_info.node_list[node_index].replication_list.resize(replication_num);
                PIMCOMP_4_first_AG_info.node_list[node_index].replication_list[replication_index] = i;
            }
        }
    }

    //// 得到周期结束需要传输数据的核
    int node_appearance_num[MAX_NODE] = {0};
    int node_appearance_element[MAX_NODE] = {0};
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        if (AG_num == 0)
            continue;
        int pre_node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[0];
        int pre_replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[0].replication_index;
        int pre_AG_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[0].AG_index_in_total;
        int pre_AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[0].AG_index_in_replication;

        int pre_core_index = i;
        int pre_AG_height = PIMCOMP_3_hierarchy_map.whole[pre_AG_index][0].height_end - PIMCOMP_3_hierarchy_map.whole[pre_AG_index][0].height_start + 1;
        struct node_recv_info RecvInfo;
        RecvInfo.replication_index = pre_replication_index;
        RecvInfo.AG_index = pre_AG_index;
        RecvInfo.AG_index_in_replication = pre_AG_index_in_replication;
        RecvInfo.core_index = pre_core_index;
        RecvInfo.node_index = pre_node_index;
        if ( PIMCOMP_node_list[pre_node_index].operation == "OP_FC")
        {
            RecvInfo.start_offset_num = node_appearance_num[pre_node_index];
            RecvInfo.start_offset_element = node_appearance_element[pre_node_index];
            RecvInfo.recv_num = 1;
            RecvInfo.recv_element = pre_AG_height;
        }
        PIMCOMP_4_recv_info.node_list[pre_node_index].push_back(RecvInfo);
        node_appearance_num[pre_node_index]++;
        node_appearance_element[pre_node_index] += pre_AG_height;
        for (int j = 1; j < AG_num; ++j)
        {
            int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
            int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
            int AG_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
            int AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication;
            int AG_height = PIMCOMP_3_hierarchy_map.whole[AG_index][0].height_end - PIMCOMP_3_hierarchy_map.whole[AG_index][0].height_start + 1;

            if (node_index != pre_node_index || pre_replication_index != replication_index)
            {
                struct node_recv_info RecvInfo2;
                RecvInfo2.replication_index = replication_index;
                RecvInfo2.AG_index = AG_index;
                RecvInfo2.AG_index_in_replication = AG_index_in_replication;
                RecvInfo2.core_index = i;
                RecvInfo2.node_index = node_index;
                if (PIMCOMP_node_list[node_index].operation == "OP_FC")
                {
                    RecvInfo2.start_offset_num = node_appearance_num[node_index];
                    RecvInfo2.start_offset_element = node_appearance_element[node_index];
                    RecvInfo2.recv_num = 1;
                    RecvInfo2.recv_element = AG_height;
                }
                PIMCOMP_4_recv_info.node_list[node_index].push_back(RecvInfo2);
            }
            else
            {
                if (PIMCOMP_node_list[node_index].operation == "OP_FC")
                {
                    int already_num = PIMCOMP_4_recv_info.node_list[node_index].size();
                    PIMCOMP_4_recv_info.node_list[node_index][already_num - 1].recv_num += 1;
                    PIMCOMP_4_recv_info.node_list[node_index][already_num - 1].recv_element += AG_height;
                }
            }
            node_appearance_num[node_index]++;
            node_appearance_element[node_index] += AG_height;
            pre_replication_index = replication_index;
            pre_node_index = node_index;
        }
    }


    //// 每个AG的input_cycle_index
    PIMCOMP_4_AG_input_cycle_index.resize(PIMCOMP_2_resource_info.AGs);
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_CONV" || PIMCOMP_node_list[i].operation == "OP_FC")
        {
            // 预处理
            int effective_node_index = PIMCOMP_node_list[i].effective_node_index;
            int replication_num = PIMCOMP_node_list[i].replication_num;
            int input_cycle_num; // input_cycle_num = output_channel_num
            if (PIMCOMP_node_list[i].operation == "OP_CONV")
                input_cycle_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            else
                input_cycle_num = 1;
            std::vector<int> AG0_num_per_core;
            std::vector<std::vector<int>> replication_index_per_core;
//            std::vector<int> input_cycle_num_per_core;
//            std::vector<int> start_input_cycle_index_per_core;
            AG0_num_per_core.resize(ChipH * ChipW);
            replication_index_per_core.resize(ChipH * ChipW);
//            input_cycle_num_per_core.resize(ChipH * ChipW);
//            start_input_cycle_index_per_core.resize(ChipH * ChipW);
            for (int j = 0; j < ChipW * ChipH; ++j)
            {
                for (int k = 0; k < PIMCOMP_4_virtual_core_AG_map.core_list[j].AG_list.size(); ++k)
                {
                    int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[j].AG_list[k].node_index;
                    if (node_index == i)
                    {
                        int AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[j].AG_list[k].AG_index_in_replication;
                        int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[j].AG_list[k].replication_index;
                        int AG_index_in_total = PIMCOMP_4_virtual_core_AG_map.core_list[j].AG_list[k].AG_index_in_total;
                        if (AG_index_in_replication == 0)
                        {
                            AG0_num_per_core[j]++;
                            replication_index_per_core[j].push_back(replication_index);
                        }
                    }
                }
            }
            // 得到每个核分配了多少计算任务
            int accumulated_input_cycle_num = 0;
            for (int j = 0; j < ChipW * ChipH; ++j)
            {
                int input_cycle_num_this_core = std::ceil(float(input_cycle_num) / float(replication_num) * float(AG0_num_per_core[j]));
                if (input_cycle_num_this_core + accumulated_input_cycle_num > input_cycle_num)
                    input_cycle_num_this_core = input_cycle_num - accumulated_input_cycle_num;
//                input_cycle_num_per_core[j] = input_cycle_num_this_core;
//                start_input_cycle_index_per_core[j] = accumulated_input_cycle_num;
                if (input_cycle_num_this_core != 0)
                {
                    int current_input_cycle_index = accumulated_input_cycle_num;
                    bool stop = false;
                    while (!stop)
                    {
                        for (int k = 0; k < replication_index_per_core[j].size(); ++k)
                        {
                            int replication_index = replication_index_per_core[j][k];
                            int AG_num = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_index.size();
                            for (int l = 0; l < AG_num; ++l)
                            {
                                int AG_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_index[l];
                                PIMCOMP_4_AG_input_cycle_index[AG_index].push_back(current_input_cycle_index);
                            }
                            current_input_cycle_index++;
                            if (current_input_cycle_index == accumulated_input_cycle_num + input_cycle_num_this_core)
                            {
                                stop = true;
                                break;
                            }
                        }
                    }
                }
                accumulated_input_cycle_num += input_cycle_num_this_core;
            }
        }
    }
//    //// 打印映射信息
//    std::ofstream OutFile("../output/TaskAllocating.txt", std::ios::out | std::ios::trunc);
//    for (int i = 0; i < PIMCOMP_4_AG_input_cycle_index.size(); ++i)
//    {
//        OutFile << "AG_" << i << std::endl;
//        for (int j = 0; j < PIMCOMP_4_AG_input_cycle_index[i].size(); ++j)
//        {
//            OutFile << "    " << PIMCOMP_4_AG_input_cycle_index[i][j] << std::endl;
//        }
//    }
}

void BatchPipelineSchedule::ScheduleStage0(int instruction_group_index, bool append_instruction)
{
    bias_address_map.resize(ChipW * ChipH);
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        bias_address_map[i].resize(node_num);
        if (PIMCOMP_4_core_instruction_group_num[i] < instruction_group_index)
            continue;
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        int AG_num = current_core.AG_list.size();
        if (AG_num == 0)
            continue;
        int start_address = 0;
        std::set<int> core_node; // 为每个核上的每个节点都LOAD一次bias
        for (int j = 0; j < AG_num; ++j)
        {
            int node_index = current_core.node_list[j];
            int level_index = PIMCOMP_node_list[node_index].level_index;
            if (core_node.count(node_index) == 0)
            {
                // TODO 这里可能需要添加一个判断该节点是否需要加载偏置，有可能有的CONV或FC不需要BIAS。需要生成JSON时确定。
                if (PIMCOMP_node_list[node_index].with_bias)
                {
                    core_node.insert(node_index);
                    bias_address_map[i][node_index] = start_address;
                    struct INST Instruction_ld;
                    Instruction_ld.node_index = node_index;
                    Instruction_ld.type = MEM;
                    Instruction_ld.level_index = level_index;
                    Instruction_ld.operation = "LD";
                    Instruction_ld.stage = "BIAS";
                    Instruction_ld.source = -1 * node_index;
                    Instruction_ld.source_address = -1 * node_index;
                    Instruction_ld.source_offset = 0;
                    Instruction_ld.destination = node_index;
                    Instruction_ld.destination_address = start_address;
                    Instruction_ld.destination_offset = 0;
                    Instruction_ld.element_num =  current_core.AG_list[j].output_element_num;
                    start_address += Instruction_ld.element_num;
                    Instruction_ld.instruction_group_index = instruction_group_index;
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[i].instruction_ir_list_body.push_back(Instruction_ld);
                }
            }
        }
        PIMCOMP_5_memory_start_address[i] = start_address;
    }
}

static int MVMUL_num_every_node[MAX_NODE] = {0};
void BatchPipelineSchedule::ScheduleStage1( int instruction_group_index, bool append_instruction)
{
    //// 首先为每个AG生成MVMUL操作
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (PIMCOMP_4_core_instruction_group_num[i] < instruction_group_index)
            continue;
        int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        for (int j = 0; j < AG_num; ++j)
        {
            int AG_index_in_total = current_core.AG_list[j].AG_index_in_total;
            int input_cycle_in_total = current_core.AG_list[j].input_cycle_in_total;
            int node_index = current_core.node_list[j];
            int AG_num_per_replication = current_core.AG_list[j].AG_num_per_replication;
//            if (MVMUL_num_every_node[node_index]  < input_cycle_in_total * AG_num_per_replication)
            if (node_offset_inference[AG_index_in_total] < PIMCOMP_4_AG_input_cycle_index[AG_index_in_total].size())
            {
                int replication_index = current_core.AG_list[j].replication_index;
                int AG_index_in_replication = current_core.AG_list[j].AG_index_in_replication;
                if (append_instruction)
                {
                    int replication_num = current_core.AG_list[j].replication_num;
                    int AGP = current_core.AG_list[j].AGP;
                    int agp_index = current_core.AG_list[j].agp_index;
                    int agp_offset = current_core.AG_list[j].agp_offset;
                    int input_element_num = current_core.AG_list[j].input_element_num;
                    int output_element_num = current_core.AG_list[j].output_element_num;
                    int level_index = PIMCOMP_node_list[node_index].level_index;

                    // For MVMUL
                    struct INST Instruction;
                    Instruction.type = MVMUL;
                    Instruction.operation = "MVMUL";
                    Instruction.stage = "MAIN";
                    Instruction.AG_num_per_replication = AG_num_per_replication;
//                    Instruction.input_cycle_index = MVMUL_num_every_node[node_index] / AG_num_per_replication;
                    Instruction.input_cycle_index = PIMCOMP_4_AG_input_cycle_index[AG_index_in_total][node_offset_inference[AG_index_in_total]];
                    Instruction.AG_index_in_total = AG_index_in_total;
                    Instruction.replication_index = replication_index;
                    Instruction.replication_num = replication_num;
                    Instruction.input_cycle_in_total = input_cycle_in_total;
                    Instruction.AG_index_in_replication = AG_index_in_replication;
                    Instruction.conv_or_fc = PIMCOMP_node_list[node_index].operation;
                    Instruction.node_index = node_index;
                    Instruction.AGP = AGP;
                    Instruction.agp_index = agp_index;
                    Instruction.destination = Instruction.AG_index_in_total;
                    Instruction.source = Instruction.AG_index_in_total;
                    Instruction.level_index = level_index;
                    Instruction.input_element_num = input_element_num;
                    Instruction.output_element_num = output_element_num;
                    Instruction.source_offset = (node_offset_inference[AG_index_in_total] % operation_cycle_before_comm % 2) * input_element_num + agp_offset; // 这里的2是因为double buffer
                    Instruction.destination_offset = (node_offset_inference[AG_index_in_total] % operation_cycle_before_comm) * output_element_num + agp_offset;
                    Instruction.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[i].instruction_ir_list_body.push_back(Instruction);
                    MVMUL_num_every_node[node_index]++;

                    // For BIAS
                    if (PIMCOMP_node_list[node_index].with_bias)
                    {
                        if(AG_index_in_replication == 0)
                        {
                            struct INST Instruction_bias;
                            Instruction_bias.type = VEC2OP;
                            Instruction_bias.operation = "VVADD";
                            Instruction_bias.stage = "MAIN-B";
                            Instruction_bias.node_index = node_index;
                            //destination_address在后面分配
                            Instruction_bias.destination = AG_index_in_total;
                            Instruction_bias.destination_offset = (node_offset_inference[AG_index_in_total] % operation_cycle_before_comm) * output_element_num;
                            //source_1_address在后面分配
                            Instruction_bias.source_1 = Instruction_bias.destination;
                            Instruction_bias.source_1_offset = Instruction_bias.destination_offset;
                            //source_2_address当下就分配。就是核上保存bias的地址。
                            Instruction_bias.source_2 = -1;
                            Instruction_bias.source_2_address = bias_address_map[i][node_index];
                            Instruction_bias.source_2_offset = 0;
                            Instruction_bias.level_index = level_index;
                            Instruction_bias.relative_length = 1;
                            Instruction_bias.element_num = output_element_num;
                            Instruction_bias.instruction_group_index = instruction_group_index;
                            if(append_instruction)
                                PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[i].instruction_ir_list_body.push_back(Instruction_bias);
                        }
                    }
                }
//                for AG_output_element_size
                if (AG_accumulated_num[AG_index_in_total] == 0)
                {
                    int effective_node_index = PIMCOMP_node_list[node_index].effective_node_index;
                    int crossbar_num = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list.size();
                    int crossbar_start_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[0];
                    int crossbar_end_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[crossbar_num - 1];
                    int input_element_num = PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_start + 1;
                    int output_element_num = PIMCOMP_2_virtual_crossbar[crossbar_end_index].width_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].width_start + 1;
                    AG_output_element_size[AG_index_in_total] = output_element_num;
                }

//                 for stage_2 ADD
                if (j != 0)
                {
                    if (node_index == PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j-1] && replication_index == PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j-1].replication_index)
                    {
                        add_flag[AG_index_in_total] = 1;
                    }
                }

                if (AG_index_in_replication == 0)
                {
                    activate_flag[AG_index_in_total] = 1;
                }

                //// for stage_3 SEND/RECV (具体的判断条件在函数中写)
                comm_flag[AG_index_in_total] = 1;

                //// for stage_4 WB (具体的判断条件在函数中写)
                wb_flag[AG_index_in_total] = 1;

                // consider the offset
                node_offset_instruction_group[AG_index_in_total] += 1;
                node_offset_inference[AG_index_in_total] += 1;

                // record the input_cycle_index
//                if (AG_index_in_replication == 0)
//                    PIMCOMP_4_input_cycle_record[node_index].push_back(input_cycle_this_replication_start+AG_accumulated_num[AG_index_in_total]);

                AG_accumulated_num[AG_index_in_total] += 1;
            }
        }
    }
}



void BatchPipelineSchedule::ScheduleStage2( int instruction_group_index, bool append_instruction)
{
    //// 同一结点且同一权重块的AG之间的结果融合（VVADD）
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (PIMCOMP_4_core_instruction_group_num[i] < instruction_group_index)
            continue;
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        int AG_num = current_core.AG_list.size();
        if (AG_num == 0)
            continue;
        int node_index = current_core.node_list[0];
        int replication_index = current_core.AG_list[0].replication_index;
        int p = 1;
        for (int j = 1; j < AG_num; ++j)
        {
            int current_node_index = current_core.node_list[j];
            int current_replication_index = current_core.AG_list[j].replication_index;
            int AG_index_in_total = current_core.AG_list[j].AG_index_in_total;
            int level_index = PIMCOMP_node_list[node_index].level_index;
            if (node_index == current_node_index && replication_index == current_replication_index)
            {
                if (add_flag[AG_index_in_total] == 1)
                {
                    struct INST Instruction;
                    Instruction.type = VEC2OP;
                    Instruction.operation = "VVADD";
                    Instruction.stage = "MAIN-A";
                    Instruction.node_index = current_node_index;
                    Instruction.destination = current_core.AG_list[j-p].AG_index_in_total;
                    Instruction.source_1 = current_core.AG_list[j-p].AG_index_in_total;
                    Instruction.source_2 = current_core.AG_list[j].AG_index_in_total;
                    Instruction.AGP = current_core.AG_list[j].AGP;
                    Instruction.agp_index = current_core.AG_list[j].agp_index;
                    Instruction.level_index = level_index;
                    Instruction.relative_length = 1;
                    Instruction.element_num = Instruction.relative_length * AG_output_element_size[Instruction.source_1];
                    Instruction.instruction_group_index = instruction_group_index;
//                    Instruction.destination_offset = (node_offset_inference[AG_index_in_total] - 1)* Instruction.element_num;
                    Instruction.destination_offset = ((node_offset_inference[AG_index_in_total] - 1) % operation_cycle_before_comm )* Instruction.element_num; //// 【【【【这个修改不确定！！】】】】
                    Instruction.source_1_offset = Instruction.destination_offset;
                    Instruction.source_2_offset = Instruction.destination_offset;
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_vvadd[instruction_group_index].core_list[i].instruction_ir_list_vvadd.push(Instruction);
//                        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction);
                    p += 1;
                }
            }
            else
            {
                node_index = current_node_index;
                replication_index = current_replication_index;
                p = 1;
            }
        }
    }
}


void BatchPipelineSchedule::ScheduleStage3(int instruction_group_index, bool append_instruction)
{
    //// 结果发送与融合
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (PIMCOMP_4_core_instruction_group_num[i] < instruction_group_index)
            continue;
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        int AG_num = current_core.AG_list.size();
        if (AG_num == 0)
            continue;
        int j = 0;
        while(j < AG_num)
//        for (int j = 0; j < AG_num; ++j)
        {
            int node_index = current_core.node_list[j];
            int replication_index = current_core.AG_list[j].replication_index;
            int AG_index_in_replication = current_core.AG_list[j].AG_index_in_replication;
            int AG_index_in_total = current_core.AG_list[j].AG_index_in_total;
            int level_index = PIMCOMP_node_list[node_index].level_index;

//            if (instruction_group_index == 0 && i == 130)
//                std::cout << node_index << " " << AG_index_in_total << std::endl;
            if (AG_index_in_replication != 0  && comm_flag[AG_index_in_total] == 1)
            {
                //// 这样要保证同一Rep的AG连续且递增，不够灵活
//                bool SendRecv = false;
//                int estimate_first_AG_index =  j - AG_index_in_replication;
//                if (estimate_first_AG_index < 0 ||
//                    current_core.AG_list[estimate_first_AG_index].AG_index_in_replication != 0 ||
//                    current_core.AG_list[estimate_first_AG_index].replication_index != replication_index ||
//                    current_core.node_list[estimate_first_AG_index] != node_index )
//                {
//                    SendRecv = true;
//                }
//                if(SendRecv)

                //// 只要求递增即可，可以不连续
                bool SendRecv = true;
                for (int k = 1; k <= AG_index_in_replication; ++k)
                {
                    int estimate_first_AG_index = j - k;
                    if (estimate_first_AG_index >= 0 &&
                        current_core.AG_list[estimate_first_AG_index].AG_index_in_replication == 0 &&
                        current_core.AG_list[estimate_first_AG_index].replication_index == replication_index &&
                        current_core.node_list[estimate_first_AG_index] == node_index )
                    {
                        SendRecv = false;
                        break;
                    }
                }
                if (SendRecv)
                {
                    int RecvCore = PIMCOMP_4_first_AG_info.node_list[node_index].replication_list[replication_index];
                    // 添加发送接收指令和结果融合指令。
                    struct INST Instruction_send;
                    Instruction_send.type = COMM;
                    Instruction_send.level_index = level_index;
                    Instruction_send.operation = "SEND";
                    Instruction_send.stage = "MAIN";
                    Instruction_send.to_core = RecvCore;
                    Instruction_send.from_core = i;
                    Instruction_send.source = AG_index_in_total;
                    Instruction_send.relative_length = node_offset_instruction_group[AG_index_in_total];
                    Instruction_send.element_num = Instruction_send.relative_length * AG_output_element_size[AG_index_in_total];
                    Instruction_send.instruction_group_index = instruction_group_index;
                    Instruction_send.AGP = current_core.AG_list[j].AGP;
                    Instruction_send.agp_index = current_core.AG_list[j].agp_index;
                    Instruction_send.comm_index = comm_index;
                    Instruction_send.instruction_index_in_core = PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[i].instruction_ir_list_body.size();
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[i].instruction_ir_list_body.push_back(Instruction_send);
                    PIMCOMP_6_inter_core_communication[i][RecvCore] += Instruction_send.element_num;

                    struct INST Instruction_recv;
                    Instruction_recv.type = COMM;
                    Instruction_recv.level_index = level_index;
                    Instruction_recv.operation = "RECV";
                    Instruction_recv.stage = "MAIN";
                    Instruction_recv.from_core = i;
                    Instruction_recv.to_core = RecvCore;
                    // 注意，另一个核的AGx对应的AG_index_in_total一定没在该Core中出现过。所以可以作为地址的一种表示。
                    Instruction_recv.destination = AG_index_in_total;
                    Instruction_recv.relative_length = node_offset_instruction_group[AG_index_in_total];
                    Instruction_recv.element_num = Instruction_recv.relative_length * AG_output_element_size[AG_index_in_total];
                    Instruction_recv.instruction_group_index = instruction_group_index;
                    Instruction_recv.AGP = current_core.AG_list[j].AGP;
                    Instruction_recv.agp_index = current_core.AG_list[j].agp_index;
                    Instruction_recv.comm_index = comm_index;
                    Instruction_recv.instruction_index_in_core = PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[RecvCore].instruction_ir_list_body.size();
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[RecvCore].instruction_ir_list_body.push_back(Instruction_recv);

                    struct INST Instruction_vvadd;
                    Instruction_vvadd.type = VEC2OP;
                    Instruction_vvadd.level_index = level_index;
                    Instruction_vvadd.node_index = node_index;
                    Instruction_vvadd.operation = "VVADD";
                    Instruction_vvadd.stage = "MAIN-C";
                    struct core_schedule recv_core = PIMCOMP_4_virtual_core_AG_map.core_list[RecvCore];
                    int tmp_AG_num = recv_core.AG_list.size();
                    // tmp_ag_total_index是RecvCore中同node同rep的AG0对应的位置
                    int tmp_ag_total_index = 0;
                    for (int k = 0; k < tmp_AG_num; ++k)
                    {
                        if( recv_core.node_list[k] == node_index &&
                            recv_core.AG_list[k].AG_index_in_replication == 0 &&
                            recv_core.AG_list[k].replication_index == replication_index )
                        {
                            tmp_ag_total_index = recv_core.AG_list[k].AG_index_in_total;
                        }
                    }
                    Instruction_vvadd.source_1 = tmp_ag_total_index;
                    Instruction_vvadd.source_2 = AG_index_in_total;
                    Instruction_vvadd.destination = tmp_ag_total_index;
                    Instruction_vvadd.relative_length = node_offset_instruction_group[AG_index_in_total];
                    Instruction_vvadd.element_num = Instruction_vvadd.relative_length * AG_output_element_size[AG_index_in_total];
                    Instruction_vvadd.AGP = current_core.AG_list[j].AGP;
                    Instruction_vvadd.agp_index = current_core.AG_list[j].agp_index;
                    Instruction_vvadd.destination_offset = 0;
                    Instruction_vvadd.source_1_offset = 0;
                    Instruction_vvadd.source_2_offset = 0;
                    Instruction_vvadd.relative_length = node_offset_instruction_group[node_index];
                    Instruction_vvadd.instruction_group_index = instruction_group_index;
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list[RecvCore].instruction_ir_list_body.push_back(Instruction_vvadd);

                    comm_index++;
                    // 因为之前经过了信息融合，所以不需要多次发送接收。跳过后面同一Rep的其他AG。
                    j = j + 1;
                    while (j < AG_num && current_core.AG_list[j].node_index == node_index && current_core.AG_list[j].replication_index == replication_index)
                        j += 1;
                }
                else
                    j = j + 1;
            }
            else
                j = j + 1;
        }
    }
}



void BatchPipelineSchedule::ScheduleStageAct(int instruction_group_index, bool append_instruction)
{
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (PIMCOMP_4_core_instruction_group_num[i] < instruction_group_index)
            continue;
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        int AG_num = current_core.AG_list.size();
        if (AG_num == 0)
            continue;
        for (int j = 0; j < AG_num; ++j)
        {
            int AG_index_in_total = current_core.AG_list[j].AG_index_in_total;
            int AG_index_in_replication = current_core.AG_list[j].AG_index_in_replication;
            int AG_num_per_replication = current_core.AG_list[j].AG_num_per_replication;
            int node_index = current_core.node_list[j];
            int level_index = PIMCOMP_node_list[node_index].level_index;
            int output_element_num = AG_output_element_size[AG_index_in_total];

            if (activate_flag[AG_index_in_total] == 1)
            {
                if (PIMCOMP_node_list[node_index].with_act)
                {
                    struct INST Instruction_act;
                    Instruction_act.type = VEC1OP;
                    Instruction_act.level_index = level_index;
                    int act_type = PIMCOMP_node_list[node_index].act_type;
                    Instruction_act.operation = act_type == 0 ? "VRELU" : (act_type == 1? "VTANH" : "VSIGMOID");
                    Instruction_act.stage = "MAIN";
                    Instruction_act.relative_length = node_offset_instruction_group[AG_index_in_total];
                    Instruction_act.source = AG_index_in_total;
                    Instruction_act.destination = AG_index_in_total;
                    Instruction_act.element_num = Instruction_act.relative_length * AG_output_element_size[Instruction_act.source];
                    Instruction_act.instruction_group_index = instruction_group_index;
                    Instruction_act.destination_offset = 0;
                    Instruction_act.source_offset = 0;
                    if(append_instruction)
                        PIMCOMP_4_base_instruction_ir_vrelu[instruction_group_index].core_list[i].instruction_ir_list_vrelu.push(Instruction_act);
                }
            }
        }
    }
}



int BatchPipelineSchedule::GetEffectiveInstructionGroupNum()
{
    int effective_instruction_group_num = 0;
    int AG_num_in_total = PIMCOMP_3_hierarchy_map.whole.size();
    for (int i = 0; i < AG_num_in_total; ++i)
    {
        // 得到整个结构最终instruction_group_num
        int AG_index = PIMCOMP_3_hierarchy_map.whole_index[i];
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        int AG_instruction_group_num = ceil(float(PIMCOMP_node_list[node_index].input_cycle_in_total) / float(PIMCOMP_node_list[node_index].replication_num));
        int core_index = PIMCOMP_3_hierarchy_map.whole[i][0].vcore_index;
        if (AG_instruction_group_num > effective_instruction_group_num)
            effective_instruction_group_num = AG_instruction_group_num;
        // 得到全部出现过的instruction_group_num
        PIMCOMP_4_unique_instruction_group_index.insert(AG_instruction_group_num);
        // 为每个CONV或FC节点生成instruction_group_num
        if (PIMCOMP_node_list[node_index].instruction_group_num == 0)
            PIMCOMP_node_list[node_index].instruction_group_num = AG_instruction_group_num;
        else if (PIMCOMP_node_list[node_index].instruction_group_num < AG_instruction_group_num)
            PIMCOMP_node_list[node_index].instruction_group_num = AG_instruction_group_num;
        if (PIMCOMP_4_core_instruction_group_num[core_index] == 0)
            PIMCOMP_4_core_instruction_group_num[core_index] = AG_instruction_group_num;
        else if (PIMCOMP_4_core_instruction_group_num[core_index] < AG_instruction_group_num)
            PIMCOMP_4_core_instruction_group_num[core_index] = AG_instruction_group_num;
    }
    return effective_instruction_group_num;
}

void BatchPipelineSchedule::ResetPostStartAndEndAddress(int origin_length, int assumed_core_num)
{
    post_start_address.clear();
    post_end_address.clear();
    std::vector<int> output_core_allocated;
    for (int i = 0; i < assumed_core_num; ++i)
        output_core_allocated.push_back(ceil(float(origin_length) / float(assumed_core_num)));
    int minus_num = ceil(float(origin_length) / float(assumed_core_num)) * assumed_core_num - origin_length;
    for (int i = 0; i < minus_num; ++i)
        output_core_allocated[assumed_core_num-1-i] -= 1;
    int start_address;
    int end_address = -1;
    for (int i = 0; i < assumed_core_num; ++i)
    {
        start_address = end_address + 1;
        end_address = start_address + output_core_allocated[i] - 1;
        post_start_address.push_back(start_address);
        post_end_address.push_back(end_address);
    }
}


void BatchPipelineSchedule::ScheduleScheduleOnePostOperation(int instruction_group_index, int post_node_index)
{
    int appointed_post_core_num = ChipH * ChipW;
    struct PIMCOMP_node PostOperationNode = PIMCOMP_node_list[post_node_index];
    int level_index = PostOperationNode.level_index;
    int copy_offset_flag = PostOperationNode.copy_offset_flag;
    int AG0_index_in_total = PostOperationNode.AG0_index_in_total;
    if (PostOperationNode.operation == "OP_POOL")
    {
        int split_num = 2; // split_num表示把原先分配到该核的任务分成几次完成（防止片上内存不够）
        int output_channel_length = PostOperationNode.output_dim[1];
        int output_channel_num_total = PostOperationNode.output_dim[2] * PostOperationNode.output_dim[3]; // == input_sliding_window_num
        int real_post_core_num = appointed_post_core_num  > std::ceil(float(output_channel_num_total) / float(split_num)) ? static_cast<int>(std::ceil(float(output_channel_num_total) / float(split_num))) : appointed_post_core_num;
        int real_segment_num = real_post_core_num * split_num > output_channel_num_total ? output_channel_num_total : real_post_core_num * split_num;
        ResetPostStartAndEndAddress(output_channel_num_total, real_segment_num);

        for (int i = 0; i < real_post_core_num; ++i)
        {
            for (int s = 0; s < split_num; ++s)
            {
                if (i * split_num + s >= real_segment_num)
                    break;
                int load_offset = 0;
                std::queue<int> load_offset_queue;
                int output_channel_start = post_start_address[i * split_num + s];
                int output_channel_end = post_end_address[i * split_num + s];

                // 这里是先遍历一遍，看看一共需要多少输入。然后依次基础上确定输出的位置
                int total_load_channel = 0;
                for (int j = output_channel_start; j <= output_channel_end; ++j)
                {
                    int associated_input_num = PIMCOMP_conv_pool_input_output_info[post_node_index].output_index[j].size();
                    total_load_channel += associated_input_num;
                }

                for (int j = output_channel_start; j <= output_channel_end; ++j)
                {
                    int associated_input_num = PIMCOMP_conv_pool_input_output_info[post_node_index].output_index[j].size();
                    for (int k = 0; k < associated_input_num; k++)
                    {
                        int input_channel_index = PIMCOMP_conv_pool_input_output_info[post_node_index].output_index[j][k];
                        struct INST Instruction_ld;
                        Instruction_ld.node_index = post_node_index;
                        Instruction_ld.type = MEM;
                        Instruction_ld.level_index = level_index;
                        Instruction_ld.level_diff = 0;
                        Instruction_ld.operation = "LD";
                        Instruction_ld.stage = "POST";
                        int provider_node_index = PIMCOMP_node_list[post_node_index].provider_index[0];
                        Instruction_ld.provider_node_index = provider_node_index;
                        Instruction_ld.source = -1 * provider_node_index;
                        Instruction_ld.source_address = -1 * provider_node_index;
                        Instruction_ld.source_offset = input_channel_index * output_channel_length; // POOL:input_channel_length == output_channel_length
                        Instruction_ld.destination = AG0_index_in_total;
                        Instruction_ld.destination_address = 0;
                        Instruction_ld.destination_offset = load_offset;
                        load_offset_queue.push(load_offset);
                        load_offset += output_channel_length;
                        Instruction_ld.element_num =  output_channel_length;
                        Instruction_ld.instruction_group_index = instruction_group_index;
                        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);
                    }
                    int rs_offset_for_avg_pool;
                    for (int k = 0; k < associated_input_num; ++k)
                    {
                        int output_channel_index = j;
                        int input_channel_index = PIMCOMP_conv_pool_input_output_info[post_node_index].output_index[j][k];
                        struct INST Instruction_pool;
                        Instruction_pool.stage = "POST";
                        Instruction_pool.input_cycle_index = j;
                        Instruction_pool.node_index = PostOperationNode.index;
                        Instruction_pool.level_index = level_index;
                        Instruction_pool.element_num = output_channel_length;
                        Instruction_pool.copy_offset_flag = PostOperationNode.copy_offset_flag;
                        if (k == 0) // 如果提前没有访问过，就先把输入向量搬运过去。
                        {
                            Instruction_pool.type = VEC1OP;
                            Instruction_pool.operation = "LMV";
                            Instruction_pool.source = AG0_index_in_total;
                            Instruction_pool.source_address = 0;
                            Instruction_pool.destination = AG0_index_in_total;
                            Instruction_pool.destination_address = 0;
                            int rs_offset = load_offset_queue.front();
                            load_offset_queue.pop();
                            Instruction_pool.source_offset =  rs_offset;
                            Instruction_pool.destination_offset = total_load_channel * output_channel_length + (j - output_channel_start) * output_channel_length;
                        }
                        else
                        {
                            Instruction_pool.type = VEC2OP;
                            if (PIMCOMP_node_list[PostOperationNode.index].param.pool_method == 0)
                                Instruction_pool.operation = "VVMAX";
                            else
                                Instruction_pool.operation = "VVADD";
                            Instruction_pool.source_1 = AG0_index_in_total;
                            Instruction_pool.source_1_address = 0;
                            Instruction_pool.source_2 = AG0_index_in_total;
                            Instruction_pool.source_2_address = 0;
                            Instruction_pool.destination = AG0_index_in_total;
                            Instruction_pool.destination_address = 0;
                            int rs_offset = load_offset_queue.front();
                            load_offset_queue.pop();
                            Instruction_pool.source_1_offset = rs_offset;
                            Instruction_pool.source_2_offset = total_load_channel * output_channel_length + (j - output_channel_start) * output_channel_length;
                            Instruction_pool.destination_offset = Instruction_pool.source_2_offset;
                            rs_offset_for_avg_pool = rs_offset;
                        }
                        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_pool);

                        if (k == associated_input_num - 1 && PIMCOMP_node_list[PostOperationNode.index].param.pool_method == 1)
                        {
                            struct INST Instruction_lldi;
                            Instruction_lldi.type = LLDI;
                            Instruction_lldi.level_index = level_index;
                            Instruction_lldi.input_cycle_index = j;
                            Instruction_lldi.operation = "LLDI";
                            Instruction_lldi.node_index = PostOperationNode.index;
                            Instruction_lldi.stage = "POST";
                            Instruction_lldi.destination_address = 0;
                            Instruction_lldi.destination_offset = rs_offset_for_avg_pool; // 紧邻前一个VVADD的rs_offset
                            Instruction_lldi.element_num = output_channel_length;
                            float kernel_h = PIMCOMP_node_list[PostOperationNode.index].param.kernel_h;
                            float kernel_w = PIMCOMP_node_list[PostOperationNode.index].param.kernel_w;
                            if (model_name == "inception_v3")
                                Instruction_lldi.imm_value = 1.0 / (kernel_h * kernel_w);
                            else
                                Instruction_lldi.imm_value = 1.0 / associated_input_num;
                            Instruction_lldi.instruction_group_index = instruction_group_index;
                            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_lldi);

                            struct INST Instruction_vvmul;
                            Instruction_vvmul.type = VEC2OP;
                            Instruction_vvmul.operation = "VVMUL";
                            Instruction_vvmul.stage = "POST";
                            Instruction_vvmul.level_index = level_index;
                            Instruction_vvmul.node_index = PostOperationNode.index;
                            Instruction_vvmul.element_num = output_channel_length;
                            Instruction_vvmul.instruction_group_index = instruction_group_index;
                            Instruction_vvmul.source_1_address = 0;
                            Instruction_vvmul.source_1_offset = rs_offset_for_avg_pool; // 紧邻前一个VVADD的rs_offset
                            Instruction_vvmul.source_2_address = 0;
                            Instruction_vvmul.source_2_offset = total_load_channel * output_channel_length + (j - output_channel_start) * output_channel_length;
                            Instruction_vvmul.destination_address = 0;
                            Instruction_vvmul.destination_offset = total_load_channel * output_channel_length + (j - output_channel_start) * output_channel_length;
                            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_vvmul);
                        }
                    }
                }

                struct INST Instruction_st;
                Instruction_st.node_index = post_node_index;
                Instruction_st.type = MEM;
                Instruction_st.level_index = level_index;
                Instruction_st.level_diff = 0;
                Instruction_st.operation = "ST";
                Instruction_st.stage = "POST";
                Instruction_st.source = AG0_index_in_total;
                Instruction_st.source_address = 0;
                Instruction_st.destination = -1 * post_node_index;
                Instruction_st.destination_address = -1 * post_node_index;
                Instruction_st.source_offset = load_offset;
                Instruction_st.destination_offset = output_channel_start * output_channel_length; // POOL:input_channel_length == output_channel_length
                Instruction_st.element_num = (output_channel_end - output_channel_start + 1) * output_channel_length;
                Instruction_st.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_st);
            }
        }
    }
    else if (PostOperationNode.operation ==  "OP_RELU" || PostOperationNode.operation == "OP_TANH" || PostOperationNode.operation == "OP_SIGMOID" )
    {
        int split_num = 1;
        int output_channel_length = PostOperationNode.output_dim[1];
        int output_channel_num_total = PostOperationNode.output_dim[2] * PostOperationNode.output_dim[3];
        int real_post_core_num = appointed_post_core_num  > std::ceil(float(output_channel_num_total) / float(split_num)) ? static_cast<int>(std::ceil(float(output_channel_num_total) / float(split_num))) : appointed_post_core_num;
        int real_segment_num = real_post_core_num * split_num > output_channel_num_total ? output_channel_num_total : real_post_core_num * split_num;
        ResetPostStartAndEndAddress(output_channel_num_total, real_segment_num);

        for (int i = 0; i < real_post_core_num; ++i)
        {
            for (int s = 0; s < split_num; ++s)
            {
                if (i * split_num + s >= real_segment_num)
                    break;
                int input_channel_start = post_start_address[i * split_num + s];
                int input_channel_end = post_end_address[i * split_num + s];

                int load_offset = 0;
                std::vector<int> load_address;
                struct INST Instruction_ld;
                Instruction_ld.node_index = post_node_index;
                Instruction_ld.type = MEM;
                Instruction_ld.level_index = level_index;
                Instruction_ld.level_diff = 0;
                Instruction_ld.operation = "LD";
                Instruction_ld.stage = "POST";
                int provider_node_index = PIMCOMP_node_list[post_node_index].provider_index[0];
                Instruction_ld.provider_node_index = provider_node_index;
                Instruction_ld.source = -1 * provider_node_index;
                Instruction_ld.source_address = -1 * provider_node_index;
                Instruction_ld.source_offset = input_channel_start * output_channel_length;
                Instruction_ld.destination = AG0_index_in_total;
                Instruction_ld.destination_address = 0;
                Instruction_ld.destination_offset = load_offset;
                Instruction_ld.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                load_address.push_back(load_offset);
                load_offset += Instruction_ld.element_num;
                Instruction_ld.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);

                for (int k = input_channel_start; k <= input_channel_end; ++k)
                {
                    struct INST Instruction_act;
                    std::string act_type;
                    act_type = PostOperationNode.operation == "OP_RELU" ? "VRELU" : (PostOperationNode.operation == "OP_TANH" ? "VTANH" : "VSIGM");
                    Instruction_act.type = VEC1OP;
                    Instruction_act.level_index = level_index;
                    Instruction_act.level_diff = 0;
                    Instruction_act.operation = act_type;
                    Instruction_act.stage = "POST";
                    Instruction_act.source = AG0_index_in_total;
                    Instruction_act.source_address = 0;
                    Instruction_act.destination = AG0_index_in_total;
                    Instruction_act.destination_address = 0;
                    Instruction_act.source_offset = (k-input_channel_start) * output_channel_length;
                    Instruction_act.destination_offset = load_offset + (k-input_channel_start) * output_channel_length;
                    Instruction_act.element_num = output_channel_length;
                    Instruction_act.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_act);
                }

                struct INST Instruction_st;
                Instruction_st.node_index = post_node_index;
                Instruction_st.type = MEM;
                Instruction_st.level_index = level_index;
                Instruction_st.level_diff = 0;
                Instruction_st.operation = "ST";
                Instruction_st.stage = "POST";
                Instruction_st.source = AG0_index_in_total;
                Instruction_st.source_address = 0;
                Instruction_st.source_offset = load_offset; // 最终该load_offset就是load全部元素量
                Instruction_st.destination = -1 * post_node_index;
                Instruction_st.destination_address = -1 * post_node_index;
                Instruction_st.destination_offset = input_channel_start * output_channel_length;
                Instruction_st.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                Instruction_st.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_st);
            }
        }
    }
    else if (PostOperationNode.operation ==  "OP_ELTWISE")
    {
        int split_num = 1;
        int output_channel_length = PostOperationNode.output_dim[1];
        int output_channel_num_total = PostOperationNode.output_dim[2] * PostOperationNode.output_dim[3];
        int real_post_core_num = appointed_post_core_num  > std::ceil(float(output_channel_num_total) / float(split_num)) ? static_cast<int>(std::ceil(float(output_channel_num_total) / float(split_num))) : appointed_post_core_num;
        int real_segment_num = real_post_core_num * split_num > output_channel_num_total ? output_channel_num_total : real_post_core_num * split_num;
        ResetPostStartAndEndAddress(output_channel_num_total, real_segment_num);
        int elt_type = PostOperationNode.param.eletype;
        std::string elt_operation;
        switch (elt_type)
        {   case 2: elt_operation = "VVADD"; break;
            case 4: elt_operation = "VSUB"; break; }

        for (int i = 0; i < real_post_core_num; ++i)
        {
            for (int s = 0; s < split_num; ++s)
            {
                if (i * split_num + s >= real_segment_num)
                    break;
                int input_channel_start = post_start_address[i * split_num + s];
                int input_channel_end = post_end_address[i * split_num + s];
                int load_offset = 0;
                std::vector<int> load_address;
                for (int j = 0; j < PostOperationNode.provider_num; ++j)
                {
                    int provider_index = PostOperationNode.provider_index[j];
                    int provider_channel_length = PIMCOMP_node_list[provider_index].output_dim[1];
                    struct INST Instruction_ld;
                    Instruction_ld.type = MEM;
                    Instruction_ld.node_index = post_node_index;
                    Instruction_ld.level_index = level_index;
                    Instruction_ld.level_diff = 0;
                    Instruction_ld.provider_node_index = provider_index;
                    Instruction_ld.operation = "LD";
                    Instruction_ld.stage = "POST";
                    Instruction_ld.source = -1 * provider_index;
                    Instruction_ld.source_address = -1 * provider_index;
                    Instruction_ld.source_offset = input_channel_start * provider_channel_length;
                    Instruction_ld.destination = AG0_index_in_total;
                    Instruction_ld.destination_address = 0;
                    Instruction_ld.destination_offset = load_offset;
                    Instruction_ld.element_num = (input_channel_end - input_channel_start + 1) * provider_channel_length;
                    load_address.push_back(load_offset);
                    load_offset += Instruction_ld.element_num;
                    Instruction_ld.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);
                }

                for (int j = 1; j < PostOperationNode.provider_num; ++j)
                {
                    int provider_index = PostOperationNode.provider_index[j];
                    int provider_channel_length = PIMCOMP_node_list[provider_index].output_dim[1];
                    for (int k = input_channel_start; k <= input_channel_end; ++k)
                    {
                        struct INST Instruction_elt;
                        Instruction_elt.type = VEC2OP;
                        Instruction_elt.level_index = level_index;
                        Instruction_elt.level_diff = 0;
                        Instruction_elt.operation = elt_operation;
                        Instruction_elt.stage = "POST";
                        Instruction_elt.source_1 = AG0_index_in_total;
                        Instruction_elt.source_1_address = 0;
                        Instruction_elt.source_2 = AG0_index_in_total;
                        Instruction_elt.source_2_address = 0;
                        Instruction_elt.destination = AG0_index_in_total;
                        Instruction_elt.destination_address = 0;
                        if (j == 1)
                            Instruction_elt.source_1_offset = load_address[0] + (k-input_channel_start) * output_channel_length;
                        else
                            Instruction_elt.source_1_offset = load_offset + (k-input_channel_start) * output_channel_length;
                        Instruction_elt.source_2_offset = load_address[j] + (k-input_channel_start) * output_channel_length;
                        Instruction_elt.destination_offset = load_offset + (k-input_channel_start) * output_channel_length;
                        Instruction_elt.element_num = provider_channel_length;
                        Instruction_elt.instruction_group_index = instruction_group_index;
                        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_elt);
                    }
                }

                struct INST Instruction_st;
                Instruction_st.node_index = post_node_index;
                Instruction_st.type = MEM;
                Instruction_st.level_index = level_index;
                Instruction_st.level_diff = 0;
                Instruction_st.operation = "ST";
                Instruction_st.stage = "POST";
                Instruction_st.source = AG0_index_in_total;
                Instruction_st.source_address = 0;
                Instruction_st.source_offset = load_offset; // 最终该load_offset就是load全部元素量
                Instruction_st.destination = -1 * post_node_index;
                Instruction_st.destination_address = -1 * post_node_index;
                Instruction_st.destination_offset = input_channel_start * output_channel_length;
                Instruction_st.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                Instruction_st.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_st);
            }
        }
    }
    else if (PostOperationNode.operation ==  "OP_CONCAT")
    {
        int split_num = 2;
        int output_channel_length = PostOperationNode.output_dim[1];
        int output_channel_num_total = PostOperationNode.output_dim[2] * PostOperationNode.output_dim[3];
        int real_post_core_num = appointed_post_core_num  > std::ceil(float(output_channel_num_total) / float(split_num)) ? static_cast<int>(std::ceil(float(output_channel_num_total) / float(split_num))) : appointed_post_core_num;
        int real_segment_num = real_post_core_num * split_num > output_channel_num_total ? output_channel_num_total : real_post_core_num * split_num;
        ResetPostStartAndEndAddress(output_channel_num_total, real_segment_num);

        int store_offset = 0;
        for (int i = 0; i < real_post_core_num; ++i)
        {
            // For CONCAT, input_channel_index == output_channel_index
            for (int s = 0; s < split_num; ++s)
            {
                if (i * split_num + s >= real_segment_num)
                    break;
                int input_channel_start = post_start_address[i * split_num + s];
                int input_channel_end = post_end_address[i * split_num + s];
                int load_offset = 0;
                for (int j = 0; j < PostOperationNode.provider_num; ++j)
                {
                    int provider_index = PostOperationNode.provider_index[j];
                    int provider_channel_length = PIMCOMP_node_list[provider_index].output_dim[1];
                    struct INST Instruction_ld;
                    Instruction_ld.type = MEM;
                    Instruction_ld.node_index = post_node_index;
                    Instruction_ld.level_index = level_index;
                    Instruction_ld.level_diff = 0;
                    Instruction_ld.provider_node_index = provider_index;
                    Instruction_ld.operation = "LD";
                    Instruction_ld.stage = "POST";
                    Instruction_ld.source = -1 * provider_index;
                    Instruction_ld.source_address = -1 * provider_index;
                    Instruction_ld.source_offset = input_channel_start * provider_channel_length;
                    Instruction_ld.destination = PIMCOMP_node_list[post_node_index].AG0_index_in_total;
                    Instruction_ld.destination_address = 0;
                    Instruction_ld.destination_offset = load_offset;
                    Instruction_ld.element_num = (input_channel_end - input_channel_start + 1) * provider_channel_length;
                    load_offset += Instruction_ld.element_num;
                    Instruction_ld.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);
                }

                int source_offset = 0;
                int destination_offset = 0;
                for (int j = 0; j < PostOperationNode.provider_num; ++j)
                {
                    int provider_index = PostOperationNode.provider_index[j];
                    int provider_channel_length = PIMCOMP_node_list[provider_index].output_dim[1];
                    for (int k = input_channel_start; k <= input_channel_end; ++k)
                    {
                        struct INST Instruction_vm;
                        Instruction_vm.type = VEC1OP;
                        Instruction_vm.level_index = level_index;
                        Instruction_vm.level_diff = 0;
                        Instruction_vm.operation = "LMV";
                        Instruction_vm.stage = "POST";
                        Instruction_vm.source = AG0_index_in_total;
                        Instruction_vm.source_address = 0;
                        Instruction_vm.destination = AG0_index_in_total;
                        Instruction_vm.destination_address = 0;
                        Instruction_vm.source_offset = source_offset;
                        Instruction_vm.destination_offset = (k-input_channel_start) * output_channel_length + destination_offset + (input_channel_end-input_channel_start+1)*output_channel_length;
                        Instruction_vm.element_num = provider_channel_length;
                        source_offset += Instruction_vm.element_num;
                        Instruction_vm.instruction_group_index = instruction_group_index;
                        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_vm);
                    }
                    destination_offset += provider_channel_length;
                }

                struct INST Instruction_st;
                Instruction_st.type = MEM;
                Instruction_st.node_index = post_node_index;
                Instruction_st.level_index = level_index;
                Instruction_st.level_diff = 0;
                Instruction_st.operation = "ST";
                Instruction_st.stage = "POST";
                Instruction_st.source = AG0_index_in_total;
                Instruction_st.source_address = 0;
                Instruction_st.source_offset = load_offset; // 最终该load_offset就是load全部元素量
                Instruction_st.destination = -1 * post_node_index;
                Instruction_st.destination_address = -1 * post_node_index;
                Instruction_st.destination_offset = store_offset;
                Instruction_st.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                store_offset += Instruction_st.element_num;
                Instruction_st.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_st);
            }
        }
    }
    else if (PostOperationNode.operation ==  "OP_SHUFFLE")
    {
        int split_num = 1;
        int output_channel_length = PostOperationNode.output_dim[1];
        int output_channel_num_total = PostOperationNode.output_dim[2] * PostOperationNode.output_dim[3];
        int real_post_core_num = appointed_post_core_num  > std::ceil(float(output_channel_num_total) / float(split_num)) ? static_cast<int>(std::ceil(float(output_channel_num_total) / float(split_num))) : appointed_post_core_num;
        int real_segment_num = real_post_core_num * split_num > output_channel_num_total ? output_channel_num_total : real_post_core_num * split_num;
        ResetPostStartAndEndAddress(output_channel_num_total, real_segment_num);

        for (int i = 0; i < real_post_core_num; ++i)
        {
            for (int s = 0; s < split_num; ++s)
            {
                if (i * split_num + s >= real_segment_num)
                    break;
                int input_channel_start = post_start_address[i * split_num + s];
                int input_channel_end = post_end_address[i * split_num + s];

                int load_offset = 0;
                struct INST Instruction_ld;
                Instruction_ld.node_index = post_node_index;
                Instruction_ld.type = MEM;
                Instruction_ld.level_index = level_index;
                Instruction_ld.level_diff = 0;
                Instruction_ld.operation = "LD";
                Instruction_ld.stage = "POST";
                int provider_node_index = PIMCOMP_node_list[post_node_index].provider_index[0];
                Instruction_ld.provider_node_index = provider_node_index;
                Instruction_ld.source = -1 * provider_node_index;
                Instruction_ld.source_address = -1 * provider_node_index;
                Instruction_ld.source_offset = input_channel_start * output_channel_length;
                Instruction_ld.destination = AG0_index_in_total;
                Instruction_ld.destination_address = 0;
                Instruction_ld.destination_offset = load_offset;
                Instruction_ld.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                load_offset += Instruction_ld.element_num;
                Instruction_ld.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);

                int split_factor = PIMCOMP_node_list[post_node_index].param.split_factor;
                int input_channel_element_num = PIMCOMP_node_list[post_node_index].param.input_channel;
                int split_channel_element_num = input_channel_element_num / split_factor;
                for (int k = input_channel_start; k <= input_channel_end; ++k)
                {
                    for (int m = 0; m < split_factor; ++m)
                    {
                        for (int n = 0; n < split_channel_element_num; ++n)
                        {
                            //// 添加指令
                            struct INST Instruction_lmv;
                            Instruction_lmv.level_index = level_index;
                            Instruction_lmv.level_diff = 0;
                            Instruction_lmv.type = VEC1OP;
                            Instruction_lmv.stage = "POST";
                            Instruction_lmv.operation = "LMV";
                            Instruction_lmv.output_channel_index = k;
                            Instruction_lmv.node_index = post_node_index;
                            Instruction_lmv.source_address = 0;
                            Instruction_lmv.source_offset = ((k-input_channel_start) * output_channel_length) + (m * split_channel_element_num + n);
                            Instruction_lmv.destination_address = 0;
                            Instruction_lmv.destination_offset = (load_offset + (k-input_channel_start) * output_channel_length) + (n * split_factor + m);
                            Instruction_lmv.element_num = 1;
                            Instruction_lmv.instruction_group_index = instruction_group_index;
                            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_lmv);
                        }
                    }
                }

                struct INST Instruction_st;
                Instruction_st.node_index = post_node_index;
                Instruction_st.type = MEM;
                Instruction_st.level_index = level_index;
                Instruction_st.level_diff = 0;
                Instruction_st.operation = "ST";
                Instruction_st.stage = "POST";
                Instruction_st.source = AG0_index_in_total;
                Instruction_st.source_address = 0;
                Instruction_st.source_offset = load_offset; // 最终该load_offset就是load全部元素量
                Instruction_st.destination = -1 * post_node_index;
                Instruction_st.destination_address = -1 * post_node_index;
                Instruction_st.destination_offset = input_channel_start * output_channel_length;
                Instruction_st.element_num = (input_channel_end - input_channel_start + 1) * output_channel_length;
                Instruction_st.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_st);
            }
        }
    }
}


void BatchPipelineSchedule::SchedulePickOnePostOperation(int start_instruction_group_index)
{
    std::set <int> complete_node;
    std::set <int> wait_node;
    std::set <int> ready_node;
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list[i].operation;
        if (operation == "OP_INPUT" || operation == "OP_CONV"|| operation == "OP_FC")
        {
            complete_node.insert(i);
        }
        else if(operation == "OP_RELU"  || operation == "OP_TANH" || operation == "OP_SIGMOID")
        {
            int provider_node_index = PIMCOMP_node_list[i].provider_index[0];
            std::string provider_operation = PIMCOMP_node_list[provider_node_index].operation;
            if (provider_operation ==  "OP_CONV" || provider_operation ==  "OP_FC")
            {
                complete_node.insert(i);
            }
            else
            {
                wait_node.insert(i);
            }
        }
        else
        {
            wait_node.insert(i);
        }
    }
    for (std::set<int>::iterator i = wait_node.begin(); i != wait_node.end(); i++)
    {
        int node_index = *i;
        bool ready = true;
        for (int j = 0; j < PIMCOMP_node_list[node_index].provider_num; ++j)
        {
            int provider_index = PIMCOMP_node_list[node_index].provider_index[j];
            if (complete_node.count(provider_index) == 0)
                ready = false;
        }
        if (ready)
        {
            ready_node.insert(node_index);
        }
    }

    int post_cycle = 0;
    while (wait_node.size() != 0)
    {
//        std::cout << "post_cycle " << post_cycle << std::endl;

//        std::cout << "  ready : ";
//        for (std::set<int>::iterator i = ready_node.begin(); i != ready_node.end(); i++)
//        {
//            std::cout << *i << " ";
//        }
//        std::cout << std::endl;

        int pick = *ready_node.begin();
//        std::cout << "  pick : " << pick << std::endl;
        std::string post_operation = PIMCOMP_node_list[pick].operation;
        if (post_operation == "OP_CONCAT" ||  post_operation == "OP_POOL" || post_operation == "OP_ELTWISE"
            || post_operation == "OP_RELU" || post_operation == "OP_TANH" || post_operation == "OP_SIGMOID"
            || post_operation == "OP_SHUFFLE")
        {
            PIMCOMP_4_base_instruction_ir[start_instruction_group_index + post_cycle].core_list.resize(ChipH * ChipW);
            ScheduleScheduleOnePostOperation(start_instruction_group_index + post_cycle, pick);
        }
        else
            std::cout << post_operation << std::endl;
        wait_node.erase(pick);
        complete_node.insert(pick);
        ready_node.clear();

        for (std::set<int>::iterator i = wait_node.begin(); i != wait_node.end(); i++)
        {
            int node_index = *i;
            bool ready = true;
            for (int j = 0; j < PIMCOMP_node_list[node_index].provider_num; ++j)
            {
                int provider_index = PIMCOMP_node_list[node_index].provider_index[j];
                if (complete_node.count(provider_index) == 0)
                    ready = false;
            }
            if (ready)
            {
                ready_node.insert(node_index);
            }
        }
        post_cycle++;
    }
    PIMCOMP_post_instruction_num = post_cycle;
}


void BatchPipelineSchedule::ScheduleMerge()
{
    for (int i = 0; i < PIMCOMP_base_instruction_num; ++i)
    {
        for (int j = 0; j < ChipH * ChipW; ++j)
        {
            for (auto list_index = PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.begin(); list_index != PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.end(); ++list_index)
            {
                struct INST tmpInstruction = *list_index;
                if (tmpInstruction.operation == "MVMUL")
                {
                    if (PIMCOMP_4_base_instruction_ir_vvadd[i].core_list[j].instruction_ir_list_vvadd.size())
                    {
                        int MVMUL_rd = tmpInstruction.destination;
                        if (MVMUL_rd == PIMCOMP_4_base_instruction_ir_vvadd[i].core_list[j].instruction_ir_list_vvadd.front().source_2)
                        {
                            struct INST tmpInstructionVVADD = PIMCOMP_4_base_instruction_ir_vvadd[i].core_list[j].instruction_ir_list_vvadd.front();
                            PIMCOMP_4_base_instruction_ir_vvadd[i].core_list[j].instruction_ir_list_vvadd.pop();
                            PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.insert(++list_index, tmpInstructionVVADD);
                            list_index--;
                        }
                    }
                }
            }

            for (auto list_index = PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.rbegin(); list_index != PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.rend(); ++list_index)
            {
                if (PIMCOMP_4_base_instruction_ir_vrelu[i].core_list[j].instruction_ir_list_vrelu.size())
                {
                    struct INST tmpInstruction = *list_index;
                    int destination = tmpInstruction.destination; // 这种写法可以和加上BIAS后的写法兼容，不用改。因为bias的vvadd的destination和第一个MVMUL的destination是一样的。
                    if (destination == PIMCOMP_4_base_instruction_ir_vrelu[i].core_list[j].instruction_ir_list_vrelu.top().destination)
                    {
                        struct INST tmpInstructionVRELU = PIMCOMP_4_base_instruction_ir_vrelu[i].core_list[j].instruction_ir_list_vrelu.top();
                        PIMCOMP_4_base_instruction_ir_vrelu[i].core_list[j].instruction_ir_list_vrelu.pop();
                        PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.insert(list_index.base(), tmpInstructionVRELU);
                        list_index--;
                    }
                }
            }

            for (auto list_index = PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.begin(); list_index != PIMCOMP_4_base_instruction_ir_body[i].core_list[j].instruction_ir_list_body.end(); ++list_index)
            {
                struct INST tmpInstruction = *list_index;
                if (tmpInstruction.operation == "RECV" || tmpInstruction.operation == "SEND")
                    tmpInstruction.instruction_index_in_core = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list.size();
                PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list.push_back(tmpInstruction);
            }
        }
    }
}


void BatchPipelineSchedule::ScheduleMain()
{
    PIMCOMP_5_memory_start_address.resize(ChipH * ChipW);
    int effective_instruction_group_num = GetEffectiveInstructionGroupNum();
    PIMCOMP_base_instruction_num = effective_instruction_group_num;
    std::cout << "effective_instruction_group_num:" << effective_instruction_group_num << std::endl;
    int instruction_group_num = user_given_instruction_group_num > effective_instruction_group_num ? effective_instruction_group_num : user_given_instruction_group_num;
    int max_post_operation_instruction_group_num = 500;
    PIMCOMP_4_base_instruction_ir.resize(instruction_group_num + max_post_operation_instruction_group_num);
    PIMCOMP_4_base_instruction_ir_body.resize(instruction_group_num);
    PIMCOMP_4_base_instruction_ir_vvadd.resize(instruction_group_num);
    PIMCOMP_4_base_instruction_ir_vrelu.resize(instruction_group_num);
    PIMCOMP_4_input_cycle_record.resize(node_num);
    bool append_instruction = 1;
    for (int j = 0; j < instruction_group_num; ++j)
    {
        int instruction_group_index;
        instruction_group_index = j;
        PIMCOMP_4_base_instruction_ir_body[instruction_group_index].core_list.resize(ChipW * ChipH);
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list.resize(ChipW * ChipH);
        PIMCOMP_4_base_instruction_ir_vvadd[instruction_group_index].core_list.resize(ChipW * ChipH);
        PIMCOMP_4_base_instruction_ir_vrelu[instruction_group_index].core_list.resize(ChipW * ChipH);
        if (j == 0)
            ScheduleStage0(j, append_instruction);
        for (int k = 0; k < operation_cycle_before_comm; k++)
        {
            ScheduleStage1(instruction_group_index, append_instruction);
            ScheduleStage2(instruction_group_index, append_instruction);
            for (int & n : add_flag) {n = 0;}
        }
        //// Stage3的作用是融合同一个复制块的计算结果，得到完整的结果
        ScheduleStage3(instruction_group_index, append_instruction);
        for (int & n : comm_flag) {n = 0;}
        //// StageACT的作用是为每个复制块的计算结果添加激活层
        ScheduleStageAct(instruction_group_index, append_instruction);
        for (int & n : activate_flag) {n = 0;}
        for (int & n : node_offset_instruction_group) {n = 0;}
    }

    //// 将MVMUL指令和VVADD指令合并到PIMCOMP_4_base_instruction_ir
    ScheduleMerge();
    //// 最终流水线设计的后处理操作
//    SchedulePickOnePostOperation(instruction_group_num);
    //// 将PIMCOMP_4_base_instruction_group截断
    PIMCOMP_4_base_instruction_ir.resize(PIMCOMP_base_instruction_num + PIMCOMP_post_instruction_num);
}



void BatchPipelineSchedule::Clear()
{
    for (int & n : AG_output_element_size) {n = 0;}
    for (int & n : node_offset_inference) {n = 0;}
    for (int & n : AG_accumulated_num) {n = 0;}
    comm_index = 0;
    PIMCOMP_4_base_instruction_ir_body.resize(0);
    PIMCOMP_4_base_instruction_ir_vvadd.resize(0);
}


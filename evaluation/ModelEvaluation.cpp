//
// Created by SXT on 2022/9/20.
//

#include "ModelEvaluation.h"

double tmp_practical;

ModelEvaluation::ModelEvaluation(int index, double given_interval)
{
    EVA_MVMUL_start_interval = given_interval;
    EvaluationIndex = index;
    instruction_group_num = PIMCOMP_8_base_instruction_ir_with_placement.size();
    std::cout << "instruction_group_num:" << instruction_group_num << std::endl;
    COMM_cycle_flag = new bool *[MAX_CHIP];
    MEM_cycle_flag = new bool *[MAX_CHIP];
    WB_cycle_flag = new bool *[MAX_CHIP];
    PIMCOMP_GUI_evaluation_of_core.resize(ChipH * ChipW);
    for (int i = 0; i < MAX_CHIP; ++i)
    {
        COMM_cycle_flag[i] = new bool[500000000];
        MEM_cycle_flag[i] = new bool[500000000];
        WB_cycle_flag[i] = new bool[500000000];
    }
}

ModelEvaluation::~ModelEvaluation()
{
    for (int i = 0; i < MAX_CHIP; ++i)
    {
        delete [] COMM_cycle_flag[i];
        delete [] MEM_cycle_flag[i];
    }
    delete [] COMM_cycle_flag;
    delete [] MEM_cycle_flag;
}


static std::vector<long long> MEM_volume_per_core;
static std::vector<long long> COMM_volume_per_core;

void ModelEvaluation::EvaluateCompute()
{
    PIMCOMP_GUI_execution_time_of_core.resize(ChipH * ChipW);
    PIMCOMP_GUI_execution_time_of_node.resize(PIMCOMP_node_list.size());
    PIMCOMP_GUI_inter_core_communication_volume.resize(ChipH * ChipW);
    for (int i = 0; i < ChipH * ChipW; ++i)
        PIMCOMP_GUI_inter_core_communication_volume[i].resize(ChipH * ChipW);

    MEM_volume_per_core.resize(ChipH * ChipW);
    COMM_volume_per_core.resize(ChipH * ChipW);

    for (int i = 0; i < instruction_group_num; ++i)
    {
        EvaluateRecursionSingleInstructionGroup(i, 0, 0);
        if (i == instruction_group_num-1)
        {
            ShowEvaluationResultSingleInstructionGroup();
        }
        ResetSingleInstructionGroup(0);
    }
//    if(element_pipeline)
//        ShowVisualResultOfCore();
    Clear();
}



void ModelEvaluation::ShowVisualResultOfCore()
{
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (PIMCOMP_GUI_execution_time_of_core[i].size() != 0)
        {
            std::cout << "core_" << i << ":";
            int print_len = 100;
            int begin = PIMCOMP_GUI_execution_time_of_core[i].front().second;
            int end = PIMCOMP_GUI_execution_time_of_core[i].back().second;
            int begin_print = floor(float(begin) / tmp_practical * print_len);
            int end_print = ceil(float(end) / tmp_practical * print_len);
            for (int j = 0; j < print_len; ++j)
            {
                if (j < begin_print)
                    std::cout << " ";
                else if (j < end_print)
                    std::cout << "#";
                else
                    std::cout << " ";
            }
            std::cout << std::endl;
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Memory Evaluation ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static int base_visited_record[MAX_AG] = {0};
//// Core Memory
static float base_core_memory_usage[MAX_CORE] = {0};
static float base_core_memory_usage_recv[MAX_CORE] = {0};
static float base_core_memory_usage_input[MAX_CORE] = {0};
static float base_core_memory_usage_output[MAX_CORE] = {0};
//// Node Memory
static float base_node_memory_usage[MAX_NODE] = {0};
/// AG memory
static float base_AG_memory_usage[MAX_AG] = {0};
static float base_AG_memory_usage_input[MAX_AG] = {0};
static float base_AG_memory_usage_output[MAX_AG] = {0};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Compute Evaluation ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static int MVMUL_single[MAX_CORE] = {0};
static int MVMUL_AG[MAX_AG] = {0};
static int VVADD_single[MAX_CORE] = {0};
static int COMM_single[MAX_CORE] = {0};
static int MEM_single[MAX_CORE] = {0};
static double LATENCY_single[MAX_CORE] = {0};
static double LATENCY_single_old[MAX_CORE] = {0};
static int visited_single[MAX_CORE] = {0};

static long long AG_last_exec_start_time[MAX_AG] = {0};
static long long AG_last_complete_time[MAX_AG] = {0};
static long long last_MVMUL_exec_start_time[MAX_CORE] = {0};
static long long last_synchronous_time[MAX_CORE] = {0};
// #MVMUL for each core of every instruction group
//  每个instruction_group中，每个核的MVMUL数量
static int MVMUL_instruction_group_core[50000][MAX_CORE] = {0};
// Preparation time for MVMUL. If this time exceeds given_interval, then MVMUL interval is preparation_time_single[core_index]
// 每个MVMUL用于准备的时间。如果这个时间超过了given_interval，那么两个MVMUL的间隔就是这个时间
static long long preparation_time_single[MAX_CORE] = {0};
static long long preparation_timeline[MAX_CORE] = {0};

static long long COMM_volume = 0;
static long long MEM_volume = 0;
static long long HT_volume = 0;

static long long COMM_cycle_total = 0;
static long long MEM_cycle_total = 0;
static long long HT_cycle_total = 0;

static double Dyn_Pow_Local_Memory_Volume = 0.0;
static double Dyn_Pow_Vector_Volume = 0.0;
static double Dyn_Pow_MVMUL_crossbar_Num = 0.0;

static double MVMUL_latency_core[MAX_CORE] = {0};
static double VEC_latency_core[MAX_CORE] = {0};
static double COMM_syn_latency_core[MAX_CORE] = {0};
static double COMM_trans_latency_core[MAX_CORE] = {0};
static double MEM_latency_core[MAX_CORE] = {0};


void ModelEvaluation::EvaluateRecursionSingleInstructionGroup(int instruction_group_index, int core_index, int index_in_core)
{
    if (core_index >= ChipW * ChipH)
        return;
    if (PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list.size() == 0)
        return;
    visited_single[core_index] = 1;
    int instruction_ir_num = PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list.size();
    for (int k = index_in_core; k < instruction_ir_num; ++k)
    {
        struct INST tmpInstruction = PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k];
        if (tmpInstruction.operation == "SEND" || tmpInstruction.operation == "RECV")
        {
            int comm_index = tmpInstruction.comm_index;
            int instruction_index_in_core = tmpInstruction.instruction_index_in_core;
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = LATENCY_single[core_index];
            if (comm_index_2_index_in_core.count(comm_index) == 0)
            {
                comm_index_2_index_in_core.insert(std::pair<int,int>(comm_index, instruction_index_in_core));
                comm_index_2_core_index.insert(std::pair<int,int>(comm_index, core_index));
                int next_core_index = core_index+1;
                while (visited_single[next_core_index] != 0)
                {
                    next_core_index++;
                }
                EvaluateRecursionSingleInstructionGroup(instruction_group_index, next_core_index, 0);
            }
            else
            {
                int corresponding_core_index = comm_index_2_core_index[comm_index];
                int corresponding_instruction_index_in_core = comm_index_2_index_in_core[comm_index];

                //// Synchronization Mechanism
                //// 完全同步机制
                if (LATENCY_single[core_index] > LATENCY_single[corresponding_core_index])
                {
                    COMM_syn_latency_core[corresponding_core_index] += LATENCY_single[core_index] - LATENCY_single[corresponding_core_index];
                    LATENCY_single[corresponding_core_index] = LATENCY_single[core_index];
                }
                else
                {
                    COMM_syn_latency_core[core_index] += LATENCY_single[corresponding_core_index] - LATENCY_single[core_index];
                    LATENCY_single[core_index] = LATENCY_single[corresponding_core_index];
                }
                last_synchronous_time[core_index] = LATENCY_single[core_index];
                last_synchronous_time[corresponding_core_index] = LATENCY_single[corresponding_core_index];

                int element_num = tmpInstruction.element_num;
                int communication_bytes_num = element_num * ArithmeticPrecision / 8;

                // Bus Cost
                double effective_BUS_bandwidth = BUS_bandwidth;
                int communication_needed_cycle = std::ceil((double(communication_bytes_num)) / (effective_BUS_bandwidth / Frequency));
                COMM_cycle_total += communication_needed_cycle;
                int chip_index = core_index / (OneChipWidth * OneChipHeight);
                int real_comm_latency = CheckBusBandwidth(chip_index, LATENCY_single[core_index], communication_needed_cycle);

                // NoC Cost
//                int hamming_distance = std::abs(corresponding_core_index % (OneChipWidth * OneChipHeight) % OneChipWidth - core_index % (OneChipWidth * OneChipHeight) % OneChipWidth)  + std::abs(corresponding_core_index % (OneChipWidth * OneChipHeight) / OneChipWidth - core_index % (OneChipWidth * OneChipHeight) / OneChipWidth) ;
//                double hop_hamming_distance = std::abs(corresponding_core_index % (OneChipWidth * OneChipHeight) % OneChipWidth / RouterWidth - core_index % (OneChipWidth * OneChipHeight) % OneChipWidth / RouterWidth )  + std::abs(corresponding_core_index % (OneChipWidth * OneChipHeight) / OneChipWidth / RouterHeight - core_index % (OneChipWidth * OneChipHeight) / OneChipWidth / RouterHeight );
//                double real_comm_latency = hop_hamming_distance * std::ceil((double(communication_bytes_num)) / (LINK_bandwidth / Frequency));

//                // Consider multiple chips which communicate through Hyper-Transport
//                // 考虑多个芯片，通过Hyper-Transport进行通信
                if ( (core_index / (OneChipHeight * OneChipWidth)) != (corresponding_core_index / (OneChipHeight * OneChipWidth)) )
                {
                    HT_volume += communication_bytes_num;
                    HT_cycle_total += communication_bytes_num  / (Hyper_Transport_bandwidth / Frequency);
                    real_comm_latency += communication_bytes_num / (Hyper_Transport_bandwidth / Frequency);
                }

                COMM_cycle_total += real_comm_latency;
                LATENCY_single[corresponding_core_index] += real_comm_latency;
                LATENCY_single[core_index] += real_comm_latency;
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[instruction_index_in_core].latency_end = LATENCY_single[core_index];
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[corresponding_core_index].instruction_ir_list[corresponding_instruction_index_in_core].latency_end = LATENCY_single[corresponding_core_index];

                COMM_trans_latency_core[core_index] += real_comm_latency;
                COMM_trans_latency_core[corresponding_core_index] += real_comm_latency;

                //// Half-Synchronization Mechanism
                //// 非完全同步机制
//                int element_num = tmpInstruction.element_num;
//                int communication_bytes_num = element_num * ArithmeticPrecision / 8;
//                double effective_BUS_bandwidth = BUS_bandwidth;
//                int communication_needed_cycle = std::ceil((double(communication_bytes_num)) / (effective_BUS_bandwidth / Frequency));
//                int chip_index = core_index / (OneChipWidth * OneChipHeight);
//                int real_comm_latency = 0;
//                if (tmpInstruction.operation == "RECV")     // Corresponding Core Send to Current Core
//                {
//                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[corresponding_core_index], communication_needed_cycle);
//                    LATENCY_single[corresponding_core_index] += real_comm_latency;  // Send Finish Time
//                    COMM_trans_latency_core[corresponding_core_index] += real_comm_latency;
//                    if (LATENCY_single[core_index] < LATENCY_single[corresponding_core_index])
//                    {
//                        if (LATENCY_single[corresponding_core_index] - real_comm_latency < LATENCY_single[core_index])
//                        {
//                            COMM_trans_latency_core[core_index] += LATENCY_single[corresponding_core_index] - LATENCY_single[core_index];
//                        }
//                        else
//                        {
//                            COMM_syn_latency_core[core_index] += LATENCY_single[corresponding_core_index] - LATENCY_single[core_index] - real_comm_latency  ;
//                            COMM_trans_latency_core[core_index] += real_comm_latency;
//                        }
//                        LATENCY_single[core_index] = LATENCY_single[corresponding_core_index];
//                        last_synchronous_time[core_index] = LATENCY_single[core_index];
//                    }
//                }
//                else    // Current Core Send to Corresponding Core
//                {
//                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[core_index], communication_needed_cycle);
//                    LATENCY_single[core_index] += real_comm_latency;  // Send Finish Time
//                    COMM_trans_latency_core[core_index] += real_comm_latency;
//                    if (LATENCY_single[corresponding_core_index] < LATENCY_single[core_index])
//                    {
//                        if (LATENCY_single[core_index] - real_comm_latency < LATENCY_single[corresponding_core_index])
//                        {
//                            COMM_trans_latency_core[corresponding_core_index] += LATENCY_single[core_index] - LATENCY_single[corresponding_core_index];
//                        }
//                        else
//                        {
//                            COMM_syn_latency_core[corresponding_core_index] += LATENCY_single[core_index] - LATENCY_single[corresponding_core_index] - real_comm_latency;
//                            COMM_trans_latency_core[corresponding_core_index] += real_comm_latency;
//                        }
//                        LATENCY_single[corresponding_core_index] = LATENCY_single[core_index];
//                        last_synchronous_time[corresponding_core_index] = LATENCY_single[corresponding_core_index];
//                    }
//                }
//                COMM_cycle_total += real_comm_latency;
//                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[instruction_index_in_core].latency_end = LATENCY_single[core_index];
//                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[corresponding_core_index].instruction_ir_list[corresponding_instruction_index_in_core].latency_end = LATENCY_single[corresponding_core_index];


                // Syn and Half-Syn share the below codes:
                if (tmpInstruction.operation == "SEND")
                    PIMCOMP_GUI_inter_core_communication_volume[core_index][corresponding_core_index] += communication_bytes_num;
                else
                    PIMCOMP_GUI_inter_core_communication_volume[corresponding_core_index][core_index] += communication_bytes_num;
                COMM_volume += communication_bytes_num;
                COMM_single[corresponding_core_index]++;
                COMM_single[core_index]++;
                COMM_volume_per_core[core_index] += communication_bytes_num;
                COMM_volume_per_core[corresponding_core_index] += communication_bytes_num;
                // Make sure the program is estimated from the beginning in every instruction group.
                // 这是为了确保每个instruction_group都从头开始。
                if (instruction_index_in_core == instruction_ir_num-1)
                {
                    preparation_timeline[core_index] = LATENCY_single[core_index];
                    preparation_time_single[core_index] = 0;
                }
                if (corresponding_instruction_index_in_core == PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[corresponding_core_index].instruction_ir_list.size()-1)
                {
                    preparation_timeline[corresponding_core_index] = LATENCY_single[corresponding_core_index];
                    preparation_time_single[corresponding_core_index] = 0;
                }
                EvaluateRecursionSingleInstructionGroup(instruction_group_index, corresponding_core_index, corresponding_instruction_index_in_core+1);
                EvaluateRecursionSingleInstructionGroup(instruction_group_index, core_index, instruction_index_in_core+1);
            }
            return;
        }
        else if (tmpInstruction.operation == "MVMUL")
        {
            int AG_index = tmpInstruction.destination;
            int node_index = tmpInstruction.node_index;
            if (MVMUL_instruction_group_core[instruction_group_index][core_index] != 0)
            {
                long long tmp = last_MVMUL_exec_start_time[core_index];
//                int real_interval = EVA_MVMUL_start_interval > preparation_time_single[core_index] ? EVA_MVMUL_start_interval : preparation_time_single[core_index];
                int real_interval = EVA_MVMUL_start_interval > (preparation_timeline[core_index] - last_MVMUL_exec_start_time[core_index]) ? EVA_MVMUL_start_interval : (preparation_timeline[core_index] - last_MVMUL_exec_start_time[core_index]);
                if (AG_last_exec_start_time[AG_index] != 0 && AG_last_exec_start_time[AG_index] + MVMUL_latency > last_MVMUL_exec_start_time[core_index] + real_interval)
                    AG_last_exec_start_time[AG_index] = AG_last_exec_start_time[AG_index] + MVMUL_latency;
                else
                    AG_last_exec_start_time[AG_index] = last_MVMUL_exec_start_time[core_index] + real_interval;
                last_MVMUL_exec_start_time[core_index] = AG_last_exec_start_time[AG_index];
                if (last_MVMUL_exec_start_time[core_index] < last_synchronous_time[core_index])
                {
                    AG_last_exec_start_time[AG_index] = last_synchronous_time[core_index] + preparation_time_single[core_index];
                    last_MVMUL_exec_start_time[core_index] = last_synchronous_time[core_index] + preparation_time_single[core_index];
                }
                MVMUL_latency_core[core_index] += (last_MVMUL_exec_start_time[core_index] - tmp) > MVMUL_latency ? MVMUL_latency : (last_MVMUL_exec_start_time[core_index] - tmp) ;
                LATENCY_single[core_index] = AG_last_exec_start_time[AG_index] + MVMUL_latency;
            }
            else
            {
//                AG_last_exec_start_time[AG_index] = LATENCY_single[core_index];
                AG_last_exec_start_time[AG_index] = preparation_timeline[core_index];
                last_MVMUL_exec_start_time[core_index] = AG_last_exec_start_time[AG_index];
                LATENCY_single[core_index] += MVMUL_latency;
                MVMUL_latency_core[core_index] += MVMUL_latency;
            }
            preparation_time_single[core_index] = 0;
            PIMCOMP_GUI_execution_time_of_core[core_index].push_back(std::make_pair(node_index,last_MVMUL_exec_start_time[core_index]));
            PIMCOMP_GUI_execution_time_of_node[node_index].push_back(last_MVMUL_exec_start_time[core_index]);
            MVMUL_instruction_group_core[instruction_group_index][core_index]++;
            MVMUL_single[core_index]++;
            MVMUL_AG[AG_index]++;
            Dyn_Pow_Local_Memory_Volume += tmpInstruction.input_element_num;
            Dyn_Pow_Local_Memory_Volume += tmpInstruction.output_element_num;
            Dyn_Pow_MVMUL_crossbar_Num += PIMCOMP_node_crossbar_num_per_AG[node_index];
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = AG_last_exec_start_time[AG_index];
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = LATENCY_single[core_index];
        }
        else if (tmpInstruction.operation == "VVADD")
        {
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = LATENCY_single[core_index];
            VVADD_single[core_index]++;
            int element_num = tmpInstruction.element_num;
            int real_vector_latency = ceil(double(element_num) * double(ArithmeticPrecision) / double(VECTOR_process_bytes * 8)) * VECTOR_unit_latency;
            if (tmpInstruction.stage == "MAIN-C")
            {
                last_synchronous_time[core_index] += real_vector_latency;
                LATENCY_single[core_index] += real_vector_latency;
            }
            else if (tmpInstruction.stage == "POST" || tmpInstruction.stage == "MAIN-A" || tmpInstruction.stage == "MAIN-B")
            {
                LATENCY_single[core_index] += real_vector_latency;
            }
            if (tmpInstruction.stage == "POST" || tmpInstruction.stage == "MAIN-C")
                VEC_latency_core[core_index] += real_vector_latency;
            Dyn_Pow_Local_Memory_Volume += 2 * tmpInstruction.element_num;
            Dyn_Pow_Vector_Volume += tmpInstruction.element_num;
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = LATENCY_single[core_index];
        }
        else if (tmpInstruction.operation == "LD" || tmpInstruction.operation == "ST")
        {
            if (tmpInstruction.operation == "LD")
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = preparation_timeline[core_index];
            else
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = LATENCY_single[core_index];

            int element_num = tmpInstruction.element_num;
            int memory_bytes_num = element_num * ArithmeticPrecision / 8;
            int memory_needed_cycle = std::ceil((double(memory_bytes_num)) / (GLOBAL_MEMORY_bandwidth / Frequency));
            int chip_index = core_index / (OneChipWidth * OneChipHeight);

            int real_mem_latency = 0;
            if (tmpInstruction.operation == "LD") // BIAS and INPUT
            {
                real_mem_latency = CheckGlobalMemoryBandwidth(chip_index, preparation_timeline[core_index], memory_needed_cycle);
//                real_mem_latency = CheckGlobalMemoryBandwidth(core_index, preparation_timeline[core_index], memory_needed_cycle);
                preparation_timeline[core_index] = preparation_timeline[core_index] + real_mem_latency;
                if (preparation_timeline[core_index] > LATENCY_single[core_index])
                {
                    MEM_latency_core[core_index] += preparation_timeline[core_index] - LATENCY_single[core_index];
                    LATENCY_single[core_index] = preparation_timeline[core_index];
                }
                preparation_time_single[core_index] += real_mem_latency;
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = preparation_timeline[core_index];
            }
            else
            {
                real_mem_latency = CheckGlobalMemoryBandwidth(chip_index, LATENCY_single[core_index], memory_needed_cycle);
//                real_mem_latency = CheckGlobalMemoryBandwidth(core_index, LATENCY_single[core_index], memory_needed_cycle);
                MEM_latency_core[core_index] += real_mem_latency;
                LATENCY_single[core_index] += real_mem_latency;
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = LATENCY_single[core_index];
            }

            MEM_volume_per_core[core_index] += memory_bytes_num;
            MEM_cycle_total += real_mem_latency;
            MEM_single[core_index] ++;
            if (tmpInstruction.operation == "LD")
                MEM_volume += memory_bytes_num;
        }
        else if (tmpInstruction.operation == "VRELU" || tmpInstruction.operation == "VTANH" || tmpInstruction.operation == "VSIGMOID" || tmpInstruction.operation == "VVMAX")
        {
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = LATENCY_single[core_index];
            int element_num = tmpInstruction.element_num;
            int real_vector_latency = ceil(double(element_num) * double(ArithmeticPrecision) / double(VECTOR_process_bytes * 8)) * VECTOR_unit_latency;
            LATENCY_single[core_index] += real_vector_latency;
            Dyn_Pow_Local_Memory_Volume += 2 * tmpInstruction.element_num;
            Dyn_Pow_Vector_Volume += tmpInstruction.element_num;
            if (tmpInstruction.stage == "POST")
                VEC_latency_core[core_index] += real_vector_latency;
            PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = LATENCY_single[core_index];
        }
        else if (tmpInstruction.operation == "LMV" || tmpInstruction.operation == "LLDI")
        {
            if (tmpInstruction.stage == "MAIN")
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = preparation_timeline[core_index];
            else
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_start = LATENCY_single[core_index];
            int element_num = tmpInstruction.element_num;
            int real_vector_latency;
            if (tmpInstruction.operation == "LMV")
                real_vector_latency = 2 * ceil(double(element_num) * double(ArithmeticPrecision) / 8 / (double(LOCAL_MEMORY_bandwidth) / double(Frequency) )) ;
            else
                real_vector_latency = ceil(double(element_num) * double(ArithmeticPrecision) / 8 / (double(LOCAL_MEMORY_bandwidth) / double(Frequency) )) ;
            if (tmpInstruction.stage == "MAIN")
            {
                preparation_time_single[core_index] += real_vector_latency;
                preparation_timeline[core_index] += real_vector_latency;
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = preparation_timeline[core_index];
            }
            else
            {
                LATENCY_single[core_index] += real_vector_latency;
                VEC_latency_core[core_index] += real_vector_latency;
                PIMCOMP_8_base_instruction_ir_with_placement[instruction_group_index].core_list[core_index].instruction_ir_list[k].latency_end = LATENCY_single[core_index];
            }
            Dyn_Pow_Local_Memory_Volume += 2 * tmpInstruction.element_num;
            Dyn_Pow_Vector_Volume += tmpInstruction.element_num;
        }
        // Make sure the program is estimated from the beginning in every instruction group.
        // 这是为了确保每个instruction_group都从头开始。
        if (k == instruction_ir_num-1)
        {
            preparation_timeline[core_index] = LATENCY_single[core_index];
            preparation_time_single[core_index] = 0;
        }
    }
    // Maybe the instruction_ir_num of some core is 0, but the next core still has instructions. So we have to consider this situation.
    // 这样是因为有的core的instruction_ir_num是0，但是下一个core仍然有指令。如果这部分在循环内，则后面核的指令都访问不到。
    int next_core_index = core_index+1;
    while (visited_single[next_core_index] != 0)
    {
        next_core_index++;
    }
    EvaluateRecursionSingleInstructionGroup(instruction_group_index, next_core_index, 0);
}

int ModelEvaluation::CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle)
{
    int needed_cycle = communication_needed_cycle;
    int forward_cycle = 0;
    while(needed_cycle > 0)
    {
        if (COMM_cycle_flag[chip_index][current_time + forward_cycle] == 0)
        {
            COMM_cycle_flag[chip_index][current_time + forward_cycle] = 1;
            needed_cycle --;
        }
        forward_cycle++;
    }
    return forward_cycle;
}

int ModelEvaluation::CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle)
{
    int needed_cycle = global_memory_needed_cycle;
    int forward_cycle = 0;
    while(needed_cycle > 0)
    {
        if (MEM_cycle_flag[chip_index][current_time + forward_cycle] == 0)
        {
            MEM_cycle_flag[chip_index][current_time + forward_cycle] = 1;
            needed_cycle --;
        }
        forward_cycle++;
    }
    return forward_cycle;
}

int ModelEvaluation::CheckGlobalMemoryBandwidthWB(int chip_index, long long current_time, int global_memory_needed_cycle)
{
    int needed_cycle = global_memory_needed_cycle;
    int forward_cycle = 0;
    while(needed_cycle > 0)
    {
        if (WB_cycle_flag[chip_index][current_time + forward_cycle] == 0)
        {
            WB_cycle_flag[chip_index][current_time + forward_cycle] = 1;
            needed_cycle --;
        }
        forward_cycle++;
    }
    return forward_cycle;
}

void ModelEvaluation::ShowEvaluationResultSingleInstructionGroup()
{
    double practical = 0;
    int MVMUL_num = 0;
    int VVADD_num = 0;
    int COMM_num = 0;
    int MEM_num = 0;
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if (LATENCY_single[i] > practical)
            practical = LATENCY_single[i];
        std::cout  << i << "  MVMUL:" << std::left << std::setw(5) << MVMUL_single[i]
//                        << "  VVADD:" << std::left << std::setw(5) << VVADD_single[i]
//                        << " COMM:" << std::left << std::setw(5) << COMM_single[i]
                        << " MEM:" << std::left << std::setw(4) << MEM_single[i]
                        << " LATENCY: " << std::left << std::setw(8) << std::setprecision(1) << std::fixed <<  LATENCY_single[i]
//                        << "   COMM_SYN:" <<  COMM_syn_latency_core[i]
                        << "   COMM_SYN:" <<  std::setprecision(2) << COMM_syn_latency_core[i] / double(LATENCY_single[i]) * 100 << "%"
//                        << "   COMM_TRANS:" <<  COMM_trans_latency_core[i]
                        << "   COMM_TRANS:" <<  std::setprecision(2) << COMM_trans_latency_core[i] / double(LATENCY_single[i]) * 100 << "%"
                        << "   MVMUL:" <<   MVMUL_latency_core[i]
                        << "   MVMUL:" <<  std::setprecision(2) << MVMUL_latency_core[i] / double(LATENCY_single[i]) * 100 << "%"
//                        << "   VEC:" <<  VEC_latency_core[i]
                        << "   VEC:" <<  std::setprecision(2) << VEC_latency_core[i] / double(LATENCY_single[i]) * 100 << "%"
                        << "   MEM:" <<  MEM_latency_core[i]
                        << "   MEM:" << std::setprecision(2) << MEM_latency_core[i] / double(LATENCY_single[i]) * 100 << "%"
                        << "   ratio:" <<  std::setprecision(2) << (COMM_syn_latency_core[i]+COMM_trans_latency_core[i]+MVMUL_latency_core[i]+VEC_latency_core[i]+MEM_latency_core[i]) / double(LATENCY_single[i]) * 100 << "%"
                        << "  COMM Volume:" << double(COMM_volume_per_core[i]) / 1024.0  << "kB"
                        << "  MEM Volume:" << double(MEM_volume_per_core[i]) / 1024.0  << "kB"
                        << std::endl;
        MVMUL_num += MVMUL_single[i];
        VVADD_num += VVADD_single[i];
        COMM_num += COMM_single[i];
        MEM_num += MEM_single[i];
        // For GUI
        PIMCOMP_GUI_evaluation_of_core[i].MVMUL_num = MVMUL_single[i];
        PIMCOMP_GUI_evaluation_of_core[i].VVADD_num = VVADD_single[i];
        PIMCOMP_GUI_evaluation_of_core[i].COMM_num = COMM_single[i];
        PIMCOMP_GUI_evaluation_of_core[i].MEM_num = MEM_single[i];
        PIMCOMP_GUI_evaluation_of_core[i].overall_latency = LATENCY_single[i];
        PIMCOMP_GUI_evaluation_of_core[i].comm_syn_latency = COMM_syn_latency_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].comm_trans_latency = COMM_trans_latency_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].mvmul_latency = MVMUL_latency_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].vec_latency = VEC_latency_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].mem_latency = MEM_latency_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].comm_volume = COMM_volume_per_core[i];
        PIMCOMP_GUI_evaluation_of_core[i].mem_volume = MEM_volume_per_core[i];
    }


    std::cout << std::endl;
    std::cout << std::fixed << "practical (ns)" << std::endl << practical << std::endl;
    if (element_pipeline)
        std::cout << std::fixed << "IG average latency (ns)" << std::endl << practical / instruction_group_num << std::endl;
    tmp_practical = practical;
    std::cout << "MVMUL (#)" << std::endl << MVMUL_num << std::endl;
    std::cout << "VVADD (#)" << std::endl << VVADD_num << std::endl;
    std::cout << "COMM (#)" << std::endl << COMM_num << std::endl;
    std::cout << "Volume (bytes)"  << std::endl << COMM_volume << std::endl;
    std::cout << "Cycles (ns)" << std::endl << COMM_cycle_total << std::endl;
    std::cout << "MEM (#)"  << std::endl << MEM_num << std::endl;
    std::cout << "Volume (bytes)" << std::endl << MEM_volume  << std::endl;
    std::cout << "Cycles (ns)" << std::endl << MEM_cycle_total << std::endl;
    std::cout << "HT Volume (bytes)" << std::endl << HT_volume << std::endl;
    std::cout << "Cycles (ns)" << std::endl <<  HT_cycle_total << std::endl;
}

void ModelEvaluation::SaveEvaluation()
{
    std::ofstream OutFile("../output/EvaluationResult.txt", std::ios::out | std::ios::trunc);
    for (int i = 0; i < instruction_group_num; ++i)
    {
        OutFile << "========================================= base instruction_group " << i << " =========================================" << std::endl;
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            int instruction_num = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list.size();
            if (instruction_num == 0)
                continue;
            OutFile << "Core " << j << " Start" << std::endl;
            for (int k = 0; k < instruction_num; ++k)
            {
                struct INST Instruction = PIMCOMP_8_base_instruction_ir_with_placement[i].core_list[j].instruction_ir_list[k];
                SaveSingleInstructionWithAddress(OutFile, Instruction, i, j);
            }
            OutFile << "Core " << j << " Over" << std::endl;
        }
    }
    OutFile.close();
}

void ModelEvaluation::ResetSingleInstructionGroup(bool clear_other_info)
{
    for (int & n : visited_single) {n = 0;}
    if (clear_other_info)
    {
        for (int & n : MVMUL_single) {n = 0;}
        for (int & n : VVADD_single) {n = 0;}
        for (int & n : COMM_single) {n = 0;}
        for (int & n : MEM_single) {n = 0;}
        for (int i = 0; i < ChipW * ChipH; ++i)  LATENCY_single_old[i] = LATENCY_single[i];
    }
}

void ModelEvaluation::Clear()
{
    for (int & n : MVMUL_single) {n = 0;}
    for (int & n : MVMUL_AG) {n = 0;}
    for (int & n : VVADD_single) {n = 0;}
    for (int & n : COMM_single) {n = 0;}
    for (int & n : MEM_single) {n = 0;}
    for (double & n : LATENCY_single) {n = 0;}
    for (double & n : LATENCY_single_old) {n = 0;}
    for (int & n : visited_single) {n = 0;}

    for (long long & n : AG_last_exec_start_time) {n = 0;}
    for (long long & n : AG_last_complete_time) {n = 0;}
    for (long long & n : last_MVMUL_exec_start_time) {n = 0;}
    for (long long & n : last_synchronous_time) {n = 0;}
    for (int i = 0; i < 50000; ++i)
        for (int j = 0; j < MAX_CORE; ++j)
            MVMUL_instruction_group_core[i][j] = 0;
    for (long long & n : preparation_time_single) {n = 0;}
    for (long long & n : preparation_timeline) {n = 0;}


    for (int & n : base_visited_record) {n = 0;}
    for (float & n : base_AG_memory_usage) {n = 0;}
    for (float & n : base_AG_memory_usage_input) {n = 0;}
    for (float & n : base_AG_memory_usage_output) {n = 0;}
    for (float & n : base_node_memory_usage) {n = 0;}
    for (float & n : base_core_memory_usage) {n = 0;}
    for (float & n : base_core_memory_usage_recv) {n = 0;}
    for (float & n : base_core_memory_usage_input) {n = 0;}
    for (float & n : base_core_memory_usage_output) {n = 0;}


    tmp_practical = 0.0;
    COMM_volume = 0;
    MEM_volume = 0;
    HT_volume = 0;
    COMM_cycle_total = 0;
    MEM_cycle_total = 0;
    HT_cycle_total = 0;

    Dyn_Pow_Local_Memory_Volume = 0;
    Dyn_Pow_Vector_Volume = 0;
    Dyn_Pow_MVMUL_crossbar_Num = 0;
}
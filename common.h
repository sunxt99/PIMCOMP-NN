// 123
// Created by SXT on 2022/8/18.
//

#ifndef _COMMON
#define _COMMON

#include <iomanip>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <list>
#include <cmath>
#include "./backend/json/json.h"
#include <ctime>
#include <random>
#include <stdexcept>
#include <numeric>
#include <queue>

/////////////////////////////////////////////////////////  NODE  /////////////////////////////////////////////////////////
struct param
{
    //// FC
    int num_input;
    int num_output;
    //// CONV and POOL
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_h1;
    int pad_w0;
    int pad_w1;
    int dilation_h;
    int dilation_w;
    int input_channel;
    int output_channel;
    int group;
    int activation;
    int wino_off;
    int pool_method;
    //// FLATTEN
    int axis;
    int end_axis;
    //// ELTWISE
    int eletype;
    int caffe_flavor;
    float shift;
    float power;
    float scale;
    //// PAD
    int mode;
    int pad_0_h;
    int pad_0_w;
    int pad_1_h;
    int pad_1_w;
    int pad_2_h;
    int pad_2_w;
    int pad_3_h;
    int pad_3_w;
    int value;
    //// SHUFFLE
    int split_factor;
};


struct PIMCOMP_node
{
    int AG0_core_index;
    int AG0_index_in_total;
    int AG0_node_index;
    int AGP;
    int H;
    int W;
    int bitwidth;
    std::vector<int> consumer_index;
    int consumer_num;
    int copy_offset_flag;
    int effective_node_index;
    int index;
    int index_in_level;
    int input_cycle_in_total;
    std::vector<int> input_dim;
    int input_dim_num;
    int level_index;
    std::string name;
    std::string operation;
    std::vector<int> output_dim;
    struct param param;
    int output_dim_num;
    std::vector<int> provider_index;
    int provider_num;
    int replication_num;
    int replication_num_origin;
    int instruction_group_num = 0;
    bool with_bias = false;
    bool with_act = false;
    int act_type;
    bool with_bn = false;
    bool with_clip = false;
    float clip_min;
    float clip_max;
    int max_level_gap = -1;
};

/////////////////////////////////////////////////////////  2  /////////////////////////////////////////////////////////
struct AG_list
{
    int AG_index;
    std::vector<int> virtual_crossbar_list;
    std::vector<int> virtual_core_list;
};

struct replication
{
    std::vector<int> AG_index;
    std::vector<struct AG_list> AG_list;
    int agp_index;
};

struct PIMCOMP_2_AG_partition
{
    int AGP_num;
    int Height;
    int Width;
    int index;
    int input_cycle_in_total;
    int crossbar_num_per_AG;
    int AG_num_per_replication;
    std::string name;
    std::string operation;
    std::vector<struct replication> replication;
    int replication_num;
    int replication_num_origin;
};

struct PIMCOMP_2_virtual_crossbar
{
    int index_in_weight;
    int virtual_index;
    int replication_index;
    int array_group_in_weight;
    int array_group_total;
    int height_start;
    int height_end;
    int width_start;
    int width_end;
    int weight_index;
    int node_index;
    int AG_num_per_replication;
    int agp_index;
    int agp_offset;
    // added in hierarchy mapping
    // vcore and vcore_index are equivalent
    int vcore;            // in 2_virtual_crossbar
    int vcore_index;      // in 3_hierarchy_map/whole
    int index_in_vcore;
    // added in element placement
    int physical_core;
    int physical_position_in_core;
    // added in schedule
    int instruction_group_num;
};

struct PIMCOMP_2_resource_info
{
    int AGs;
    int RRAMS;
    int Core;
};

/////////////////////////////////////////////////////////  3  /////////////////////////////////////////////////////////
struct PIMCOMP_3_hierarchy_map
{
    std::vector<std::vector<struct PIMCOMP_2_virtual_crossbar>> whole;
//    std::vector<struct whole> whole;
    std::vector<int> whole_index;
    std::vector<std::vector<struct PIMCOMP_2_virtual_crossbar>> split;
//    std::vector<struct whole> split;
    std::vector<int> split_index;
};

/////////////////////////////////////////////////////////  5  /////////////////////////////////////////////////////////
struct AG_base_info
{
    int node_index;
    int core_index;
    int AG_index;
    int AG_index_in_replication;
    int input_element_num;
    int output_element_num;
    int AG_num_per_replication;
    int replication_index;
    int replication_num;
};

struct AG_memory_info
{
    int AG_index;
    int input_element_num;
    int output_element_num;
    int appearance_num_in_one_core;
    int total_input_element_num; //
    long long memory_start_address;
    long long memory_element_length;
    long long memory_output_start_address;
    bool is_double_buffer;
};

struct AG_memory_info_of_one_core_struct
{
    std::vector<struct AG_memory_info> AG_memory_info_of_one_core;
};

struct AG_memory_info_of_one_IG_struct
{
    std::vector<struct AG_memory_info_of_one_core_struct> AG_memory_info_of_one_IG;
};




/////////////////////////////////////////////////////////  6  /////////////////////////////////////////////////////////

struct PIMCOMP_conv_pool_input_output_info
{
    std::vector<std::vector<int>> input_index;
    std::vector<std::vector<int>> output_index;
};


struct AG_input_info
{
    int node_index;
    int AG_index;
    int AG_index_in_replication;
    // CONV
    bool is_CONV;
    int start_position;
    int end_position;
    int start_channel_index;
    int end_channel_index;
    int start_channel_start_to_end;
    int end_channel_start_to_end;
    // FC
    bool is_FC;
    int FC_start_position;
    int FC_end_position;
};



/////////////////////////////////////////////////////////  6  /////////////////////////////////////////////////////////
struct replication_list_schedule
{
    std::vector<int> replication_list;
};

struct PIMCOMP_4_first_AG_info
{
    std::vector<struct replication_list_schedule> node_list;
};

struct AG_info_schedule
{
    int AGP;
    int AG_index_in_replication;
    int AG_index_in_total;
    int AG_num_per_replication;
    int agp_index;
    int agp_offset;
    int input_cycle_in_total;
    int level_index;
    int replication_index;
    int replication_num;
    int replication_num_origin;

    int width_start;
    int width_end;
    int height_start;
    int height_end;
    int input_element_num;
    int output_element_num;
    int core_index;
    int node_index;
    bool first_layer;

    // for Element Memory Allocation
    // flatten one sliding window to a vector with k*k*channel_input elements, with index of [0:k*k*channel_input-1],
    // then determine the position of the element that this AG needs in this vector
    // 将一个window的所有元素看做0 ~ k*k*channel_input-1，然后看AG所需的输入在一整个窗口中的位置。
    int start_input_element_num_in_window;
    int end_input_element_num_in_window;
};

struct core_schedule
{
    std::vector<struct AG_info_schedule> AG_list;
    std::vector<int> node_list;
};

struct PIMCOMP_4_virtual_core_AG_map
{
    std::vector<struct core_schedule> core_list;
};

struct node_recv_info
{
    // For CONV
    int AG_index;
    int AG_index_in_replication;
    int core_index;
    int node_index;
    int replication_index;
    // For FC
    int recv_element;
    int recv_num;
    int start_offset_element;
    int start_offset_num;
};

struct PIMCOMP_4_recv_info
{
    std::map<int, std::vector<struct node_recv_info>> node_list;
};


////////////////////////////////////////////////////  Instruction  ////////////////////////////////////////////////////
enum InstType {MVMUL, VEC1OP, VEC2OP, COMM, MEM, LLDI, VER};

struct INST
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int AGP;
    int AG_index_in_replication;
    int AG_index_in_total;
    int AG_num_per_replication;
    int agp_index;
    int destination;
    int input_cycle_in_total;
    int input_cycle_index;
    int input_element_num;
    int instruction_group_index;
    int level_index;
    int node_index;
    int output_element_num;
    int replication_index;
    int replication_num;
    int source;
    long long source_offset;
    long long destination_offset;
    //// Added VEC_1OP
    int element_num;
    int level_diff;
    int relative_length;
    //// Added VEC_2op
    int source_1;
    int source_2;
    long long source_1_offset;
    long long source_2_offset;
    //// Added COMM
    int comm_index;
    int instruction_index_in_core;
    int from_core; // RECV
    int to_core;  // SEND
    ////
    int copy_offset_flag;
    //// Load Input Batch
    int source_offset_between_batch = 0;
    //// Store Output Batch
    int destination_offset_between_batch = 0;
    //// CONCAT
    int input_cycle;
    //// POOL
    int input_channel_index;
    int output_channel_index;
    // Prepare For Input
    int rs_offset_in_channel;
    //// LLDI
    float imm_value;
    //// for batch
    int provider_node_index;

    long long source_address;
    long long destination_address;
    long long source_1_address;
    long long source_2_address;

    long long latency_start = 0;
    long long latency_end = 0;
};

struct INST_MVMUL
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int AGP;
    int AG_index_in_replication;
    int AG_index_in_total;
    int AG_num_per_replication;
    int agp_index;
    int destination;
    int input_cycle_in_total;
    int input_cycle_index;
    int input_element_num;
    int instruction_group_index;
    int level_index;
    int node_index;
    int output_element_num;
    int replication_index;
    int replication_num;
    int source;
    int rs_offset;
    int rd_offset;
};

struct INST_VEC_1OP
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int destination;
    int element_num;
    int instruction_group_index;
    int level_index;
    int level_diff;
    int relative_length;
    int source;
    int rd_offset;
    int rs_offset;
    int copy_offset_flag;
};

struct INST_VEC_2OP
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int AGP;
    int agp_index;
    int destination;
    int element_num;
    int instruction_group_index;
    int level_index;
    int level_diff;
    int relative_length;
    int source_1;
    int source_2;
    int rs1_offset;
    int rs2_offset;
    int rd_offset;
    int copy_offset_flag;
};

struct INST_COMM
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int comm_index;
    int element_num;
    int relative_length;
    int instruction_group_index;
    int instruction_index_in_core;
    int level_index;
    int level_diff;
    int AGP;
    int agp_index;
    // RECV
    int destination;
    int from_core;
    // SEND
    int source;
    int to_core;
    int copy_offset_flag;
};

struct INST_MEM
{
    InstType type;
    std::string operation;
    std::string stage;
    std::string conv_or_fc;
    std::string note;
    int destination;
    int element_num;
    int instruction_group_index;
    int level_index;
    int level_diff;
    int relative_length;
    int source;
    int rd_offset;
    int rs_offset;
    int copy_offset_flag;
};

////// BODY
struct instruction_ir_list_body_struct
{
    std::list<struct INST> instruction_ir_list_body;
};
struct PIMCOMP_4_instruction_ir_body
{
    std::vector<struct instruction_ir_list_body_struct> core_list;
};

//// VVADD
struct instruction_ir_list_vvadd_struct
{
    std::queue<struct INST> instruction_ir_list_vvadd;
};
struct PIMCOMP_4_instruction_ir_vvadd
{
    std::vector<struct instruction_ir_list_vvadd_struct> core_list;
};

//// VRELU
struct instruction_ir_list_vrelu_struct
{
    std::stack<struct INST> instruction_ir_list_vrelu;
};
struct PIMCOMP_4_instruction_ir_vrelu
{
    std::vector<struct instruction_ir_list_vrelu_struct> core_list;
};

////// Merged
struct instruction_ir_list_struct
{
    std::vector<struct INST> instruction_ir_list;
};
struct PIMCOMP_4_instruction_ir
{
    std::vector<struct instruction_ir_list_struct> core_list;
};



////////////////////////////////////////////////////  7  ////////////////////////////////////////////////////



////////////////////////////////////////////////////  Fast Evaluation  ////////////////////////////////////////////////////

struct MapSortStruct
{
    int input_cycle_per_replication;
    int crossbar_num_per_AG;
    int AG_num_per_replication;
    int crossbar_num_per_replication;
    int node_index;
    int height;
    std::string operation;
    float ratio;
};

struct AGMapStruct
{
    int node_index;
    int replication_index;
    int index_in_replication;
    int AG_index_in_total;
//    int input_cycle_num; // MVMUL_num
//    int instruction_group_num;
    int output_element_num;
    int input_element_num;
};


/////////////////////////////////////////// Design Space Exploration ///////////////////////////////////////////////////


struct DSE_AG_struct
{
    int node_index;
    int core_index;
    int replication_index;
    int index_in_replication;
    int input_cycle_num; // MVMUL_num
    int crossbar_num_of_AG;
};

struct DSE_result_info
{
    int max_MVMUL_num;
    int crossbar_num;
    int AG_num;
    int MVMUL_num;
};


//////////////////////////////////////////////////////// GUI DESIGN ////////////////////////////////////////////////////////////////

struct evaluation_info
{
    int MVMUL_num;
    int VVADD_num;
    int COMM_num;
    int MEM_num;
    double overall_latency;
    double comm_syn_latency;
    double comm_trans_latency;
    double mvmul_latency;
    double vec_latency;
    double mem_latency;
    double comm_volume;
    double mem_volume;
};

//////////////////////////////////////////////////////// SHOW INSTRUCTION ////////////////////////////////////////////////////////////////

static void SaveSingleInstructionWithAddress(std::ofstream & OutFile, struct INST Instruction, int instruction_group_index, int core_index)
{
    std::string Operation = Instruction.operation;
    switch (Instruction.type)
    {
        case MVMUL:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " stage:" << Instruction.stage
                    << " node:" << Instruction.node_index
                    << " input:" << Instruction.input_cycle_index
                    << " rs:" << Instruction.source
                    << " rs_addr:" << Instruction.source_address
                    << " rd:" << Instruction.destination
                    << " rd_addr:" << Instruction.destination_address
                    << " input_element_num:" << Instruction.input_element_num
                    << " output_element_num:" << Instruction.output_element_num
                    << " IG:" << instruction_group_index;
            if (Instruction.latency_start != 0)
                OutFile << " S:" << Instruction.latency_start;
            if (Instruction.latency_end != 0)
                OutFile << " E:" << Instruction.latency_end;
            OutFile << std::endl;
            break;
        }
        case VEC1OP:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " stage:" << Instruction.stage
                    << " node:" << Instruction.node_index
                    << " rs_addr:" << Instruction.source_address
                    << " rs_offset:" << Instruction.source_offset
                    << " rd_addr:" << Instruction.destination_address
                    << " rd_offset:" << Instruction.destination_offset
                    << " element_num:" << Instruction.element_num
                    << " IG:" << instruction_group_index;
            if (Instruction.latency_start != 0)
                OutFile << " S:" << Instruction.latency_start;
            if (Instruction.latency_end != 0)
                OutFile << " E:" << Instruction.latency_end;
            OutFile << std::endl;
            break;
        }
        case VEC2OP:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " stage:" << Instruction.stage
                    << " node:" << Instruction.node_index;
            if (Operation == "VVMAX")
                OutFile << " input_cycle_index:" << Instruction.input_cycle_index;
            OutFile << " rs1_addr:" << Instruction.source_1_address
                    << " rs1_offset:" << Instruction.source_1_offset
                    << " rs2_addr:" << Instruction.source_2_address
                    << " rs2_offset:" << Instruction.source_2_offset
                    << " rd_addr:" << Instruction.destination_address
                    << " rd_offset:" << Instruction.destination_offset
                    << " element_num:" << Instruction.element_num
                    << " IG:" << instruction_group_index;
            if (Instruction.latency_start != 0)
                OutFile << " S:" << Instruction.latency_start;
            if (Instruction.latency_end != 0)
                OutFile << " E:" << Instruction.latency_end;
            OutFile << std::endl;
            break;
        }
        case COMM:
        {
            if (Operation == "SEND")
            {
                OutFile << "    [" << Operation << "]"
                        << " core:" << core_index
                        << " stage:" << Instruction.stage
                        << " comm_index:" << Instruction.comm_index
                        << " to Core:" << Instruction.to_core
                        << " rs_addr:" << Instruction.source_address
                        << " element_num:" << Instruction.element_num
                        << " IG:" << instruction_group_index;
                if (Instruction.latency_start != 0)
                    OutFile << " S:" << Instruction.latency_start;
                if (Instruction.latency_end != 0)
                    OutFile << " E:" << Instruction.latency_end;
                OutFile << std::endl;
            }
            else
            {
                OutFile << "    [" << Operation << "]"
                        << " core:" << core_index
                        << " stage:" << Instruction.stage
                        << " comm_index:" << Instruction.comm_index
                        << " from Core:" << Instruction.from_core
                        << " rd_addr:" << Instruction.destination_address
                        << " element_num:" << Instruction.element_num
                        << " IG:" << instruction_group_index;
                if (Instruction.latency_start != 0)
                    OutFile << " S:" << Instruction.latency_start;
                if (Instruction.latency_end != 0)
                    OutFile << " E:" << Instruction.latency_end;
                OutFile << std::endl;
            }
            break;
        }
        case MEM:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " stage:" << Instruction.stage
                    << " node:" << Instruction.node_index
                    << " rs_addr:" << Instruction.source_address
                    << " rs_offset:" << Instruction.source_offset
                    << " rd_addr:" << Instruction.destination_address
                    << " rd_offset:" << Instruction.destination_offset
                    << " element_num:" << Instruction.element_num
                    << " IG:" << instruction_group_index;
            if (Instruction.latency_start != 0)
                OutFile << " S:" << Instruction.latency_start;
            if (Instruction.latency_end != 0)
                OutFile << " E:" << Instruction.latency_end;
            OutFile << std::endl;
            break;
        }
        case LLDI:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " stage:" << Instruction.stage
                    << " node:" << Instruction.node_index
                    << " rd_addr:" << Instruction.destination_address
                    << " rd_offset:" << Instruction.destination_offset
                    << " imm_value:" << Instruction.imm_value
                    << " element_num:" << Instruction.element_num
                    << " IG:" << instruction_group_index;
            if (Instruction.latency_start != 0)
                OutFile << " S:" << Instruction.latency_start;
            if (Instruction.latency_end != 0)
                OutFile << " E:" << Instruction.latency_end;
            OutFile << std::endl;
            break;
        }
        case VER:
        {
            OutFile << "    [" << Operation << "]"
                    << " core:" << core_index
                    << " node:" << Instruction.node_index
                    << " input_cycle:" << Instruction.input_cycle_index
                    << " source_address:" << Instruction.source_address
                    << " element_num:" << Instruction.element_num
                    << " IG:" << instruction_group_index
                    << std::endl;
            break;
        };
    }
}



static Json::Value SaveInstructionWithAddressInJSON(struct INST Instruction)
{
    Json::Value JsonInstruction;
    std::string Operation = Instruction.operation;
    switch (Instruction.type)
    {
        case MVMUL:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["stage"] = Instruction.stage;
            JsonInstruction["source"] = Instruction.source;
            Json::Int64 source_address = Instruction.source_address;
            JsonInstruction["source_address"] = source_address;
            Json::Int64 source_offset = Instruction.source_offset;
            JsonInstruction["source_offset"] = source_offset;
            JsonInstruction["destination"] = Instruction.destination;
            Json::Int64 destination_address = Instruction.destination_address;
            JsonInstruction["destination_address"] = destination_address;
            Json::Int64 destination_offset = Instruction.destination_offset;
            JsonInstruction["destination_offset"] = destination_offset;
            JsonInstruction["node_index"] = Instruction.node_index;
            JsonInstruction["input_cycle_index"] = Instruction.input_cycle_index;
            JsonInstruction["input_element_num"] = Instruction.input_element_num;
            JsonInstruction["output_element_num"] = Instruction.output_element_num;
            JsonInstruction["input_cycle_index"] = Instruction.input_cycle_index;
            return JsonInstruction;
        }
        case VEC1OP:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["node_index"] = Instruction.node_index;
            JsonInstruction["stage"] = Instruction.stage; // Only For ReLU of MVMUL stage
            JsonInstruction["input_cycle_index"] = Instruction.input_cycle_index; // Only For ReLU of MVMUL stage
            Json::Int64 source_address = Instruction.source_address;
            JsonInstruction["source_address"] = source_address;
            Json::Int64 source_offset = Instruction.source_offset;
            JsonInstruction["source_offset"] = source_offset;
            Json::Int64 destination_address = Instruction.destination_address;
            JsonInstruction["destination_address"] = destination_address;
            Json::Int64 destination_offset = Instruction.destination_offset;
            JsonInstruction["destination_offset"] = destination_offset;
            JsonInstruction["element_num"] = Instruction.element_num;
            JsonInstruction["imm_value"] = Instruction.imm_value;
            return JsonInstruction;
        }
        case VEC2OP:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["node_index"] = Instruction.node_index;
            JsonInstruction["stage"] = Instruction.stage;
            Json::Int64 source_1_address = Instruction.source_1_address;
            JsonInstruction["source_1_address"] = source_1_address;
            Json::Int64 source_1_offset = Instruction.source_1_offset;
            JsonInstruction["source_1_offset"] = source_1_offset;
            Json::Int64 source_2_address = Instruction.source_2_address;
            JsonInstruction["source_2_address"] = source_2_address;
            Json::Int64 source_2_offset = Instruction.source_2_offset;
            JsonInstruction["source_2_offset"] = source_2_offset;
            Json::Int64 destination_address = Instruction.destination_address;
            JsonInstruction["destination_address"] = destination_address;
            Json::Int64 destination_offset = Instruction.destination_offset;
            JsonInstruction["destination_offset"] = destination_offset;
            JsonInstruction["element_num"] = Instruction.element_num;
            if ((Operation == "VVMAX" || Operation == "VVMUL" || Operation == "VVADD") && Instruction.stage == "POST")
                JsonInstruction["input_cycle_index"] = Instruction.input_cycle_index; // Only For POST
            return JsonInstruction;
        }
        case COMM:
        {
            if (Operation == "SEND")
            {
                JsonInstruction["operation"] = Operation;
                JsonInstruction["to_core"] = Instruction.to_core;
                Json::Int64 source_address = Instruction.source_address;
                JsonInstruction["source_address"] = source_address;
            }
            else
            {
                JsonInstruction["operation"] = Operation;
                JsonInstruction["from_core"] = Instruction.from_core;
                Json::Int64 destination_address = Instruction.destination_address;
                JsonInstruction["destination_address"] = destination_address;
            }
            JsonInstruction["stage"] = Instruction.stage;
            JsonInstruction["element_num"] = Instruction.element_num;
            JsonInstruction["comm_index"] = Instruction.comm_index;
            JsonInstruction["instruction_index_in_core"] = Instruction.instruction_index_in_core;
            return JsonInstruction;
        }
        case MEM:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["node_index"] = Instruction.node_index;
            JsonInstruction["stage"] = Instruction.stage;
            Json::Int64 source_address = Instruction.source_address;
            JsonInstruction["source_address"] = source_address;
            Json::Int64 source_offset = Instruction.source_offset;
            JsonInstruction["source_offset"] = source_offset;
            Json::Int64 destination_address = Instruction.destination_address;
            JsonInstruction["destination_address"] = destination_address;
            Json::Int64 destination_offset = Instruction.destination_offset;
            JsonInstruction["destination_offset"] = destination_offset;
            JsonInstruction["element_num"] = Instruction.element_num;
            return JsonInstruction;
        }
        case LLDI:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["stage"] = Instruction.stage;
            JsonInstruction["node_index"] = Instruction.node_index;
            Json::Int64 destination_address = Instruction.destination_address;
            JsonInstruction["destination_address"] = destination_address;
            Json::Int64 destination_offset = Instruction.destination_offset;
            JsonInstruction["destination_offset"] = destination_offset;
            JsonInstruction["imm_value"] = Instruction.imm_value;
            JsonInstruction["element_num"] = Instruction.element_num;
            return JsonInstruction;
        }
        case VER:
        {
            JsonInstruction["operation"] = Operation;
            JsonInstruction["node_index"] = Instruction.node_index;
            JsonInstruction["stage"] = Instruction.stage; // Only For ReLU of MVMUL stage
            JsonInstruction["input_cycle_index"] = Instruction.input_cycle_index; // Only For ReLU of MVMUL stage
            Json::Int64 source_address = Instruction.source_address;
            JsonInstruction["source_address"] = source_address;
            Json::Int64 source_offset = Instruction.source_offset;
            JsonInstruction["source_offset"] = source_offset;
            JsonInstruction["element_num"] = Instruction.element_num;
            return JsonInstruction;
        }
        default:
            return JsonInstruction;
    }
}

static void MyDivide(std::vector<int> &start_address_vector, std::vector<int> &end_address_vector, int output_channel_num_total, int replication_num)
{
    //// 如果有output_channel_num < replication_num的情况，那么处理长度为0的rep的start会比end高1。可以根据这一点来跳过这些rep。
    start_address_vector.clear();
    end_address_vector.clear();
    std::vector<int> channel_allocated;
    for (int i = 0; i < replication_num; ++i)
        channel_allocated.push_back(ceil(float(output_channel_num_total) / float(replication_num)));
    int minus_num = ceil(float(output_channel_num_total) / float(replication_num)) * replication_num - output_channel_num_total;
    for (int i = 0; i < minus_num; ++i)
        channel_allocated[replication_num-1-i] -= 1;
    int start_address;
    int end_address = -1;
    for (int i = 0; i < replication_num; ++i)
    {
        start_address = end_address + 1;
        end_address = start_address + channel_allocated[i] - 1;
        start_address_vector.push_back(start_address);
        end_address_vector.push_back(end_address);
    }
}

#endif
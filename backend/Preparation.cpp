//
// Created by SXT on 2022/10/5.
//

#include "Preparation.h"

Json::Value PIMCOMP_VERIFICATION_INFO;
std::map<int, struct PIMCOMP_node> PIMCOMP_node_list_origin;
std::map<int, struct PIMCOMP_node> PIMCOMP_node_list;
std::vector<struct PIMCOMP_conv_pool_input_output_info> PIMCOMP_conv_pool_input_output_info;
std::vector<struct PIMCOMP_conv_pool_input_output_info> PIMCOMP_conv_pool_full_output_info;
std::vector<struct PIMCOMP_2_AG_partition> PIMCOMP_2_AG_partition;
std::vector<struct PIMCOMP_2_virtual_crossbar> PIMCOMP_2_virtual_crossbar;
struct PIMCOMP_2_resource_info PIMCOMP_2_resource_info;
std::vector<int> PIMCOMP_2_effective_node;
struct PIMCOMP_3_hierarchy_map PIMCOMP_3_hierarchy_map;
std::vector<std::vector<int>> PIMCOMP_3_virtual_core_crossbar_map;
struct PIMCOMP_4_first_AG_info PIMCOMP_4_first_AG_info;
struct PIMCOMP_4_virtual_core_AG_map PIMCOMP_4_virtual_core_AG_map;
std::vector<int> PIMCOMP_4_AG_num_of_same_rep_in_core;
std::vector<int> PIMCOMP_4_AG_input_element_num;
struct PIMCOMP_4_recv_info PIMCOMP_4_recv_info;
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_4_base_instruction_ir;
std::vector<std::vector<int>> PIMCOMP_4_input_cycle_record;
std::map<int, struct PIMCOMP_4_instruction_ir> PIMCOMP_4_post_instruction_ir;

int PIMCOMP_base_instruction_num;
int PIMCOMP_post_instruction_num;
int comm_pair_total_num;

//// Added for Memory Allocation
//// Activate in the batch schedule stage. First save the bias data, and then allocate memory.
std::vector<int> PIMCOMP_5_memory_start_address;
std::vector<struct AG_memory_info_of_one_IG_struct> PIMCOMP_5_AG_memory_info;
std::vector<struct AG_base_info> PIMCOMP_5_AG_base_info;
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_5_base_instruction_with_address;

//// Added for Data Reload
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_6_base_instruction_with_input;
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_6_base_instruction_with_input_batch;
std::vector<struct AG_input_info> PIMCOMP_6_AG_input_info;
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_7_base_instruction_ir_with_optimization;
std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_8_base_instruction_ir_with_placement;
std::vector<int> PIMCOMP_8_virtual_core_to_physical_core_map;
std::vector<std::vector<long long>> PIMCOMP_6_inter_core_communication;

std::set<int> PIMCOMP_4_unique_instruction_group_index;
std::vector<int> PIMCOMP_4_evaluation_instruction_group_index;
std::vector<int> PIMCOMP_4_core_instruction_group_num;  // Activate in Batch pipeline. Record instruction_group_num of every core.

// An intermediate result of distributed mapping. Sort the nodes in the order of "compute_crossbar_ratio".
std::vector<std::pair<struct MapSortStruct, int>> PIMCOMP_3_compute_crossbar_ratio;
// The final result of distributed mapping. Get the mapping_result.
std::vector<std::vector<struct AGMapStruct>> PIMCOMP_3_mapping_result;

std::vector<std::vector<int>> PIMCOMP_topology_provider_consumer_relation;
std::vector<std::vector<int>> PIMCOMP_topology_consumer_provider_relation;
// PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_index]=0、1、2
// Get index_in_all_provider according to provider_index
std::vector<std::map<int,int>> PIMCOMP_4_element_node_provider_index_2_index_in_all_providers;

std::vector<std::vector<int>> PIMCOMP_DSE_replication_num;
std::vector<std::vector<std::vector<struct DSE_AG_struct>>> PIMCOMP_DSE_core_map_info;
std::vector<struct DSE_result_info> PIMCOMP_DSE_result_info;
std::vector<std::vector<std::vector<std::vector<int>>>> PIMCOMP_DSE_node_map_info;

std::vector<int> PIMCOMP_node_crossbar_num_per_AG;

// EP stands for Element-Pipeline
std::vector<double> PIMCOMP_EP_delay_for_conv_and_pool;
std::vector<std::vector<std::vector<int>>> PIMCOMP_EP_path_to_conv_or_pool;

//// For QT GUI output
// PIMCOMP_GUI_memory_usage_every_instruction_group[core_index][instruction_group_index]
std::vector<std::vector<double>> PIMCOMP_GUI_memory_usage_every_instruction_group;
std::vector<struct evaluation_info> PIMCOMP_GUI_evaluation_of_core;
std::vector<std::vector<std::pair<int, long long>>> PIMCOMP_GUI_execution_time_of_core;
std::vector<std::vector<long long>> PIMCOMP_GUI_execution_time_of_node;
std::vector<std::vector<int>> PIMCOMP_GUI_inter_core_communication_volume;

bool instruction_with_reload;
bool element_pipeline;

void EliminatePaddingOperator(std::string model_name, Json::Value & DNNInfo)
{
    std::set<int> pad_node_index_set;
    if (model_name == "inception_v3")
    {
        for (int i = 0; i < DNNInfo["node_list"].size(); ++i)
        {
            if (DNNInfo["node_list"][i]["operation"].asString() == "OP_PAD")
            {
                pad_node_index_set.insert(i);
                std::string pad_node_name = DNNInfo["node_list"][i]["name"].asString();
                // Check if there is only one provider and one consumer
                int provider_num = DNNInfo["node_list"][i]["provider"].size();
                int consumer_num = DNNInfo["node_list"][i]["consumer"].size();
                if (provider_num != 1 || consumer_num != 1)
                {
                    fprintf(stderr, "padding operator has more than one provider or consumer\n");
                    abort();
                }
                // idx of provider
                std::string provider_name = DNNInfo["node_list"][i]["provider"][0].asString();
                int provider_index = 0;
                for (int j = 0; j < DNNInfo["node_list"].size(); ++j)
                    if (DNNInfo["node_list"][j]["name"].asString() == provider_name)
                        provider_index = j;
                // idx of consumer
                std::string consumer_name = DNNInfo["node_list"][i]["consumer"][0].asString();
                int consumer_index = 0;
                for (int j = 0; j < DNNInfo["node_list"].size(); ++j)
                    if (DNNInfo["node_list"][j]["name"].asString() == consumer_name)
                        consumer_index = j;
                // checks if the values can be substituted
                int pad_0_h = DNNInfo["node_list"][i]["param"]["pad_0_h"].asInt();
                int pad_0_w = DNNInfo["node_list"][i]["param"]["pad_0_w"].asInt();
                int pad_1_h = DNNInfo["node_list"][i]["param"]["pad_1_h"].asInt();
                int pad_1_w = DNNInfo["node_list"][i]["param"]["pad_1_w"].asInt();
                int pad_2_h = DNNInfo["node_list"][i]["param"]["pad_2_h"].asInt();
                int pad_2_w = DNNInfo["node_list"][i]["param"]["pad_2_w"].asInt();
                int pad_3_h = DNNInfo["node_list"][i]["param"]["pad_3_h"].asInt();
                int pad_3_w = DNNInfo["node_list"][i]["param"]["pad_3_w"].asInt();
                if (pad_0_h != 0 || pad_0_w != 0 || pad_1_h != 0 || pad_1_w != 0)
                {
                    fprintf(stderr, "padding operator parameter is illegal\n");
                    abort();
                }
                int input_dim_num = DNNInfo["node_list"][provider_index]["output_dim_num"].asInt();
                int output_dim_num = DNNInfo["node_list"][i]["output_dim_num"].asInt();
                if (output_dim_num != 4 || input_dim_num != 4)
                {
                    fprintf(stderr, "padding operator dimension is illegal\n");
                    abort();
                }
                // Checks if its consumer is CONV or POOL
                std::string consumer_operation = DNNInfo["node_list"][consumer_index]["operation"].asString();
                if (consumer_operation == "OP_CONV" || consumer_operation == "OP_POOL")
                {
                    DNNInfo["node_list"][consumer_index]["param"]["pad_h0"] = DNNInfo["node_list"][consumer_index]["param"]["pad_h0"].asInt() + pad_2_h;
                    DNNInfo["node_list"][consumer_index]["param"]["pad_h1"] = DNNInfo["node_list"][consumer_index]["param"]["pad_h1"].asInt() + pad_2_w;
                    DNNInfo["node_list"][consumer_index]["param"]["pad_w0"] = DNNInfo["node_list"][consumer_index]["param"]["pad_w0"].asInt() + pad_3_h;
                    DNNInfo["node_list"][consumer_index]["param"]["pad_w1"] = DNNInfo["node_list"][consumer_index]["param"]["pad_w1"].asInt() + pad_3_w;
                }
                else
                {
                    fprintf(stderr, "padding's consumer is illegal\n");
                    abort();
                }
                // Rewrite the provider-consumer relationship
                // Change the consumer of the provider of the pad (the provider of the pad may have multiple consumers, so find the correct one) to the consumer of the pad
                // 将pad的生产者的消费者（pad的生产者可能有多个消费者，所以要找到正确的一个）改为pad的消费者
                int provider_consumer_num = DNNInfo["node_list"][provider_index]["consumer"].size();
                for (int j = 0; j < provider_consumer_num; ++j)
                {
                    std::string provider_consumer_name = DNNInfo["node_list"][provider_index]["consumer"][j].asString();
                    if (provider_consumer_name == pad_node_name)
                    {
                        DNNInfo["node_list"][provider_index]["consumer"][j] = consumer_name;
                        break;
                    }
                }
                // Change the provider of the consumer of the pad to the provider of the pad
                // 将pad的消费者的生产者改为pad的生产者
                DNNInfo["node_list"][consumer_index]["provider"][0] = provider_name;
            }
        }

        Json::Value copy_DNNInfo;
        for (int i = 0; i < DNNInfo["node_list"].size(); ++i)
        {
            if (!pad_node_index_set.count(i))
            {
                copy_DNNInfo["node_list"].append(DNNInfo["node_list"][i]);
            }
        }

        DNNInfo["node_list"] = copy_DNNInfo["node_list"];
    }
}

void EliminateBatchNormOperator(std::string model_name, Json::Value & DNNInfo)
{
    std::set<int> bn_node_index_set;
    if (model_name == "resnet18")
    {
        for (int i = 0; i < DNNInfo["node_list"].size(); ++i)
        {
            if (DNNInfo["node_list"][i]["operation"].asString() == "OP_BN")
            {
                bn_node_index_set.insert(i);
                std::string bn_node_name = DNNInfo["node_list"][i]["name"].asString();
                // Check if there is only one provider and one consumer
                int provider_num = DNNInfo["node_list"][i]["provider"].size();
                int consumer_num = DNNInfo["node_list"][i]["consumer"].size();
                if (provider_num != 1 || consumer_num != 1)
                {
                    fprintf(stderr, "BN operator has more than one provider or consumer\n");
                    abort();
                }
                // idx of provider
                std::string provider_name = DNNInfo["node_list"][i]["provider"][0].asString();
                int provider_index = 0;
                for (int j = 0; j < DNNInfo["node_list"].size(); ++j)
                    if (DNNInfo["node_list"][j]["name"].asString() == provider_name)
                        provider_index = j;
                // idx of consumer
                std::string consumer_name = DNNInfo["node_list"][i]["consumer"][0].asString();
                int consumer_index = 0;
                for (int j = 0; j < DNNInfo["node_list"].size(); ++j)
                    if (DNNInfo["node_list"][j]["name"].asString() == consumer_name)
                        consumer_index = j;
                // Rewrite the provider-consumer relationship
                // Change the consumer of the provider of the BN (the provider of the BN may have multiple consumers, so find the correct one) to the consumer of the BN
                // 将BN的生产者的消费者（BN的生产者可能有多个消费者，所以要找到正确的一个）改为BN的消费者
                int provider_consumer_num = DNNInfo["node_list"][provider_index]["consumer"].size();
                for (int j = 0; j < provider_consumer_num; ++j)
                {
                    std::string provider_consumer_name = DNNInfo["node_list"][provider_index]["consumer"][j].asString();
                    if (provider_consumer_name == bn_node_name)
                    {
                        DNNInfo["node_list"][provider_index]["consumer"][j] = consumer_name;
                        break;
                    }
                }
                // Change the provider of the consumer of the BN to the provider of the BN
                // 将bn的消费者的生产者改为pad的生产者
                DNNInfo["node_list"][consumer_index]["provider"][0] = provider_name;
            }
        }

        Json::Value copy_DNNInfo;
        for (int i = 0; i < DNNInfo["node_list"].size(); ++i)
        {
            if (!bn_node_index_set.count(i))
            {
                copy_DNNInfo["node_list"].append(DNNInfo["node_list"][i]);
            }
        }

        DNNInfo["node_list"] = copy_DNNInfo["node_list"];
    }
}

void PreProcess(Json::Value & DNNInfo)
{
    Json::Value NodeList = DNNInfo["node_list"];
    int node_num = NodeList.size();
    // Save the "Name-Index" key-value map
    std::map<std::string, int> name2index_map;
    for (int i = 0; i < node_num; ++i)
        name2index_map.insert(std::pair<std::string, int>(NodeList[i]["name"].asCString(),i));
    // Reorder the Index
    for (int i = 0; i < node_num; ++i)
    {
        DNNInfo["node_list"][i]["new_node_index"] = i;
    }
    // Get the Provider_Index and Consumer_Index
    for (int i = 0; i < node_num; ++i)
    {
        Json::Value Node = NodeList[i];
        if (strcmp(Node["operation"].asCString(), "OP_INPUT") == 0)
            continue;
        int provider_index = name2index_map[Node["provider"][0].asCString()];
        int input_dim_num = NodeList[provider_index]["output_dim_num"].asInt();
        DNNInfo["node_list"][i]["input_dim_num"] = input_dim_num;
        for (int j = 0; j < input_dim_num; ++j)
        {
            DNNInfo["node_list"][i]["input_dim"][j] = NodeList[provider_index]["output_dim"][j].asInt();
        }
    }
    for (int i = 0; i < node_num; ++i)
    {
        Json::Value Node = NodeList[i];
        int consumer_num = Node["consumer_num"].asInt();
        for (int j = 0; j < consumer_num; ++j)
        {
            std::string consumer_name = Node["consumer"][j].asCString();
            int consumer_index = name2index_map[consumer_name];
            DNNInfo["node_list"][i]["consumer_index"].append(consumer_index);
        }

        int provider_num = Node["provider_num"].asInt();
        for (int j = 0; j < provider_num; ++j)
        {
            std::string provider_name = Node["provider"][j].asCString();
            int provider_index = name2index_map[provider_name];
            DNNInfo["node_list"][i]["provider_index"].append(provider_index);
        }
    }
}

void GetStructNodeListFromJson(Json::Value DNNInfo)
{
    Json::Value NodeList = DNNInfo["node_list"];
    int node_num = NodeList.size();
    for (int i = 0; i < node_num; ++i)
    {
        PIMCOMP_node_list_origin[i].bitwidth = NodeList[i]["bitwidth"].asInt();
        PIMCOMP_node_list_origin[i].consumer_num = NodeList[i]["consumer_num"].asInt();
        PIMCOMP_node_list_origin[i].index = NodeList[i]["new_node_index"].asInt();
        PIMCOMP_node_list_origin[i].input_dim_num = NodeList[i]["input_dim_num"].asInt();
        PIMCOMP_node_list_origin[i].name = NodeList[i]["name"].asCString();
        PIMCOMP_node_list_origin[i].operation = NodeList[i]["operation"].asCString();
        PIMCOMP_node_list_origin[i].output_dim_num = NodeList[i]["output_dim_num"].asInt();
        PIMCOMP_node_list_origin[i].provider_num = NodeList[i]["provider_num"].asInt();

        for (int j = 0; j < PIMCOMP_node_list_origin[i].provider_num; ++j)
            PIMCOMP_node_list_origin[i].provider_index.push_back(NodeList[i]["provider_index"][j].asInt());
        for (int j = 0; j < PIMCOMP_node_list_origin[i].consumer_num; ++j)
            PIMCOMP_node_list_origin[i].consumer_index.push_back(NodeList[i]["consumer_index"][j].asInt());
        for (int j = 0; j < PIMCOMP_node_list_origin[i].input_dim_num; ++j)
            PIMCOMP_node_list_origin[i].input_dim.push_back(NodeList[i]["input_dim"][j].asInt());
        for (int j = 0; j < PIMCOMP_node_list_origin[i].output_dim_num; ++j)
            PIMCOMP_node_list_origin[i].output_dim.push_back(NodeList[i]["output_dim"][j].asInt());

        if (strcmp(NodeList[i]["operation"].asCString(), "OP_FC") == 0)
        {
            PIMCOMP_node_list_origin[i].param.num_input = NodeList[i]["param"]["num_input"].asInt();
            PIMCOMP_node_list_origin[i].param.num_output = NodeList[i]["param"]["num_output"].asInt();
            int Height = PIMCOMP_node_list_origin[i].param.num_input;
            int Width = PIMCOMP_node_list_origin[i].param.num_output;
            PIMCOMP_node_list_origin[i].H = Height;
            PIMCOMP_node_list_origin[i].W = Width;
            PIMCOMP_node_list_origin[i].with_bias = NodeList[i]["param"]["with_bias"].asInt();
            PIMCOMP_node_list_origin[i].with_bn = NodeList[i]["with_bn"].asInt();
            PIMCOMP_node_list_origin[i].with_act = NodeList[i]["with_act"].asInt();
            PIMCOMP_node_list_origin[i].act_type = NodeList[i]["act_type"].asInt();
            PIMCOMP_node_list_origin[i].with_clip = NodeList[i]["with_clip"].asInt();
            PIMCOMP_node_list_origin[i].clip_min = NodeList[i]["clip_min"].asFloat();
            PIMCOMP_node_list_origin[i].clip_max = NodeList[i]["clip_max"].asFloat();
        }
        else if (strcmp(NodeList[i]["operation"].asCString(), "OP_CONV")==0 || strcmp(NodeList[i]["operation"].asCString(), "OP_POOL")==0)
        {
            PIMCOMP_node_list_origin[i].param.kernel_h = NodeList[i]["param"]["kernel_h"].asInt();
            PIMCOMP_node_list_origin[i].param.kernel_w = NodeList[i]["param"]["kernel_w"].asInt();
            PIMCOMP_node_list_origin[i].param.stride_h = NodeList[i]["param"]["stride_h"].asInt();
            PIMCOMP_node_list_origin[i].param.stride_w = NodeList[i]["param"]["stride_w"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_h0 = NodeList[i]["param"]["pad_h0"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_h1 = NodeList[i]["param"]["pad_h1"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_w0 = NodeList[i]["param"]["pad_w0"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_w1 = NodeList[i]["param"]["pad_w1"].asInt();

            if (strcmp(NodeList[i]["operation"].asCString(), "OP_CONV")==0)
            {
                PIMCOMP_node_list_origin[i].param.dilation_h = NodeList[i]["param"]["dilation_h"].asInt();
                PIMCOMP_node_list_origin[i].param.dilation_w = NodeList[i]["param"]["dilation_w"].asInt();
                PIMCOMP_node_list_origin[i].param.input_channel = NodeList[i]["param"]["input_channel"].asInt();
                PIMCOMP_node_list_origin[i].param.output_channel = NodeList[i]["param"]["output_channel"].asInt();
                PIMCOMP_node_list_origin[i].param.group = NodeList[i]["param"]["group"].asInt();
                PIMCOMP_node_list_origin[i].param.activation = NodeList[i]["param"]["activation"].asInt();
                PIMCOMP_node_list_origin[i].param.wino_off = NodeList[i]["param"]["wino_off"].asInt();
                int Height = PIMCOMP_node_list_origin[i].param.kernel_h * PIMCOMP_node_list_origin[i].param.kernel_w * PIMCOMP_node_list_origin[i].param.input_channel;
                int Width = PIMCOMP_node_list_origin[i].param.output_channel;
                PIMCOMP_node_list_origin[i].H = Height;
                PIMCOMP_node_list_origin[i].W = Width;
                PIMCOMP_node_list_origin[i].with_bias = NodeList[i]["param"]["with_bias"].asInt();
                PIMCOMP_node_list_origin[i].with_bn = NodeList[i]["with_bn"].asInt();
                PIMCOMP_node_list_origin[i].with_act = NodeList[i]["with_act"].asInt();
                PIMCOMP_node_list_origin[i].act_type = NodeList[i]["act_type"].asInt();
                PIMCOMP_node_list_origin[i].with_clip = NodeList[i]["with_clip"].asInt();
                PIMCOMP_node_list_origin[i].clip_min = NodeList[i]["clip_min"].asFloat();
                PIMCOMP_node_list_origin[i].clip_max = NodeList[i]["clip_max"].asFloat();
            }
            else
            {
                PIMCOMP_node_list_origin[i].param.pool_method = NodeList[i]["param"]["pool_method"].asInt();
            }
        }
        else if (strcmp(NodeList[i]["operation"].asCString(), "OP_FLATTEN")==0)
        {
            PIMCOMP_node_list_origin[i].param.axis = NodeList[i]["param"]["axis"].asInt();
            PIMCOMP_node_list_origin[i].param.end_axis = NodeList[i]["param"]["end_axis"].asInt();
        }
        else if (strcmp(NodeList[i]["operation"].asCString(), "OP_ELTWISE")==0)
        {
            PIMCOMP_node_list_origin[i].param.eletype = NodeList[i]["param"]["eletype"].asInt();
            PIMCOMP_node_list_origin[i].param.caffe_flavor = NodeList[i]["param"]["caffe_flavor"].asInt();
            PIMCOMP_node_list_origin[i].param.shift = NodeList[i]["param"]["shift"].asFloat();
            PIMCOMP_node_list_origin[i].param.power = NodeList[i]["param"]["power"].asFloat();
            PIMCOMP_node_list_origin[i].param.scale = NodeList[i]["param"]["scale"].asFloat();
        }
        else if (strcmp(NodeList[i]["operation"].asCString(), "OP_PAD")==0)
        {
            PIMCOMP_node_list_origin[i].param.mode = NodeList[i]["param"]["mode"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_0_h = NodeList[i]["param"]["pad_0_h"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_0_w = NodeList[i]["param"]["pad_0_w"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_1_h = NodeList[i]["param"]["pad_1_h"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_1_w = NodeList[i]["param"]["pad_1_w"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_2_h = NodeList[i]["param"]["pad_2_h"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_2_w = NodeList[i]["param"]["pad_2_w"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_3_h = NodeList[i]["param"]["pad_3_h"].asInt();
            PIMCOMP_node_list_origin[i].param.pad_3_w = NodeList[i]["param"]["pad_3_w"].asInt();
            PIMCOMP_node_list_origin[i].param.value = NodeList[i]["param"]["value"].asInt();
        }
        else if (strcmp(NodeList[i]["operation"].asCString(), "OP_SHUFFLE")==0)
        {
            PIMCOMP_node_list_origin[i].param.split_factor = NodeList[i]["param"]["split_factor"].asInt();
            PIMCOMP_node_list_origin[i].param.input_channel = NodeList[i]["param"]["input_channel"].asInt();
        }
    }
}

void ShowModelInfo()
{
    int node_num = PIMCOMP_node_list_origin.size();
    std::cout << "#Nodes in total: " << node_num << std::endl;
    float weight_precession = 16;
    float weights = 0.0;
    float FC_weights = 0.0;
    float Output_Sum = 0.0;
    int min_crossbar_need_one_core = 0;
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list_origin[i].operation == "OP_CONV")
        {
            std::cout << i <<std::endl;
            float kernel = PIMCOMP_node_list_origin[i].param.kernel_h;
            float input_channel = PIMCOMP_node_list_origin[i].param.input_channel;
            float output_channel = PIMCOMP_node_list_origin[i].param.output_channel;
            weights += kernel * kernel * input_channel * output_channel;
//            std::cout << "weight: " << kernel * kernel * input_channel * output_channel*weight_precession/8/1024/1024 << "MB" << std::endl;
            std::vector<int> Input = PIMCOMP_node_list_origin[i].input_dim;
            std::cout << "input: " << float(Input[0]) * float(Input[1]) * float(Input[2]) * float(Input[3]) *weight_precession/8/1024 << "kB" << std::endl;
            std::vector<int> Output = PIMCOMP_node_list_origin[i].output_dim;
            std::cout << "output: " << float(Output[0]) * float(Output[1]) * float(Output[2]) * float(Output[3]) *weight_precession/8/1024 << "kB" << std::endl;
            Output_Sum += float(Output[0]) * float(Output[1]) * float(Output[2]) * float(Output[3]);
            float kernel_h = PIMCOMP_node_list_origin[i].param.kernel_h;
            float kernel_w = PIMCOMP_node_list_origin[i].param.kernel_w;
            std::cout << "Weight Matrix:" << kernel_h * kernel_w * input_channel << " × " << output_channel << std::endl;
            std::cout << "min crossbar:" <<  ceil(output_channel / float(CrossbarW)) << std::endl;
            if (min_crossbar_need_one_core < ceil(output_channel / float(CrossbarW)) )
                min_crossbar_need_one_core = ceil(output_channel / float(CrossbarW));
        }
        else if (PIMCOMP_node_list_origin[i].operation == "OP_FC")
        {
            std::cout << i <<std::endl;
            float input_num = PIMCOMP_node_list_origin[i].param.num_input;
            float output_num = PIMCOMP_node_list_origin[i].param.num_output;
            weights += input_num * output_num;
            FC_weights += input_num * output_num;
//            std::cout << "weight: " << input_num*output_num*weight_precession/8/1024/1024 << "MB" << std::endl;
            std::cout << "output: " << PIMCOMP_node_list_origin[i].param.num_output *weight_precession/8/1024/1024 << "MB" << std::endl;
            Output_Sum += PIMCOMP_node_list_origin[i].param.num_output;
            std::cout << "Weight Matrix:" << input_num << " × " << output_num << std::endl;
            std::cout << "min crossbar:" << ceil(output_num / float(CrossbarW)) << std::endl;
            if (min_crossbar_need_one_core < ceil(output_num / float(CrossbarW)) )
                min_crossbar_need_one_core = ceil(output_num / float(CrossbarW));
        }
    }
    std::cout << "FC Weight: " << FC_weights*weight_precession/8/1024/1024 << "MB" << std::endl;
    std::cout << "Sum Weight: " << weights*weight_precession/8/1024/1024 << "MB" << std::endl;
    std::cout << "FC Ratio: " << FC_weights/weights*100 << "%" << std::endl;
    std::cout << "Output Sum:" << Output_Sum*weight_precession/8/1024/1024 << "MB" << std::endl;
    std::cout << "Minimum Crossbar Num in One Core:" << min_crossbar_need_one_core << std::endl;
}


void GetTopologyInformation()
{
    // Activation Nodes following CONV or FC are not counted
    // CONV或FC后面紧跟的激活函数不算在内
    int node_num = PIMCOMP_node_list_origin.size();
    PIMCOMP_topology_provider_consumer_relation.resize(node_num);
    PIMCOMP_topology_consumer_provider_relation.resize(node_num);
    PIMCOMP_4_element_node_provider_index_2_index_in_all_providers.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list_origin[i].operation;
        if (PIMCOMP_node_list_origin[i].consumer_num == 0)
        {
            PIMCOMP_topology_provider_consumer_relation[i].push_back(-1);
            continue;
        }
        if (operation == "OP_CONV" || operation == "OP_FC")
        {
            for (int j = 0; j < PIMCOMP_node_list_origin[i].consumer_index.size(); ++j)
            {
                int consumer_index = PIMCOMP_node_list_origin[i].consumer_index[j];
                std::string consumer_operation = PIMCOMP_node_list_origin[consumer_index].operation;
                if (consumer_operation == "OP_RELU" || consumer_operation == "OP_TANH" || consumer_operation == "OP_SIGMOID")
                {
//                    PIMCOMP_node_list_origin[i].with_act = true;
//                    PIMCOMP_node_list_origin[i].act_type = consumer_operation == "OP_RELU" ? 0 : (consumer_operation == "OP_TANH" ? 1 : 2);
                    for (int k = 0; k < PIMCOMP_node_list_origin[consumer_index].consumer_index.size(); ++j)
                    {
                        int consumer_consumer_index = PIMCOMP_node_list_origin[consumer_index].consumer_index[k];
                        PIMCOMP_topology_provider_consumer_relation[i].push_back(consumer_consumer_index);
                        PIMCOMP_topology_consumer_provider_relation[consumer_consumer_index].push_back(i);
                        PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_consumer_index][i] = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_consumer_index].size();
                    }
                }
                else
                {
//                    PIMCOMP_node_list_origin[i].with_act = false;
                    PIMCOMP_topology_provider_consumer_relation[i].push_back(consumer_index);
                    PIMCOMP_topology_consumer_provider_relation[consumer_index].push_back(i);
                    PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_index][i] = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_index].size();
                }
            }
        }
        else
        {
            if (operation == "OP_RELU" || operation == "OP_TANH" || operation == "OP_SIGMOID")
            {
                int provider_index = PIMCOMP_node_list_origin[i].provider_index[0];
                if (PIMCOMP_node_list_origin[provider_index].operation == "OP_CONV" || PIMCOMP_node_list_origin[provider_index].operation == "OP_FC")
                    continue;
            }
            int consumer_num = PIMCOMP_node_list_origin[i].consumer_num;
            for (int j = 0; j < consumer_num; ++j)
            {
                int consumer_index = PIMCOMP_node_list_origin[i].consumer_index[j];
                PIMCOMP_topology_provider_consumer_relation[i].push_back(consumer_index);
                PIMCOMP_topology_consumer_provider_relation[consumer_index].push_back(i);
                PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_index][i] = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[consumer_index].size();
            }
        }
    }

    std::cout << "provider - consumer" << std::endl;
    for (int i = 0; i < node_num; ++i)
    {
        for (int j = 0; j < PIMCOMP_topology_provider_consumer_relation[i].size(); ++j)
        {
            std::cout << i << " :" << PIMCOMP_topology_provider_consumer_relation[i][j] << std::endl;
        }
    }
    std::cout << " consumer - provider" << std::endl;
    for (int i = 0; i < node_num; ++i)
    {
        for (int j = 0; j < PIMCOMP_topology_consumer_provider_relation[i].size(); ++j)
        {
            std::cout << i << " :" << PIMCOMP_topology_consumer_provider_relation[i][j] << std::endl;
        }
    }


}


void CopyFromOriginNodeList()
{
    PIMCOMP_node_list.clear();
    for (auto iter = PIMCOMP_node_list_origin.begin(); iter != PIMCOMP_node_list_origin.end() ; ++iter)
    {
        PIMCOMP_node_list[iter->first] = iter->second;
    }
}


void GetConvPoolInputOutputInfo()
{
    int node_num = PIMCOMP_node_list_origin.size();
    PIMCOMP_conv_pool_input_output_info.resize(node_num);
    for (int n = 0; n < node_num; ++n)
    {
        struct PIMCOMP_node Node = PIMCOMP_node_list_origin[n];
        if (Node.operation != "OP_POOL" && Node.operation != "OP_CONV")
            continue;

        struct param Params = Node.param;
        int input_H = Node.input_dim[2];
        int input_W = Node.input_dim[3];
        int pool_kernel_w = Params.kernel_w;
        int pool_kernel_h = Params.kernel_h;
        int pool_padding_h0 = Params.pad_h0;
        int pool_padding_h1 = Params.pad_h1;
        int pool_padding_w0 = Params.pad_w0;
        int pool_padding_w1 = Params.pad_w1;
        int pool_stride_w = Params.stride_w;
        int pool_stride_h = Params.stride_h;

        int output_W = floor(float(input_W + pool_padding_w0 + pool_padding_w1 - pool_kernel_w) / float(pool_stride_w)) + 1;
        int output_H = floor(float(input_H + pool_padding_h0 + pool_padding_h1 - pool_kernel_h) / float(pool_stride_h)) + 1;
        int info_output_W = Node.output_dim[3];
        int info_output_H = Node.output_dim[2];
        if (info_output_W != output_W || info_output_H != output_H)
        {
            std::cout << " Output Size Doesn't Match" << std::endl;
            return;
        }

        PIMCOMP_conv_pool_input_output_info[n].input_index.resize(input_H * input_W);
        PIMCOMP_conv_pool_input_output_info[n].output_index.resize(info_output_W * info_output_H);

        int output_index = 0;
        int normal_start_index_in_w = pool_padding_w0/pool_stride_w + (pool_padding_w0 % pool_stride_w == 0 ? 0 : 1);
        int normal_start_index_in_h = pool_padding_h0/pool_stride_h + (pool_padding_h0 % pool_stride_h == 0 ? 0 : 1);
        for (int i = 0; i < output_H; ++i)
        {
            for (int j = 0; j < output_W; ++j)
            {
                int start_address = i * pool_stride_h * input_W + j *  pool_stride_w;

                if (j < normal_start_index_in_w)
                    start_address -= (j * pool_stride_w);
                else
                    start_address -= pool_padding_w0;

                if (i < normal_start_index_in_h)
                    start_address -= (i * pool_stride_h * input_W);
                else
                    start_address -= pool_padding_h0 * input_W;

                int start_row = start_address / input_W;
                int start_col = start_address % input_W;

                int pool_w_num = pool_kernel_w;
                if (j < normal_start_index_in_w)
                    pool_w_num = pool_w_num - pool_padding_w0 + j * pool_stride_w;
                if (start_col + pool_w_num > input_W)
                    pool_w_num = pool_w_num - (start_col + pool_w_num - input_W);

                int pool_h_num = pool_kernel_h;
                if (i < normal_start_index_in_h)
                    pool_h_num = pool_h_num - pool_padding_h0 + i * pool_stride_h;
                if (start_row + pool_h_num > input_H)
                    pool_h_num = pool_h_num - (start_row + pool_h_num - input_H);


                for (int h = 0; h < pool_h_num ; ++h)
                {
                    for (int w = 0; w < pool_w_num; ++w)
                    {
                        int position = start_address + w + h * input_W;
                        PIMCOMP_conv_pool_input_output_info[n].input_index[position].push_back(output_index);
                        PIMCOMP_conv_pool_input_output_info[n].output_index[output_index].push_back(position);
                    }
                }
                output_index += 1;
            }
        }
    }
}



void EP_delay_for_conv_and_pool()
{
    PIMCOMP_EP_delay_for_conv_and_pool.resize(PIMCOMP_node_list_origin.size());
    for (int n = 0; n < PIMCOMP_node_list_origin.size(); ++n)
    {
        if (PIMCOMP_node_list_origin[n].operation !=  "OP_CONV" && PIMCOMP_node_list_origin[n].operation !=  "OP_POOL")
            continue;
        struct param Params = PIMCOMP_node_list_origin[n].param;
        int input_h = PIMCOMP_node_list_origin[n].input_dim[2];
        int input_w = PIMCOMP_node_list_origin[n].input_dim[3];
        int kernel_w = Params.kernel_w;
        int kernel_h = Params.kernel_h;
        int padding_h0 = Params.pad_h0;
        int padding_h1 = Params.pad_h1;
        int padding_w0 = Params.pad_w0;
        int padding_w1 = Params.pad_w1;
        int stride_w = Params.stride_w;
        int stride_h = Params.stride_h;

        int output_w = floor(float(input_w + padding_w0 + padding_w1 - kernel_w) / float(stride_w)) + 1;
        int output_h = floor(float(input_h + padding_h0 + padding_h1 - kernel_h) / float(stride_h)) + 1;
        int info_output_w = PIMCOMP_node_list_origin[n].output_dim[3];
        int info_output_h = PIMCOMP_node_list_origin[n].output_dim[2];
        if (info_output_w != output_w || info_output_h != output_h)
        {
            std::cout << " Output Size Doesn't Match" << std::endl;
            return;
        }
        int rest_input_w = input_w - (kernel_w - padding_w0 - 1);
        int rest_input_h = input_h - (kernel_h - padding_h0 - 1);
        int effective_input_w = (kernel_w - padding_w0 - 1);
        int effective_input_h = (kernel_h - padding_h0 - 1);
        int blank_element_w = rest_input_w - output_w;
        int blank_element_h = rest_input_h - output_h;
        if (blank_element_w < 0)  // stride = 1 and padding_w1 > 0
        {
            effective_input_w += rest_input_w;
            output_w = rest_input_w;
        }
        else if (blank_element_w <= output_w - 1)
            effective_input_w += rest_input_w;
        else
            effective_input_w += output_w + (output_w-1) * (stride_w-1);
        if (blank_element_h < 0) // stride = 1 and padding_h1 > 0
        {
            effective_input_h += rest_input_h;
            output_h = rest_input_h;
        }
        else if (blank_element_h <= output_h - 1)
            effective_input_h += rest_input_h;
        else
            effective_input_h += output_h + (output_h-1) * (stride_h-1);

        int provider_index = PIMCOMP_node_list_origin[n].provider_index[0];
        if (PIMCOMP_node_list_origin[provider_index].operation == "OP_INPUT")
            PIMCOMP_EP_delay_for_conv_and_pool[n] = 0.0;
        else
            PIMCOMP_EP_delay_for_conv_and_pool[n] = float(effective_input_w * effective_input_h - (output_w * output_h))/ float(effective_input_w * effective_input_h);
//        std::cout << n << " " << PIMCOMP_node_list_origin[n].operation << " " << PIMCOMP_EP_delay_for_conv_and_pool[n] << std::endl;
    }
}

static int node_visit_num_record[3000];
void EP_pre_assign_size(int node_index)
{
    if (node_index == -1)
        return;
    for (int i = 0; i < PIMCOMP_topology_provider_consumer_relation[node_index].size(); ++i)
    {
        int consumer_index = PIMCOMP_topology_provider_consumer_relation[node_index][i];
        {
            EP_pre_assign_size(consumer_index);
        }
    }
    node_visit_num_record[node_index] += 1;
}

static int tmp_node_visit_num_record[3000];
void EP_path_to_conv_or_pool(int node_index)
{
    for (int i = 0; i < PIMCOMP_topology_provider_consumer_relation[node_index].size(); ++i)
    {
        int node_visit_num = tmp_node_visit_num_record[node_index];
        int consumer_index = PIMCOMP_topology_provider_consumer_relation[node_index][i];
        if (consumer_index == -1)
        {
            tmp_node_visit_num_record[node_index]++;
            return;
        }
        int consumer_visit_num = tmp_node_visit_num_record[consumer_index];
        if (consumer_visit_num > 50)
            continue;
        else
        {
            PIMCOMP_EP_path_to_conv_or_pool[consumer_index][consumer_visit_num].assign(PIMCOMP_EP_path_to_conv_or_pool[node_index][node_visit_num].begin(),PIMCOMP_EP_path_to_conv_or_pool[node_index][node_visit_num].end());
            if (PIMCOMP_node_list_origin[node_index].operation == "OP_CONV" || PIMCOMP_node_list_origin[node_index].operation == "OP_POOL")
                PIMCOMP_EP_path_to_conv_or_pool[consumer_index][consumer_visit_num].push_back(node_index);
            EP_path_to_conv_or_pool(consumer_index);
        }
    }
    tmp_node_visit_num_record[node_index]++;
}


void GetPriorInfoForElementPipeline()
{
    // First determine the delay
    // 首先确定delay
    EP_delay_for_conv_and_pool();
    PIMCOMP_EP_path_to_conv_or_pool.resize(PIMCOMP_node_list_origin.size());
    // Pre-allocate the vector length by EP_pre_assign_size
    // 通过EP_pre_assign_size预先分配vector长度
    EP_pre_assign_size(0);
    for (int i = 0; i < PIMCOMP_node_list_origin.size(); ++i)
        if (node_visit_num_record[i] != 0)
            PIMCOMP_EP_path_to_conv_or_pool[i].resize(node_visit_num_record[i]);
    // Determine all paths to each pool or conv
    // 确定到达每个pool或conv的路径（所有路径）
    EP_path_to_conv_or_pool(0);
//    std::cout << "Element Pipeline: Path To The End Node" << std::endl;
//    for (int i = 0; i < PIMCOM_EP_path_to_conv_or_pool.size(); ++i)
//    {
//        for (int j = 0; j < PIMCOM_EP_path_to_conv_or_pool[i].size(); ++j)
//        {
//            std::cout << i << ": " ;
//            for (int k = 0; k < PIMCOM_EP_path_to_conv_or_pool[i][j].size(); ++k)
//            {
//                std::cout << PIMCOM_EP_path_to_conv_or_pool[i][j][k] << " " ;
//            }
//            std::cout << std::endl;
//        }
//    }
}

void GetConvPoolInputOutputInfoForInputPreparation()
{
    int node_num = PIMCOMP_node_list_origin.size();
    PIMCOMP_conv_pool_full_output_info.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        struct PIMCOMP_node Node = PIMCOMP_node_list_origin[i];
        if (Node.operation != "OP_POOL" && Node.operation != "OP_CONV")
            continue;

        struct param Params = Node.param;
        int input_H = Node.input_dim[2];
        int input_W = Node.input_dim[3];
        int pool_kernel_w = Params.kernel_w;
        int pool_kernel_h = Params.kernel_h;
        int pool_padding_h0 = Params.pad_h0;
        int pool_padding_h1 = Params.pad_h1;
        int pool_padding_w0 = Params.pad_w0;
        int pool_padding_w1 = Params.pad_w1;
        int pool_stride_w = Params.stride_w;
        int pool_stride_h = Params.stride_h;

        int output_W = floor(float(input_W + pool_padding_w0 + pool_padding_w1 - pool_kernel_w) / float(pool_stride_w)) + 1;
        int output_H = floor(float(input_H + pool_padding_h0 + pool_padding_h1 - pool_kernel_h) / float(pool_stride_h)) + 1;
        int info_output_W = Node.output_dim[3];
        int info_output_H = Node.output_dim[2];
        if (info_output_W != output_W || info_output_H != output_H)
        {
            std::cout << " Output Size Doesn't Match" << std::endl;
            return;
        }

        PIMCOMP_conv_pool_full_output_info[i].output_index.resize(output_W * output_H);

        std::vector<std::vector<int>> Matrix;
        std::vector<int> Row;
        for (int j = 0; j < input_W + pool_padding_w0 + pool_padding_w1; ++j)
            Row.push_back(-1);
        for (int j = 0; j < input_H + pool_padding_h0 + pool_padding_h1; ++j)
            Matrix.push_back(Row);

        int index = 0;
        for (int h = pool_padding_h0; h < pool_padding_h0 + input_H; ++h)
        {
            for (int w = pool_padding_w0; w < pool_padding_w0 + input_W; ++w)
            {
                Matrix[h][w] = index;
                index++;
            }
        }

        for (int h = 0; h < output_H; ++h)
        {
            for (int w = 0; w < output_W; ++w)
            {
                int start_position_w = pool_stride_w * w;
                int start_position_h = pool_stride_h * h;
                PIMCOMP_conv_pool_full_output_info[i].output_index[h * output_W + w].resize(pool_kernel_h * pool_kernel_w);
                for (int k_h = 0; k_h < pool_kernel_h; ++k_h)
                {
                    for (int k_w = 0; k_w < pool_kernel_w; ++k_w)
                    {
                        int position_w = start_position_w + k_w;
                        int position_h = start_position_h + k_h;
                        PIMCOMP_conv_pool_full_output_info[i].output_index[h * output_W + w][k_h * pool_kernel_w + k_w] = Matrix[position_h][position_w];
                    }
                }
            }
        }
    }
}


void SaveIntermediateInfo(Json::Value DNNInfo)
{
    Json::Value PIMCOMP_Intermediate_INFO;
    //// PIMCOMP_node_list
    int node_num = DNNInfo["node_list"].size();
    PIMCOMP_Intermediate_INFO["node_list"] = DNNInfo["node_list"];
    //// PIMCOMP_conv_pool_input_output_info
    {
        PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"].resize(node_num);
        for (int i = 0; i < node_num; ++i)
        {
            //// input_index
            int input_index_num = PIMCOMP_conv_pool_input_output_info[i].input_index.size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["input_index"].resize(input_index_num);
            for (int j = 0; j < input_index_num; ++j)
            {
                int related_output_channel_num = PIMCOMP_conv_pool_input_output_info[i].input_index[j].size();
                PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["input_index"][j].resize(related_output_channel_num);
                for (int k = 0; k < related_output_channel_num; ++k)
                    PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["input_index"][j][k] = PIMCOMP_conv_pool_input_output_info[i].input_index[j][k];
            }
            //// output_index
            int output_index_num = PIMCOMP_conv_pool_input_output_info[i].output_index.size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["output_index"].resize(output_index_num);
            for (int j = 0; j < output_index_num; ++j)
            {
                int related_input_channel_num = PIMCOMP_conv_pool_input_output_info[i].output_index[j].size();
                PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["output_index"][j].resize(related_input_channel_num);
                for (int k = 0; k < related_input_channel_num; ++k)
                    PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_input_output_info"][i]["output_index"][j][k] = PIMCOMP_conv_pool_input_output_info[i].output_index[j][k];
            }
        }
    }
    //// PIMCOMP_conv_pool_full_output_info
    for (int i = 0; i < node_num; ++i)
    {
        int output_index_num = PIMCOMP_conv_pool_full_output_info[i].output_index.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_full_output_info"][i]["output_index"].resize(output_index_num);
        for (int j = 0; j < output_index_num; ++j)
        {
            int related_input_channel_num = PIMCOMP_conv_pool_full_output_info[i].output_index[j].size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_full_output_info"][i]["output_index"][j].resize(related_input_channel_num);
            for (int k = 0; k < related_input_channel_num; ++k)
                PIMCOMP_Intermediate_INFO["PIMCOMP_conv_pool_full_output_info"][i]["output_index"][j][k] = PIMCOMP_conv_pool_full_output_info[i].output_index[j][k];
        }
    }
    //// PIMCOMP_2_AG_partition
    {
        int num_ag_partition = PIMCOMP_2_AG_partition.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"].resize(num_ag_partition);
        for (int i = 0; i < num_ag_partition; ++i)
        {
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["AGP_num"] = PIMCOMP_2_AG_partition[i].AGP_num;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["Height"] = PIMCOMP_2_AG_partition[i].Height;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["Width"] = PIMCOMP_2_AG_partition[i].Width;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["index"] = PIMCOMP_2_AG_partition[i].index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["node_index"] = PIMCOMP_2_AG_partition[i].index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["input_cycle_in_total"] = PIMCOMP_2_AG_partition[i].input_cycle_in_total;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["crossbar_num_per_AG"] = PIMCOMP_2_AG_partition[i].crossbar_num_per_AG;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["AG_num_per_replication"] = PIMCOMP_2_AG_partition[i].AG_num_per_replication;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["name"] = PIMCOMP_2_AG_partition[i].name;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["operation"] = PIMCOMP_2_AG_partition[i].operation;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication_num"] = PIMCOMP_2_AG_partition[i].replication_num;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication_num_origin"] = PIMCOMP_2_AG_partition[i].replication_num_origin;
            int replication_num = PIMCOMP_2_AG_partition[i].replication.size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"].resize(replication_num);
            for (int j = 0; j < replication_num; ++j)
            {
                PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["agp_index"] = PIMCOMP_2_AG_partition[i].replication[j].agp_index;
                int AG_num = PIMCOMP_2_AG_partition[i].replication[j].AG_index.size();
                PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_index"].resize(AG_num);
                PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"].resize(AG_num);
                for (int k = 0; k < AG_num; ++k)
                {
                    PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_index"][k] = PIMCOMP_2_AG_partition[i].replication[j].AG_index[k];
                    PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"][k]["AG_index"] = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].AG_index;
                    int crossbar_num = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list.size();
                    PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"][k]["virtual_crossbar_list"].resize(crossbar_num);
                    PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"][k]["virtual_core_list"].resize(crossbar_num);
                    for (int l = 0; l < crossbar_num; ++l)
                    {
                        PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"][k]["virtual_crossbar_list"][l] = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list[l];
                        PIMCOMP_Intermediate_INFO["PIMCOMP_2_AG_partition"][i]["replication"][j]["AG_list"][k]["virtual_core_list"][l] = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_core_list[l];
                    }
                }
            }
        }
    }
    //// PIMCOMP_2_virtual_crossbar
    {
        int crossbar_num = PIMCOMP_2_virtual_crossbar.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"].resize(crossbar_num);
        for (int i = 0; i < crossbar_num; ++i)
        {
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["index_in_weight"] = PIMCOMP_2_virtual_crossbar[i].index_in_weight;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["virtual_index"] = PIMCOMP_2_virtual_crossbar[i].virtual_index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["replication_index"] = PIMCOMP_2_virtual_crossbar[i].replication_index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["array_group_in_weight"] = PIMCOMP_2_virtual_crossbar[i].array_group_in_weight;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["array_group_total"] = PIMCOMP_2_virtual_crossbar[i].array_group_total;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["height_start"] = PIMCOMP_2_virtual_crossbar[i].height_start;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["height_end"] = PIMCOMP_2_virtual_crossbar[i].height_end;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["width_start"] = PIMCOMP_2_virtual_crossbar[i].width_start;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["width_end"] = PIMCOMP_2_virtual_crossbar[i].width_end;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["weight_index"] = PIMCOMP_2_virtual_crossbar[i].weight_index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["node_index"] = PIMCOMP_2_virtual_crossbar[i].node_index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["AG_num_per_replication"] = PIMCOMP_2_virtual_crossbar[i].AG_num_per_replication;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["agp_index"] = PIMCOMP_2_virtual_crossbar[i].agp_index;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["agp_offset"] = PIMCOMP_2_virtual_crossbar[i].agp_offset;
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["vcore_index"] = PIMCOMP_2_virtual_crossbar[i].vcore_index;  // 这个出现在3_hierarchy_map/whole
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_virtual_crossbar"][i]["index_in_vcore"] = PIMCOMP_2_virtual_crossbar[i].index_in_vcore;
        }
    }
    //// PIMCOMP_2_resource_info
    {
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_resource_info"]["RRAMS"] = PIMCOMP_2_resource_info.RRAMS;
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_resource_info"]["AGs"] = PIMCOMP_2_resource_info.AGs;
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_resource_info"]["Core"] = PIMCOMP_2_resource_info.Core;
    }
    //// PIMCOMP_2_effective_node
    {
        int effective_node_num = PIMCOMP_2_effective_node.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_2_effective_node"].resize(effective_node_num);
        for (int i = 0; i < effective_node_num; ++i)
            PIMCOMP_Intermediate_INFO["PIMCOMP_2_effective_node"][i] = PIMCOMP_2_effective_node[i];
    }
    //// PIMCOMP_3_hierarchy_map
    {
        int AG_num = PIMCOMP_3_hierarchy_map.whole.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_3_hierarchy_map"]["whole"].resize(AG_num);
        PIMCOMP_Intermediate_INFO["PIMCOMP_3_hierarchy_map"]["whole_index"].resize(AG_num);
        for (int i = 0; i < AG_num; ++i)
        {
            PIMCOMP_Intermediate_INFO["PIMCOMP_3_hierarchy_map"]["whole_index"][i] = PIMCOMP_3_hierarchy_map.whole_index[i];
            int crossbar_num = PIMCOMP_3_hierarchy_map.whole[i].size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_3_hierarchy_map"]["whole"][i].resize(crossbar_num);
            for (int j = 0; j < crossbar_num; ++j)
            {
                PIMCOMP_Intermediate_INFO["PIMCOMP_3_hierarchy_map"]["whole"][i][j]["virtual_index"] = PIMCOMP_3_hierarchy_map.whole[i][j].virtual_index; // 这是crossbar的virtual_index
                // Other information can be obtained from PIMCOMP_2_virtual_crossbar.
                // 剩下的信息可以从PIMCOMP_2_virtual_crossbar中获取，这里没必要重复一遍。
            }
        }
    }
    //// PIMCOMP_3_virtual_core_crossbar_map
    {
        int core_num = PIMCOMP_3_virtual_core_crossbar_map.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_3_virtual_core_crossbar_map"].resize(core_num);
        for (int i = 0; i < core_num; ++i)
        {
            int crossbar_num = PIMCOMP_3_virtual_core_crossbar_map[i].size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_3_virtual_core_crossbar_map"][i].resize(crossbar_num);
            for (int j = 0; j < crossbar_num; ++j)
                PIMCOMP_Intermediate_INFO["PIMCOMP_3_virtual_core_crossbar_map"][i][j] = PIMCOMP_3_virtual_core_crossbar_map[i][j];
        }
    }
    //// PIMCOMP_4_virtual_core_AG_map
    {
        int core_num = PIMCOMP_4_virtual_core_AG_map.core_list.size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"].resize(core_num);
        for (int i = 0; i < core_num; ++i)
        {
            int AG_num_this_core = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
            PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"].resize(AG_num_this_core);
            PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["node_list"].resize(AG_num_this_core);
            for (int j = 0; j < AG_num_this_core; ++j)
            {
                // node_list and AG_list here correspond. node_list[x] indicates the node_index of AG_list[x].
                // 注意这里的node_list和AG_list是对应的。就是指示每个AG的node是多少。
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["node_list"][j] = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["AGP"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AGP;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["AG_index_in_replication"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["AG_index_in_total"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["AG_num_per_replication"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_num_per_replication;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["agp_index"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].agp_index;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["agp_offset"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].agp_offset;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["input_cycle_in_total"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_cycle_in_total;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["replication_num"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["level_index"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].level_index;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["replication_index"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["replication_num"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["replication_num_origin"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num_origin;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["width_start"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].width_start;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["width_end"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].width_end;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["height_start"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].height_start;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["height_end"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].height_end;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["input_element_num"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_element_num;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["output_element_num"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].output_element_num;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["core_index"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].core_index;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["node_index"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].node_index;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["first_layer"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].first_layer;
                // For Element Memory Allocation
                // Flatten one sliding window to a vector with k*k*channel_input elements, with index of [0:k*k*channel_input-1], then determine the position of the element that this AG needs in this vector
                // 将一个window的所有元素看做0 ~ k*k*channel_input-1，然后看AG所需的输入在一整个窗口中的位置。
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["start_input_element_num_in_window"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].start_input_element_num_in_window;
                PIMCOMP_Intermediate_INFO["PIMCOMP_4_virtual_core_AG_map"]["core_list"][i]["AG_list"][j]["end_input_element_num_in_window"] = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].end_input_element_num_in_window;
            }
        }
    }
    //// PIMCOMP_topology_provider_consumer_relation
    PIMCOMP_Intermediate_INFO["PIMCOMP_topology_provider_consumer_relation"].resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        int consumer_number = PIMCOMP_topology_provider_consumer_relation[i].size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_topology_provider_consumer_relation"][i].resize(consumer_number);
        for (int j = 0; j < consumer_number; ++j)
            PIMCOMP_Intermediate_INFO["PIMCOMP_topology_provider_consumer_relation"][i][j] = PIMCOMP_topology_provider_consumer_relation[i][j];
    }
    //// PIMCOMP_topology_consumer_provider_relation
    PIMCOMP_Intermediate_INFO["PIMCOMP_topology_consumer_provider_relation"].resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        int provider_num = PIMCOMP_topology_consumer_provider_relation[i].size();
        PIMCOMP_Intermediate_INFO["PIMCOMP_topology_consumer_provider_relation"][i].resize(provider_num);
        for (int j = 0; j < provider_num; ++j)
            PIMCOMP_Intermediate_INFO["PIMCOMP_topology_consumer_provider_relation"][i][j] = PIMCOMP_topology_consumer_provider_relation[i][j];
    }

    std::string strJson = PIMCOMP_Intermediate_INFO.toStyledString();
    std::ofstream fob("../output/IntermediateInfo.json", std::ios::trunc | std::ios::out);
    if (fob.is_open())
    {
        fob.write(strJson.c_str(), strJson.length());
        fob.close();
    }
}
//
// Created by SXT on 2022/10/11.
//

#include "ElementPipelineSchedule.h"

//// KEY ASSUMPTION: FC and CONV nodes have only one provider

// bias_address_map[core_index][node_index] is the bias address of this node in this core
static std::vector<std::vector<long long>> bias_address_map;
// bias_element_num_map[core_index][node_index] is the bias element number of this node in this core
static std::vector<std::vector<int>> bias_element_num_map;
static int pipeline_effective_instruction_group_num;
ElementPipelineSchedule::ElementPipelineSchedule(std::string model_name_)
{
    node_num = PIMCOMP_node_list.size();
    model_name = model_name_;
    // last_node_index is the last node in the entire model, after which the result is written back to DRAM.
    // last_node_index需要是整个模型中最后一个节点，之后结果就写回DRAM。
    if (model_name == "alexnet")
    {
        last_node_index = 16;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "vgg16")
    {
        last_node_index = 26;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "resnet18")
    {
        last_node_index = 40;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "resnet34")
    {
        last_node_index = 72;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "resnet50")
    {
        last_node_index = 89;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "googlenet")
    {
        last_node_index = 89;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "squeezenet")
    {
        last_node_index = 39;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "inception_v3")
    {
        last_node_index = 121;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "mobilenetv2")
    {
        last_node_index = 100;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "shufflenet")
    {
        last_node_index = 104;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "caffenet")
    {
        last_node_index = 16;
        last_node_output_channel_num = 1;
    }
    else if (model_name == "rcnn-ilsvrc13")
    {
        last_node_index = 16;
        last_node_output_channel_num = 1;
    }
    else
    {
        fprintf(stderr, "Please Indicate The Last Node Index And Its Output Channel Num. \n");
        abort();
    }

    PIMCOMP_6_inter_core_communication.resize(ChipW * ChipH);
    for (int i = 0; i < ChipH * ChipW; ++i)
        PIMCOMP_6_inter_core_communication[i].resize(ChipW * ChipH);

    // 128kB / 16bit = 64k
//    MM.SetParameter(ChipH * ChipW, 64 * 1024);
    MM.SetParameter(ChipH * ChipW, 1024 * 1024 * 128);
}


/////////////////////////////////////// NODE ///////////////////////////////////////
std::set<std::string> no_consider_node_set = {"OP_INPUT", "OP_FLATTEN", "OP_RESHAPE", "OP_DROPOUT", "OP_LRN",
                                              "OP_SOFTMAX", "OP_TRANSPOSE", "OP_SQUEEZE", "OP_MATMUL", "OP_BN",
                                              "OP_CLIP", "OP_SQUEEZE", "OP_MATMUL"};
std::set<int> post_node_index;
std::vector<int> node_output_channel_num;
std::vector<std::vector<int>> node_AG_mapping;               // 记录每个节点有哪些AG (AG in each node)
std::vector<std::vector<int>> node_AG0_index_in_replication; // 记录每个节点若干个rep的AG0的AG_index_in_total (AG_index of several AG0 of each node)
std::vector<int> node_replication_num;                      // 记录生产-消费关系中每个结点的复制倍数（包括pool、vec等）(replication_num of each node)
std::vector<int> post_node_map;                             // 记录每个POST节点放到哪个核上进行 (POST node)
/////////////////////////////////////// SPLIT ///////////////////////////////////////
std::vector<std::vector<std::vector<int>>> node_rep_split_output_channel_index_list;  // node_rep_split_output_channel_index_list[node_index][replication_index] records output_channel_index_list
std::vector<std::vector<std::vector<int>>> node_rep_split_key_input_channel_index_list; // node_rep_split_key_input_channel_index_list[node_index][replication_index] records key_input_channel_index of output_channel_index
std::vector<std::vector<std::vector<int>>> node_rep_split_ready_input_channel_index_list;
std::vector<std::vector<std::vector<std::vector<int>>>> node_rep_split_ready_input_channel_index_list_for_vec;
std::vector<std::vector<int>> node_rep_split_produce_output_channel_num;
std::vector<std::vector<int>> node_rep_split_complete_output_channel_index_flag;
std::vector<std::vector<int>> node_rep_split_output_channel_num_list;

//// Whole AG Info
std::vector<struct AG_info_schedule> PIMCOMP_4_element_AG_info_list;

//// struct for Memory Allocation
std::vector<std::vector<std::vector<int>>> node_rep_AG_list; // node_rep_AG_list[node_index][replication_list] records a series of AG_index
std::vector<std::vector<int>> split_output_channel_to_rep_index_list; // mapping relationship from node's 'output_channel_index' to 'replication_index'

void ElementPipelineSchedule::SchedulePreparation()
{
    node_replication_num.resize(node_num);
    node_rep_AG_list.resize(node_num);

    //// construct an AG_info_list, with all AG useful info
    int AG_num = PIMCOMP_2_resource_info.AGs;
    int AG_input_element_num_in_window = 0;
    node_AG0_index_in_replication.resize(node_num);
    for (int i = 0; i < AG_num; ++i)
    {
        int core_index = PIMCOMP_3_hierarchy_map.whole[i][0].vcore;
        int rep_index = PIMCOMP_3_hierarchy_map.whole[i][0].replication_index;
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        int AG_index_in_total = PIMCOMP_3_hierarchy_map.whole[i][0].array_group_total;
        int AG_index_in_replication = PIMCOMP_3_hierarchy_map.whole[i][0].array_group_in_weight;
        int AG_num_per_replication = PIMCOMP_3_hierarchy_map.whole[i][0].AG_num_per_replication;
        int replication_index = PIMCOMP_3_hierarchy_map.whole[i][0].replication_index;
        bool first_layer = PIMCOMP_node_list[node_index].provider_index[0] == 0;
        if (first_layer)
            first_node_index = node_index;
        node_replication_num[node_index] = PIMCOMP_node_list[node_index].replication_num;
        if (node_rep_AG_list[node_index].size() != node_replication_num[node_index])
            node_rep_AG_list[node_index].resize(node_replication_num[node_index]);
        node_rep_AG_list[node_index][replication_index].push_back(AG_index_in_total);

        struct AG_info_schedule AGInfo;
        AGInfo.AG_index_in_total = AG_index_in_total;
        AGInfo.AG_index_in_replication = AG_index_in_replication;
        AGInfo.AG_num_per_replication = AG_num_per_replication;
        AGInfo.replication_index = rep_index;
        AGInfo.replication_num = PIMCOMP_node_list[node_index].replication_num;
        AGInfo.replication_num_origin = PIMCOMP_node_list[node_index].replication_num_origin;
        AGInfo.input_cycle_in_total = PIMCOMP_node_list[node_index].input_cycle_in_total;
        AGInfo.core_index = core_index;
        AGInfo.node_index = node_index;
        AGInfo.first_layer = first_layer;

        int effective_node_index = PIMCOMP_node_list[node_index].effective_node_index;
        int crossbar_num_AG = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list.size();
        int crossbar_start_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[0];
        int crossbar_end_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[crossbar_num_AG - 1];
        int input_element_num = PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_start + 1;
        int output_element_num = PIMCOMP_2_virtual_crossbar[crossbar_end_index].width_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].width_start + 1;

        AGInfo.input_element_num = input_element_num;
        AGInfo.output_element_num = output_element_num;

        // add for memory allocation
        if (AG_index_in_replication == 0)
            AG_input_element_num_in_window = 0;
        AGInfo.start_input_element_num_in_window = AG_input_element_num_in_window;
        AGInfo.end_input_element_num_in_window = AG_input_element_num_in_window + input_element_num - 1;
        AG_input_element_num_in_window += input_element_num;
        // save this info in PIMCOMP_4_virtual_core_AG_map
        for (int j = 0; j < PIMCOMP_4_virtual_core_AG_map.core_list[core_index].AG_list.size(); ++j)
        {
            if (PIMCOMP_4_virtual_core_AG_map.core_list[core_index].AG_list[j].AG_index_in_total == AG_index_in_total)
            {
                PIMCOMP_4_virtual_core_AG_map.core_list[core_index].AG_list[j].start_input_element_num_in_window = AGInfo.start_input_element_num_in_window;
                PIMCOMP_4_virtual_core_AG_map.core_list[core_index].AG_list[j].end_input_element_num_in_window = AGInfo.end_input_element_num_in_window;
                break;
            }
        }

        PIMCOMP_4_element_AG_info_list.push_back(AGInfo);

        // Record AG_index_in_total for each AG0 of each node.
        // 记录每个结点、每个rep中AG0的AG_index_in_total。
        if (AG_index_in_replication == 0)
            node_AG0_index_in_replication[node_index].push_back(AG_index_in_total);
    }

//    std::cout << "provider - consumer" << std::endl;
//    for (int i = 0; i < node_num; ++i)
//    {
//        for (int j = 0; j < PIMCOMP_topology_provider_consumer_relation[i].size(); ++j)
//        {
//            std::cout << i << " :" << PIMCOMP_topology_provider_consumer_relation[i][j] << std::endl;
//        }
//    }

    //// Other AG Information（for data preparation for CONV and FC layer）
    node_AG_mapping.resize(node_num);
    for (int i = 0; i < AG_num; ++i)
    {
        int AG_index = PIMCOMP_3_hierarchy_map.whole_index[i];
        int node_index = PIMCOMP_3_hierarchy_map.whole[i][0].node_index;
        node_AG_mapping[node_index].push_back(AG_index);
    }

    srand((unsigned)time(NULL));
    post_node_map.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list[i].operation;
        if (operation != "OP_CONV" && operation != "OP_FC")
        {
            post_node_map[i] = i % (ChipH * ChipW);
            std::cout << "node:" << i  << "  operation:" << operation << "  core:" << post_node_map[i] << std::endl;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////// SPLIT //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    node_rep_split_output_channel_index_list.resize(node_num);
    node_rep_split_key_input_channel_index_list.resize(node_num);
    node_rep_split_ready_input_channel_index_list.resize(node_num);
    node_rep_split_ready_input_channel_index_list_for_vec.resize(node_num);
    node_rep_split_produce_output_channel_num.resize(node_num);
    node_rep_split_complete_output_channel_index_flag.resize(node_num);
    node_rep_split_output_channel_num_list.resize(node_num);
    split_output_channel_to_rep_index_list.resize(node_num);
    for (int i = 1; i < node_num; ++i)
    {
        if (PIMCOMP_topology_provider_consumer_relation[i].size() > 0)
        {
            int replication_num = node_replication_num[i];
            std::string operation = PIMCOMP_node_list[i].operation;
            if (no_consider_node_set.count(operation))
                continue;
            if (operation != "OP_FC" && operation != "OP_CONV")
            {
                replication_num = 1;
                node_replication_num[i] = 1;
            }
            node_rep_split_output_channel_index_list[i].resize(replication_num);
            node_rep_split_key_input_channel_index_list[i].resize(replication_num);
            node_rep_split_ready_input_channel_index_list[i].resize(replication_num);
            node_rep_split_ready_input_channel_index_list_for_vec[i].resize(replication_num);
            node_rep_split_produce_output_channel_num[i].resize(replication_num);
            node_rep_split_output_channel_num_list[i].resize(replication_num);
            int output_channel_num;
            if (PIMCOMP_node_list[i].operation == "OP_FC")
                output_channel_num = 1;
            else
                output_channel_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            split_output_channel_to_rep_index_list[i].resize(output_channel_num);
            node_rep_split_complete_output_channel_index_flag[i].resize(output_channel_num);
            //// For each node, each replication block, select the output_channel_index they need to be responsible for, and determine the key_input_channel_index
            //// 为每个结点，每个复制倍数，选定它们需要负责的output_channel_index，同时确定key_input_channel_index
            for (int j = 0; j < output_channel_num; ++j)
            {
                int rep_index = j % replication_num;
                node_rep_split_output_channel_index_list[i][rep_index].push_back(j);
                node_rep_split_output_channel_num_list[i][rep_index]++;
                split_output_channel_to_rep_index_list[i][j] = rep_index;

                int key_input_channel_index;
                if (PIMCOMP_node_list[i].operation == "OP_FC")
                {
                    //// FC has only one provider
                    int provider_node_index = PIMCOMP_topology_consumer_provider_relation[i][0];
                    //// Find the previous conv or pool or other producer of the fc, which is related to key_input_channel_index
                    //// 找到该fc的前面conv或pool或fc的生产者，这关系到key_input_channel_index
                    while (PIMCOMP_node_list[provider_node_index].operation != "OP_CONV" && PIMCOMP_node_list[provider_node_index].operation != "OP_POOL" && PIMCOMP_node_list[provider_node_index].operation != "OP_FC")
                        provider_node_index = PIMCOMP_topology_consumer_provider_relation[provider_node_index][0];
                    //// If FC is preceded by CONV, key_channel_index = output_channel_num-1. It means that you need to wait for the previous layer to pass through all the results before processing. And if the front of FC is FC, it is 0, as long as the previous FC is passed once.
                    //// 如果FC的前面是CONV，则key_channel_index是output_channel_num-1。意思是需要等前一个层把全部结果都穿过来之后再进行处理。而如果FC的前面是FC，则为0，只要前面FC传一次就够。
                    if (PIMCOMP_node_list[provider_node_index].operation == "OP_CONV" || PIMCOMP_node_list[provider_node_index].operation == "OP_POOL")
                    {
                        int tmp_output_channel_num = PIMCOMP_node_list[provider_node_index].output_dim[2] * PIMCOMP_node_list[provider_node_index].output_dim[3];
                        key_input_channel_index = tmp_output_channel_num - 1; // output_channel_index = output_channel_num - 1
                    }
                    else if (PIMCOMP_node_list[provider_node_index].operation == "OP_FC")
                    {
                        key_input_channel_index = 0;
                    }
                }
                else if (PIMCOMP_node_list[i].operation == "OP_CONV" || PIMCOMP_node_list[i].operation == "OP_POOL")
                {
                    key_input_channel_index = GetInputChannelFromOutputIndex(i, j, 1);
                }
                else
                {
                    key_input_channel_index = j;
                }
                node_rep_split_key_input_channel_index_list[i][rep_index].push_back(key_input_channel_index);
            }

            int input_channel_num;
            if (PIMCOMP_node_list[i].operation == "OP_FC")
                input_channel_num = 50000;
            else
                input_channel_num = PIMCOMP_node_list[i].input_dim[2] * PIMCOMP_node_list[i].input_dim[3];
            for (int j = 0; j < replication_num; ++j)
            {
                node_rep_split_ready_input_channel_index_list[i][j].resize(input_channel_num);
                // Add For VEC Node
                if (PIMCOMP_node_list[i].operation != "OP_FC" && PIMCOMP_node_list[i].operation != "OP_CONV"
                    && PIMCOMP_node_list[i].operation != "OP_POOL" && (!no_consider_node_set.count(PIMCOMP_node_list[i].operation))) // VEC
                {
                    int provider_num = PIMCOMP_topology_consumer_provider_relation[i].size();
                    node_rep_split_ready_input_channel_index_list_for_vec[i][j].resize(provider_num);
                    for (int k = 0; k < provider_num; ++k)
                        node_rep_split_ready_input_channel_index_list_for_vec[i][j][k].resize(input_channel_num);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////// OTHERS: node_output_channel_num /////////////////////////////////////////
    node_output_channel_num.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_FC")
            node_output_channel_num[i] = 1;
        else if (no_consider_node_set.count(PIMCOMP_node_list[i].operation))
            node_output_channel_num[i] = 0;
        else
            node_output_channel_num[i] = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
    }
}


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Memory Allocation ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


////////////////////////// PIMCOMP_4_Element_Memory_Dependency //////////////////////////
struct PIMCOMP_4_Element_Memory_Dep_input_segment
{
    int input_cycle_index;
    int AG_index;
    int start_element_index_in_AG;
    int end_element_index_in_AG;
    int start_element_index_in_channel;
    int end_element_index_in_channel;
    int input_channel_index;
    bool padding = false;
};

struct PIMCOMP_4_Element_Memory_Dep_input_cycle_info
{
    std::vector<struct PIMCOMP_4_Element_Memory_Dep_input_segment> input_segment;
};

struct PIMCOMP_4_Element_Memory_Dep_AG_list
{
    std::vector<struct PIMCOMP_4_Element_Memory_Dep_input_cycle_info> input_cycle_list;
};


struct PIMCOMP_4_Element_Memory_Dep_core_list
{
    std::vector<int> core_list;
};

struct PIMCOMP_4_Element_Memory_Dep_input_channel_list
{
    int input_channel_element_num;
    std::vector<struct PIMCOMP_4_Element_Memory_Dep_core_list> input_channel_list;
};


struct PIMCOMP_Element_Memory_Dep_pool_input_channel
{
    // The length of flag_core_list is Chip*Chip, which records how many times an input_channel needs to be used in each core before it can be released
    // flag_core_list长度是Chip*Chip，记录了一个input_channel在各个核中都需要使用多少次然后才能被回收
    std::vector<int> flag_core_list;
    // effective_core_list records which cores an input_channel is sent to
    // effective_core_list记录了一个input_channel要发送到哪些核中
    std::set<int> effective_core_list;
};

struct PIMCOMP_Element_Memory_Dep_pool_info
{
    std::vector<struct PIMCOMP_Element_Memory_Dep_pool_input_channel> input_channel_list;
};

struct PIMCOMP_4_Element_Memory_Dep
{
    std::vector<struct PIMCOMP_Element_Memory_Dep_pool_info> pool_list;
    std::vector<struct PIMCOMP_4_Element_Memory_Dep_input_channel_list> node_list;
    std::vector<struct PIMCOMP_4_Element_Memory_Dep_AG_list> AG_list;
} PIMCOMP_4_Element_Memory_Dependency;

void ElementPipelineSchedule::PrepareDependency()
{
    PIMCOMP_4_Element_Memory_Dependency.AG_list.resize(PIMCOMP_2_resource_info.AGs);
    PIMCOMP_4_Element_Memory_Dependency.pool_list.resize(node_num);
    for (int i = PIMCOMP_4_element_AG_info_list.size()-1; i >= 0 ; --i)
    {
        int input_cycle_in_total = PIMCOMP_4_element_AG_info_list[i].input_cycle_in_total;
        PIMCOMP_4_Element_Memory_Dependency.AG_list[i].input_cycle_list.resize(input_cycle_in_total);
    }

    PIMCOMP_4_Element_Memory_Dependency.node_list.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list[i].operation;
        int node_index = i;
        if (operation == "OP_CONV")
        {
            int input_channel_num = PIMCOMP_node_list[i].input_dim[2] * PIMCOMP_node_list[i].input_dim[3];
            int input_channel_element_num = PIMCOMP_node_list[node_index].param.input_channel;
            int output_channel_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list.resize(input_channel_num);
            PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_element_num = input_channel_element_num;
//            std::cout << "node:" << i << " " << operation << std::endl;
            for (int j = 0; j < input_channel_num; ++j)
            {
                std::vector<int> core_list;
                core_list.resize(ChipH * ChipW);
                int input_channel_index = j;
//                std::cout << "  input_channel:" << j << std::endl;
                int related_output_channel_num = PIMCOMP_conv_pool_input_output_info[node_index].input_index[input_channel_index].size();
                for (int m = 0; m < related_output_channel_num; ++m)
                {
                    int related_output_channel_index = PIMCOMP_conv_pool_input_output_info[node_index].input_index[input_channel_index][m];
//                    std::cout << "    related_output_channel: " << related_output_channel_index << "       ";
                    int input_channel_index_in_window = 0;
                    for (int n = 0; n < PIMCOMP_conv_pool_full_output_info[node_index].output_index[related_output_channel_index].size(); ++n)
                    {
//                        std::cout << PIMCOMP_conv_pool_full_output_info[node_index].output_index[related_output_channel_index][n] << " ";
                        if (PIMCOMP_conv_pool_full_output_info[node_index].output_index[related_output_channel_index][n] == input_channel_index)
                            input_channel_index_in_window = n;
                    }
//                    std::cout << "  channel_index_in_window: " << input_channel_index_in_window;
                    int start_element_of_this_channel_in_window = input_channel_index_in_window * input_channel_element_num;
                    int end_element_of_this_channel_in_window = (input_channel_index_in_window+1) * input_channel_element_num - 1;
//                    std::cout << "  from element: " << start_element_of_this_channel_in_window << "   to element " << end_element_of_this_channel_in_window;
                    // The input cycle of the node has several segments, and each segment corresponds to a series of AGs. The relationship between each segment and several AGs is represented by a map.
                    // 该node的input cycle具有若干个分段，每个分段都对应一系列AG。每个分段与若干个AG对应的关系用map来表示。
                    int replication_index = split_output_channel_to_rep_index_list[node_index][related_output_channel_index];
                    int probable_AG_num = node_rep_AG_list[node_index][replication_index].size();
                    // Here iter->second is the AG_list corresponding to the segment.
                    // 这里的iter->second就是该分段所对应的AG_list
                    for (int n = 0; n < probable_AG_num; ++n)
                    {
                        int probable_AG_index = node_rep_AG_list[node_index][replication_index][n];
                        int start_effective = -1;
                        int end_effective = -1;
                        int start_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].start_input_element_num_in_window;
                        int end_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].end_input_element_num_in_window;
                        if (end_element_of_this_AG_in_window >= start_element_of_this_channel_in_window && end_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                        {
                            if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                            {
                                start_effective = start_element_of_this_AG_in_window;
                                end_effective = end_element_of_this_AG_in_window;
                            }
                            else
                            {
                                start_effective = start_element_of_this_channel_in_window;
                                end_effective = end_element_of_this_AG_in_window;
                            }
                        }
                        else if (end_element_of_this_AG_in_window > end_element_of_this_channel_in_window && start_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                        {
                            if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                            {
                                start_effective = start_element_of_this_AG_in_window;
                                end_effective = end_element_of_this_channel_in_window;
                            }
                            else
                            {
                                start_effective = start_element_of_this_channel_in_window;
                                end_effective = end_element_of_this_channel_in_window;
                            }
                        }
                        if (start_effective != -1 && end_effective != -1)
                        {
                            struct PIMCOMP_4_Element_Memory_Dep_input_segment base_info;
                            base_info.padding = false;
                            base_info.input_channel_index = input_channel_index;
                            base_info.AG_index = probable_AG_index;
                            base_info.input_cycle_index = related_output_channel_index;
                            int AG_input_element_num = PIMCOMP_4_element_AG_info_list[probable_AG_index].input_element_num;
                            base_info.start_element_index_in_AG = start_effective % CrossbarH % AG_input_element_num;
                            base_info.end_element_index_in_AG = end_effective % CrossbarH % AG_input_element_num;
                            base_info.start_element_index_in_channel = start_effective % input_channel_element_num;
                            base_info.end_element_index_in_channel = end_effective % input_channel_element_num;
                            PIMCOMP_4_Element_Memory_Dependency.AG_list[probable_AG_index].input_cycle_list[related_output_channel_index].input_segment.push_back(base_info);
                            core_list[(PIMCOMP_4_element_AG_info_list[probable_AG_index].core_index)] ++;
//                            if (probable_AG_index == 0 && related_output_channel_index == 0)
//                                std::cout << "  probably related AG: " << probable_AG_index << "  AG_input_element_num:" << AG_input_element_num
//                                          << "  input_channel_index:" << input_channel_index
//                                          << "  input_cycle_index:" << related_output_channel_index
//                                          << "  ori:"  << start_effective << "-"  << end_effective
//                                          << "  AG:" << base_info.start_element_index_in_AG  << "-" << base_info.end_element_index_in_AG
//                                          << "  channel:" << base_info.start_element_index_in_channel << "-" << base_info.end_element_index_in_channel << std::endl;
                        }
                    }

//                    std::cout << std::endl;
                }
                for (auto iter = core_list.begin(); iter != core_list.end(); ++iter)
                {
                    PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list[j].core_list.push_back(*iter);
                }
            }
        }
        else if (operation == "OP_FC")
        {
            //// The special case here is because the front of FC may be POOL or CONV. Special handling is required for this type of FC. A POOL or CONV changes the input of the last layer of FC to 1*1. So special handling is required.
            //// 这里特殊处理是因为FC前面可能是POOL或CONV。对于这类FC需要特殊处理。虽然最后一层FC前虽然不是CONV，但是有一个POOL，将最后一层FC的输入变为1*1。所以需要特殊处理。
            int provider_node_index = PIMCOMP_topology_consumer_provider_relation[node_index][0]; //// 应该没有问题，FC的生产者一般只有一个。
            //// Find the previous conv or pool or other producer of the fc, which is related to key_input_channel_index
            //// 找到该fc的前面conv或pool或fc的生产者，这关系到key_input_channel_index
            while (PIMCOMP_node_list[provider_node_index].operation != "OP_CONV" && PIMCOMP_node_list[provider_node_index].operation != "OP_POOL" && PIMCOMP_node_list[provider_node_index].operation != "OP_FC")
                provider_node_index = PIMCOMP_topology_consumer_provider_relation[provider_node_index][0];
            //// If FC is preceded by CONV, key_channel_index = output_channel_num-1. It means that you need to wait for the previous layer to pass through all the results before processing. And if the front of FC is FC, it is 0, as long as the previous FC is passed once.
            //// 如果FC的前面是CONV，则key_channel_index是output_channel_num-1。意思是需要等前一个层把全部结果都穿过来之后再进行处理。而如果FC的前面是FC，则为0，只要前面FC传一次就够。
            int input_channel_num = 0;
            int input_channel_element_num = 0;
            if (PIMCOMP_node_list[provider_node_index].operation == "OP_CONV" || PIMCOMP_node_list[provider_node_index].operation == "OP_POOL")
            {
                input_channel_num = PIMCOMP_node_list[provider_node_index].output_dim[2] * PIMCOMP_node_list[provider_node_index].output_dim[3];
                input_channel_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
            }
            else if (PIMCOMP_node_list[provider_node_index].operation == "OP_FC")
            {
                input_channel_num = 1;
                input_channel_element_num = PIMCOMP_node_list[i].param.num_input;
            }


            PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list.resize(input_channel_num);
            PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_element_num = input_channel_element_num;
            for (int j = 0; j < input_channel_num; ++j)
            {
                std::vector<int> core_list;
                core_list.resize(ChipW * ChipH);
                int start_element_of_this_channel_in_window = j * input_channel_element_num;
                int end_element_of_this_channel_in_window = (j+1) * input_channel_element_num - 1;
                // The node__input_cycle__AG_list[i] of the FC layer has only one segment, which includes all AGs.
                // FC层的node__input_cycle__AG_list[i]只有一段，其中包括全部的AG。
                for (int k = 0; k < node_AG_mapping[i].size(); ++k)
                {
                    int start_effective = -1;
                    int end_effective = -1;
                    int probable_AG_index = node_AG_mapping[i][k];
                    int start_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].start_input_element_num_in_window;
                    int end_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].end_input_element_num_in_window;
                    if (end_element_of_this_AG_in_window >= start_element_of_this_channel_in_window && end_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                    {
                        if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                        {
                            start_effective = start_element_of_this_AG_in_window;
                            end_effective = end_element_of_this_AG_in_window;
                        }
                        else
                        {
                            start_effective = start_element_of_this_channel_in_window;
                            end_effective = end_element_of_this_AG_in_window;
                        }
                    }
                    else if (end_element_of_this_AG_in_window > end_element_of_this_channel_in_window && start_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                    {
                        if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                        {
                            start_effective = start_element_of_this_AG_in_window;
                            end_effective = end_element_of_this_channel_in_window;
                        }
                        else
                        {
                            start_effective = start_element_of_this_channel_in_window;
                            end_effective = end_element_of_this_channel_in_window;
                        }
                    }
                    if (start_effective != -1 && end_effective != -1)
                    {
                        struct PIMCOMP_4_Element_Memory_Dep_input_segment base_info;
                        base_info.padding = false;
                        base_info.input_channel_index = j;
                        base_info.AG_index = probable_AG_index;
                        base_info.input_cycle_index = 0;
                        int AG_input_element_num = PIMCOMP_4_element_AG_info_list[probable_AG_index].input_element_num;
                        base_info.start_element_index_in_AG = start_effective % CrossbarH % AG_input_element_num;
                        base_info.end_element_index_in_AG = end_effective % CrossbarH % AG_input_element_num;
                        base_info.start_element_index_in_channel = start_effective % input_channel_element_num;
                        base_info.end_element_index_in_channel = end_effective % input_channel_element_num;
                        PIMCOMP_4_Element_Memory_Dependency.AG_list[probable_AG_index].input_cycle_list[0].input_segment.push_back(base_info);
                        core_list[PIMCOMP_4_element_AG_info_list[probable_AG_index].core_index]++;
//                        std::cout << "node:" << i << "  AG: " << AG_index  << "  input_channel_index:" << j
//                            << "  ori:"  << start_effective << "-"  << end_effective
//                            << "  AG:" << base_info.start_element_index_in_AG  << "-" << base_info.end_element_index_in_AG
//                            << "  channel:" << base_info.start_element_index_in_channel << "-" << base_info.end_element_index_in_channel << std::endl;
                    }
                }
                for (auto iter = core_list.begin(); iter != core_list.end(); ++iter)
                {
                    PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list[j].core_list.push_back(*iter);
                }
            }
        }
    }
}


////////////////////////// PIMCOMP_4_Element_Memory_INFO //////////////////////////
struct PIMCOMP_4_Element_Memory_type_2_base_info
{
    long long start_address = -1;
};

struct PIMCOMP_4_Element_Memory_type_2_vec_output_channel_info
{

    std::vector<struct PIMCOMP_4_Element_Memory_type_2_base_info> input_channel_list;
};

struct PIMCOMP_4_Element_Memory_type_2_vec_info
{
//    long long output_start_address = -1;
    int output_channel_element_num = 0;
    std::vector<struct PIMCOMP_4_Element_Memory_type_2_vec_output_channel_info> provider_list;
    std::vector<long long> output_start_address_record;
};


struct PIMCOMP_4_Element_Memory_type_1_input_channel_info
{
    long long start_address = -1;
    int rest_use_times = 0;
};

struct PIMCOMP_4_Element_Memory_type_1_main_info
{
    int input_channel_element_num = 0;
    std::vector<struct PIMCOMP_4_Element_Memory_type_1_input_channel_info> input_channel_list;
};

struct PIMCOMP_4_Element_Memory_type_3_pool_info // 基本与type_1_main_info一致
{
//    long long output_start_address = -1;
    int output_channel_element_num = 0;
    int input_channel_element_num = 0;
    std::vector<struct PIMCOMP_4_Element_Memory_type_1_input_channel_info> input_channel_list;
    std::vector<long long> output_start_address_record;
};

struct PIMCOMP_4_Element_Memory_type_0_AG_info
{
    // output_start_address and output_element_num. Because each AG on each core has a fixed location for saving output.
    // output_start_address 和 output_element_num是因为每个核上每个AG都有一个固定保存output的位置。
    // The results of different output_channels are stored here. [Don't worry about being covered. Because the result will be passed to the next node first]
    // 不同output_channel的结果都保存在这里。[不担心被覆盖。因为结果会先传到下一节点]
    int input_start_address = -1;
    int input_element_num = 0;
    int output_start_address = -1;
    int output_element_num = 0;
};

struct PIMCOMP_4_Element_Memory_core_info
{
    std::vector<struct PIMCOMP_4_Element_Memory_type_3_pool_info> type_3_pool_list; // POOL节点的输入、输出信息 (POOL input and output info)
    std::vector<struct PIMCOMP_4_Element_Memory_type_2_vec_info> type_2_vec_list;   // VEC节点的输入、输出信息  (VEC input and output info)
    std::vector<struct PIMCOMP_4_Element_Memory_type_1_main_info> type_1_main_list; // CONV和FC节点的输入信息  (CONV/FC input info)
    std::vector<struct PIMCOMP_4_Element_Memory_type_0_AG_info> type_0_AG_list;     // CONV和FC节点的输出信息 (以AG为单位进行保存) (CONV/FC output info)
};

struct PIMCOMP_4_Element_Memory_INFO
{
    std::vector<struct PIMCOMP_4_Element_Memory_core_info> core_list;
} PIMCOMP_4_Element_Memory_INFO;

struct OP_output_channel_split_info_begin_end
{
    int start_output_channel_index;
    int end_output_channel_index;
};

std::vector<std::vector<int>> POOL_input_channel_flag;
std::vector<std::vector<std::vector<int>>> VEC_input_channel_flag;
// Record the relationship from output_channel to core (POST nodes such as pool and vec)
// 记录从output_channel到core的关系 (pool和vec等POST节点)
std::vector<std::vector<int>> PIMCOMP_4_Element_Memory_type_2_output_channel_to_core;

void ElementPipelineSchedule::PrepareMemoryINFO()
{
    PIMCOMP_4_Element_Memory_INFO.core_list.resize(ChipW * ChipH);
    POOL_input_channel_flag.resize(node_num);
    VEC_input_channel_flag.resize(node_num);
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        PIMCOMP_4_Element_Memory_INFO.core_list[i].type_3_pool_list.resize(node_num);
        PIMCOMP_4_Element_Memory_INFO.core_list[i].type_2_vec_list.resize(node_num);
        PIMCOMP_4_Element_Memory_INFO.core_list[i].type_1_main_list.resize(node_num);
        PIMCOMP_4_Element_Memory_INFO.core_list[i].type_0_AG_list.resize(PIMCOMP_2_resource_info.AGs);
    }
    PIMCOMP_4_Element_Memory_type_2_output_channel_to_core.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list[i].operation;
        // Record which cores the node is scattered on
        // 记录该节点分散在哪些核上
        std::set<int> node_split_info_core_list;
        // 这里需要注意的是split可能是不连续的几段。比如rep0的第一个AG和rep3的第一个AG在同一核上。因此需要一个OP_output_channel_split_info的vector来存储多个split_info。
        // node_split_info_begin_end是一个map，记录了节点i在若干核上的若干个split。key是core_index，value是在该core上的任务（可能是若干个任务）。
        std::map<int, std::vector<struct OP_output_channel_split_info_begin_end>> node_split_info_begin_end;
        // 从output_channel_index到core的映射
        if (PIMCOMP_node_list[i].output_dim_num == 4)
            PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].resize(PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3]);
        else
            PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].resize(1);

        //// VEC operator and POOL operator (这里要包括上CONV和FC之后的RELU层)
        if (operation != "OP_CONV"  && operation != "OP_FC"  && !no_consider_node_set.count(operation))
        {
            //// 原来写的不符合了。目前认为一个POST节点只在一个核上完成。所以在下面重写了。
//            // 找到VEC算子对应的CONV生产者
//            int conv_consumer_node_index = PIMCOMP_node_list[i].AG0_node_index;
//            int conv_consumer_replication_num = node_replication_num[conv_consumer_node_index];
//            // VEC的分散数目和其CONV生产者的复制倍数一致
//            for (int j = 0; j < conv_consumer_replication_num; ++j)
//            {
//                int AG0_index_this_replication = node_AG0_index_in_replication[conv_consumer_node_index][j];
//                int AG0_core_index = PIMCOMP_4_element_AG_info_list[AG0_index_this_replication].core_index;
//                int start_channel_index, end_channel_index;
//                if (operation != "OP_POOL")
//                {
//                    start_channel_index = vec_rep_min_max_channel_index[i][j][0];
//                    end_channel_index = vec_rep_min_max_channel_index[i][j][1];
//                }
//                else
//                {
//                    start_channel_index = pool_rep_min_max_output_channel_index[i][j][0];
//                    end_channel_index = pool_rep_min_max_output_channel_index[i][j][1];
//                }
//                node_split_info_core_list.insert(AG0_core_index);
//                struct OP_output_channel_split_info_begin_end tmp_split_info;
//                tmp_split_info.start_output_channel_index = start_channel_index;
//                tmp_split_info.end_output_channel_index = end_channel_index;
//                node_split_info_begin_end[AG0_core_index].push_back(tmp_split_info);
//            }

            //// 重新写这一部分
//            int AG0_index_in_total = PIMCOMP_node_list[i].AG0_index_in_total;
//            int AG0_core_index = PIMCOMP_4_element_AG_info_list[AG0_index_in_total].core_index;
//            node_split_info_core_list.insert(AG0_core_index);

            int AG0_core_index = post_node_map[i];
            node_split_info_core_list.insert(AG0_core_index);

            int start_channel_index = 0;
            int end_channel_index = PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].size();
            struct OP_output_channel_split_info_begin_end tmp_split_info;
            tmp_split_info.start_output_channel_index = start_channel_index;
            tmp_split_info.end_output_channel_index = end_channel_index;
            node_split_info_begin_end[AG0_core_index].push_back(tmp_split_info);

            // node_split_info_begin_end记录了多个片段。其中键为core_index,值是每一段的begin和end
            for(auto iter = node_split_info_begin_end.begin(); iter != node_split_info_begin_end.end(); iter++)
            {
                for (int j = 0; j < iter->second.size(); ++j)
                {
                    for (int k = iter->second[j].start_output_channel_index; k < iter->second[j].end_output_channel_index; ++k)
                    {
                        PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i][k] = iter->first;
                    }
                }
            }

            if (operation != "OP_POOL")
            {
                int provider_num = PIMCOMP_node_list[i].provider_num;
                for (auto iter = node_split_info_core_list.begin(); iter != node_split_info_core_list.end(); iter++)
                {
                    int core_index = *iter;
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_2_vec_list[i].output_channel_element_num = PIMCOMP_node_list[i].output_dim[1];
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_2_vec_list[i].provider_list.resize(provider_num);
                    int output_channel_num = PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].size();
                    for (int j = 0; j < provider_num; ++j)
                    {
                        PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_2_vec_list[i].provider_list[j].input_channel_list.resize(output_channel_num);
                    }
                }
                VEC_input_channel_flag[i].resize(provider_num);
                for (int j = 0; j < provider_num; ++j)
                    VEC_input_channel_flag[i][j].resize(PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].size());
            }
            else
            {
                int total_input_channel_num = PIMCOMP_node_list[i].input_dim[2] * PIMCOMP_node_list[i].input_dim[3];
                POOL_input_channel_flag[i].resize(total_input_channel_num);
                PIMCOMP_4_Element_Memory_Dependency.pool_list[i].input_channel_list.resize(total_input_channel_num);
                for (int j = 0; j < total_input_channel_num; ++j)
                    PIMCOMP_4_Element_Memory_Dependency.pool_list[i].input_channel_list[j].flag_core_list.resize(ChipW * ChipH);

                int output_channel_num = PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i].size();
                for (int j = 0; j < output_channel_num; ++j)
                {
                    int execution_core_index = PIMCOMP_4_Element_Memory_type_2_output_channel_to_core[i][j];
                    int related_input_channel_num = PIMCOMP_conv_pool_input_output_info[i].output_index[j].size();
                    for (int k = 0; k < related_input_channel_num; ++k)
                    {
                        int related_input_channel_index = PIMCOMP_conv_pool_input_output_info[i].output_index[j][k];
                        PIMCOMP_4_Element_Memory_Dependency.pool_list[i].input_channel_list[related_input_channel_index].flag_core_list[execution_core_index] ++;
                        PIMCOMP_4_Element_Memory_Dependency.pool_list[i].input_channel_list[related_input_channel_index].effective_core_list.insert(execution_core_index);
                    }
                }

                for (auto iter = node_split_info_core_list.begin(); iter != node_split_info_core_list.end(); iter++)
                {
                    int core_index = *iter;
                    int input_channel_element_num = PIMCOMP_node_list[i].input_dim[1];
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_3_pool_list[i].input_channel_list.resize(total_input_channel_num);
                    for (int j = 0; j < total_input_channel_num; ++j)
                    {
                        PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_3_pool_list[i].input_channel_list[j].rest_use_times = PIMCOMP_4_Element_Memory_Dependency.pool_list[i].input_channel_list[j].flag_core_list[core_index];
                    }
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_3_pool_list[i].input_channel_element_num = input_channel_element_num;
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_3_pool_list[i].output_channel_element_num = input_channel_element_num;
                }
            }
        }
        else if (operation == "OP_CONV" || operation == "OP_FC")
        {
            int input_channel_num = PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list.size();
            // 先初始化一遍
            for (int k = 0; k < ChipH * ChipW; ++k)
                for (int j = 0; j < input_channel_num; ++j)
                    if (PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list[j].core_list[k] != 0) // 第i个节点的第j个input_channel在第k个核上会被利用的次数
                    {
                        PIMCOMP_4_Element_Memory_INFO.core_list[k].type_1_main_list[i].input_channel_list.resize(input_channel_num);
                        PIMCOMP_4_Element_Memory_INFO.core_list[k].type_1_main_list[i].input_channel_element_num = PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_element_num;
                        break;
                    }
            // 再对于rest_use进行赋值
            for (int j = 0; j < input_channel_num; ++j)
                for (int k = 0; k < ChipH * ChipW; ++k)
                    if (PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list[j].core_list[k] != 0)
                        PIMCOMP_4_Element_Memory_INFO.core_list[k].type_1_main_list[i].input_channel_list[j].rest_use_times = PIMCOMP_4_Element_Memory_Dependency.node_list[i].input_channel_list[j].core_list[k];
        }
    }
}

void ElementPipelineSchedule::PreparePadding()
{
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            int output_channel_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            int input_channel_element_num = PIMCOMP_node_list[i].input_dim[1];
            for (int j = 0; j < output_channel_num; ++j)
            {
                for (int k = 0; k < PIMCOMP_conv_pool_full_output_info[i].output_index[j].size(); ++k)
                {
                    if (PIMCOMP_conv_pool_full_output_info[i].output_index[j][k] == -1)
                    {
                        int start_element_of_this_channel_in_window = k * input_channel_element_num;
                        int end_element_of_this_channel_in_window = (k+1) * input_channel_element_num - 1;

                        int replication_index = split_output_channel_to_rep_index_list[i][j];
                        int probable_AG_num = node_rep_AG_list[i][replication_index].size();
                        for (int n = 0; n < probable_AG_num; ++n)
                        {
                            int probable_AG_index = node_rep_AG_list[i][replication_index][n];
                            int start_effective = -1;
                            int end_effective = -1;
                            int start_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].start_input_element_num_in_window;
                            int end_element_of_this_AG_in_window = PIMCOMP_4_element_AG_info_list[probable_AG_index].end_input_element_num_in_window;
                            if (end_element_of_this_AG_in_window >= start_element_of_this_channel_in_window && end_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                            {
                                if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                                {
                                    start_effective = start_element_of_this_AG_in_window;
                                    end_effective = end_element_of_this_AG_in_window;
                                }
                                else
                                {
                                    start_effective = start_element_of_this_channel_in_window;
                                    end_effective = end_element_of_this_AG_in_window;
                                }
                            }
                            else if (end_element_of_this_AG_in_window > end_element_of_this_channel_in_window && start_element_of_this_AG_in_window <= end_element_of_this_channel_in_window)
                            {
                                if (start_element_of_this_AG_in_window >= start_element_of_this_channel_in_window)
                                {
                                    start_effective = start_element_of_this_AG_in_window;
                                    end_effective = end_element_of_this_channel_in_window;
                                }
                                else
                                {
                                    start_effective = start_element_of_this_channel_in_window;
                                    end_effective = end_element_of_this_channel_in_window;
                                }
                            }
                            if (start_effective != -1 && end_effective != -1)
                            {
                                //// 加入DEP数据结构中
                                struct PIMCOMP_4_Element_Memory_Dep_input_segment dep_base_info;
                                int AG_input_element_num = PIMCOMP_4_element_AG_info_list[probable_AG_index].input_element_num;
                                dep_base_info.padding = true;
                                dep_base_info.AG_index = probable_AG_index;
                                dep_base_info.input_cycle_index = j;
                                dep_base_info.start_element_index_in_AG = start_effective % CrossbarH % AG_input_element_num;
                                dep_base_info.end_element_index_in_AG = end_effective % CrossbarH % AG_input_element_num;
                                dep_base_info.start_element_index_in_channel = 0;
                                dep_base_info.end_element_index_in_channel = input_channel_element_num-1;
                                PIMCOMP_4_Element_Memory_Dependency.AG_list[probable_AG_index].input_cycle_list[j].input_segment.push_back(dep_base_info);

//                                if (probable_AG_index == 0 && j == 0)
//                                    std::cout << "node:" << i
//                                              << "  AG_index:" << probable_AG_index
//                                              << "  input_cycle_index:" << j
//                                              << "  ori:"  << start_effective << "-"  << end_effective
//                                              << "  AG:" << dep_base_info.start_element_index_in_AG  << "-" << dep_base_info.end_element_index_in_AG << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<bool> PIMCOMP_4_Element_Memory_No_Duplication; // 只有第一层需要

void ElementPipelineSchedule::MemoryPreparation()
{
    PrepareDependency();
    PrepareMemoryINFO();
    PreparePadding();

    int first_node_input_channel_num = PIMCOMP_node_list[first_node_index].input_dim[2] * PIMCOMP_node_list[first_node_index].input_dim[3];
    PIMCOMP_4_Element_Memory_No_Duplication.resize(first_node_input_channel_num);
}


void ElementPipelineSchedule::SavePreparation()
{
    Json::Value JsonPreparation;

    ///////////////////////////////////////// NODE ///////////////////////////////////////
    //std::vector<std::vector<int>> node_AG_mapping;               //// 记录每个节点有哪些AG
    //std::vector<std::vector<int>> node_AG0_index_in_replication; //// 记录每个节点若干个rep的AG0的AG_index_in_total
    //std::vector<int> node_replication_num;                      //// 记录生产-消费关系中每个结点的复制倍数（包括pool、vec等）
    //std::vector<std::map<int,int>> PIMCOMP_4_element_node_provider_index_2_index_in_all_providers; //// PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_index]=0、1、2（也就是根据provider_index得到index_in_all_provider）

    JsonPreparation["node_AG_mapping"]["node"].resize(node_num);
    JsonPreparation["node_AG0_index_in_replication"]["node"].resize(node_num);
    JsonPreparation["node_replication_num"]["node"].resize(node_num);
    JsonPreparation["node_provider_index_2_index_in_all_providers"]["node"].resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        for (int j = 0; j < node_AG_mapping[i].size(); ++j)
        {
            JsonPreparation["node_AG_mapping"]["node"][i]["AG_index"][j] = node_AG_mapping[i][j];
        }
        for (int j = 0; j < node_AG0_index_in_replication[i].size(); ++j)
        {
            JsonPreparation["node_AG0_index_in_replication"]["node"][i]["AG0_index"][j] = node_AG0_index_in_replication[i][j];
        }
        JsonPreparation["node_replication_num"]["node"][i] = node_replication_num[i];

        int ii = 0;
        for (auto iter = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[i].begin(); iter != PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[i].end(); iter++)
        {
            int provider_index = iter->first;
            int index_in_providers = iter->second;
            JsonPreparation["node_provider_index_2_index_in_all_providers"]["node"][i]["providers"][ii]["index_in_all_providers_of_this_node"] = index_in_providers;
            JsonPreparation["node_provider_index_2_index_in_all_providers"]["node"][i]["providers"][ii]["original_provider_index"] = provider_index;
            ii ++;
        }
    }


    ///////////////////////////////////////// SPLIT ///////////////////////////////////////
    //std::vector<std::vector<std::vector<int>>> node_rep_split_output_channel_index_list;  //// node_rep_split_output_channel_index_list[node_index][replication_index]是其负责的output_channel_index list
    //std::vector<std::vector<std::vector<int>>> node_rep_split_key_input_channel_index_list; //// node_rep_split_key_input_channel_index_list[node_index][replication_index]是对应output_channel_index的key input channel index
    //std::vector<std::vector<int>> node_rep_split_output_channel_num_list;
    JsonPreparation["split_node_rep_split_output_channel_index_list"].resize(node_num);
    JsonPreparation["split_node_rep_split_key_input_channel_index_list"].resize(node_num);
    JsonPreparation["split_node_rep_split_output_channel_num_list"].resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        for (int j = 0; j < node_rep_split_output_channel_index_list[i].size(); j++)
        {
            JsonPreparation["split_node_rep_split_output_channel_num_list"][i][j] = node_rep_split_output_channel_num_list[i][j];
            for (int k = 0; k < node_rep_split_output_channel_index_list[i][j].size(); ++k)
            {
                JsonPreparation["split_node_rep_split_output_channel_index_list"][i][j][k] = node_rep_split_output_channel_index_list[i][j][k];
                JsonPreparation["split_node_rep_split_key_input_channel_index_list"][i][j][k] = node_rep_split_key_input_channel_index_list[i][j][k];
            }
        }
    }

    Json::Reader jsonReader;
    Json::Value DNNInfo;
    std::ifstream jsonFile("../models/JSON/"+model_name+".json");
    if(!jsonReader.parse(jsonFile, DNNInfo, true))
    {
        std::cout << "error" << std::endl;
        return ;
    }

    JsonPreparation["0_node_list"] = DNNInfo["node_list"];

    std::string strJson = JsonPreparation.toStyledString();
    std::ofstream fob("../output/Preparation.json", std::ios::trunc | std::ios::out);
    if (fob.is_open())
    {
        fob.write(strJson.c_str(), strJson.length());
        fob.close();
    }
}


std::vector<std::set<int>> CheckForIndex;
std::vector<int> CheckForNum;



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Instruction (Detail) ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ElementPipelineSchedule::CheckInputPrepared(int node_index, int replication_index, int input_cycle_index)
{
    std::string operation = PIMCOMP_node_list[node_index].operation;
    if (operation == "OP_CONV" || operation == "OP_POOL")
    {
        int related_input_channel_num = PIMCOMP_conv_pool_input_output_info[node_index].output_index[input_cycle_index].size();
        for (int i = 0; i < related_input_channel_num; ++i)
        {
            int related_input_channel_index = PIMCOMP_conv_pool_input_output_info[node_index].output_index[input_cycle_index][i];
            if (node_rep_split_ready_input_channel_index_list[node_index][replication_index][related_input_channel_index] != 1)
                return false;
        }
        return true;
    }
    else if (operation == "OP_FC")
    {
        int key_input_channel_index = node_rep_split_key_input_channel_index_list[node_index][replication_index][input_cycle_index];
        for (int i = 0; i <= key_input_channel_index; ++i)
        {
            if (node_rep_split_ready_input_channel_index_list[node_index][replication_index][i] != 1)
                return false;
        }
        return true;
    }
    else
    {
        int provider_num = PIMCOMP_topology_consumer_provider_relation[node_index].size();
        for (int i = 0; i < provider_num; ++i)
        {
            int provider_index = PIMCOMP_topology_consumer_provider_relation[node_index][i];
            int tmp_index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[node_index][provider_index];
            if (node_rep_split_ready_input_channel_index_list_for_vec[node_index][replication_index][tmp_index_in_all_providers][input_cycle_index] != 1)
                return false;
        }
        return true;
    }
}



std::vector<std::vector<int>> instruction_group_index_of_node;

struct Comm_struct
{
    int node_index;
    int core_index;
    long long start_address;
    int element_num;
};

struct Post_Comm_struct
{
    bool is_pool;
    int node_index;
    int replication_index;
    int input_channel_index;
    int index_in_all_providers = -1;
    int input_channel_element_num;
    struct Comm_struct COMM;
};
std::vector<struct Post_Comm_struct> Post_Comm_Vector;

struct Main_Comm_struct
{
    int next_node_index;
    int input_channel_index;
    struct Comm_struct COMM;
};
std::vector<struct Main_Comm_struct> Main_Comm_Vector;



void ElementPipelineSchedule::ScheduleSplitInstructionCommForMain(int instruction_group_index, struct Main_Comm_struct Main_Comm)
{
    int next_node_index = Main_Comm.next_node_index;
    int input_channel_index = Main_Comm.input_channel_index;

    //// 指示位
    int replication_num = node_rep_split_ready_input_channel_index_list[next_node_index].size();
    for (int i = 0; i < replication_num; ++i)
    {
        node_rep_split_ready_input_channel_index_list[next_node_index][i][input_channel_index] = 1;
    }

    //// 内存和传输
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        if(PIMCOMP_4_Element_Memory_Dependency.node_list[next_node_index].input_channel_list[input_channel_index].core_list[i] != 0)
        {
            int recv_core = i;
            int recv_length = PIMCOMP_4_Element_Memory_INFO.core_list[recv_core].type_1_main_list[next_node_index].input_channel_element_num;
            if (PIMCOMP_4_Element_Memory_INFO.core_list[recv_core].type_1_main_list[next_node_index].input_channel_list[input_channel_index].start_address == -1)
            {
                long long result = MM.MemoryAllocate(recv_core, recv_length);
                if (result == -1)
                {
                    fprintf(stderr, "Allocate Memory Error For Next Node.\n");
                    abort();
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[recv_core].type_1_main_list[next_node_index].input_channel_list[input_channel_index].start_address = result;

                ScheduleSplitInstructionCOMM(instruction_group_index,
                                                  Main_Comm.COMM.node_index, Main_Comm.COMM.core_index, Main_Comm.COMM.start_address,
                                                  next_node_index, recv_core, result,
                                                  Main_Comm.COMM.element_num);
            }
        }
    }
}


void ElementPipelineSchedule::ScheduleSplitInstructionCOMMForPost(int instruction_group_index, struct Post_Comm_struct Post_Comm)
{
    bool is_pool = Post_Comm.is_pool;
    int node_index = Post_Comm.node_index;
    int replication_index = Post_Comm.replication_index;
    int input_channel_index = Post_Comm.input_channel_index;
    int index_in_all_providers = Post_Comm.index_in_all_providers;
    int input_channel_element_num = Post_Comm.input_channel_element_num;
    if (is_pool)
    {
        // 指示位
        node_rep_split_ready_input_channel_index_list[node_index][replication_index][input_channel_index] = 1;
        // 传输
        MemoryAllocationForPool(instruction_group_index, node_index, input_channel_index, Post_Comm.COMM);
    }
    else
    {
        // 指示位
        node_rep_split_ready_input_channel_index_list_for_vec[node_index][replication_index][index_in_all_providers][input_channel_index] = 1;
        // 传输
        MemoryAllocationForVec(instruction_group_index, node_index, index_in_all_providers, input_channel_index, input_channel_element_num, Post_Comm.COMM);
    }
}


int write_back_element_total = 0;
void ElementPipelineSchedule::ScheduleSplitInstructionWriteBack(int instruction_group_index, int input_channel_index, struct Comm_struct Comm)
{
    int core_index = Comm.core_index;
    int node_index = Comm.node_index;
    int element_num = Comm.element_num;
    long long start_address = Comm.start_address;
    struct INST Instruction_st;
    Instruction_st.type = MEM;
    Instruction_st.operation = "ST";
    Instruction_st.node_index = node_index;
    Instruction_st.stage = "OUTPUT";
    Instruction_st.source_address = start_address;
    Instruction_st.source_offset = 0;
    Instruction_st.destination_address = -1 * node_index;
    Instruction_st.destination_offset = input_channel_index * element_num;
    Instruction_st.element_num = element_num;
    Instruction_st.instruction_group_index = instruction_group_index;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_st);
    write_back_element_total += element_num;
}


void ElementPipelineSchedule::ScheduleSplitInstructionStage0LoadBias(int instruction_group_index)
{
    bias_address_map.resize(ChipH * ChipW);
    bias_element_num_map.resize(ChipH * ChipW);
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        bias_address_map[i].resize(node_num);
        bias_element_num_map[i].resize(node_num);
        struct core_schedule current_core = PIMCOMP_4_virtual_core_AG_map.core_list[i];
        int AG_num = current_core.AG_list.size();
        if (AG_num == 0)
            continue;
        std::set<int> core_node; // 为每个核上的每个节点都LOAD一次bias
        for (int j = 0; j < AG_num; ++j)
        {
            int node_index = current_core.node_list[j];
            int bias_element_num = current_core.AG_list[j].output_element_num;
            if (core_node.count(node_index) == 0)
            {
                // TODO 这里可能需要添加一个判断该节点是否需要加载偏置，有可能有的CONV或FC不需要BIAS。需要生成JSON时确定。
                if (PIMCOMP_node_list[node_index].with_bias)
                {
                    core_node.insert(node_index);
                    long long bias_start_address = MM.MemoryAllocate(i, bias_element_num);
                    if (bias_start_address == -1)
                    {
                        fprintf(stderr, "Allocate Memory Error For BIAS.\n");
                        abort();
                    }
                    bias_address_map[i][node_index] = bias_start_address;
                    bias_element_num_map[i][node_index] = bias_element_num;
                    struct INST Instruction_ld;
                    Instruction_ld.node_index = node_index;
                    Instruction_ld.type = MEM;
                    Instruction_ld.operation = "LD";
                    Instruction_ld.stage = "BIAS";
                    Instruction_ld.source = -1 * node_index;
                    Instruction_ld.source_address = -1 * node_index;
                    Instruction_ld.source_offset = 0;
                    Instruction_ld.destination = node_index;
                    Instruction_ld.destination_address = bias_start_address;
                    Instruction_ld.destination_offset = 0;
                    Instruction_ld.element_num =  bias_element_num;
                    Instruction_ld.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[i].instruction_ir_list.push_back(Instruction_ld);
                }
            }
        }
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage1MVMUL(int instruction_group_index, int start_AG_index_in_total, int AG_num_this_replication, int input_cycle_index)
{
    int replication_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].replication_index;
    std::map<int, std::vector<int>> core_AG_map; //core_AG_map[core_index]保存了一系列AG

    for (int i = 0; i < AG_num_this_replication; ++i)
    {
        int AG_index_in_total = start_AG_index_in_total + i;
        int node_index = PIMCOMP_4_element_AG_info_list[AG_index_in_total].node_index;
        int core_index = PIMCOMP_4_element_AG_info_list[AG_index_in_total].core_index;
        //// 为每个AG生成一个保存输出的位置，并且这个位置保持不变。未来始终有效。
        MemoryAllocationForAG(AG_index_in_total, core_index, 1);
        //// 将输入排好，这其中包括正常输入和padding两种segment。
        int input_segment_num = PIMCOMP_4_Element_Memory_Dependency.AG_list[AG_index_in_total].input_cycle_list[input_cycle_index].input_segment.size();
        int ready_element_num = 0;
        for (int j = 0; j < input_segment_num; ++j)
        {
            struct PIMCOMP_4_Element_Memory_Dep_input_segment input_segment_info = PIMCOMP_4_Element_Memory_Dependency.AG_list[AG_index_in_total].input_cycle_list[input_cycle_index].input_segment[j];
            int end_element_index_in_AG = input_segment_info.end_element_index_in_AG;
            int start_element_index_in_AG = input_segment_info.start_element_index_in_AG;
            if (!input_segment_info.padding)
            {
                int input_channel_index = input_segment_info.input_channel_index;
                int start_element_index_in_channel = input_segment_info.start_element_index_in_channel;
                int end_element_index_in_channel = input_segment_info.end_element_index_in_channel;
                int start_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_list[input_channel_index].start_address;
                if (start_address == -1)
                {
                    fprintf(stderr, "Memory Preparation Error \n" );
                    std::cout << core_index << "  " << node_index << "  " << input_channel_index <<  "  " << std::endl;
                    abort();
                }
                else
                {
                    // ready_element_num增加数量
                    ready_element_num +=  end_element_index_in_AG - start_element_index_in_AG + 1;
                    // 添加搬移指令
                    struct INST Instruction_vm;
                    Instruction_vm.type = VEC1OP;
                    Instruction_vm.operation = "LMV";
                    Instruction_vm.node_index = node_index;
                    Instruction_vm.stage = "MAIN";
                    Instruction_vm.source = AG_index_in_total;
                    Instruction_vm.source_address = start_address;
                    Instruction_vm.destination = AG_index_in_total;
                    Instruction_vm.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_start_address;
                    Instruction_vm.source_offset = start_element_index_in_channel;
                    Instruction_vm.destination_offset = start_element_index_in_AG;
                    Instruction_vm.element_num = end_element_index_in_AG - start_element_index_in_AG + 1;
                    Instruction_vm.instruction_group_index = instruction_group_index;
                    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_vm);
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_list[input_channel_index].rest_use_times--;
                // 若rest_use_times==0，则回收内存
                if (PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_list[input_channel_index].rest_use_times == 0)
                {
                    long long free_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_list[input_channel_index].start_address;
                    int free_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_element_num;
                    if (!MM.MemoryFree(core_index, free_start_address, free_element_num))
                    {
                        fprintf(stderr, "Memory Recycle Error \n");
//                        std::cout << core_index << "  " << node_index << "  " << input_channel_index <<  "  " << free_element_num << std::endl;
                        abort();
                    }
                    // 将原本分配的内存地址设为-1
                    PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_1_main_list[node_index].input_channel_list[input_channel_index].start_address = -1;
                }
            }
            else
            {
                // ready_element_num增加数量
                ready_element_num +=  end_element_index_in_AG - start_element_index_in_AG + 1;
                // 增加padding指令
                struct INST Instruction_lldi;
                Instruction_lldi.type = LLDI;
                Instruction_lldi.operation = "LLDI";
                Instruction_lldi.node_index = node_index;
                Instruction_lldi.stage = "MAIN";
                Instruction_lldi.destination = AG_index_in_total;
                Instruction_lldi.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_start_address;
                Instruction_lldi.destination_offset = start_element_index_in_AG;
                Instruction_lldi.element_num = end_element_index_in_AG - start_element_index_in_AG + 1;
                Instruction_lldi.imm_value = 0;
                Instruction_lldi.instruction_group_index = i;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_lldi);
            }
        }
        // 当进行完所有的常规input和padding后，检查AG的input element是否准备好，能否进行后面的计算
        if (ready_element_num != PIMCOMP_4_element_AG_info_list[AG_index_in_total].input_element_num)
        {
            fprintf(stderr, "Input Preparation Error \n");
//            std::cout << ready_element_num << "  " << AG_index_in_total << std::endl;
            abort();
        }

        //// 首先为每个AG生成MVMUL操作
        int input_element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].input_element_num;
        int output_element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].output_element_num;

        struct INST Instruction;
        Instruction.type = MVMUL;
        Instruction.operation = "MVMUL";
        Instruction.stage = "MAIN";
        Instruction.input_cycle_index = input_cycle_index;
        Instruction.AG_index_in_total = AG_index_in_total;
        Instruction.replication_index = replication_index;
        Instruction.AG_index_in_replication = i;
        Instruction.conv_or_fc = PIMCOMP_node_list[node_index].operation;
        Instruction.node_index = node_index;
        Instruction.source = Instruction.AG_index_in_total;
        Instruction.source_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_start_address;
        Instruction.source_offset = 0;
        Instruction.destination = Instruction.AG_index_in_total;
        Instruction.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_start_address;
        Instruction.destination_offset = 0;
        Instruction.input_element_num = input_element_num;
        Instruction.output_element_num = output_element_num;
        Instruction.instruction_group_index = instruction_group_index;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction);


        //// For BIAS
        if (PIMCOMP_node_list[node_index].with_bias)
        {
            if(i == 0)
            {
                struct INST Instruction_bias;
                Instruction_bias.type = VEC2OP;
                Instruction_bias.operation = "VVADD";
                Instruction_bias.stage = "MAIN-B";
                Instruction_bias.node_index = node_index;
                Instruction_bias.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_start_address;
                Instruction_bias.destination_offset = 0;
                Instruction_bias.source_1_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_start_address;
                Instruction_bias.source_1_offset = 0;
                Instruction_bias.source_2_address = bias_address_map[core_index][node_index];
                Instruction_bias.source_2_offset = 0;
                Instruction_bias.element_num = output_element_num;
                Instruction_bias.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_bias);
            }
        }

        //// VVADD
        core_AG_map[core_index].push_back(AG_index_in_total);
        if (core_AG_map[core_index].size() > 1)
        {
            struct INST Instruction_vvadd;
            Instruction_vvadd.type = VEC2OP;
            Instruction_vvadd.operation = "VVADD";
            Instruction_vvadd.stage = "MAIN-A";
            Instruction_vvadd.node_index = node_index;
            Instruction_vvadd.input_cycle_index = input_cycle_index;
            Instruction_vvadd.destination = core_AG_map[core_index][0];
            Instruction_vvadd.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_vvadd.destination].output_start_address;
            Instruction_vvadd.destination_offset = 0;
            Instruction_vvadd.source_1 = core_AG_map[core_index][0];
            Instruction_vvadd.source_1_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_vvadd.source_1].output_start_address;
            Instruction_vvadd.source_1_offset = 0;
            Instruction_vvadd.source_2 = AG_index_in_total;
            Instruction_vvadd.source_2_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_vvadd.source_2].output_start_address;
            Instruction_vvadd.source_2_offset = 0;
            Instruction_vvadd.element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].output_element_num; // 同一rep的output element num一致
            Instruction_vvadd.instruction_group_index = instruction_group_index;
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_vvadd);
        }
    }
}


static int comm_index = 0; //// SEND/RECV对的编号。为了后续评估模型而设置。


void ElementPipelineSchedule::ScheduleSplitInstructionStage3ACC(int instruction_group_index, int start_AG_index_in_total, int AG_num_this_replication)
{
    int RecvCore = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].core_index;
    int node_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].node_index;

    std::map<int, int> core_AG_map; //core_AG_map[core_index]保存了第一个AG，也就是前面累加结果的那个AG
    core_AG_map[RecvCore] = start_AG_index_in_total;
    for (int i = 1; i < AG_num_this_replication; ++i)
    {
        int AG_index_in_total = i + start_AG_index_in_total;
        int current_core = PIMCOMP_4_element_AG_info_list[AG_index_in_total].core_index;
        int output_element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].output_element_num;
        if (core_AG_map.count(current_core) == 0)
        {
            struct INST Instruction_send;
            Instruction_send.type = COMM;
            Instruction_send.operation = "SEND";
            Instruction_send.stage = "MAIN";
            Instruction_send.node_index = node_index;
            Instruction_send.to_core = RecvCore;
            Instruction_send.from_core = current_core;
            Instruction_send.source = AG_index_in_total;
            Instruction_send.source_address = PIMCOMP_4_Element_Memory_INFO.core_list[current_core].type_0_AG_list[AG_index_in_total].output_start_address;
            Instruction_send.node_index = node_index;
            Instruction_send.element_num = output_element_num;
            Instruction_send.instruction_group_index = instruction_group_index;
            Instruction_send.comm_index = comm_index;
            Instruction_send.instruction_index_in_core = PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[current_core].instruction_ir_list.size();
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[current_core].instruction_ir_list.push_back(Instruction_send);
            PIMCOMP_6_inter_core_communication[current_core][RecvCore] += Instruction_send.element_num;

            //// 为该AG在ReCVCore分配内存
            MemoryAllocationForAG(AG_index_in_total, RecvCore, 0);

            struct INST Instruction_recv;
            Instruction_recv.type = COMM;
            Instruction_recv.operation = "RECV";
            Instruction_recv.stage = "MAIN";
            Instruction_recv.node_index = node_index;
            Instruction_recv.from_core = current_core;
            Instruction_recv.to_core = RecvCore;
            Instruction_recv.destination = AG_index_in_total;
            Instruction_recv.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[AG_index_in_total].output_start_address;
            Instruction_recv.node_index = node_index;
            Instruction_recv.element_num = output_element_num;
            Instruction_recv.instruction_group_index = instruction_group_index;
            Instruction_recv.comm_index = comm_index;
            Instruction_recv.instruction_index_in_core = PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[RecvCore].instruction_ir_list.size();
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[RecvCore].instruction_ir_list.push_back(Instruction_recv);

            struct INST Instruction_vvadd;
            Instruction_vvadd.type = VEC2OP;
            Instruction_vvadd.operation = "VVADD";
            Instruction_vvadd.stage = "MAIN-C";
            Instruction_vvadd.node_index = node_index;
            Instruction_vvadd.source_1 = start_AG_index_in_total;
            Instruction_vvadd.source_1_address = PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[Instruction_vvadd.source_1].output_start_address;
            Instruction_vvadd.source_1_offset = 0;
            Instruction_vvadd.source_2 = AG_index_in_total;
            Instruction_vvadd.source_2_address = PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[Instruction_vvadd.source_2].output_start_address;
            Instruction_vvadd.source_2_offset = 0;
            Instruction_vvadd.destination = start_AG_index_in_total;
            Instruction_vvadd.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[Instruction_vvadd.destination].output_start_address;
            Instruction_vvadd.destination_offset = 0;
            Instruction_vvadd.element_num = output_element_num;
            Instruction_vvadd.instruction_group_index = instruction_group_index;
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[RecvCore].instruction_ir_list.push_back(Instruction_vvadd);

            comm_index++;
            core_AG_map[current_core] = AG_index_in_total;

            //// 回收刚刚分配的AG的内存，下次重新分配 (分配时未分配input，所以不用回收input)
            int output_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[AG_index_in_total].output_start_address;
            bool result_output = MM.MemoryFree(RecvCore, output_start_address, output_element_num);
            if (!result_output)
            {
                fprintf(stderr, "COMM Recycle Failed \n");
                abort();
            }
            PIMCOMP_4_Element_Memory_INFO.core_list[RecvCore].type_0_AG_list[AG_index_in_total].output_start_address = -1;

        }
        else
        {
            continue;
        }
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage3ACT(int instruction_group_index, int start_AG_index_in_total, int input_cycle_index)
{
    int core_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].core_index;
    int output_element_num = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].output_element_num;
    int node_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].node_index;
    struct INST Instruction_act;
    Instruction_act.type = VEC1OP;
    Instruction_act.node_index = node_index;
    int act_type = PIMCOMP_node_list[node_index].act_type;
    Instruction_act.operation = act_type == 0 ? "VRELU" : (act_type == 1? "VTANH" : "VSIGMOID");
    Instruction_act.stage = "MAIN";
    Instruction_act.input_cycle_index = input_cycle_index;
    Instruction_act.source = start_AG_index_in_total;
    Instruction_act.source_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_act.source].output_start_address;
    Instruction_act.source_offset = 0;
    Instruction_act.destination = start_AG_index_in_total;
    Instruction_act.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_act.destination].output_start_address;
    Instruction_act.destination_offset = 0;
    Instruction_act.element_num = output_element_num;
    Instruction_act.instruction_group_index = instruction_group_index;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_act);
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage3CLIP(int instruction_group_index, int start_AG_index_in_total, int input_cycle_index)
{
    int core_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].core_index;
    int output_element_num = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].output_element_num;
    int node_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].node_index;
    struct INST Instruction_vrs;
    Instruction_vrs.type = VEC1OP;
    Instruction_vrs.node_index = node_index;
    Instruction_vrs.operation = "VRSU";
    Instruction_vrs.stage = "MAIN";
    Instruction_vrs.input_cycle_index = input_cycle_index;
    Instruction_vrs.source = start_AG_index_in_total;
    Instruction_vrs.source_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_vrs.source].output_start_address;
    Instruction_vrs.source_offset = 0;
    Instruction_vrs.destination = start_AG_index_in_total;
    Instruction_vrs.destination_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_vrs.destination].output_start_address;
    Instruction_vrs.destination_offset = 0;
    Instruction_vrs.imm_value = PIMCOMP_node_list[node_index].clip_max;
    Instruction_vrs.element_num = output_element_num;
    Instruction_vrs.instruction_group_index = instruction_group_index;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_vrs);

    Instruction_vrs.operation = "VRSL";
    Instruction_vrs.imm_value = PIMCOMP_node_list[node_index].clip_min;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_vrs);
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage3VER(int instruction_group_index, int start_AG_index_in_total, int input_cycle_index)
{
    int core_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].core_index;
    int output_element_num = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].output_element_num;
    int node_index = PIMCOMP_4_element_AG_info_list[start_AG_index_in_total].node_index;
    struct INST Instruction_ver;
    Instruction_ver.type = VER;
    Instruction_ver.node_index = node_index;
    Instruction_ver.operation = "VER";
    Instruction_ver.stage = "MAIN";
    Instruction_ver.input_cycle_index = input_cycle_index;
    Instruction_ver.source = start_AG_index_in_total;
    Instruction_ver.source_address = PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[Instruction_ver.source].output_start_address;
    Instruction_ver.source_offset = 0;
    Instruction_ver.element_num = output_element_num;
    Instruction_ver.instruction_group_index = instruction_group_index;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_ver);
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage4Verify(int instruction_group_index, int node_index, int core_index, long long source_address, int source_offset, int element_num, int input_cycle_index)
{
    struct INST Instruction_ver;
    Instruction_ver.type = VER;
    Instruction_ver.node_index = node_index;
    Instruction_ver.operation = "VER";
    Instruction_ver.stage = "VER";
    Instruction_ver.input_cycle_index = input_cycle_index;
    Instruction_ver.source_address = source_address;
    Instruction_ver.source_offset = source_offset;
    Instruction_ver.element_num = element_num;
    Instruction_ver.instruction_group_index = instruction_group_index;
    PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[core_index].instruction_ir_list.push_back(Instruction_ver);
}


void ElementPipelineSchedule::ScheduleSplitInstructionCOMM(int instruction_group_index,
                                                                int send_node_index, int from_core, long long source_address,
                                                                int recv_node_index, int to_core, long long destination_address,
                                                                int element_num)
{
    if (to_core != from_core)
    {
        struct INST Instruction_send;
        Instruction_send.type = COMM;
        Instruction_send.operation = "SEND";
        Instruction_send.stage = "POST";
        Instruction_send.node_index = send_node_index;
        Instruction_send.to_core = to_core;
        Instruction_send.from_core = from_core;
        Instruction_send.source_address = source_address;
        Instruction_send.element_num = element_num;
        Instruction_send.instruction_group_index = instruction_group_index;
        Instruction_send.comm_index = comm_index;
        Instruction_send.instruction_index_in_core = PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[from_core].instruction_ir_list.size();
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[from_core].instruction_ir_list.push_back(Instruction_send);

        struct INST Instruction_recv;
        Instruction_recv.type = COMM;
        Instruction_recv.operation = "RECV";
        Instruction_recv.stage = "POST";
        Instruction_recv.node_index = recv_node_index;
        Instruction_recv.from_core = from_core;
        Instruction_recv.to_core = to_core;
        Instruction_recv.destination_address = destination_address;
        Instruction_recv.element_num = element_num;
        Instruction_recv.instruction_group_index = instruction_group_index;
        Instruction_recv.comm_index = comm_index;
        Instruction_recv.instruction_index_in_core = PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[to_core].instruction_ir_list.size();
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[to_core].instruction_ir_list.push_back(Instruction_recv);

        comm_index ++;
    }
    else
    {
        struct INST Instruction_vm;
        Instruction_vm.type = VEC1OP;
        Instruction_vm.operation = "LMV";
        Instruction_vm.input_cycle_index = -1; // disable the verification process
        Instruction_vm.stage = "POST";
        Instruction_vm.node_index = recv_node_index;
        Instruction_vm.source = -1;
        Instruction_vm.source_address = source_address;
        Instruction_vm.source_offset = 0;
        Instruction_vm.destination = -1;
        Instruction_vm.destination_address = destination_address;
        Instruction_vm.destination_offset = 0;
        Instruction_vm.element_num = element_num;
        Instruction_vm.instruction_group_index = instruction_group_index;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[from_core].instruction_ir_list.push_back(Instruction_vm);
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage4Pool(int instruction_group_index, int pool_node_index, int input_cycle_index)
{
    int input_channel_num = PIMCOMP_conv_pool_input_output_info[pool_node_index].output_index[input_cycle_index].size();
    int execution_core = post_node_map[pool_node_index];
    int element_num = PIMCOMP_node_list[pool_node_index].output_dim[1];

    //// 为该POOL算子分配一个地址保存输出
    long long output_start_address = MM.MemoryAllocate(execution_core, element_num);
    if (output_start_address == -1)
    {
        fprintf(stderr, "Allocate Memory Error For POOL Output.\n");
        abort();
    }
    PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].output_start_address_record.push_back(output_start_address);

    for (int i = 0; i < input_channel_num; ++i)
    {
        //// 首先判断该input_channel数据是否准备好
        int input_channel_index = PIMCOMP_conv_pool_input_output_info[pool_node_index].output_index[input_cycle_index][i];
        long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].input_channel_list[input_channel_index].start_address;
        if(input_start_address == -1)
        {
            fprintf(stderr, "POOL Preparation Failed \n");
            abort();
        }

        //// 添加指令，执行操作
        struct INST Instruction;
        Instruction.stage = "POST";
        Instruction.input_channel_index = input_channel_index;
        Instruction.output_channel_index = input_cycle_index;
        Instruction.node_index = pool_node_index;
        Instruction.element_num = element_num;
        Instruction.instruction_group_index = instruction_group_index;
        Instruction.input_cycle_index = input_cycle_index; // For Debug
        if (i == 0)
        {
            Instruction.type = VEC1OP;
            Instruction.operation = "LMV";
            Instruction.source_address = input_start_address;
            Instruction.source_offset = 0;
            Instruction.destination_offset = output_start_address;
            Instruction.destination_address = 0;
        }
        else
        {
            Instruction.type = VEC2OP;
            if (PIMCOMP_node_list[pool_node_index].param.pool_method == 0)
                Instruction.operation = "VVMAX";
            else
                Instruction.operation = "VVADD";
            Instruction.source_1_address = output_start_address;
            Instruction.source_1_offset = 0;
            Instruction.source_2_address = input_start_address;
            Instruction.source_2_offset = 0;
            Instruction.destination_address = output_start_address;
            Instruction.destination_offset = 0;
        }
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction);

        if (i == input_channel_num-1 && PIMCOMP_node_list[pool_node_index].param.pool_method == 1) //利用剩余空间计算AvgPOOL的最后一步
        {
            long long LLDI_addr = MM.MemoryAllocate(execution_core, element_num);
            if (LLDI_addr == -1)
            {
                fprintf(stderr, "Allocate Memory Error For LLDI.\n");
                abort();
            }

            struct INST Instruction_lldi;
            Instruction_lldi.type = LLDI;
            Instruction_lldi.operation = "LLDI";
            Instruction_lldi.node_index = pool_node_index;
            Instruction_lldi.stage = "POST";
            Instruction_lldi.destination_address = LLDI_addr;
            Instruction_lldi.destination_offset = 0;
            Instruction_lldi.element_num = element_num;
            float kernel_h = PIMCOMP_node_list[pool_node_index].param.kernel_h;
            float kernel_w = PIMCOMP_node_list[pool_node_index].param.kernel_w;
            if (model_name == "inception_v3")
                Instruction_lldi.imm_value = 1.0 / (kernel_h * kernel_w);
            else
                Instruction_lldi.imm_value = 1.0 / input_channel_num;
            Instruction_lldi.instruction_group_index = instruction_group_index;
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_lldi);

            struct INST Instruction_vvmul;
            Instruction_vvmul.type = VEC2OP;
            Instruction_vvmul.operation = "VVMUL";
            Instruction_vvmul.stage = "POST";
            Instruction_vvmul.input_cycle_index = input_cycle_index;
            Instruction_vvmul.node_index = pool_node_index;
            Instruction_vvmul.element_num = element_num;
            Instruction_vvmul.instruction_group_index = instruction_group_index;
            Instruction_vvmul.source_1_address = output_start_address;
            Instruction_vvmul.source_1_offset = 0;
            Instruction_vvmul.source_2_address = LLDI_addr;
            Instruction_vvmul.source_2_offset = 0;
            Instruction_vvmul.destination_address = output_start_address;
            Instruction_vvmul.destination_offset = 0;
            PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_vvmul);

            bool result = MM.MemoryFree(execution_core, LLDI_addr, element_num);
            if (!result)
            {
                fprintf(stderr, "LLDI Recycle Failed \n");
                abort();
            }
        }

        //// For Verification
        ScheduleSplitInstructionStage4Verify(instruction_group_index, pool_node_index, execution_core, output_start_address, 0, element_num, input_cycle_index);

        //// 判断能否释放POOL INPUT所占用的内存
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].input_channel_list[input_channel_index].rest_use_times--;
        if (PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].input_channel_list[input_channel_index].rest_use_times == 0)
        {
            int channel_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].output_channel_element_num;
            bool result = MM.MemoryFree(execution_core, input_start_address, channel_element_num);
            if (!result)
            {
                fprintf(stderr, "POOL Recycle Failed \n");
                abort();
            }
            PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_3_pool_list[pool_node_index].input_channel_list[input_channel_index].start_address = -1;
        }
    }
}


void ElementPipelineSchedule::ScheduleSplitInstructionStage4Activate(int instruction_group_index, int vec_node_index, int input_cycle_index)
{
    int execution_core = post_node_map[vec_node_index];
    int provider_num = 1;
    int element_num = PIMCOMP_node_list[vec_node_index].output_dim[1];

    //// 分配一个地址保存输出
    long long output_start_address = MM.MemoryAllocate(execution_core, element_num);
    if (output_start_address == -1)
    {
        fprintf(stderr, "Allocate Memory Error For POOL Output.\n");
        abort();
    }
    PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_start_address_record.push_back(output_start_address);


    for (int i = 0; i < provider_num; ++i)
    {
        //// 首先判断该input_channel数据是否准备好
        int provider_node_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        int input_channel_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
        int index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_node_index];
        long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address;
        int output_channel_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_channel_element_num;
        if (input_start_address == -1)
        {
            fprintf(stderr, "VEC Preparation Failed \n");
            std::cout << "core:" << execution_core << "  node:" << vec_node_index << "  index_in_all_providers:" << index_in_all_providers << "  input_cycle_index:" << input_cycle_index << std::endl;
            abort();
        }

        //// 添加指令
        std::string act_type;
        std::string consumer_op = PIMCOMP_node_list[vec_node_index].operation;
        act_type = consumer_op == "OP_RELU" ? "VRELU" : (consumer_op == "OP_TANH" ? "VTANH" : "VSIGM");
        struct INST Instruction_act;
        Instruction_act.type = VEC1OP;
        Instruction_act.stage = "POST";
        Instruction_act.operation = act_type;
        Instruction_act.output_channel_index = input_cycle_index;
        Instruction_act.node_index = vec_node_index;
        Instruction_act.source_address = input_start_address;
        Instruction_act.source_offset = 0;
        Instruction_act.destination_address = output_start_address;
        Instruction_act.destination_offset = 0;
        Instruction_act.element_num = element_num;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_act);

        //// For Verification
        ScheduleSplitInstructionStage4Verify(instruction_group_index, vec_node_index, execution_core, output_start_address, 0, element_num, input_cycle_index);

        //// 释放VEC INPUT所占用的内存
        bool result = MM.MemoryFree(execution_core, input_start_address, input_channel_element_num);
        if (!result)
        {
            fprintf(stderr, "VEC Input Recycle Failed \n");
            std::cout << execution_core << "  " << input_start_address << "  " << input_channel_element_num << std::endl;
            std::cout << MM.FindElementLength(execution_core, input_start_address) << std::endl;
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address = -1;
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage4Eltwise(int instruction_group_index, int vec_node_index, int input_cycle_index)
{
    int execution_core = post_node_map[vec_node_index];
    int element_num = PIMCOMP_node_list[vec_node_index].output_dim[1];
    int provider_num = PIMCOMP_topology_consumer_provider_relation[vec_node_index].size();

    int elt_type = PIMCOMP_node_list[vec_node_index].param.eletype;
    std::string elt_operation;
    switch (elt_type)
    {
        case 2: elt_operation = "VVADD"; break;
        case 4: elt_operation = "VSUB"; break;
    }

    //// 分配一个地址保存输出
    long long output_start_address = MM.MemoryAllocate(execution_core, element_num);
    if (output_start_address != -1)
    {
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_start_address_record.push_back(output_start_address);
        // 先把结果全写成0，方便后面计算。【当然，如果是VMUL则需要先写成1。】
        struct INST Instruction_lldi;
        Instruction_lldi.type = LLDI;
        Instruction_lldi.operation = "LLDI";
        Instruction_lldi.stage = "POST";
        Instruction_lldi.node_index = vec_node_index;
        Instruction_lldi.destination_address = output_start_address;
        Instruction_lldi.destination_offset = 0;
        Instruction_lldi.element_num = element_num;
        Instruction_lldi.imm_value = 0;
        Instruction_lldi.instruction_group_index = instruction_group_index;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_lldi);
    }
    else
    {
        fprintf(stderr, "Allocate Memory Error For ELTWISE Output.\n");
        abort();
    }

    //// 添加VEC指令
    for (int i = 0; i < provider_num; ++i)
    {
        //// 首先判断该input_channel数据是否准备好
        int provider_node_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        int index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_node_index];
        int input_channel_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
        long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address;
        int output_channel_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_channel_element_num;
        if (input_start_address == -1)
        {
            fprintf(stderr, "VEC Preparation Failed \n");
            std::cout << "core:" << execution_core << "  node:" << vec_node_index << "  index_in_all_providers:" << index_in_all_providers << "  input_cycle_index:" << input_cycle_index << std::endl;
            abort();
        }

        // 添加指令
        int provider_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        struct INST Instruction_elt;
        Instruction_elt.type = VEC2OP;
        Instruction_elt.operation = elt_operation;
        Instruction_elt.stage = "POST";
        Instruction_elt.node_index = vec_node_index;
        Instruction_elt.input_cycle_index = input_cycle_index;
        Instruction_elt.source_1_address = output_start_address;
        Instruction_elt.source_1_offset = 0;
        Instruction_elt.source_2_address = input_start_address;
        Instruction_elt.source_2_offset = 0;
        Instruction_elt.destination_address = output_start_address;
        Instruction_elt.destination_offset = 0;
        Instruction_elt.element_num = element_num;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_elt);

        //// For Verification
        ScheduleSplitInstructionStage4Verify(instruction_group_index, vec_node_index, execution_core, output_start_address, 0, element_num, input_cycle_index);

        //// 释放VEC INPUT所占用的内存
        bool result = MM.MemoryFree(execution_core, input_start_address, input_channel_element_num);
        if (!result)
        {
            fprintf(stderr, "VEC Input Recycle Failed \n");
            std::cout << execution_core << "  " << input_start_address << "  " << input_channel_element_num << std::endl;
            std::cout << MM.FindElementLength(execution_core, input_start_address) << std::endl;
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address = -1;
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage4Concat(int instruction_group_index, int vec_node_index, int input_cycle_index)
{
    int execution_core = post_node_map[vec_node_index];
    int provider_num = PIMCOMP_topology_consumer_provider_relation[vec_node_index].size();
    int output_element_num = PIMCOMP_node_list[vec_node_index].output_dim[1];

    //// 分配一个地址保存输出
    long long output_start_address = MM.MemoryAllocate(execution_core, output_element_num);
    if (output_start_address == -1)
    {
        fprintf(stderr, "Allocate Memory Error For POOL Output.\n");
        abort();
    }
    PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_start_address_record.push_back(output_start_address);

    int destination_offset = 0;
    for (int i = 0; i < provider_num; ++i)
    {
        //// 首先判断该input_channel数据是否准备好
        int provider_node_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        int index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_node_index];
        int input_channel_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
        long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address;
        int output_channel_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_channel_element_num;
        if (input_start_address == -1)
        {
            fprintf(stderr, "VEC Preparation Failed \n");
            std::cout << "core:" << execution_core << "  node:" << vec_node_index << "  index_in_all_providers:" << index_in_all_providers << "  input_cycle_index:" << input_cycle_index << std::endl;
            abort();
        }

        //// 添加指令
        int provider_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        struct INST Instruction_concat;
        Instruction_concat.type = VEC1OP;
        Instruction_concat.operation = "LMV";
        Instruction_concat.input_cycle_index = input_cycle_index;
        Instruction_concat.stage = "POST";
        Instruction_concat.node_index = vec_node_index;
        int this_provider_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
        Instruction_concat.source_address = input_start_address;
        Instruction_concat.source_offset = 0;
        Instruction_concat.destination_address = output_start_address;
        Instruction_concat.destination_offset = destination_offset;
        Instruction_concat.element_num = this_provider_element_num;
        destination_offset += this_provider_element_num;
        PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_concat);

        //// 释放VEC INPUT所占用的内存
        bool result = MM.MemoryFree(execution_core, input_start_address, input_channel_element_num);
        if (!result)
        {
            fprintf(stderr, "VEC Input Recycle Failed \n");
            std::cout << execution_core << "  " << input_start_address << "  " << input_channel_element_num << std::endl;
            std::cout << MM.FindElementLength(execution_core, input_start_address) << std::endl;
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address = -1;
    }

    //// For Verification
    ScheduleSplitInstructionStage4Verify(instruction_group_index, vec_node_index, execution_core, output_start_address, 0, output_element_num, input_cycle_index);

    if (destination_offset != output_element_num)
    {
        fprintf(stderr, "CONCAT Operator Failed\n");
        abort();
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionStage4Shuffle(int instruction_group_index, int vec_node_index, int input_cycle_index)
{
    int execution_core = post_node_map[vec_node_index];
    int provider_num = 1;
    int element_num = PIMCOMP_node_list[vec_node_index].output_dim[1];
    int split_factor = PIMCOMP_node_list[vec_node_index].param.split_factor;

    //// 分配一个地址保存输出
    long long output_start_address = MM.MemoryAllocate(execution_core, element_num);
    if (output_start_address == -1)
    {
        fprintf(stderr, "Allocate Memory Error For POOL Output.\n");
        abort();
    }
    PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_start_address_record.push_back(output_start_address);


    for (int i = 0; i < provider_num; ++i)
    {
        //// 首先判断该input_channel数据是否准备好
        int provider_node_index = PIMCOMP_topology_consumer_provider_relation[vec_node_index][i];
        int input_channel_element_num = PIMCOMP_node_list[provider_node_index].output_dim[1];
        int index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[vec_node_index][provider_node_index];
        long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address;
        int output_channel_element_num = PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].output_channel_element_num;
        if (input_start_address == -1)
        {
            fprintf(stderr, "VEC Preparation Failed \n");
            std::cout << "core:" << execution_core << "  node:" << vec_node_index << "  index_in_all_providers:" << index_in_all_providers << "  input_cycle_index:" << input_cycle_index << std::endl;
            abort();
        }

        int segment_element_num = input_channel_element_num / split_factor;
        for (int j = 0; j < split_factor; ++j)
        {
            for (int k = 0; k < segment_element_num; ++k)
            {
                //// 添加指令
                struct INST Instruction_act;
                Instruction_act.type = VEC1OP;
                Instruction_act.stage = "POST";
                Instruction_act.operation = "LMV";
                Instruction_act.output_channel_index = input_cycle_index;
                Instruction_act.node_index = vec_node_index;
                Instruction_act.source_address = input_start_address;
                Instruction_act.source_offset = j * segment_element_num + k;
                Instruction_act.destination_address = output_start_address;
                Instruction_act.destination_offset = k * split_factor + j;
                Instruction_act.element_num = 1;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[execution_core].instruction_ir_list.push_back(Instruction_act);
            }
        }

        //// 释放VEC INPUT所占用的内存
        bool result = MM.MemoryFree(execution_core, input_start_address, input_channel_element_num);
        if (!result)
        {
            fprintf(stderr, "VEC Input Recycle Failed \n");
            std::cout << execution_core << "  " << input_start_address << "  " << input_channel_element_num << std::endl;
            std::cout << MM.FindElementLength(execution_core, input_start_address) << std::endl;
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[execution_core].type_2_vec_list[vec_node_index].provider_list[index_in_all_providers].input_channel_list[input_cycle_index].start_address = -1;
    }
}

////////////////////////////// Memory Allocation //////////////////////////////


void ElementPipelineSchedule::MemoryAllocationForFirstLayer(int instruction_group_index, int input_cycle_index)
{
    // 第一次就要根据input_cycle_index来读取数据。不用担心重复读取，因为PIMCOMP_4_Element_Memory_No_Duplication会保证每个input_channel只加载一次。
    int node_index = first_node_index;
    int related_input_channel_num = PIMCOMP_conv_pool_input_output_info[node_index].output_index[input_cycle_index].size();
    int input_channel_element_num = PIMCOMP_4_Element_Memory_Dependency.node_list[node_index].input_channel_element_num;
    //// LOAD INPUT
    for (int i = 0; i < related_input_channel_num; ++i)
    {
        int related_input_channel_index = PIMCOMP_conv_pool_input_output_info[node_index].output_index[input_cycle_index][i];
        if (PIMCOMP_4_Element_Memory_No_Duplication[related_input_channel_index] == 0)
            PIMCOMP_4_Element_Memory_No_Duplication[related_input_channel_index] = 1;
        else
            continue;
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            if (PIMCOMP_4_Element_Memory_Dependency.node_list[node_index].input_channel_list[related_input_channel_index].core_list[j] == 0)
                continue;
            int related_core_index = j;
            if (PIMCOMP_4_Element_Memory_INFO.core_list[related_core_index].type_1_main_list[node_index].input_channel_list[related_input_channel_index].start_address == -1)
            {
                long long result = MM.MemoryAllocate(related_core_index, input_channel_element_num);
                if (result == -1)
                {
                    fprintf(stderr, "Allocate Memory Error For First Layer.\n");
                    MM.PrintMemoryInfo(related_core_index);
                    abort();
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[related_core_index].type_1_main_list[node_index].input_channel_list[related_input_channel_index].start_address = result;

                struct INST Instruction_ld;
                Instruction_ld.input_cycle_index = input_cycle_index;
                Instruction_ld.type = MEM;
                Instruction_ld.level_diff = 0;
                Instruction_ld.operation = "LD";
                Instruction_ld.stage = "INPUT";
                Instruction_ld.node_index = node_index;
                Instruction_ld.source = -1; // OP_INPUT
                Instruction_ld.source_address = -1;
                Instruction_ld.source_offset = related_input_channel_index * input_channel_element_num;
                Instruction_ld.destination_address = result;
                Instruction_ld.destination_offset = 0;
                Instruction_ld.element_num = PIMCOMP_4_Element_Memory_INFO.core_list[related_core_index].type_1_main_list[node_index].input_channel_element_num;
                Instruction_ld.instruction_group_index = instruction_group_index;
                PIMCOMP_4_base_instruction_ir[instruction_group_index].core_list[related_core_index].instruction_ir_list.push_back(Instruction_ld);
            }
        }
    }
}


void ElementPipelineSchedule::MemoryAllocationForAG(int AG_index_in_total, int core_index, bool need_input)
{
    //// 为每个AG生成一个用于保存输出的位置，并且这个位置保持不变。
    if (PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_start_address == -1)
    {
        int output_element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].output_element_num;
        long long result = MM.MemoryAllocate(core_index, output_element_num);
        if (result == -1)
        {
            fprintf(stderr, "Allocate Memory Error For AG Output.\n");
            MM.PrintMemoryInfo(core_index);
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_start_address = result;
        PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].output_element_num = output_element_num;
    }
    if (need_input) // 并非所有情况都需要input。比如recv时只需要output。
    {
        //// 为每个AG生成一个用于整理输入（img2col）的位置，并且这个位置保持不变。
        if (PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_start_address == -1)
        {
            int input_element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].input_element_num;
            long long result = MM.MemoryAllocate(core_index, input_element_num);
            if (result == -1)
            {
                fprintf(stderr, "Allocate Memory Error For AG Input.\n");
                MM.PrintMemoryInfo(core_index);
                abort();
            }
            PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_start_address = result;
            PIMCOMP_4_Element_Memory_INFO.core_list[core_index].type_0_AG_list[AG_index_in_total].input_element_num = input_element_num;
        }
    }
}

void ElementPipelineSchedule::MemoryAllocationForVec(int instruction_group_index, int node_index, int index_in_all_providers, int input_channel_index,int input_channel_element_num, struct Comm_struct Comm)
{
    int recv_core_index = post_node_map[node_index];
    if (PIMCOMP_4_Element_Memory_INFO.core_list[recv_core_index].type_2_vec_list[node_index].provider_list[index_in_all_providers].input_channel_list[input_channel_index].start_address == -1
        && VEC_input_channel_flag[node_index][index_in_all_providers][input_channel_index] == 0)
    {
        VEC_input_channel_flag[node_index][index_in_all_providers][input_channel_index] = 1;
        int element_num = input_channel_element_num;
        long long result = MM.MemoryAllocate(recv_core_index, element_num);
        if (result == -1)
        {
            fprintf(stderr, "Allocate Memory Error For Pool Input Failed.\n");
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[recv_core_index].type_2_vec_list[node_index].provider_list[index_in_all_providers].input_channel_list[input_channel_index].start_address = result;

        ScheduleSplitInstructionCOMM(instruction_group_index,
                                          Comm.node_index, Comm.core_index, Comm.start_address,
                                          node_index, recv_core_index, result,
                                          Comm.element_num);
    }
}


void ElementPipelineSchedule::MemoryAllocationForPool(int instruction_group_index, int node_index, int input_channel_index, struct Comm_struct Comm)
{
    int recv_core_index = post_node_map[node_index];
    if (PIMCOMP_4_Element_Memory_INFO.core_list[recv_core_index].type_3_pool_list[node_index].input_channel_list[input_channel_index].start_address == -1
        && POOL_input_channel_flag[node_index][input_channel_index] == 0 )
    {
        POOL_input_channel_flag[node_index][input_channel_index] = 1;
        int element_num = PIMCOMP_4_Element_Memory_INFO.core_list[recv_core_index].type_3_pool_list[node_index].input_channel_element_num;
        long long result = MM.MemoryAllocate(recv_core_index, element_num);
        if (result == -1)
        {
            fprintf(stderr, "Allocate Memory For Pool Input Failed.\n");
            abort();
        }
        PIMCOMP_4_Element_Memory_INFO.core_list[recv_core_index].type_3_pool_list[node_index].input_channel_list[input_channel_index].start_address = result;

        ScheduleSplitInstructionCOMM(instruction_group_index,
                                          Comm.node_index, Comm.core_index, Comm.start_address,
                                          node_index, recv_core_index, result,
                                          Comm.element_num);
    }
}


void ElementPipelineSchedule::MemoryFreeForPost()
{
    for (int j = 0; j < node_num; ++j)
    {
        if (post_node_index.count(j))
        {
            std::string operation = PIMCOMP_node_list[j].operation;
            int free_core_index = post_node_map[j];
            int free_element_num = PIMCOMP_node_list[j].output_dim[1];
            if (operation == "OP_POOL")
            {
                // 回收POOL分配的内存
                int pool_output_channel_num = PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_3_pool_list[j].output_start_address_record.size();
                for (int k = 0; k < pool_output_channel_num; ++k)
                {
                    long long output_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_3_pool_list[j].output_start_address_record[k];
                    bool result = MM.MemoryFree(free_core_index, output_start_address, free_element_num);
                    if (!result)
                    {
                        fprintf(stderr, "POOL Output Recycle Failed \n");
                        std::cout << free_core_index << " " << output_start_address << " " << free_element_num << std::endl;
                        abort();
                    }
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_3_pool_list[j].output_start_address_record.resize(0);
            }
            else
            {
                // 回收VEC分配的内存
                int vec_output_channel_num = PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_2_vec_list[j].output_start_address_record.size();
                for (int k = 0; k < vec_output_channel_num; ++k)
                {
                    long long output_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_2_vec_list[j].output_start_address_record[k];
                    bool result = MM.MemoryFree(free_core_index, output_start_address, free_element_num);
                    if (!result)
                    {
                        fprintf(stderr, "VEC Output Recycle Failed \n");
                        std::cout << free_core_index << " " << output_start_address << " " << free_element_num << std::endl;
                        abort();
                    }
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[free_core_index].type_2_vec_list[j].output_start_address_record.resize(0);
            }
        }
    }
}


void ElementPipelineSchedule::MemoryFreeForMain()
{
    for (int c = 0; c < ChipW * ChipH; ++c)
    {
        int AG_num = PIMCOMP_4_Element_Memory_INFO.core_list[c].type_0_AG_list.size();
        for (int AG_index = 0; AG_index < AG_num; ++AG_index)
        {
            long long input_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[c].type_0_AG_list[AG_index].input_start_address;
            int input_element_num = PIMCOMP_4_element_AG_info_list[AG_index].input_element_num;
            if (input_start_address != -1)
            {
                bool result = MM.MemoryFree(c, input_start_address, input_element_num);
                if (!result)
                {
                    fprintf(stderr, "AG Input Recycle Failed \n");
                    std::cout << c << " " << input_start_address << " " << input_element_num << std::endl;
                    abort();
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[c].type_0_AG_list[AG_index].input_start_address = -1;
            }

            long long output_start_address = PIMCOMP_4_Element_Memory_INFO.core_list[c].type_0_AG_list[AG_index].output_start_address;
            int output_element_num = PIMCOMP_4_element_AG_info_list[AG_index].output_element_num;
            if (output_start_address != -1)
            {
                bool result = MM.MemoryFree(c, output_start_address, output_element_num);
                if (!result)
                {
                    fprintf(stderr, "AG Output Recycle Failed \n");
                    std::cout << c << " " << output_start_address << " " << output_element_num << std::endl;
                    abort();
                }
                PIMCOMP_4_Element_Memory_INFO.core_list[c].type_0_AG_list[AG_index].output_start_address = -1;
            }
        }

        //// 回收BIAS
        for (int n = 0; n < bias_element_num_map[c].size(); ++n)
        {
            if(bias_element_num_map[c][n] != 0)
            {
                long long bias_start_address = bias_address_map[c][n];
                int bias_element_num = bias_element_num_map[c][n];
                bool result = MM.MemoryFree(c, bias_start_address, bias_element_num);
                if (!result)
                {
                    fprintf(stderr, "BIAS Recycle Failed \n");
                    std::cout << c << " " << bias_start_address << " " << bias_element_num << std::endl;
                    abort();
                }
                bias_address_map[c][n] = -1;
                bias_element_num_map[c][n] = 0;
            }
        }
    }
}


void ElementPipelineSchedule::ScheduleSplitInstructionPost(int instruction_group_index, int this_node_index, int next_node_index, int input_channel_index, struct Comm_struct COMM)
{
    if (next_node_index < 0 || next_node_index >= node_num)
        return;
    std::string next_operation = PIMCOMP_node_list[next_node_index].operation;
    if (next_operation == "OP_POOL")
    {
        int replication_num = node_replication_num[next_node_index];
        for (int k = 0; k < replication_num; ++k)
        {
            struct Post_Comm_struct post_comm;
            post_comm.node_index = next_node_index;
            post_comm.replication_index = k;
            post_comm.input_channel_index = input_channel_index;
            post_comm.is_pool = true;
            // post_comm.COMM包括了Send信息，比如send core、source address、element number
            post_comm.COMM.core_index = COMM.core_index;
            post_comm.COMM.start_address = COMM.start_address;
            post_comm.COMM.element_num = COMM.element_num;
            post_comm.COMM.node_index = COMM.node_index;
            Post_Comm_Vector.push_back(post_comm);
        }
    }
    else if (next_operation == "OP_ELTWISE" || next_operation == "OP_CONCAT" || next_operation == "OP_RELU" || next_operation == "OP_TANH" || next_operation == "OP_SIGMOID" || next_operation == "OP_PAD" || next_operation == "OP_SHUFFLE")
    {
        int replication_num = node_replication_num[next_node_index];
        int effective_provider_node_index = this_node_index; // 这里的this_node_index不会是CONV后面的FC。所以可以直接使用
        int input_channel_element_num = PIMCOMP_node_list[effective_provider_node_index].output_dim[1];
        int index_in_all_providers = PIMCOMP_4_element_node_provider_index_2_index_in_all_providers[next_node_index][effective_provider_node_index];

        for (int k = 0; k < replication_num; ++k)
        {
            struct Post_Comm_struct post_comm;
            post_comm.node_index = next_node_index;
            post_comm.replication_index = k;
            post_comm.input_channel_index = input_channel_index;
            post_comm.index_in_all_providers = index_in_all_providers;
            post_comm.input_channel_element_num = input_channel_element_num;
            post_comm.is_pool = false;
            post_comm.COMM.core_index = COMM.core_index;
            post_comm.COMM.start_address = COMM.start_address;
            post_comm.COMM.element_num = COMM.element_num;
            post_comm.COMM.node_index = COMM.node_index;
            Post_Comm_Vector.push_back(post_comm);
        }
    }
    else if (no_consider_node_set.count(next_operation))
    {
        if (next_node_index == last_node_index)
        {
            ScheduleSplitInstructionWriteBack(instruction_group_index, input_channel_index, COMM);
        }
        else
        {
            for (auto iter = PIMCOMP_topology_provider_consumer_relation[next_node_index].begin(); iter != PIMCOMP_topology_provider_consumer_relation[next_node_index].end() ; ++iter)
            {
                int next_next_node_index = *iter;
                ScheduleSplitInstructionPost(instruction_group_index, next_node_index, next_next_node_index, input_channel_index, COMM);
            }
        }
    }
    else if (next_operation == "OP_CONV" || next_operation == "OP_FC")
    {
        struct Main_Comm_struct CONV_FC_COMM;
        CONV_FC_COMM.next_node_index = next_node_index;
        CONV_FC_COMM.input_channel_index = input_channel_index;
        CONV_FC_COMM.COMM.core_index = COMM.core_index;
        CONV_FC_COMM.COMM.start_address = COMM.start_address;
        CONV_FC_COMM.COMM.element_num = COMM.element_num;
        CONV_FC_COMM.COMM.node_index = COMM.node_index;
        Main_Comm_Vector.push_back(CONV_FC_COMM);
    }
    else
    {
        fprintf(stderr, "Post Node Type Not Supported.\n");
        std::cout << "node_index:" << next_node_index << "    operation:" << next_operation << std::endl;
        abort();
    }
}

void ElementPipelineSchedule::ScheduleSplitInstructionMain(int instruction_group_index)
{
    for (int n = 0; n < node_num; ++n)
    {
        std::string operation = PIMCOMP_node_list[n].operation;
        if (operation == "OP_CONV" || operation == "OP_FC")
        {
            int effective_node_index = PIMCOMP_node_list[n].effective_node_index;
            int replication_num = PIMCOMP_node_list[n].replication_num;
            for (int r = 0; r < replication_num; ++r)
            {
                int first_AG_in_replication = PIMCOMP_2_AG_partition[effective_node_index].replication[r].AG_index[0];
                struct AG_info_schedule thisAG = PIMCOMP_4_element_AG_info_list[first_AG_in_replication];
                int node_index = thisAG.node_index;
                bool first_layer = thisAG.first_layer;
                int replication_index = thisAG.replication_index;
                int produced_output_channel_num = node_rep_split_produce_output_channel_num[node_index][replication_index];
                if (produced_output_channel_num >= node_rep_split_output_channel_num_list[node_index][replication_index])
                    continue;
                int output_channel_index = node_rep_split_output_channel_index_list[node_index][replication_index][produced_output_channel_num];
                if ( first_layer || CheckInputPrepared(node_index, replication_index, output_channel_index) )
                {
                    int AG_index_in_total = thisAG.AG_index_in_total; // AG_index_in_total == first_AG_in_replication
                    int AG_num_this_replication = thisAG.AG_num_per_replication;

                    if (first_layer)
                        MemoryAllocationForFirstLayer(instruction_group_index, output_channel_index);

                    ScheduleSplitInstructionStage1MVMUL(instruction_group_index, AG_index_in_total, AG_num_this_replication, output_channel_index);

                    ScheduleSplitInstructionStage3ACC(instruction_group_index, AG_index_in_total, AG_num_this_replication);

                    if (node_index != last_node_index && PIMCOMP_node_list[node_index].with_act)
                        ScheduleSplitInstructionStage3ACT(instruction_group_index, AG_index_in_total, output_channel_index);

                    ScheduleSplitInstructionStage3VER(instruction_group_index, AG_index_in_total, output_channel_index);

                    if (PIMCOMP_node_list[node_index].with_clip)
                        ScheduleSplitInstructionStage3CLIP(instruction_group_index, AG_index_in_total, output_channel_index);

                    CheckForIndex[node_index].insert(output_channel_index);
                    CheckForNum[node_index]++;
                    node_rep_split_produce_output_channel_num[node_index][replication_index] ++;
                    instruction_group_index_of_node[node_index].push_back(instruction_group_index);

                    struct Comm_struct COMM;
                    COMM.core_index = PIMCOMP_4_element_AG_info_list[AG_index_in_total].core_index;
                    COMM.start_address = PIMCOMP_4_Element_Memory_INFO.core_list[COMM.core_index].type_0_AG_list[AG_index_in_total].output_start_address;
                    COMM.element_num = PIMCOMP_4_element_AG_info_list[AG_index_in_total].output_element_num;
                    COMM.node_index = node_index;

                    if (node_index == last_node_index)
                    {
                        ScheduleSplitInstructionWriteBack(instruction_group_index, output_channel_index, COMM);
                    }
                    else  //// 传递给下一个节点
                    {
                        for (int next = 0; next < PIMCOMP_topology_provider_consumer_relation[node_index].size(); ++next)
                        {
                            int consumer_index = PIMCOMP_topology_provider_consumer_relation[node_index][next]; //// 一般情况CONV或FC都只有一个消费者
                            ScheduleSplitInstructionPost(instruction_group_index, node_index, consumer_index, output_channel_index, COMM);
                        }
                    }
                }
            }
        }
        else if (operation == "OP_POOL")
        {
            post_node_index.insert(n);
            int replication_num = node_replication_num[n];
            for (int k = 0; k < replication_num; ++k)
            {
                int try_output_channel_num = 0;
                while (try_output_channel_num < node_rep_split_output_channel_num_list[n][k])
                {
                    int output_channel_index_in_total = node_rep_split_output_channel_index_list[n][k][try_output_channel_num];
                    //// 进行池化操作
                    if (node_rep_split_complete_output_channel_index_flag[n][output_channel_index_in_total] == 0 && CheckInputPrepared(n, k, output_channel_index_in_total))
                    {
                        int input_channel_index_for_next = output_channel_index_in_total;
                        ScheduleSplitInstructionStage4Pool(instruction_group_index, n, input_channel_index_for_next);
                        node_rep_split_produce_output_channel_num[n][k]++;
                        node_rep_split_complete_output_channel_index_flag[n][output_channel_index_in_total] = 1;

                        CheckForIndex[n].insert(input_channel_index_for_next);
                        CheckForNum[n]++;
                        instruction_group_index_of_node[n].push_back(instruction_group_index);

                        // 传输给下一个节点的信息
                        struct Comm_struct pool_COMM;
                        pool_COMM.core_index = post_node_map[n];
                        pool_COMM.node_index = n;
                        pool_COMM.element_num = PIMCOMP_node_list[n].output_dim[1];
                        int current_output_num = PIMCOMP_4_Element_Memory_INFO.core_list[pool_COMM.core_index].type_3_pool_list[pool_COMM.node_index].output_start_address_record.size();
                        pool_COMM.start_address = PIMCOMP_4_Element_Memory_INFO.core_list[pool_COMM.core_index].type_3_pool_list[pool_COMM.node_index].output_start_address_record[current_output_num-1];

                        if (n == last_node_index) // 若该POOL节点是最后一个节点，就把结果写回DRAM
                        {
                            ScheduleSplitInstructionWriteBack(instruction_group_index, input_channel_index_for_next, pool_COMM);
                        }
                        else
                        {
                            // 考虑多个消费者
                            for (auto iter = PIMCOMP_topology_provider_consumer_relation[n].begin(); iter != PIMCOMP_topology_provider_consumer_relation[n].end() ; ++iter)
                            {
                                int next_node_index = *iter;
                                ScheduleSplitInstructionPost(instruction_group_index, n, next_node_index, input_channel_index_for_next, pool_COMM);
                            }
                        }
                    }
                    try_output_channel_num++;
                }
            }
        }
        else if (operation == "OP_ELTWISE" || operation == "OP_CONCAT" || operation == "OP_RELU" || operation == "OP_TANH" || operation == "OP_SIGMOID" || operation == "OP_PAD" || operation == "OP_SHUFFLE")
        {
            post_node_index.insert(n);
            int replication_num = node_replication_num[n];
            for (int k = 0; k < replication_num; ++k)
            {
                int try_output_channel_num = 0;
                while (try_output_channel_num < node_rep_split_output_channel_num_list[n][k])
                {
                    //// 进行VEC操作
                    int vec_rep_key_input_channel_index_current = node_rep_split_output_channel_index_list[n][k][try_output_channel_num];

                    if (node_rep_split_complete_output_channel_index_flag[n][vec_rep_key_input_channel_index_current] == 0 && CheckInputPrepared(n, k, vec_rep_key_input_channel_index_current))
                    {
                        int input_channel_index_for_next = vec_rep_key_input_channel_index_current;

                        if (operation == "OP_ELTWISE")
                        {
                            ScheduleSplitInstructionStage4Eltwise(instruction_group_index, n, input_channel_index_for_next);
                        }
                        else if (operation == "OP_CONCAT")
                        {
                            ScheduleSplitInstructionStage4Concat(instruction_group_index, n, input_channel_index_for_next);
                        }
                        else if(operation == "OP_RELU" || operation == "OP_TANH" || operation == "OP_SIGMOID")
                        {
                            ScheduleSplitInstructionStage4Activate(instruction_group_index, n, input_channel_index_for_next);
                        }
                        else if(operation == "OP_SHUFFLE")
                        {
                            ScheduleSplitInstructionStage4Shuffle(instruction_group_index, n, input_channel_index_for_next);
                        }
                        node_rep_split_produce_output_channel_num[n][k]++;
                        node_rep_split_complete_output_channel_index_flag[n][vec_rep_key_input_channel_index_current] = 1;
                        CheckForIndex[n].insert(input_channel_index_for_next);
                        CheckForNum[n]++;
                        instruction_group_index_of_node[n].push_back(instruction_group_index);

                        struct Comm_struct vec_COMM;
                        vec_COMM.core_index = post_node_map[n];
                        vec_COMM.element_num = PIMCOMP_node_list[n].output_dim[1];
                        vec_COMM.node_index = n;
                        int current_output_num = PIMCOMP_4_Element_Memory_INFO.core_list[vec_COMM.core_index].type_2_vec_list[vec_COMM.node_index].output_start_address_record.size();
                        vec_COMM.start_address = PIMCOMP_4_Element_Memory_INFO.core_list[vec_COMM.core_index].type_2_vec_list[vec_COMM.node_index].output_start_address_record[current_output_num-1];

                        if (n == last_node_index) // 若该VEC节点是最后一个节点，就把结果写回DRAM
                        {
                            int core_index = post_node_map[n];
                            ScheduleSplitInstructionWriteBack(instruction_group_index, input_channel_index_for_next, vec_COMM);
                        }
                        else
                        {
                            //// 考虑多个消费者
                            for (auto iter = PIMCOMP_topology_provider_consumer_relation[n].begin(); iter != PIMCOMP_topology_provider_consumer_relation[n].end() ; ++iter)
                            {
                                int next_node_index = *iter;
                                ScheduleSplitInstructionPost(instruction_group_index, n, next_node_index, input_channel_index_for_next, vec_COMM);
                            }
                        }
                    }
                    try_output_channel_num++;
                }
            }
        }
        else if (no_consider_node_set.count(operation))
        {
            continue;
        }
        else
        {
            fprintf(stderr, "Post Node Type Not Supported.\n");
            std::cout << "node_index:" << n << "    operation:" << operation << std::endl;
            abort();
        }
    }
}


void ElementPipelineSchedule::ScheduleSplitInstruction()
{
    int instruction_group_num = model_name == "vgg16" ? 15000 : 2000;
    PIMCOMP_4_base_instruction_ir.resize(instruction_group_num);
    PIMCOMP_GUI_memory_usage_every_instruction_group.resize(ChipW * ChipH);
    for (int i = 0; i < instruction_group_num; ++i)
    {
        std::cout << "Element Pipeline Instruction Group Index: " << i << std::endl;
        PIMCOMP_4_base_instruction_ir[i].core_list.resize(ChipH * ChipW);
        if (i == 0)
            ScheduleSplitInstructionStage0LoadBias(i); // Load BIAS
        // 执行Split数据流
        ScheduleSplitInstructionMain(i);
        // 判断是否运行到最后一个节点，并且最后一个节点已经生成完毕
        if (last_node_index > 0 && CheckForIndex[last_node_index].size() == last_node_output_channel_num && CheckForNum[last_node_index] == last_node_output_channel_num)
        {
            effective_instruction_group_num = i+1;
            PIMCOMP_4_base_instruction_ir.resize(effective_instruction_group_num);
            int last_node_output_dim_num = PIMCOMP_node_list[last_node_index].output_dim_num;
            int final_output_element_num = 1;
            for (int j = 0; j < last_node_output_dim_num; ++j)
                final_output_element_num *= PIMCOMP_node_list[last_node_index].output_dim[j];
//            if (final_output_element_num != write_back_element_total)
//            {
//                fprintf(stderr, "Write Back Failed\n");
//                std::cout << final_output_element_num << "  " << write_back_element_total << std::endl;
//                abort();
//            }
            MemoryFreeForMain();
            MemoryFreeForPost();
            for (int c = 0; c < ChipH * ChipW; ++c)
                PIMCOMP_GUI_memory_usage_every_instruction_group[c].push_back(static_cast<float>(MM.GetCoreElementNum(c)) * ArithmeticPrecision / 8 / 1024);
            std::cout << static_cast<float>(MM.GetMaxElementNum()) * ArithmeticPrecision / 8 / 1024 << " kB" <<  std::endl;
            std::cout << std::endl << "[effective_instruction_group_num]:" << effective_instruction_group_num << std::endl;
            pipeline_effective_instruction_group_num = effective_instruction_group_num;
            break;
        }
        else
        {
            // 执行前面缓存的数据传输指令
            for (int j = 0; j < Main_Comm_Vector.size(); ++j)
            {
                struct Main_Comm_struct Main_Comm = Main_Comm_Vector[j];
                ScheduleSplitInstructionCommForMain(i, Main_Comm);
            }
            for (int j = 0; j < Post_Comm_Vector.size(); ++j)
            {
                struct Post_Comm_struct Post_Comm = Post_Comm_Vector[j];
                ScheduleSplitInstructionCOMMForPost(i, Post_Comm);
            }
            Main_Comm_Vector.clear();
            Post_Comm_Vector.clear();
        }
        // 回收POST节点分配的内存 (Main节点分配的内存在流程中会进行释放)
        MemoryFreeForPost();
        std::cout << static_cast<float>(MM.GetMaxElementNum()) * ArithmeticPrecision / 8 / 1024 << " kB" <<  std::endl;
        for (int c = 0; c < ChipH * ChipW; ++c)
            PIMCOMP_GUI_memory_usage_every_instruction_group[c].push_back(static_cast<float>(MM.GetCoreElementNum(c)) * ArithmeticPrecision / 8 / 1024);
    }
}


void ElementPipelineSchedule::ScheduleExecution()
{
    instruction_group_index_of_node.resize(node_num);
    element_pipeline = 1;
    SchedulePreparation();
    MemoryPreparation();
//    SavePreparation();
    CheckForIndex.resize(node_num);
    CheckForNum.resize(node_num);

    ScheduleSplitInstruction();
    comm_pair_total_num = comm_index;

    Check();
    Clear();
}


////////////////////////////////////// Check Clear And Save //////////////////////////////////////
void ElementPipelineSchedule::Check()
{
    std::cout << "================= Check Result =================" << std::endl;
    std::cout << "Node    Expected    Index     Num" << std::endl;
    bool pass = true;
    for (int i = 0; i < node_num; ++i)
    {
        //// 注意这里的逻辑，PIMCOMP_topology_provider_consumer_relation中是包含了那些非CONV/FC后的relu，而no_consider_node_set包括了不考虑的操作
        //// 所以PIMCOMP_topology_provider_consumer_relation[i].size() > 0是剔除了那些CONV/FC之后紧跟的relu
        if (!no_consider_node_set.count(PIMCOMP_node_list[i].operation) && PIMCOMP_topology_provider_consumer_relation[i].size() > 0)
        {
            std::cout << std::setw(3) << i << ":" << std::setw(10) <<  node_output_channel_num[i] << std::setw(10) << CheckForIndex[i].size() << std::setw(10) << CheckForNum[i] << std::endl;
            if (node_output_channel_num[i] != CheckForIndex[i].size() || CheckForIndex[i].size() != CheckForNum[i])
                pass = false;
        }
    }
    if (pass)
        std::cout << "----------------- PASS -----------------" << std::endl;
    else
        std::cout << "----------------- FAIL -----------------" << std::endl;
}


void ElementPipelineSchedule::Clear()
{
    PIMCOMP_4_element_AG_info_list.clear();

    node_AG_mapping.clear();
    node_AG0_index_in_replication.clear();
    node_replication_num.clear();
    node_output_channel_num.clear();

    node_rep_split_output_channel_index_list.clear();
    node_rep_split_output_channel_num_list.clear();
    node_rep_split_ready_input_channel_index_list.clear();
    node_rep_split_produce_output_channel_num.clear();
    node_rep_split_key_input_channel_index_list.clear();
    comm_index = 0;
    CheckForIndex.clear();
    CheckForNum.clear();

    Main_Comm_Vector.clear();
    Post_Comm_Vector.clear();
}



void ElementPipelineSchedule::SaveInstruction()
{
    std::ofstream OutFile("../output/ElementPipeline.inst", std::ios::out | std::ios::trunc);
    for (int i = 0; i < effective_instruction_group_num; ++i)
//    for (int i = 0; i < 2; ++i)
    {
        OutFile << "========================================= base instruction_group " << i << " =========================================" << std::endl;
        for (int j = 0; j < ChipW * ChipH; ++j)
        {
            int instruction_num = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list.size();
            if (instruction_num == 0)
                continue;
            OutFile << "core " << j << std::endl;
            for (int k = 0; k < instruction_num; ++k)
            {
                struct INST Instruction = PIMCOMP_4_base_instruction_ir[i].core_list[j].instruction_ir_list[k];
                SaveSingleInstructionWithAddress(OutFile, Instruction, i, j);
            }
        }
    }
    OutFile.close();
}

int ElementPipelineSchedule::GetInputChannelFromOutputIndex(int node_index, int output_index, bool is_last)
/**
 * @brief 用于CONV或POOL节点，根据其output channel index，获取该划窗在输入特征图上的第一个"或"最后一个input channel index
 * For CONV or POOL nodes, according to its output channel index, get the first "or" last input channel index of the sliding window on the input feature map
 * @param node_index    节点序号
 * index of node
 * @param output_index  节点的output channel index
 * output channel index of node
 * @param is_last       想获得的划窗中第一个还是最后一个input channel index
 * need the first or the last input channel index
 * @return 返回第一个或最后一个input channel index
 * return the first "or" last input channel index of the sliding window on the input feature map
 */
{
    struct PIMCOMP_node Node = PIMCOMP_node_list[node_index];
    struct param Params = Node.param;
    int input_H = Node.input_dim[2];
    int input_W = Node.input_dim[3];
    int conv_kernel_w = Params.kernel_w;
    int conv_kernel_h = Params.kernel_h;
    int conv_padding_h0 = Params.pad_h0;
    int conv_padding_h1 = Params.pad_h1;
    int conv_padding_w0 = Params.pad_w0;
    int conv_padding_w1 = Params.pad_w1;
    int conv_stride_w = Params.stride_w;
    int conv_stride_h = Params.stride_h;

    int output_W = floor(float(input_W + conv_padding_w0 + conv_padding_w1 - conv_kernel_w) / float(conv_stride_w)) + 1;
    int output_H = floor(float(input_H + conv_padding_h0 + conv_padding_h1 - conv_kernel_h) / float(conv_stride_h)) + 1;
    int info_output_W = Node.output_dim[3];
    int info_output_H = Node.output_dim[2];
    if (info_output_W != output_W || info_output_H != output_H)
    {
        std::cout << Node.name << std::endl;
        std::cout << "Output Size Doesn't Match" << std::endl;
        return -1;
    }
    int normal_start_index_in_w = conv_padding_w0/conv_stride_w + (conv_padding_w0 % conv_stride_w == 0 ? 0 : 1);
    int normal_start_index_in_h = conv_padding_h0/conv_stride_h + (conv_padding_h0 % conv_stride_h == 0 ? 0 : 1);

    int i = output_index / output_W;
    int j = output_index % output_W;
    int start_address = i * conv_stride_h * input_W + j *  conv_stride_w;
    if (j < normal_start_index_in_w)
        start_address -= (j * conv_stride_w);
    else
        start_address -= conv_padding_w0;
    if (i < normal_start_index_in_h)
        start_address -= (i * conv_stride_h * input_W);
    else
        start_address -= conv_padding_h0 * input_W;

    int start_row = start_address / input_W;
    int start_col = start_address % input_W;

    int conv_w_num = conv_kernel_w;
    if (j < normal_start_index_in_w)
        conv_w_num = conv_w_num - conv_padding_w0 + j * conv_stride_w;
    if (start_col + conv_w_num > input_W)
        conv_w_num = conv_w_num - (start_col + conv_w_num - input_W);

    int conv_h_num = conv_kernel_h;
    if (i < normal_start_index_in_h)
        conv_h_num = conv_h_num - conv_padding_h0 + i * conv_stride_h;
    if (start_row + conv_h_num > input_H)
        conv_h_num = conv_h_num - (start_row + conv_h_num - input_H);

    int h = 0;
    int w = 0;
    if (is_last)
    {
        h = conv_h_num-1;
        w = conv_w_num-1;
    }
    int position = start_address + w + h * input_W; // input_index
    return position;
}

void ElementPipelineSchedule::GetInputChannelFromOutputIndex(int *first_last, int node_index, int output_index)
/**
 * @brief 用于CONV或POOL节点，根据其output channel index，获取该划窗在输入特征图上的第一个"和"最后一个input channel index
 * For CONV or POOL nodes, according to its output channel index, get the first "and" last input channel index of the sliding window on the input feature map
 * @param first_last    数组，保存第一个和最后一个input channel index
 * Array , save the first and last input channel index
 * @param node_index    节点序号
 * index of node
 * @param output_index  节点的output channel index
 * output channel index of node
 * @return 返回第一个和最后一个input channel index
 * return the first and the last input channel index of the sliding window on the input feature map
 */
{
    struct PIMCOMP_node Node = PIMCOMP_node_list[node_index];
    struct param Params = Node.param;
    int input_H = Node.input_dim[2];
    int input_W = Node.input_dim[3];
    int conv_kernel_w = Params.kernel_w;
    int conv_kernel_h = Params.kernel_h;
    int conv_padding_h0 = Params.pad_h0;
    int conv_padding_h1 = Params.pad_h1;
    int conv_padding_w0 = Params.pad_w0;
    int conv_padding_w1 = Params.pad_w1;
    int conv_stride_w = Params.stride_w;
    int conv_stride_h = Params.stride_h;

    int output_W = floor(float(input_W + conv_padding_w0 + conv_padding_w1 - conv_kernel_w) / float(conv_stride_w)) + 1;
    int output_H = floor(float(input_H + conv_padding_h0 + conv_padding_h1 - conv_kernel_h) / float(conv_stride_h)) + 1;
    int info_output_W = Node.output_dim[3];
    int info_output_H = Node.output_dim[2];
    if (info_output_W != output_W || info_output_H != output_H)
    {
        std::cout << info_output_H << " " << output_W << std::endl;
        std::cout << " Output Size Doesn't Match" << std::endl;
        return ;
    }
    int normal_start_index_in_w = conv_padding_w0/conv_stride_w + (conv_padding_w0 % conv_stride_w == 0 ? 0 : 1);
    int normal_start_index_in_h = conv_padding_h0/conv_stride_h + (conv_padding_h0 % conv_stride_h == 0 ? 0 : 1);

    int i = output_index / output_W;
    int j = output_index % output_W;
    int start_address = i * conv_stride_h * input_W + j *  conv_stride_w;
    if (j < normal_start_index_in_w)
        start_address -= (j * conv_stride_w);
    else
        start_address -= conv_padding_w0;
    if (i < normal_start_index_in_h)
        start_address -= (i * conv_stride_h * input_W);
    else
        start_address -= conv_padding_h0 * input_W;

    int start_row = start_address / input_W;
    int start_col = start_address % input_W;

    int conv_w_num = conv_kernel_w;
    if (j < normal_start_index_in_w)
        conv_w_num = conv_w_num - conv_padding_w0 + j * conv_stride_w;
    if (start_col + conv_w_num > input_W)
        conv_w_num = conv_w_num - (start_col + conv_w_num - input_W);

    int conv_h_num = conv_kernel_h;
    if (i < normal_start_index_in_h)
        conv_h_num = conv_h_num - conv_padding_h0 + i * conv_stride_h;
    if (start_row + conv_h_num > input_H)
        conv_h_num = conv_h_num - (start_row + conv_h_num - input_H);

    first_last[0] = start_address;
    first_last[1] = start_address + (conv_w_num-1) + (conv_h_num-1) * input_W;
}
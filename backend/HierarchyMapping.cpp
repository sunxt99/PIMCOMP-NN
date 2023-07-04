//
// Created by SXT on 2022/8/19.
//

#include "HierarchyMapping.h"

static int ArrayGroupIndex = 0;

HierarchyMapping::HierarchyMapping()
{
    node_num = PIMCOMP_node_list.size();
    ResourceList = new int[ChipW*ChipH];
    for (int i = 0; i < ChipW * ChipH; ++i)
        ResourceList[i] = CoreW * CoreH;
    PIMCOMP_2_resource_info.Core = 0;
    PIMCOMP_3_virtual_core_crossbar_map.resize(ChipH * ChipW);
    PIMCOMP_4_virtual_core_AG_map.core_list.resize(ChipH * ChipW);
}


void HierarchyMapping::MapHierarchy(std::string replicating_method)
{
//    ShowOriginalInfo();

    if (replicating_method != "GA")
    {
        /// Baseline
//        MapBaseline();
        /// Regular
        MapBaseline2();
        /// Distributed
//        MapDistributed();
    }
    else
    {
        LoadGAMappingResult(0);
    }

    Check();
    // AllocateMapInfo() was once in BatchPipelineSchedule stage. Now we put it here.
    // 以前这部分在BatchPipelineSchedule中，现在放到该部分。
    AllocateMapInfo();
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (PIMCOMP_3_virtual_core_crossbar_map[i].size() != 0)
            PIMCOMP_2_resource_info.Core += 1;
    }
//    FastEvaluationForElement();
//    FastEvaluationForBatch();
//    FastEvaluationInstructionGroupNum();
    Clear();
}

void HierarchyMapping::FastEvaluationInstructionGroupNum()
{
    std::vector<int> tmp_replication_num_vector;
    tmp_replication_num_vector.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_CONV" || PIMCOMP_node_list[i].operation == "OP_FC")
            tmp_replication_num_vector[i] = PIMCOMP_node_list[i].replication_num;
    }
    double max_delay = 0.0;
    double effective_first_conv_replication;
    double first_conv_instruction_group_num = 0;
    for (int i = 0; i < node_num; ++i)
    {
        int first_conv_replication;
        // the first conv
        if (PIMCOMP_node_list_origin[i].provider_index.size() == 1 && PIMCOMP_node_list_origin[i].provider_index[0] == 0)
        {
            first_conv_replication = tmp_replication_num_vector[i];
            first_conv_instruction_group_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3] / first_conv_replication;
        }
        // the last conv or pool
        if (PIMCOMP_topology_provider_consumer_relation[i].size() == 1 && PIMCOMP_topology_provider_consumer_relation[i][0] == -1)
        {
//          for (int j = 0; j < PIMCOM_EP_path_to_conv_or_pool[i].size(); ++j)
            int path_num = PIMCOMP_EP_path_to_conv_or_pool[i].size();
            for (int j = 0; j < std::min(path_num, 500); ++j)
            {
                int pool_replication_num;
                std::vector<double> layer_run;
                std::vector<double> layer_wait;
                for (int k = 0; k < PIMCOMP_EP_path_to_conv_or_pool[i][j].size(); ++k)
                {
                    int conv_or_pool_node = PIMCOMP_EP_path_to_conv_or_pool[i][j][k];
                    int replication_num = tmp_replication_num_vector[conv_or_pool_node];
                    if (replication_num != 0)
                    {
                      pool_replication_num = replication_num;
//                        pool_replication_num = 1;
                    }
                    else
                    {
                        replication_num = pool_replication_num;
                    }
                    tmp_replication_num_vector[conv_or_pool_node] = replication_num;
                    if (k != 0)
                    {
                        int previous_conv_or_pool_node = PIMCOMP_EP_path_to_conv_or_pool[i][j][k-1];
                        if (tmp_replication_num_vector[conv_or_pool_node] > tmp_replication_num_vector[previous_conv_or_pool_node])
                            tmp_replication_num_vector[conv_or_pool_node] = tmp_replication_num_vector[previous_conv_or_pool_node];
//                        std::cout << conv_or_pool_node << " " << PIMCOM_node_list_origin[conv_or_pool_node].operation << " " << tmp_replication_num_vector[conv_or_pool_node] << " " << PIMCOM_EP_delay_for_conv_and_pool[conv_or_pool_node] << std::endl;
                        double wait_ratio = PIMCOMP_EP_delay_for_conv_and_pool[conv_or_pool_node];
                        double delay_factor;
                        if (tmp_replication_num_vector[conv_or_pool_node] >= tmp_replication_num_vector[previous_conv_or_pool_node])
                            delay_factor = 1;
                        else
                            delay_factor = double(tmp_replication_num_vector[previous_conv_or_pool_node]) / double(tmp_replication_num_vector[conv_or_pool_node]);
                        layer_wait.push_back(wait_ratio);
                        layer_run.push_back((1-wait_ratio)*delay_factor);
                    }
                }
                double whole_time = 1.0;
                for (int k = 0; k < layer_run.size(); ++k)
                {
                    whole_time *= layer_run[layer_run.size() - 1 - k];
                    whole_time += layer_wait[layer_wait.size() - 1 - k];
                }
//                std::cout << std::fixed << whole_time / first_conv_replication * 10000 << std::endl;
//                    if (whole_time / first_conv_replication * 10000 > max_delay)
//                        max_delay = whole_time / first_conv_replication * 10000;
//                    double effective_first_conv_replication = first_conv_replication > ChipW * ChipH ? ChipW * ChipH : first_conv_replication;
                effective_first_conv_replication = first_conv_replication;
                if (whole_time / effective_first_conv_replication > max_delay)
                    max_delay = whole_time / effective_first_conv_replication ;
            }
        }
    }
    std::cout << max_delay * effective_first_conv_replication << std::endl;
    std::cout << first_conv_instruction_group_num << std::endl;
    std::cout << max_delay * effective_first_conv_replication * first_conv_instruction_group_num << std::endl;
}

void HierarchyMapping::LoadGAMappingResult(int candidate_index)
{
    for (int i = 0; i < PIMCOMP_2_AG_partition.size(); ++i)
    {
        int node_index = PIMCOMP_2_AG_partition[i].index;
        int replication_num = PIMCOMP_2_AG_partition[i].replication_num;
        int crossbar_num_per_AG = PIMCOMP_2_AG_partition[i].crossbar_num_per_AG;
        int AG_num_per_replication = PIMCOMP_2_AG_partition[i].AG_num_per_replication;
        for (int j = 0; j < replication_num; ++j)
        {
            for (int k = 0; k < AG_num_per_replication; ++k)
            {
                struct std::vector<struct PIMCOMP_2_virtual_crossbar> whole;
                int AG_index = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].AG_index;
                for (int l = 0; l < crossbar_num_per_AG; ++l)
                {
                    int crossbar_index = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list[l];
                    int core_index = PIMCOMP_DSE_node_map_info[candidate_index][node_index][j][k];
                    PIMCOMP_2_virtual_crossbar[crossbar_index].vcore_index = core_index;
                    PIMCOMP_2_virtual_crossbar[crossbar_index].vcore = core_index;
                    PIMCOMP_2_virtual_crossbar[crossbar_index].index_in_vcore = k * crossbar_num_per_AG + l;
                    PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_core_list.push_back(core_index);
                    PIMCOMP_3_virtual_core_crossbar_map[core_index].push_back(crossbar_index);
                    whole.push_back(PIMCOMP_2_virtual_crossbar[crossbar_index]);
                }
                PIMCOMP_3_hierarchy_map.whole.push_back(whole);
                PIMCOMP_3_hierarchy_map.whole_index.push_back(AG_index);
            }
        }
    }
}

///////////////////////////////////////////////////// Baseline Method /////////////////////////////////////////////////////
void HierarchyMapping::MapBaseline()
{
    // Every core can only hold one node.
    // 因为每个核上只能有一个节点，所以用一个vector代替即可。
    std::vector<int> core_node_record;
    core_node_record.resize(ChipH * ChipW);
    for (auto & n : core_node_record)
        n = -1;
    int node_num_partition = PIMCOMP_2_AG_partition.size();
    for (int i = 0; i < node_num_partition; ++i)
    {
        int node_index = PIMCOMP_2_AG_partition[i].index;
        int replication_num = PIMCOMP_2_AG_partition[i].replication_num;
        for (int j = 0; j < replication_num; ++j)
        {
            int ag_num_current_replication = PIMCOMP_2_AG_partition[i].replication[j].AG_list.size();
            for (int k = 0; k < ag_num_current_replication; ++k)
            {
                int crossbar_need_current_ag = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list.size();
                bool mapped = false;
                for (int l = 0; l < ChipW * ChipH; ++l)
                {
                    if (crossbar_need_current_ag <= ResourceList[l])
                    {
                        // if this core has other node, then we skip this core and try the next core.
                        // 如果该核有其它节点，那么不选择该核，在下一个核映射。确保每个核只有一个节点的AG。
                        if (core_node_record[l] != -1 && core_node_record[l] != node_index )
                            continue;
                        core_node_record[l]= node_index;
                        struct std::vector<struct PIMCOMP_2_virtual_crossbar> whole;
                        for (int m = 0; m < crossbar_need_current_ag; ++m)
                        {
                            int crossbar_index = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list[m];
                            PIMCOMP_2_virtual_crossbar[crossbar_index].vcore_index = l;
                            PIMCOMP_3_virtual_core_crossbar_map[l].push_back(crossbar_index);
                            PIMCOMP_2_virtual_crossbar[crossbar_index].vcore = l;
                            PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_core_list.push_back(l);
                            PIMCOMP_2_virtual_crossbar[crossbar_index].index_in_vcore = CoreW*CoreH-ResourceList[l];
                            ResourceList[l] -= 1;
                            whole.push_back(PIMCOMP_2_virtual_crossbar[crossbar_index]);
                        }
                        PIMCOMP_3_hierarchy_map.whole.push_back(whole);
                        PIMCOMP_3_hierarchy_map.whole_index.push_back(ArrayGroupIndex);
                        mapped = true;
                        break;
                    }
                }
                ArrayGroupIndex += 1;
            }
        }
    }
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        std::cout << ResourceList[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "split AG num: " << PIMCOMP_3_hierarchy_map.split_index.size() << std::endl;
}


/////////////////////////////////////////////////////  Method /////////////////////////////////////////////////////
void HierarchyMapping::MapBaseline2()
{
    int node_num_partition = PIMCOMP_2_AG_partition.size();
    for (int i = 0; i < node_num_partition; ++i)
    {
        int replication_num = PIMCOMP_2_AG_partition[i].replication_num;
        for (int j = 0; j < replication_num; ++j)
        {
            int ag_num_current_replication = PIMCOMP_2_AG_partition[i].replication[j].AG_list.size();
            for (int k = 0; k < ag_num_current_replication; ++k)
            {
                int crossbar_need_current_ag = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list.size();
                bool mapped = false;
                for (int l = 0; l < ChipW * ChipH; ++l)
                {
                    if (crossbar_need_current_ag <= ResourceList[l])
                    {
                        struct std::vector<struct PIMCOMP_2_virtual_crossbar> whole;
                        for (int m = 0; m < crossbar_need_current_ag; ++m)
                        {
                            int crossbar_index = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list[m];
                            PIMCOMP_2_virtual_crossbar[crossbar_index].vcore_index = l;
                            PIMCOMP_3_virtual_core_crossbar_map[l].push_back(crossbar_index);
                            PIMCOMP_2_virtual_crossbar[crossbar_index].vcore = l;
                            PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_core_list.push_back(l);
                            PIMCOMP_2_virtual_crossbar[crossbar_index].index_in_vcore = CoreW*CoreH-ResourceList[l];
                            ResourceList[l] -= 1;
                            whole.push_back(PIMCOMP_2_virtual_crossbar[crossbar_index]);
                        }
                        PIMCOMP_3_hierarchy_map.whole.push_back(whole);
                        PIMCOMP_3_hierarchy_map.whole_index.push_back(ArrayGroupIndex);
//                        ResourceList[l] -= crossbar_need_current_ag;
                        mapped = true;
                        break;
                    }
                }
                ArrayGroupIndex += 1;
            }
        }
    }
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        std::cout << ResourceList[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "split AG num: " << PIMCOMP_3_hierarchy_map.split_index.size() << std::endl;
}



////////////////////////////////////////////////// Distributed Method //////////////////////////////////////////////////
int cmp(const std::pair<struct MapSortStruct, int> &x, const std::pair<struct MapSortStruct, int> &y) { return x.first.ratio > y.first.ratio; }
int cmp2(const struct AGMapStruct &x, const struct AGMapStruct &y) {return x.AG_index_in_total < y.AG_index_in_total;}

void HierarchyMapping::MapDistributed()
{
    PIMCOMP_3_hierarchy_map.whole.resize(PIMCOMP_2_resource_info.AGs);
    PIMCOMP_3_hierarchy_map.whole_index.resize(PIMCOMP_2_resource_info.AGs);
    // Preprocess
    for (int i = 0; i < PIMCOMP_2_effective_node.size(); ++i)
    {
        struct MapSortStruct tmp;
        int input_cycle_per_replication = PIMCOMP_2_AG_partition[i].input_cycle_in_total / PIMCOMP_2_AG_partition[i].replication_num;
        int crossbar_num_per_AG = PIMCOMP_2_AG_partition[i].crossbar_num_per_AG;
        int AG_num_per_replication = PIMCOMP_2_AG_partition[i].AG_num_per_replication;
        int node_index = PIMCOMP_2_AG_partition[i].index;
        int height = PIMCOMP_2_AG_partition[i].Height;
        int crossbar_num_per_replication = AG_num_per_replication * crossbar_num_per_AG;
        float ratio = float(input_cycle_per_replication) / float(crossbar_num_per_replication);
        tmp.input_cycle_per_replication = input_cycle_per_replication;
        tmp.crossbar_num_per_AG = crossbar_num_per_AG;
        tmp.AG_num_per_replication = AG_num_per_replication;
        tmp.crossbar_num_per_replication = crossbar_num_per_replication;
        tmp.node_index = node_index;
        tmp.operation = PIMCOMP_node_list[node_index].operation;
        tmp.height = height;
        tmp.ratio = ratio;
        PIMCOMP_3_compute_crossbar_ratio.push_back(std::pair<struct MapSortStruct, int>(tmp,i));
    }
//    std::sort(PIMCOMP_3_compute_crossbar_ratio.begin(), PIMCOMP_3_compute_crossbar_ratio.end(), cmp);

    GatherForAccelerate();

    // Call MapDistributedTry() multiple times to get a legal solution
    // 多次调用分散映射函数，得到合法解
    int try_index = 0;
    PIMCOMP_3_mapping_result.resize(ChipH * ChipW);
    while (!MapDistributedTry())
    {
        if (try_index > 1000)
        {
            throw std::runtime_error("Mapping Failed. Please Change Replication Num");
        }
        try_index ++ ;
        for (int i = 0; i < ChipW * ChipH; ++i)
        {
            ResourceList[i] = CoreW * CoreH;
        }
        PIMCOMP_3_mapping_result.clear();
        PIMCOMP_3_mapping_result.resize(ChipH * ChipW);
    }

    // Sort the results to ensure that the AG_index in each core is increasing
    // 对结果进行排序，保证每个核中的AG_index是递增的
    for (int i = 0; i < PIMCOMP_3_mapping_result.size(); ++i)
    {
        std::sort(PIMCOMP_3_mapping_result[i].begin(), PIMCOMP_3_mapping_result[i].end(), cmp2);
    }

    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        std::cout << ResourceList[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "distributed mapping done" << std::endl;
//    int sss = 0; for (int j = 0; j < ChipH * ChipW; ++j) { std::cout << ResourceList[j] << " "; sss += ResourceList[j]; } std::cout << "  Total:" << sss << std::endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////   Preparation  //////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int HierarchyMapping::GetInputElementNumFromAG(int node_index, int index_in_replication)
{
    int H = 0;
    if (PIMCOMP_node_list[node_index].operation == "OP_CONV")
    {
        H = PIMCOMP_node_list[node_index].param.kernel_h * PIMCOMP_node_list[node_index].param.kernel_w * PIMCOMP_node_list[node_index].param.input_channel;
    }
    else if(PIMCOMP_node_list[node_index].operation == "OP_FC")
    {
        H = PIMCOMP_node_list[node_index].param.num_input;
    }
    int AG_H_num_in_replication = ceil(float(H) / float(CrossbarH));
    if (index_in_replication == AG_H_num_in_replication-1)
    {
        return (H - (AG_H_num_in_replication-1)*CrossbarH);
    }
    else
    {
        return CrossbarH;
    }
}

// ACC means for accelerate
// ACC_input_window_element_num[i]: node_i's output element num
// ACC_input_window_element_num[i] 代表node_index为i的完整输出长度
static std::vector<int> ACC_input_window_element_num;
// ACC_input_element_num_from_AG_index_in_replication[i][j]: replication_j of node_i's input element num
// ACC_input_element_num_from_AG_index_in_replication[i][j] 代表node_index为i、AG_index_in_replication为j的AG对应的输入长度
static std::vector<std::vector<int>> ACC_input_element_num_from_AG_index_in_replication;
void HierarchyMapping::GatherForAccelerate()
{
    ACC_input_window_element_num.resize(node_num);
    ACC_input_element_num_from_AG_index_in_replication.resize(node_num);
    for (int i = 0; i < PIMCOMP_node_list.size(); ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_CONV" || PIMCOMP_node_list[i].operation == "OP_FC")
        {
            ACC_input_window_element_num[i] = PIMCOMP_node_list[i].H; // only CONV need this parameter
        }
    }
    for (int i = 0; i < PIMCOMP_3_compute_crossbar_ratio.size(); ++i)
    {
        struct MapSortStruct node_AG_hybrid_info = PIMCOMP_3_compute_crossbar_ratio[i].first;
        int node_index = node_AG_hybrid_info.node_index;
        int AG_num_per_replication = node_AG_hybrid_info.AG_num_per_replication;
        int H = node_AG_hybrid_info.height;
        std::string operation = node_AG_hybrid_info.operation;
        for (int j = 0; j < AG_num_per_replication; ++j)
        {
            if (j == AG_num_per_replication-1)
                ACC_input_element_num_from_AG_index_in_replication[node_index].push_back(H - (AG_num_per_replication-1) * CrossbarH);
            else
                ACC_input_element_num_from_AG_index_in_replication[node_index].push_back(CrossbarH);
        }
    }
}

int HierarchyMapping::MapDistributedTry()
{
    int accumulated_replication_num = 0;
    for (auto iter = PIMCOMP_3_compute_crossbar_ratio.begin();  iter != PIMCOMP_3_compute_crossbar_ratio.end() ; iter++)
    {
        int effective_node_index = iter->second;
        int AG_num_this_replication = iter->first.AG_num_per_replication;
        int crossbar_num_per_AG = iter->first.crossbar_num_per_AG;
        int crossbar_num_this_replication = iter->first.crossbar_num_per_replication;
        int node_index = PIMCOMP_2_AG_partition[effective_node_index].index;
        std::string operation = PIMCOMP_node_list[node_index].operation;
        int output_element_num_for_AG = operation == "OP_CONV" ? (PIMCOMP_node_list[node_index].param.output_channel) : (PIMCOMP_node_list[node_index].param.num_output);
        int replication_num = PIMCOMP_2_AG_partition[effective_node_index].replication_num;
        bool distributed = effective_node_index < 10 ? 1:0;
        // resnet18 distributed = 20; core_index_offset = i/8;
        // vgg16 distributed = 20; core_index_offset = i/8;

        for (int i = 0; i < replication_num; ++i)
        {
            bool replication_mapped = false;
            if (distributed)
            {
//                int core_index_offset = i / 6;
//                int core_index_offset = i / 8;
                int core_index_offset = i / 4;
                for (int j = (core_index_offset + accumulated_replication_num) % (ChipH * ChipH); j < (core_index_offset + accumulated_replication_num) % (ChipH * ChipH) + (ChipH * ChipH); ++j)
                {
                    int index_j = j;
                    if (index_j >= ChipH * ChipW)
                    {
                        index_j -= ChipH * ChipW;
                    }
                    if (ResourceList[index_j] >= crossbar_num_this_replication)
                    {
                        ResourceList[index_j] -= crossbar_num_this_replication;
                        replication_mapped = true;

                        for (int k = 0; k < AG_num_this_replication; ++k)
                        {
                            struct AGMapStruct thisAG;
                            thisAG.node_index = node_index;
                            thisAG.replication_index = i;
                            thisAG.index_in_replication = k;
                            thisAG.input_element_num = ACC_input_element_num_from_AG_index_in_replication[node_index][k];
                            thisAG.output_element_num = output_element_num_for_AG;
                            thisAG.AG_index_in_total = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_index[k];
//                            thisAG.input_cycle_num = PIMCOMP_2_AG_partition[effective_node_index].replication[i].input_cycle_this_replication;
//                            thisAG.instruction_group_num = ceil(float(thisAG.input_cycle_num) / float(operation_cycle_before_comm));
                            PIMCOMP_3_mapping_result[index_j].push_back(thisAG);
                        }

                        for (int k = 0; k < AG_num_this_replication; ++k)
                        {
                            struct std::vector<struct PIMCOMP_2_virtual_crossbar> whole;
                            int AG_index = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].AG_index;
                            for (int l = 0; l < crossbar_num_per_AG; ++l)
                            {
                                int crossbar_index = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].virtual_crossbar_list[l];
                                PIMCOMP_2_virtual_crossbar[crossbar_index].vcore_index = index_j;
                                PIMCOMP_2_virtual_crossbar[crossbar_index].vcore = index_j;
                                PIMCOMP_2_virtual_crossbar[crossbar_index].index_in_vcore = k * crossbar_num_per_AG + l;
                                PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].virtual_core_list.push_back(index_j);
                                PIMCOMP_3_virtual_core_crossbar_map[index_j].push_back(crossbar_index);
                                whole.push_back(PIMCOMP_2_virtual_crossbar[crossbar_index]);
                            }
                            PIMCOMP_3_hierarchy_map.whole[AG_index] = whole;
                            PIMCOMP_3_hierarchy_map.whole_index[AG_index] = AG_index;
                        }

                        break;
                    }
                }
            }
            if (!distributed || !replication_mapped)
            {
//                std::cout << "#########################################################" << std::endl;
//                std::cout << "crossbar_num_per_AG:" << crossbar_num_per_AG << std::endl;
//                std::cout << "AG_num_per_replication:" << AG_num_this_replication << std::endl;
                int already_AG_num = 0;
                bool AG_mapped = false;
                for (int j = (i + accumulated_replication_num) % (ChipH * ChipH); j < ((ChipH * ChipW)+(i+accumulated_replication_num)%(ChipH * ChipH)); ++j)
                {
                    int index_j = j;
                    if (index_j >= ChipH * ChipW)
                        index_j -= ChipH*ChipW;
                    for (int k = already_AG_num; k < AG_num_this_replication; ++k)
                    {
                        if (ResourceList[index_j] >= crossbar_num_per_AG)
                        {
                            ResourceList[index_j] -= crossbar_num_per_AG;
                            already_AG_num++;

                            struct AGMapStruct thisAG;
                            thisAG.node_index = node_index;
                            thisAG.replication_index = i;
                            thisAG.index_in_replication = k;
                            thisAG.input_element_num = ACC_input_element_num_from_AG_index_in_replication[node_index][k];
                            thisAG.output_element_num = output_element_num_for_AG;
                            thisAG.AG_index_in_total = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_index[k];
//                            thisAG.input_cycle_num = PIMCOMP_2_AG_partition[effective_node_index].replication[i].input_cycle_this_replication;
//                            thisAG.instruction_group_num = ceil(float(thisAG.input_cycle_num) / float(operation_cycle_before_comm));
                            PIMCOMP_3_mapping_result[index_j].push_back(thisAG);

                            struct std::vector<struct PIMCOMP_2_virtual_crossbar> whole;
                            int AG_index = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].AG_index;
                            for (int l = 0; l < crossbar_num_per_AG; ++l)
                            {
                                int crossbar_index = PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].virtual_crossbar_list[l];
                                PIMCOMP_2_virtual_crossbar[crossbar_index].vcore_index = index_j;
                                PIMCOMP_2_virtual_crossbar[crossbar_index].vcore = index_j;
                                PIMCOMP_2_virtual_crossbar[crossbar_index].index_in_vcore = k * crossbar_num_per_AG + l;
                                PIMCOMP_2_AG_partition[effective_node_index].replication[i].AG_list[k].virtual_core_list.push_back(index_j);
                                PIMCOMP_3_virtual_core_crossbar_map[index_j].push_back(crossbar_index);
                                whole.push_back(PIMCOMP_2_virtual_crossbar[crossbar_index]);
                            }
                            PIMCOMP_3_hierarchy_map.whole[AG_index] = whole;
                            PIMCOMP_3_hierarchy_map.whole_index[AG_index] = AG_index;

                            if (already_AG_num == AG_num_this_replication)
                            {
                                AG_mapped = true;
                                break;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (AG_mapped)
                    {
                        break;
                    }
                }
            }
        }
//        int sss = 0; for (int j = 0; j < ChipH * ChipW; ++j) { std::cout << ResourceList[j] << " "; sss += ResourceList[j]; } std::cout << "  Total:" << sss << std::endl;
        accumulated_replication_num += replication_num;
    }
    int sss = 0;
    for (int j = 0; j < ChipH * ChipW; ++j)
    { sss += ResourceList[j]; }
    return (ChipH*ChipW*CoreH*CoreW-sss == PIMCOMP_2_resource_info.RRAMS);
}

static int AG_flags[MAX_AG] = {0};
void HierarchyMapping::AllocateMapInfo()
{
    PIMCOMP_3_mapping_result.resize( ChipH * ChipW);
    PIMCOMP_4_AG_input_element_num.resize(PIMCOMP_2_resource_info.AGs);
    // According to the relationship of Crossbar on Core provided by 4_physical_crossbar_placement information, the relationship of AG on Core is obtained
    // 根据4_physical_crossbar_placement信息提供的Core上Crossbar的关系得到Core上AG的关系
    int crossbar_num = PIMCOMP_2_virtual_crossbar.size();
    for (int i = 0; i < crossbar_num; ++i)
    {
        int AG_index = PIMCOMP_2_virtual_crossbar[i].array_group_total;
        if (AG_flags[AG_index] != 1)
        {
            AG_flags[AG_index] = 1;
            int core_index = PIMCOMP_2_virtual_crossbar[i].vcore;
            int rep_index = PIMCOMP_2_virtual_crossbar[i].replication_index;
            int node_index = PIMCOMP_2_virtual_crossbar[i].node_index;
            int AG_index_in_total = PIMCOMP_2_virtual_crossbar[i].array_group_total;
            int AG_index_in_replication = PIMCOMP_2_virtual_crossbar[i].array_group_in_weight;
            int AG_num_per_replication = PIMCOMP_2_virtual_crossbar[i].AG_num_per_replication;
            int replication_index = PIMCOMP_2_virtual_crossbar[i].replication_index;

            struct AG_info_schedule AGInfo;
            AGInfo.AG_index_in_total = AG_index_in_total;
            AGInfo.AG_index_in_replication = AG_index_in_replication;
            AGInfo.AG_num_per_replication = AG_num_per_replication;
            AGInfo.replication_index = rep_index;
            AGInfo.node_index = node_index;
            AGInfo.AGP = PIMCOMP_node_list[node_index].AGP;
            AGInfo.agp_index = PIMCOMP_2_virtual_crossbar[i].agp_index;
            AGInfo.agp_offset = PIMCOMP_2_virtual_crossbar[i].agp_offset;
            AGInfo.replication_num = PIMCOMP_node_list[node_index].replication_num;
            AGInfo.replication_num_origin = PIMCOMP_node_list[node_index].replication_num_origin;
            AGInfo.input_cycle_in_total = PIMCOMP_node_list[node_index].input_cycle_in_total;
            AGInfo.width_start = PIMCOMP_2_virtual_crossbar[i].width_start;
            AGInfo.width_end = PIMCOMP_2_virtual_crossbar[i].width_end;
            AGInfo.height_start = PIMCOMP_2_virtual_crossbar[i].height_start;
            AGInfo.height_end = PIMCOMP_2_virtual_crossbar[i].height_end;

//            AGInfo.input_cycle_this_replication = PIMCOMP_2_virtual_crossbar[i].input_cycle_this_replication;
//            AGInfo.input_cycle_this_replication_start = PIMCOMP_2_virtual_crossbar[i].input_cycle_this_replication_start;
//            AGInfo.input_cycle_this_replication_end = PIMCOMP_2_virtual_crossbar[i].input_cycle_this_replication_end;
            AGInfo.level_index = PIMCOMP_node_list[node_index].level_index;

            int effective_node_index = PIMCOMP_node_list[node_index].effective_node_index;
            int crossbar_num_AG = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list.size();
            int crossbar_start_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[0];
            int crossbar_end_index = PIMCOMP_2_AG_partition[effective_node_index].replication[replication_index].AG_list[AG_index_in_replication].virtual_crossbar_list[crossbar_num_AG - 1];
            int input_element_num = PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].height_start + 1;
            int output_element_num = PIMCOMP_2_virtual_crossbar[crossbar_end_index].width_end - PIMCOMP_2_virtual_crossbar[crossbar_start_index].width_start + 1;
            AGInfo.input_element_num = input_element_num;
            AGInfo.output_element_num = output_element_num;

            PIMCOMP_4_virtual_core_AG_map.core_list[core_index].AG_list.push_back(AGInfo);
            PIMCOMP_4_virtual_core_AG_map.core_list[core_index].node_list.push_back(node_index);
            PIMCOMP_4_AG_input_element_num[AG_index] = input_element_num;

            // PIMCOMP_3_mapping_result allows for quick assessment of memory usage
            // 为了快速评估memory用量而添加的。因为在这个阶段就快速评估memory必须要用到PIMCOMP_3_mapping_result的信息。
            struct AGMapStruct thisAG;
            thisAG.node_index = node_index;
            thisAG.replication_index = replication_index;
            thisAG.index_in_replication = AG_index_in_replication;
            thisAG.input_element_num = input_element_num;
            thisAG.output_element_num = output_element_num;
//            thisAG.input_cycle_num = PIMCOMP_2_virtual_crossbar[i].input_cycle_this_replication;
//            thisAG.instruction_group_num = ceil(float(thisAG.input_cycle_num) / float(operation_cycle_before_comm));
            PIMCOMP_3_mapping_result[core_index].push_back(thisAG);
        }
    }

    // Get instruction_group_num for each AG
    // 得到每个AG的instruction_group_num
    for (int i = 0; i < PIMCOMP_2_AG_partition.size(); ++i)
    {
        int replication_num = PIMCOMP_2_AG_partition[i].replication_num;
        for (int j = 0; j < replication_num; ++j)
        {
            int AG_num = PIMCOMP_2_AG_partition[i].replication[j].AG_list.size();
            for (int k = 0; k < AG_num; ++k)
            {
                int AG_index = PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].AG_index;
                PIMCOMP_3_hierarchy_map.whole[AG_index][0].instruction_group_num = ceil(float(PIMCOMP_2_AG_partition[i].input_cycle_in_total) / float(PIMCOMP_2_AG_partition[i].replication_num));
            }
        }
    }

    // Get PIMCOMP_4_AG_num_of_same_rep_in_core information (needed to optimize reload later)
    // 得到PIMCOMP_4_AG_num_of_same_rep_in_core信息（后面优化reload时需要用到）
    PIMCOMP_4_AG_num_of_same_rep_in_core.resize(PIMCOMP_2_resource_info.AGs);
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        std::map<int, std::map<int,int>> node_rep_AG_num;
        for (int j = 0; j < PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size(); ++j)
        {
            int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].node_index;
            int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
            node_rep_AG_num[node_index][replication_index]++;
        }
        for (int j = 0; j < PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size(); ++j)
        {
            int AG_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
            int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].node_index;
            int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
            PIMCOMP_4_AG_num_of_same_rep_in_core[AG_index] = node_rep_AG_num[node_index][replication_index];
        }
    }
}

void HierarchyMapping::ShowOriginalInfo()
{
    int v = 0;
    int node_num_partition = PIMCOMP_2_AG_partition.size();
    for (int i = 0; i < node_num_partition; ++i)
    {
        std::cout << "Node Index: " << PIMCOMP_2_AG_partition[i].index << ", Operation: " << PIMCOMP_2_AG_partition[i].operation << std::endl;
        int replication_num = PIMCOMP_2_AG_partition[i].replication_num;
        for (int j = 0; j < replication_num; ++j)
        {
            std::cout << "|---Replication Index: " << j << std::endl;
            int ag_num_current_replication = PIMCOMP_2_AG_partition[i].replication[j].AG_list.size();
            for (int k = 0; k < ag_num_current_replication; ++k)
            {
                std::cout << "|-------Array Group Index: " << v << ", Crossbar Number: "
                          << PIMCOMP_2_AG_partition[i].replication[j].AG_list[k].virtual_crossbar_list.size() << std::endl;
                v ++;
            }
        }
    }
}


void HierarchyMapping::ShowMappingInfo()
{
    std::cout << "****************** Mapping Result ********************" << std::endl;
    std::cout << PIMCOMP_4_virtual_core_AG_map.core_list.size() << std::endl;
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size() == 0)
            continue;
        std::cout << "core_" << i << std::endl;
        std::cout << " " << "node"
                  << "  " << "r_index"
                  << "  "  << "index_r"
                  << "  | " <<  "AG_index"
                  << " " << "input_cycle" << std::endl;
        int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        for (int j = 0; j < AG_num; ++j)
        {
            std::cout << "    " << std::setw(3) << PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j]
                      << "    " << std::setw(3) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index
                      << "    " <<  std::setw(3) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication
                      << "    | " << std::setw(5) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total
                      << "    " << std::setw(5) << ceil(float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_cycle_in_total)/float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num))  << std::endl;
        }
    }
}



void HierarchyMapping::SaveMappingResult()
{
    std::ofstream OutFile("../output/MappingResult.txt", std::ios::out | std::ios::trunc);
    OutFile << "****************** Mapping Result ********************" << std::endl;
    OutFile << PIMCOMP_4_virtual_core_AG_map.core_list.size() << std::endl;
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size() == 0)
            continue;
        OutFile << "core_" << i << std::endl;
        OutFile << "   " << "node"
                  << "  " << "r_index"
                  << "  "  << "index_r"
                  << "  | " <<  "AG_index"
                  << "  " << "xbar_num"
                  << "  " << "input_cycle" << std::endl;
        int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
        for (int j = 0; j < AG_num; ++j)
        {
            int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
            OutFile << "    " << std::setw(3) << node_index
                      << "    " << std::setw(3) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index
                      << "     " <<  std::setw(3) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication
                      << "     | " << std::setw(5) << PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total
                      << "     " << std::setw(5) << ceil(float(PIMCOMP_node_list[node_index].W ) / float(CrossbarW))
                      << "      " << std::setw(5) << ceil(float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_cycle_in_total)/float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num))  << std::endl;
        }
    }
}

void HierarchyMapping::Check()
{
    // Count the number of crossbars to ensure that we map all logical crossbars
    // 统计crossbar的数量，确保每个crossbar都映射到.
    int * FLAG = new int[ChipH * ChipW * CoreH * CoreW];
    for (int i = 0; i < ChipH * ChipW * CoreH * CoreW; ++i)
        FLAG[i] = 0;
    for (int i = 0; i < PIMCOMP_3_virtual_core_crossbar_map.size(); ++i)
    {
        int rram_num = PIMCOMP_3_virtual_core_crossbar_map[i].size();
        for (int j = 0; j < rram_num; ++j)
        {
            int index = PIMCOMP_3_virtual_core_crossbar_map[i][j];
            FLAG[index] = 1;
        }
    }
    int sum = 0;
    for (int i = 0; i < ChipH * ChipW * CoreH * CoreW; ++i)
    {
        sum += FLAG[i];
    }
    if (sum == PIMCOMP_2_resource_info.RRAMS)
    {
        std::cout << "Mapping Check Pass!" << std::endl;
    }
    else
    {
        std::cout << "Expected Crossbar Num:" << PIMCOMP_2_resource_info.RRAMS << "   Mapping Crossbar Num:" <<  sum  << std::endl;
        fprintf(stderr, "Map Check Failed. Please Change Replicating Method Or Increase Crossbar Resource.\n");
        abort();
    }
}


void HierarchyMapping::Clear()
{
    ArrayGroupIndex = 0;
    for (int & n : AG_flags) {n = 0;}

    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        ResourceList[i] = CoreW * CoreH;
    }

    ACC_input_window_element_num.clear();
    ACC_input_element_num_from_AG_index_in_replication.clear();
}




struct SimpleInst
{
    std::string operation;
    int node_index;
    int AG_index;
    int comm_index;
    int instruction_index_in_core;
    int from_core; // RECV
    int to_core;  // SEND
    int element_num;
    int latency_start;
    int latency_end;
};
static std::vector<std::vector<std::vector<struct SimpleInst>>> SimpleInstructionList;
static std::vector<std::vector<std::vector<struct SimpleInst>>> SimpleCOMMInstructionList;
static std::vector<std::vector<std::vector<struct SimpleInst>>> SimpleWBInstructionList;

static const double MVMUL_read_latency = 100.0;
static const double MVMUL_process_latency_per_bit = 100.0;
static const double MVMUL_ADC_latency = 100.0;
static const double MVMUL_sa_latency = 100.0;
static const double MVMUL_write_latency = 100.0;
static const double MVMUL_latency = MVMUL_read_latency + ArithmeticPrecision * MVMUL_process_latency_per_bit + MVMUL_ADC_latency + MVMUL_sa_latency + MVMUL_write_latency;
static const double EVA_MVMUL_start_interval = 200.0;
static int visited_single[MAX_CORE] = {0};
static double LATENCY_single[MAX_CORE] = {0};
static double LATENCY_first[MAX_CORE] = {0};
static std::map<int, int> comm_index_2_index_in_core;
static std::map<int, int> comm_index_2_core_index;
static long long last_MVMUL_exec_start_time[MAX_CORE] = {0};
static int MVMUL_instruction_group_core[MAX_CORE] = {0};
static long long last_synchronous_time[MAX_CORE] = {0};
static long long preparation_timeline[MAX_CORE] = {0};

void HierarchyMapping::FastEvaluationForElement()
{
    clock_t start_time = clock();

    std::set<std::string> no_consider_node_set = {"OP_INPUT", "OP_FLATTEN", "OP_RESHAPE", "OP_DROPOUT", "OP_LRN",
                                                  "OP_SOFTMAX", "OP_TRANSPOSE", "OP_SQUEEZE", "OP_MATMUL", "OP_BN",
                                                  "OP_CLIP", "OP_SQUEEZE", "OP_MATMUL"};
    std::vector<int> post_node_map;
    post_node_map.resize(node_num);
    for (int i = 0; i < node_num; ++i)
    {
        std::string operation = PIMCOMP_node_list[i].operation;
        if (operation != "OP_CONV" && operation != "OP_FC" && no_consider_node_set.count(operation) == 0)
        {
            post_node_map[i] = i % (ChipH * ChipW);
//            std::cout << "post_node:" << i  << " " << operation <<  "  core:" << post_node_map[i] << std::endl;
        }
    }

    int appointed_instruction_group_num = 2;
    SimpleInstructionList.resize(appointed_instruction_group_num);
    SimpleCOMMInstructionList.resize(appointed_instruction_group_num);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        SimpleInstructionList[ig].resize(ChipH * ChipW);
        SimpleCOMMInstructionList[ig].resize(ChipH * ChipW);
    }
    int evaluation_instruction_group_num = 0;
    int comm_index = 0;
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            if (PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size() == 0)
                continue;
            int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
            for (int j = 0; j < AG_num; ++j)
            {
                int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
                int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
                int AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication;
                int AG_index_in_total = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
                int IG_num = ceil(float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_cycle_in_total)/float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num));
                if (IG_num > evaluation_instruction_group_num)
                    evaluation_instruction_group_num = IG_num;

                struct SimpleInst Instruction_mvmul;
                Instruction_mvmul.operation = "MVMUL";
                Instruction_mvmul.node_index = node_index;
                Instruction_mvmul.AG_index = AG_index_in_total;
                SimpleInstructionList[ig][i].push_back(Instruction_mvmul);

                if (AG_index_in_replication == 0)
                {
                    if (PIMCOMP_node_list[node_index].consumer_num != 0)
                    {
                        int consumer_index = PIMCOMP_topology_provider_consumer_relation[node_index][0];
                        std::string consumer_operation = PIMCOMP_node_list[consumer_index].operation;
                        bool no_effective_consumer = false;
                        while (no_consider_node_set.count(consumer_operation) == 1)
                        {
                            if (PIMCOMP_node_list[consumer_index].consumer_num == 0)
                            {
                                no_effective_consumer = true;
                                break;
                            }
                            consumer_index = PIMCOMP_topology_provider_consumer_relation[consumer_index][0];
                            consumer_operation = PIMCOMP_node_list[consumer_index].operation;
                        }
                        if (consumer_operation == "OP_CONV" || consumer_operation == "OP_FC")
                            continue;
                        if (no_effective_consumer)
                            continue;

                        int to_core = post_node_map[consumer_index];
                        if (to_core != i)
                        {
                            struct SimpleInst Instruction_send;
                            Instruction_send.operation = "SEND";
                            Instruction_send.to_core = to_core;
                            Instruction_send.element_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].output_element_num;
                            Instruction_send.comm_index = comm_index;
                            SimpleCOMMInstructionList[ig][i].push_back(Instruction_send);

                            struct SimpleInst Instruction_recv;
                            Instruction_recv.operation = "RECV";
                            Instruction_recv.from_core = i;
                            Instruction_recv.element_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].output_element_num;
                            Instruction_recv.comm_index = comm_index;
                            SimpleCOMMInstructionList[ig][Instruction_send.to_core].push_back(Instruction_recv);

                            comm_index++;
                        }

                    }
                }
            }
        }
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            for (int j = 0; j < SimpleCOMMInstructionList[ig][i].size(); ++j)
            {
                SimpleCOMMInstructionList[ig][i][j].instruction_index_in_core = SimpleInstructionList[ig][i].size();
                SimpleInstructionList[ig][i].push_back(SimpleCOMMInstructionList[ig][i][j]);
            }
        }
    }

    COMM_cycle_flag = new bool *[MAX_CHIP];
    for (int i = 0; i < MAX_CHIP; ++i)
        COMM_cycle_flag[i] = new bool[1000000000];
    MEM_cycle_flag = new bool *[MAX_CHIP];
    for (int i = 0; i < MAX_CHIP; ++i)
        MEM_cycle_flag[i] = new bool[1000000000];

    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        FastSingleInstructionGroupEvaluation(ig, 0, 0);
        for (int & n : visited_single) {n = 0;}
    }

    std::ofstream OutFile("../output/FastInstruction.inst", std::ios::out | std::ios::trunc);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            int instruction_num = SimpleInstructionList[ig][i].size();
            OutFile << "Core " << i << " Start" << std::endl;
            for (int j = 0; j < instruction_num; ++j)
            {
                struct SimpleInst tmpInst = SimpleInstructionList[ig][i][j];
                if (tmpInst.operation == "MVMUL")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  node:" << tmpInst.node_index
                            << "  AG_index:" << tmpInst.AG_index
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
                else if (tmpInst.operation == "SEND")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  comm_index:" << tmpInst.comm_index
                            << "  to_core:" << tmpInst.to_core
                            << "  element:" << tmpInst.element_num
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
                else if (tmpInst.operation == "RECV")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  comm_index:" << tmpInst.comm_index
                            << "  from_core:" << tmpInst.from_core
                            << "  element:" << tmpInst.element_num
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
            }
            OutFile << "Core " << i << " Over" << std::endl;
        }
    }

    double practical_time = 0.0;
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (practical_time < LATENCY_single[i])
            practical_time = LATENCY_single[i];
    }
//    std::cout << std::fixed << evaluation_instruction_group_num * practical_time / double (appointed_instruction_group_num) << std::endl
//              << std::fixed << practical_time / double (appointed_instruction_group_num) << std::endl
//              << std::fixed << evaluation_instruction_group_num << std::endl;
    std::cout << "FAST:" << std::fixed << practical_time / double (appointed_instruction_group_num) << std::endl;
    clock_t end_time = clock();
    std::cout << "quick evaluation time:" << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    for (int i = 0; i < MAX_CHIP; ++i)
    {
        delete [] COMM_cycle_flag[i];
        delete [] MEM_cycle_flag[i];
    }
    delete [] COMM_cycle_flag;
    delete [] MEM_cycle_flag;
}

static double effective_BUS_bandwidth = 25600000000.0;
static double GLOBAL_MEMORY_bandwidth = 51200000000.0;
static double Frequency = 1000000000.0;

void HierarchyMapping::FastSingleInstructionGroupEvaluation(int instruction_group_index, int core_index, int index_in_core)
{
    if (core_index >= ChipW * ChipH)
        return;
    visited_single[core_index] = 1;
    int instruction_ir_num = SimpleInstructionList[instruction_group_index][core_index].size();
    for (int k = index_in_core; k < instruction_ir_num; ++k)
    {
        struct SimpleInst tmpInstruction = SimpleInstructionList[instruction_group_index][core_index][k];
        if (tmpInstruction.operation == "SEND" || tmpInstruction.operation == "RECV")
        {
            int comm_index = tmpInstruction.comm_index;
            int instruction_index_in_core = tmpInstruction.instruction_index_in_core;
            SimpleInstructionList[instruction_group_index][core_index][k].latency_start = LATENCY_single[core_index];
            if (comm_index_2_index_in_core.count(comm_index) == 0)
            {
                comm_index_2_index_in_core.insert(std::pair<int,int>(comm_index, instruction_index_in_core));
                comm_index_2_core_index.insert(std::pair<int,int>(comm_index, core_index));
                int next_core_index = core_index+1;
                while (visited_single[next_core_index] != 0)
                {
                    next_core_index++;
                }
                FastSingleInstructionGroupEvaluation(instruction_group_index, next_core_index, 0);
            }
            else
            {
                int corresponding_core_index = comm_index_2_core_index[comm_index];
                int corresponding_instruction_index_in_core = comm_index_2_index_in_core[comm_index];

                int element_num = tmpInstruction.element_num;
                int communication_bytes_num = element_num * ArithmeticPrecision / 8;
                int communication_needed_cycle = std::ceil((double(communication_bytes_num)) / (effective_BUS_bandwidth / Frequency));
                int real_comm_latency = 0;
                // 新写的不完全同步机制
                if (tmpInstruction.operation == "RECV")  // Corresponding Core Send to Current Core
                {
                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[corresponding_core_index], communication_needed_cycle);
                    LATENCY_single[corresponding_core_index] += real_comm_latency;  // Send Finish Time
                    if (LATENCY_single[core_index] < LATENCY_single[corresponding_core_index])
                    {
                        LATENCY_single[core_index] = LATENCY_single[corresponding_core_index];
                        last_synchronous_time[core_index] = LATENCY_single[core_index];
                    }
                }
                else    // Current Core Send to Corresponding Core
                {
                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[core_index], communication_needed_cycle);
                    LATENCY_single[core_index] += real_comm_latency;  // Send Finish Time
                    if (LATENCY_single[corresponding_core_index] < LATENCY_single[core_index])
                    {
                        LATENCY_single[corresponding_core_index] = LATENCY_single[core_index];
                        last_synchronous_time[corresponding_core_index] = LATENCY_single[corresponding_core_index];
                    }
                }
                SimpleInstructionList[instruction_group_index][core_index][instruction_index_in_core].latency_end = LATENCY_single[core_index];
                SimpleInstructionList[instruction_group_index][corresponding_core_index][corresponding_instruction_index_in_core].latency_end = LATENCY_single[corresponding_core_index];

                FastSingleInstructionGroupEvaluation(instruction_group_index, corresponding_core_index, corresponding_instruction_index_in_core+1);
                FastSingleInstructionGroupEvaluation(instruction_group_index, core_index, instruction_index_in_core+1);
            }
            return;
        }
        else if (tmpInstruction.operation == "MVMUL")
        {
            int node_index = tmpInstruction.node_index;
            if (MVMUL_instruction_group_core[core_index] != 0)
            {
                long long tmp = last_MVMUL_exec_start_time[core_index];
                int real_interval = EVA_MVMUL_start_interval;
                last_MVMUL_exec_start_time[core_index] = last_MVMUL_exec_start_time[core_index] + real_interval;
                if (last_MVMUL_exec_start_time[core_index] < last_synchronous_time[core_index])
                {
                    last_MVMUL_exec_start_time[core_index] = last_synchronous_time[core_index];
                }
                LATENCY_single[core_index] = last_MVMUL_exec_start_time[core_index] + MVMUL_latency;
            }
            else
            {
//                last_MVMUL_exec_start_time[core_index] = LATENCY_single[core_index];
                last_MVMUL_exec_start_time[core_index] = preparation_timeline[core_index];
                LATENCY_single[core_index] = last_MVMUL_exec_start_time[core_index] + MVMUL_latency;
            }
            MVMUL_instruction_group_core[core_index]++;
            SimpleInstructionList[instruction_group_index][core_index][k].latency_start = last_MVMUL_exec_start_time[core_index];
            SimpleInstructionList[instruction_group_index][core_index][k].latency_end = LATENCY_single[core_index];
        }
        else if (tmpInstruction.operation == "LD")
        {
            SimpleInstructionList[instruction_group_index][core_index][k].latency_start = preparation_timeline[core_index];
            int element_num = tmpInstruction.element_num;
            int memory_bytes_num = element_num * ArithmeticPrecision / 8;
            int memory_needed_cycle = std::ceil((double(memory_bytes_num)) / (GLOBAL_MEMORY_bandwidth / Frequency));
            int chip_index = core_index / (OneChipWidth * OneChipHeight);
            int real_mem_latency = CheckGlobalMemoryBandwidth(chip_index, preparation_timeline[core_index], memory_needed_cycle);
            preparation_timeline[core_index] = preparation_timeline[core_index] + real_mem_latency;
            SimpleInstructionList[instruction_group_index][core_index][k].latency_end = preparation_timeline[core_index];
        }
        else if (tmpInstruction.operation == "ST")
        {
            SimpleInstructionList[instruction_group_index][core_index][k].latency_start = LATENCY_single[core_index];
            int element_num = tmpInstruction.element_num;
            int memory_bytes_num = element_num * ArithmeticPrecision / 8;
            int memory_needed_cycle = std::ceil((double(memory_bytes_num)) / (GLOBAL_MEMORY_bandwidth / Frequency));
            int chip_index = core_index / (OneChipWidth * OneChipHeight);
            int real_mem_latency = CheckGlobalMemoryBandwidth(chip_index, LATENCY_single[core_index], memory_needed_cycle);
            LATENCY_single[core_index] += real_mem_latency;
            SimpleInstructionList[instruction_group_index][core_index][k].latency_end = LATENCY_single[core_index];
        }
    }
    int next_core_index = core_index+1;
    while (visited_single[next_core_index] != 0)
    {
        next_core_index++;
    }
    FastSingleInstructionGroupEvaluation(instruction_group_index, next_core_index, 0);
}


void HierarchyMapping::FastEvaluationForBatch()
{
    int appointed_instruction_group_num = 2;
    SimpleInstructionList.resize(appointed_instruction_group_num);
    SimpleWBInstructionList.resize(appointed_instruction_group_num);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        SimpleInstructionList[ig].resize(ChipH * ChipW);
        SimpleWBInstructionList[ig].resize(ChipH * ChipW);
    }

    int evaluation_instruction_group_num = 0;
    int comm_index = 0;
    std::vector<int> core_IG_num;
    core_IG_num.resize(ChipH * ChipW);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            if (PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size() == 0)
                continue;
            int AG_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list.size();
            int max_IG_num = 0;
            for (int j = 0; j < AG_num; ++j)
            {
                int node_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].node_list[j];
                int replication_index = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_index;
                int AG_index_in_replication = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_replication;
                int AG_index_in_total = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].AG_index_in_total;
                int IG_num = ceil(float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_cycle_in_total)/float(PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].replication_num));
                if (IG_num > evaluation_instruction_group_num)
                    evaluation_instruction_group_num = IG_num;
                if (IG_num > max_IG_num)
                    max_IG_num = IG_num;
                int input_element_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].input_element_num;
                int output_element_num = PIMCOMP_4_virtual_core_AG_map.core_list[i].AG_list[j].output_element_num;

                struct SimpleInst Instruction_ld;
                Instruction_ld.operation = "LD";
                Instruction_ld.element_num = input_element_num * 0.7;
                SimpleInstructionList[ig][i].push_back(Instruction_ld);

                struct SimpleInst Instruction_mvmul;
                Instruction_mvmul.operation = "MVMUL";
                Instruction_mvmul.node_index = node_index;
                Instruction_mvmul.AG_index = AG_index_in_total;
                SimpleInstructionList[ig][i].push_back(Instruction_mvmul);

                if (AG_index_in_replication == 0)
                {
                    struct SimpleInst Instruction_st;
                    Instruction_st.operation = "ST";
                    Instruction_st.element_num = output_element_num;
                    SimpleWBInstructionList[ig][i].push_back(Instruction_st);
                }
            }
            if (ig == 0)
                core_IG_num[i] = max_IG_num;
        }
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            for (int j = 0; j < SimpleWBInstructionList[ig][i].size(); ++j)
            {
                SimpleInstructionList[ig][i].push_back(SimpleWBInstructionList[ig][i][j]);
            }
        }
    }

    COMM_cycle_flag = new bool *[MAX_CHIP];
    for (int i = 0; i < MAX_CHIP; ++i)
        COMM_cycle_flag[i] = new bool[1000000000];
    MEM_cycle_flag = new bool *[MAX_CHIP];
    for (int i = 0; i < MAX_CHIP; ++i)
        MEM_cycle_flag[i] = new bool[1000000000];

    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        FastSingleInstructionGroupEvaluation(ig, 0, 0);
        if (ig == 0)
            for (int c = 0; c < MAX_CORE; ++c)
                LATENCY_first[c] = LATENCY_single[c];
        for (int & n : visited_single) {n = 0;}
        for (int & n : MVMUL_instruction_group_core) {n = 0;}
    }

    std::ofstream OutFile("../output/FastInstruction.inst", std::ios::out | std::ios::trunc);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            int instruction_num = SimpleInstructionList[ig][i].size();
            OutFile << "Core " << i << " Start" << std::endl;
            for (int j = 0; j < instruction_num; ++j)
            {
                struct SimpleInst tmpInst = SimpleInstructionList[ig][i][j];
                if (tmpInst.operation == "MVMUL")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  node:" << tmpInst.node_index
                            << "  AG_index:" << tmpInst.AG_index
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
                else if (tmpInst.operation == "SEND")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  comm_index:" << tmpInst.comm_index
                            << "  to_core:" << tmpInst.to_core
                            << "  element:" << tmpInst.element_num
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
                else if (tmpInst.operation == "RECV")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  comm_index:" << tmpInst.comm_index
                            << "  from_core:" << tmpInst.from_core
                            << "  element:" << tmpInst.element_num
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
                else if (tmpInst.operation == "LD" || tmpInst.operation == "ST")
                {
                    OutFile << "    [" << tmpInst.operation << "]"
                            << "  element:" << tmpInst.element_num
                            << "  S:" << tmpInst.latency_start
                            << "  E:" << tmpInst.latency_end
                            << std::endl;
                }
            }
            OutFile << "Core " << i << " Over" << std::endl;
        }
    }

    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        std::cout << "core:" << i << "  latency:" << std::fixed << std::setprecision(1)
                  << (LATENCY_single[i] - LATENCY_first[i]) / (appointed_instruction_group_num-1) * (core_IG_num[i]-1) + LATENCY_first[i] << std::endl;
    }

    for (int i = 0; i < MAX_CHIP; ++i)
    {
        delete [] COMM_cycle_flag[i];
        delete [] MEM_cycle_flag[i];
    }
    delete [] COMM_cycle_flag;
    delete [] MEM_cycle_flag;
}


int HierarchyMapping::CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle)
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


int HierarchyMapping::CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle)
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
//
// Created by SXT on 2022/10/5.
//

#ifndef PIMCOMP_PIMCOMPVARIABLE_H
#define PIMCOMP_PIMCOMPVARIABLE_H

#include "../common.h"

extern int ArithmeticPrecision;
extern int CellPrecision;
extern int CrossbarW;
extern int CrossbarH;
extern int CoreW;  // #Crossbars every row in Core (Logical)
extern int CoreH;  // #Crossbars every column in Core (Logical)
extern int ChipW;  // #Cores every row in Chip
extern int ChipH;  // #Cores every column in Chip

extern bool instruction_with_reload;
extern bool element_pipeline;
extern int comm_pair_total_num;

extern Json::Value PIMCOMP_VERIFICATION_INFO;
extern std::map<int, struct PIMCOMP_node> PIMCOMP_node_list_origin;
extern std::map<int, struct PIMCOMP_node> PIMCOMP_node_list;
extern std::vector<struct PIMCOMP_conv_pool_input_output_info> PIMCOMP_conv_pool_input_output_info;
extern std::vector<struct PIMCOMP_conv_pool_input_output_info> PIMCOMP_conv_pool_full_output_info;
extern std::vector<struct PIMCOMP_2_AG_partition> PIMCOMP_2_AG_partition;
extern std::vector<struct PIMCOMP_2_virtual_crossbar> PIMCOMP_2_virtual_crossbar;
extern struct PIMCOMP_2_resource_info PIMCOMP_2_resource_info;
extern std::vector<int> PIMCOMP_2_effective_node;
extern struct PIMCOMP_3_hierarchy_map PIMCOMP_3_hierarchy_map;
extern std::vector<std::vector<int>> PIMCOMP_3_virtual_core_crossbar_map;
extern struct PIMCOMP_4_first_AG_info PIMCOMP_4_first_AG_info;
extern struct PIMCOMP_4_virtual_core_AG_map PIMCOMP_4_virtual_core_AG_map;
extern std::vector<int> PIMCOMP_4_AG_num_of_same_rep_in_core;
extern std::vector<int> PIMCOMP_4_AG_input_element_num;
extern struct PIMCOMP_4_recv_info PIMCOMP_4_recv_info;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_4_base_instruction_ir;
extern std::vector<std::vector<int>> PIMCOMP_4_input_cycle_record;
extern std::map<int, struct PIMCOMP_4_instruction_ir> PIMCOMP_4_post_instruction_ir;

extern int PIMCOMP_base_instruction_num; // Only For Batch Pipeline
extern int PIMCOMP_post_instruction_num; // Only For Batch Pipeline
//// Newly
extern std::vector<int> PIMCOMP_5_memory_start_address;  // 这是由于每个核都要先加载数据（bias），所以每个核可以分配的地址不是从0开始，因此需要记录。
extern std::vector<struct AG_memory_info_of_one_IG_struct> PIMCOMP_5_AG_memory_info;
extern std::vector<struct AG_base_info> PIMCOMP_5_AG_base_info;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_5_base_instruction_with_address;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_6_base_instruction_with_input;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_6_base_instruction_with_input_batch;
extern std::vector<struct AG_input_info> PIMCOMP_6_AG_input_info;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_7_base_instruction_ir_with_optimization;
extern std::vector<struct PIMCOMP_4_instruction_ir> PIMCOMP_8_base_instruction_ir_with_placement;
extern std::vector<int> PIMCOMP_8_virtual_core_to_physical_core_map;
extern std::vector<std::vector<long long>> PIMCOMP_6_inter_core_communication;

extern std::set<int> PIMCOMP_4_unique_instruction_group_index;
extern std::vector<int> PIMCOMP_4_evaluation_instruction_group_index;
extern std::vector<int> PIMCOMP_4_core_instruction_group_num;

// for distributed mapping
extern std::vector<std::pair<struct MapSortStruct, int>> PIMCOMP_3_compute_crossbar_ratio;
extern std::vector<std::vector<struct AGMapStruct>> PIMCOMP_3_mapping_result;

extern std::vector<std::vector<int>> PIMCOMP_topology_provider_consumer_relation;
extern std::vector<std::vector<int>> PIMCOMP_topology_consumer_provider_relation;
extern std::vector<std::map<int,int>> PIMCOMP_4_element_node_provider_index_2_index_in_all_providers; // 这是为element设计的。比如一个节点有三个生产者5,7,11，内部序号为0,1,2。这里的作用就是根据11得到2，根据7得到1，根据5得到0。

extern std::vector<std::vector<int>> PIMCOMP_DSE_replication_num;
extern std::vector<std::vector<std::vector<struct DSE_AG_struct>>> PIMCOMP_DSE_core_map_info;
extern std::vector<struct DSE_result_info> PIMCOMP_DSE_result_info;
extern std::vector<std::vector<std::vector<std::vector<int>>>> PIMCOMP_DSE_node_map_info;

extern std::vector<double> PIMCOMP_EP_delay_for_conv_and_pool; // EP 即 Element-Pipeline
extern std::vector<std::vector<std::vector<int>>> PIMCOMP_EP_path_to_conv_or_pool;

extern std::vector<int> PIMCOMP_node_crossbar_num_per_AG; // PIMCOMP_node_crossbar_num_per_AG[i]记录了节点i的每个AG包含的crossbar数目。

//// For QT GUI output
extern std::vector<std::vector<double>> PIMCOMP_GUI_memory_usage_every_instruction_group;
extern std::vector<struct evaluation_info> PIMCOMP_GUI_evaluation_of_core;
extern std::vector<std::vector<std::pair<int, long long>>> PIMCOMP_GUI_execution_time_of_core;
extern std::vector<std::vector<long long>> PIMCOMP_GUI_execution_time_of_node;
extern std::vector<std::vector<int>> PIMCOMP_GUI_inter_core_communication_volume;

#endif //PIMCOMP_PIMCOMPVARIABLE_H

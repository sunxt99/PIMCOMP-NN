//
// Created by SXT on 2022/8/19.
//

#include "CrossbarPartition.h"

static int ArrayGroupIndex = 0;
static int VirtualCrossbarIndex = 0;


void CrossbarPartition::PartitionCrossbar()
{
    PIMCOMP_node_crossbar_num_per_AG.resize(PIMCOMP_node_list.size());
    Partition();
    Check();
    Clear();
}


void CrossbarPartition::Partition()
{
    int node_num = PIMCOMP_node_list.size();
    int WeightIndex = 0;
    int EffectiveNodeNum = 0;
    for (int i = 0; i < node_num; ++i)
    {
        struct PIMCOMP_node Node = PIMCOMP_node_list[i];
        struct param Param = Node.param;
        int bitwidth = Node.bitwidth;
        if (Node.operation == "OP_CONV" || Node.operation == "OP_FC")
        {
            struct PIMCOMP_2_AG_partition PIMCOMP2AgPartition;

            PIMCOMP2AgPartition.index = Node.index;
            PIMCOMP2AgPartition.name = Node.name;
            PIMCOMP2AgPartition.operation = Node.operation;
            int origin_rep_num = Node.replication_num;
            int Height;
            int Width;
            int sliding_window;
            if (Node.operation == "OP_CONV")
            {
                Height = Param.kernel_h * Param.kernel_w * Param.input_channel;
                Width = Param.output_channel;
                sliding_window = Node.output_dim[2] * Node.output_dim[3];
            }
            else // FC
            {
                Height = Param.num_input;
                Width = Param.num_output;
                sliding_window = 1;
            }

            PIMCOMP_node_list[i].H = Height;
            PIMCOMP_node_list[i].W = Width;
            PIMCOMP2AgPartition.Height = Height;
            PIMCOMP2AgPartition.Width = Width;
            PIMCOMP2AgPartition.input_cycle_in_total = sliding_window;

            int HBarNum = (Height-1) / CrossbarH + 1;
            // If consider physical crossbar, you need to consider two aspects: the impact of bit precision and the impact of positive and negative coefficients
            //  考虑物理的话需要考虑两个方面：bit精度的影响和正负系数的影响
            // int WBarNum = (Width-1)  / (CrossbarW / (bitwidth/CellPrecision)) + 1;    // For physical Crossbar
            int WBarNum = (Width-1)  / CrossbarW  + 1;                                  // For logical Crossbar
            // Consider the situation where a core cannot accommodate a complete AG
            int AGP = ceil(float(WBarNum) / float(CoreW*CoreH));
            PIMCOMP2AgPartition.AGP_num = AGP;
            // If #replication < #windows, then #replication = #windows
            if (origin_rep_num > sliding_window)
                origin_rep_num = sliding_window;
            PIMCOMP2AgPartition.replication_num_origin = origin_rep_num;
            PIMCOMP2AgPartition.replication_num = origin_rep_num * AGP;

            PIMCOMP_node_list[i].AGP = AGP;
            PIMCOMP_node_list[i].replication_num_origin = origin_rep_num;
            PIMCOMP_node_list[i].replication_num = origin_rep_num*AGP;
            PIMCOMP_node_list[i].input_cycle_in_total = PIMCOMP2AgPartition.input_cycle_in_total;
            PIMCOMP2AgPartition.replication.resize(origin_rep_num * AGP);

            int input_cycle_num_total = sliding_window;
            std::vector<int> start_address_vector;
            std::vector<int> end_address_vector;
            // Divide Input Cycle
            MyDivide(start_address_vector, end_address_vector, input_cycle_num_total, origin_rep_num);

            for (int j = 0; j < origin_rep_num; ++j)
            {
                for (int agp = 0; agp < AGP; ++agp)
                {
                    int ArrayGroupIndexInWeight = 0;
                    for (int H = 0; H < HBarNum; ++H)
                    {
                        struct AG_list ag_list;
                        int WStart = agp * CoreH * CoreW;
                        int WEnd = (agp == AGP-1) ? WBarNum : (agp+1)*(CoreW*CoreH);
                        for (int W = WStart; W < WEnd; ++W)
                        {
                            struct PIMCOMP_2_virtual_crossbar PIMCOMP2VirtualCrossbar;
                            PIMCOMP2VirtualCrossbar.index_in_weight = H * WBarNum + W;
                            // virtual_index is the index of this crossbar in the whole DNN
                            PIMCOMP2VirtualCrossbar.virtual_index = VirtualCrossbarIndex;
                            ag_list.virtual_crossbar_list.push_back(VirtualCrossbarIndex);
                            VirtualCrossbarIndex += 1;
                            // a core cannot accommodate a complete AG
                            PIMCOMP2VirtualCrossbar.replication_index = j*AGP+agp;
                            // array_group_in_weight is the index of this AG in one Weight block
                            PIMCOMP2VirtualCrossbar.array_group_in_weight = ArrayGroupIndexInWeight;
                            // array_group_total is the index of this AG in the whole DNN
                            PIMCOMP2VirtualCrossbar.array_group_total = ArrayGroupIndex;
                            PIMCOMP2VirtualCrossbar.height_start = H*CrossbarH;
                            PIMCOMP2VirtualCrossbar.height_end = (H==(HBarNum-1) ? Height-1 : (H+1)*CrossbarH-1 );
                            // physical or logical
//                            VirtualBar["width_start"] = W*(CrossbarW/(bitwidth/CellPrecision));                   // physical crossbar
//                            VirtualBar["width_end"] = (W==(WBarNum-1) ? Width-1 : (W+1)*(CrossbarW/(bitwidth/CellPrecision))-1 ); // physical crossbar
                            PIMCOMP2VirtualCrossbar.width_start = W*CrossbarW;                                      // logical crossbar
                            PIMCOMP2VirtualCrossbar.width_end = (W==(WBarNum-1) ? Width-1 : (W+1)*CrossbarW-1 );    // logical crossbar
                            // weight_index is the index of weight block in the whole DNN (replicated weights are different blocks)
                            PIMCOMP2VirtualCrossbar.weight_index = WeightIndex;

                            PIMCOMP2VirtualCrossbar.node_index = Node.index;
                            // the number of AG in one replication block (weight block)
                            PIMCOMP2VirtualCrossbar.AG_num_per_replication = HBarNum;
                            // agp information
                            PIMCOMP2VirtualCrossbar.agp_index = agp;
                            PIMCOMP2VirtualCrossbar.agp_offset = WStart * CrossbarW;
                            PIMCOMP_2_virtual_crossbar.push_back(PIMCOMP2VirtualCrossbar);
                        }
                        ag_list.AG_index = ArrayGroupIndex;
                        PIMCOMP2AgPartition.replication[j*AGP+agp].AG_list.push_back(ag_list);
                        PIMCOMP2AgPartition.replication[j*AGP+agp].AG_index.push_back(ArrayGroupIndex);
                        PIMCOMP2AgPartition.replication[j*AGP+agp].agp_index = agp;

                        ArrayGroupIndex += 1;
                        ArrayGroupIndexInWeight += 1;
                    }
                }
                WeightIndex += 1;
            }
            PIMCOMP2AgPartition.AG_num_per_replication = PIMCOMP2AgPartition.replication[0].AG_list.size();
            PIMCOMP2AgPartition.crossbar_num_per_AG = PIMCOMP2AgPartition.replication[0].AG_list[0].virtual_crossbar_list.size();
            PIMCOMP_2_effective_node.push_back(i);
            PIMCOMP_2_AG_partition.push_back(PIMCOMP2AgPartition);
            PIMCOMP_node_crossbar_num_per_AG[i] = PIMCOMP2AgPartition.crossbar_num_per_AG;
            PIMCOMP_node_list[i].effective_node_index = EffectiveNodeNum;
            EffectiveNodeNum += 1;
        }
    }
    std::cout << "#RRAMs needed: " << VirtualCrossbarIndex << std::endl;
    std::cout << "#ArrayGroups needed: " << ArrayGroupIndex << std::endl;
    std::cout << "RRAMs Usage: " << double(VirtualCrossbarIndex) / double(ChipW * ChipH * CoreW * CoreH) * 100 << "%" << std::endl;
    PIMCOMP_2_resource_info.RRAMS = VirtualCrossbarIndex;
    PIMCOMP_2_resource_info.AGs = ArrayGroupIndex;
}

void CrossbarPartition::Check()
{
    if ((VirtualCrossbarIndex-1) > CoreH * CoreW * ChipH * ChipW)
    {
        fprintf(stderr, "Illegal Replication Num. \n" );
        return;
    }
    else
    {
        std::cout << "Replication Check Pass!" << std::endl;
        return;
    }
}

void CrossbarPartition::Clear()
{
    ArrayGroupIndex = 0;
    VirtualCrossbarIndex = 0;
}
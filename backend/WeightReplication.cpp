//
// Created by SXT on 2022/8/19.
//

#include "WeightReplication.h"

void WeightReplication::ReplicateWeight(std::string replicating_method)
{
    node_num = PIMCOMP_node_list.size();
    if (replicating_method == "balance")
        ReplicationByBalance();
    else if (replicating_method == "W0H0")
        ReplicationByW0H0();
    else if (replicating_method == "uniform")
        ReplicateUniformly();
    else if (replicating_method == "GA")
        LoadGAReplicationResult(0);
    else
    {
        fprintf(stderr, "Please make sure the replicating method is supported\n");
        abort();
    }
}


void WeightReplication::ReplicateUniformly()
{
    std::vector<int> node_crossbar_num;
    node_crossbar_num.resize(node_num);
    int FC_crossbar_num = 0;
    // First traverse node_list to get FC_crossbar_num and other information
    // 首先遍历一遍node_list，得到FC_crossbar_num和其他信息
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            node_crossbar_num[i] = AG_num * crossbar_num_per_AG;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            FC_crossbar_num += AG_num * crossbar_num_per_AG;
        }
    }

    // Determine the replication_time, that is, how many replications can be completed overall
    // 确定replication_time，即总体上能够完成几次复制
    int crossbar_num_one_time = 0;
    for (int i = 0; i < node_crossbar_num.size(); ++i)
    {
        if(node_crossbar_num[i] != 0)
        {
            crossbar_num_one_time += node_crossbar_num[i];
        }
    }
    float margin_factor = 1.2;
    int replication_time = std::floor(double(ChipH * ChipW * CoreH * CoreW - FC_crossbar_num) / double(crossbar_num_one_time) / margin_factor);
    if (replication_time <= 0)
    {
        fprintf(stderr, "Current replication time is illegal. Please increase crossbar source.\n");
        abort();
    }
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            PIMCOMP_node_list[i].replication_num = replication_time;
            std::cout << "node:" << i << "   relative_rep_num:" << PIMCOMP_node_list[i].replication_num << std::endl;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
        {
            PIMCOMP_node_list[i].replication_num = 1;
            std::cout << "node:" << i << "   relative_rep_num:" << PIMCOMP_node_list[i].replication_num << std::endl;
        }
    }
}

void WeightReplication::ReplicationByW0H0()
{
    std::vector<int> W0H0;
    std::set<int> W0H0_unique;
    std::vector<int> node_crossbar_num;
    W0H0.resize(node_num);
    node_crossbar_num.resize(node_num);
    int FC_crossbar_num = 0;
    // First traverse node_list to get FC_crossbar_num and other information
    // 首先遍历一遍node_list，得到FC_crossbar_num和其他信息
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            int output_H = PIMCOMP_node_list[i].output_dim[2];
            int output_W = PIMCOMP_node_list[i].output_dim[3];
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            W0H0[i] = output_H * output_W;
            W0H0_unique.insert(output_H * output_W);
            node_crossbar_num[i] = AG_num * crossbar_num_per_AG;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            FC_crossbar_num += AG_num * crossbar_num_per_AG;
        }
    }

    // get the baseline W0*H0. We can change the index. The smaller the baseline_index is, the more replication there is for the former layers.
    // 得到W0H0的baseline，这里不选择最小的，而是选择倒数第3小的（baseline_index=2）。因为如果选择更小的，那前几层复制倍数太大，装不下。
    double W0H0_baseline;
    int index = 0;
    int baseline_index = 1;
    for (auto iter = W0H0_unique.begin(); iter != W0H0_unique.end(); iter++)
    {
        std::cout << *iter << std::endl;
        if (index == baseline_index)
            W0H0_baseline = *iter;
        index++;
    }

    // Determine the replication_time, that is, how many replications can be completed overall
    // 确定replication_time，即总体上能够完成几次复制
    std::vector<int> W0H0_ratio;
    W0H0_ratio.resize(node_num);
    int crossbar_num_one_time = 0;
    for (int i = 0; i < W0H0.size(); ++i)
    {
        if(W0H0[i] != 0)
        {
            W0H0_ratio[i] = ceil(double(W0H0[i]) / double(W0H0_baseline));
            std::cout << "node:" << i << "   relative_rep_num:" << W0H0_ratio[i] << std::endl;
            crossbar_num_one_time += W0H0_ratio[i] * node_crossbar_num[i];
        }
    }
    float margin_factor = 1.2;
    int replication_time = std::floor(double(ChipH * ChipW * CoreH * CoreW - FC_crossbar_num) / double(crossbar_num_one_time) / margin_factor);
    std::cout << "replication_time:" << replication_time << std::endl;
    if (replication_time <= 0)
    {
        fprintf(stderr, "Current replication time is illegal. Please increase crossbar source.\n");
        abort();
    }
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list[i].operation == "OP_CONV")
            PIMCOMP_node_list[i].replication_num = replication_time * W0H0_ratio[i];
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
            PIMCOMP_node_list[i].replication_num = 1;
    }
}


struct conv_node_info
{
    int node_index;
    int node_crossbar_num;
    int current_sliding_window;
    int original_sliding_window;
    conv_node_info(int node_index_, int node_crossbar_num_, int current_sliding_window_, int original_sliding_window_)
    {
        node_crossbar_num = node_crossbar_num_;
        node_index = node_index_;
        current_sliding_window = current_sliding_window_;
        original_sliding_window = original_sliding_window_;
    }
    friend bool operator < (const conv_node_info &a,const conv_node_info &b)
    {
        return a.current_sliding_window < b.current_sliding_window;
    }
};

void WeightReplication::ReplicationByBalance()
{
    std::priority_queue<struct conv_node_info> priority_queue;

    int FC_crossbar_num = 0;
    // First traverse node_list to get FC_crossbar_num and other information
    // 首先遍历一遍node_list，得到FC_crossbar_num和其他信息
    for (int i = 0; i < node_num; ++i)
    {
        if(PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            int node_crossbar_num = AG_num * crossbar_num_per_AG;
            int sliding_window = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            priority_queue.push(conv_node_info(i, node_crossbar_num, sliding_window, sliding_window));
            PIMCOMP_node_list[i].replication_num = 1;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            FC_crossbar_num += AG_num * crossbar_num_per_AG;
            PIMCOMP_node_list[i].replication_num = 1;
        }
    }
    int crossbar_source_available = ChipW * ChipH * CoreW * CoreH - FC_crossbar_num;
    float margin_factor = 0.5;
    int margin_source = crossbar_source_available * margin_factor;
    while (crossbar_source_available > priority_queue.top().node_crossbar_num && crossbar_source_available > margin_source)
    {
        int node_index = priority_queue.top().node_index;
        int node_crossbar_num = priority_queue.top().node_crossbar_num;
        int original_sliding_window = priority_queue.top().original_sliding_window;
        priority_queue.pop();
        int current_sliding_window = std::ceil(float(original_sliding_window)  / float(++PIMCOMP_node_list[node_index].replication_num));
        priority_queue.push(conv_node_info(node_index, node_crossbar_num, current_sliding_window, original_sliding_window));
        crossbar_source_available -= node_crossbar_num;
    }
    for (int i = 0; i < node_num; ++i)
    {
        if (PIMCOMP_node_list[i].replication_num != 0)
            std::cout << "node:" << i << "  replication_num:" << PIMCOMP_node_list[i].replication_num << std::endl;
    }
}

void WeightReplication::LoadGAReplicationResult(int candidate_index)
{
    for (int i = 0; i < PIMCOMP_DSE_replication_num[candidate_index].size(); ++i)
    {
        if (PIMCOMP_DSE_replication_num[candidate_index][i] != 0)
        {
            PIMCOMP_node_list[i].replication_num = PIMCOMP_DSE_replication_num[candidate_index][i];
            std::cout << "node:" << i << "  " <<  PIMCOMP_node_list[i].operation << "  " << PIMCOMP_node_list[i].replication_num << std::endl;
        }
    }
    for (int i = 0; i < PIMCOMP_DSE_replication_num[candidate_index].size(); ++i)
    {
        if (PIMCOMP_DSE_replication_num[candidate_index][i] != 0)
        {
            PIMCOMP_node_list[i].replication_num = PIMCOMP_DSE_replication_num[candidate_index][i];
            std::cout << PIMCOMP_node_list[i].replication_num << ",";
        }
    }
    std::cout << std::endl;
}
#include "GeneticAlgorithm.h"

int cmp(const std::pair<double, int> &x, const std::pair<double, int> &y) { return x.first > y.first; }

GeneticAlgorithm::GeneticAlgorithm(std::string model_name_, std::string pipeline_granularity_)
{
    GA_MVMUL_start_interval  = MVMUL_start_interval;
    model_name = model_name_;
    pipeline_granularity = pipeline_granularity_;

    conv_num = 0;
    fc_num = 0;
    node_crossbar_num_per_AG.resize(node_num);
    node_AG_num.resize(node_num);
    node_cycle_num.resize(node_num);
    node_MVMUL_num.resize(node_num);
    for (int i = 0; i < PIMCOMP_node_list.size(); ++i)
    {
        if (PIMCOMP_node_list[i].operation == "OP_FC" )
        {
            int input_cycle_num = 1;
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            fc_num ++;
            node_cycle_num[i] = input_cycle_num;
            FC_index.push_back(i);
            CONV_FC_index.push_back(i);
            node_AG_num[i] = AG_num;
            node_crossbar_num_per_AG[i] = crossbar_num_per_AG;
            node_MVMUL_num[i] = AG_num * input_cycle_num;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_CONV")
        {
            int input_cycle_num = PIMCOMP_node_list[i].output_dim[2] * PIMCOMP_node_list[i].output_dim[3];
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            conv_num++;
            node_cycle_num[i] = input_cycle_num;
            CONV_index.push_back(i);
            CONV_FC_index.push_back(i);
            node_AG_num[i] = AG_num;
            node_crossbar_num_per_AG[i] = crossbar_num_per_AG;
            node_MVMUL_num[i] = AG_num * input_cycle_num;
        }
    }

    // Init current_chromosome
    current_chromosome.resize(population_num);
    node_replication_num.resize(population_num);
    core_AG_type_num.resize(population_num);
    core_crossbar_num.resize(population_num);

    SaveIntermediateInfo();
}


void GeneticAlgorithm::GeneticAlgorithmOptimizing()
{
    Init();
    for (int i = 0; i < max_iteration; ++i)
    {
        std::cout << "iteration:" << i << std::endl;
        Select();
        Mutation();
    }
    Select();
    PostProcess();
}

void GeneticAlgorithm::Init()
{
    srand((unsigned)time(NULL));
    int legal_individual_index = 0;

    while (legal_individual_index < population_num)
    {
        current_chromosome[legal_individual_index].clear();
        current_chromosome[legal_individual_index].resize(chromosome_size);
        node_replication_num[legal_individual_index].clear();
        node_replication_num[legal_individual_index].resize(node_num);
        core_AG_type_num[legal_individual_index].clear();
        core_AG_type_num[legal_individual_index].resize(core_num);
        core_crossbar_num[legal_individual_index].clear();
        core_crossbar_num[legal_individual_index].resize(core_num);
        std::vector<int> rest_crossbar_per_core;
        for (int i = 0; i < core_num; ++i)
            rest_crossbar_per_core.push_back(CoreH * CoreW);
        float margin_factor = float(rand() % 100) / (125.0) + 0.10; //  0.1 - 0.9
        InitReplicating(margin_factor, legal_individual_index);
        InitMapping(legal_individual_index);

        if (CheckLegality(current_chromosome[legal_individual_index], node_replication_num[legal_individual_index], core_AG_type_num[legal_individual_index], core_crossbar_num[legal_individual_index], "Init"))
        {
            std::cout << "Init Individual_" << legal_individual_index << " succeed!" << std::endl;
            legal_individual_index++;
        }
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

void GeneticAlgorithm::InitReplicating(float margin_factor, int individual_index)
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
            node_replication_num[individual_index][i] = 1;
        }
        else if (PIMCOMP_node_list[i].operation == "OP_FC")
        {
            int AG_num = ceil(float(PIMCOMP_node_list[i].H) / float(CrossbarH));
            int crossbar_num_per_AG = ceil(float(PIMCOMP_node_list[i].W) / float(CrossbarW));
            FC_crossbar_num += AG_num * crossbar_num_per_AG;
            node_replication_num[individual_index][i] = 1;
        }
    }
    int crossbar_source_available = ChipW * ChipH * CoreW * CoreH - FC_crossbar_num;
    int margin_source = crossbar_source_available * margin_factor;
    while (crossbar_source_available > priority_queue.top().node_crossbar_num && crossbar_source_available > margin_source)
    {
        int node_index = priority_queue.top().node_index;
        int node_crossbar_num = priority_queue.top().node_crossbar_num;
        int original_sliding_window = priority_queue.top().original_sliding_window;
        priority_queue.pop();
        int current_sliding_window = std::ceil(float(original_sliding_window)  / float(++node_replication_num[individual_index][node_index]));
        priority_queue.push(conv_node_info(node_index, node_crossbar_num, current_sliding_window, original_sliding_window));
        crossbar_source_available -= node_crossbar_num;
    }
//    for (int i = 0; i < node_num; ++i)
//    {
//        if (node_replication_num[individual_index][i] != 0)
//            std::cout << "node:" << i << "  replication_num:" << node_replication_num[individual_index][i] << std::endl;
//    }
}


void GeneticAlgorithm::InitMapping(int individual_index)
{
    if (rand() % 100 > 50)
        InitMappingDistributed(individual_index);
    else
        InitMappingClustered(individual_index);
}

void GeneticAlgorithm::InitMappingDistributed(int individual_index)
{
    std::vector<int> rest_crossbar_per_core;
    for (int i = 0; i < core_num; ++i)
        rest_crossbar_per_core.push_back(CoreH * CoreW);
    int accumulated_replication_num = 0;
    for (auto node_index : CONV_FC_index)
    {
        int AG_num_this_replication = node_AG_num[node_index];
        int crossbar_num_per_AG = node_crossbar_num_per_AG[node_index];
        int crossbar_num_this_replication = AG_num_this_replication * crossbar_num_per_AG;
        std::string operation = PIMCOMP_node_list[node_index].operation;
        int replication_num = node_replication_num[individual_index][node_index];
        bool distributed = rand() % 100 > 50 ? 1 : 0;

        for (int i = 0; i < replication_num; ++i)
        {
            bool replication_mapped = false;
            if (distributed)
            {
                int cluster_num = rand() % 31 + 1;
                int core_index_offset = i / cluster_num;
                for (int j = (core_index_offset + accumulated_replication_num) % (ChipH * ChipH); j < (core_index_offset + accumulated_replication_num) % (ChipH * ChipH) + (ChipH * ChipH); ++j)
                {
                    int index_j = j;
                    if (index_j >= ChipH * ChipW)
                        index_j -= ChipH * ChipW;
                    int selected_core = index_j;
                    if (rest_crossbar_per_core[index_j] >= crossbar_num_this_replication && core_AG_type_num[individual_index][selected_core] < max_AG_kind_per_core)
                    {
                        rest_crossbar_per_core[index_j] -= crossbar_num_this_replication;
                        replication_mapped = true;
                        int selected_position = selected_core * max_AG_kind_per_core + core_AG_type_num[individual_index][selected_core];
                        core_AG_type_num[individual_index][selected_core]++;
                        core_crossbar_num[individual_index][selected_core] += AG_num_this_replication * crossbar_num_per_AG;
                        current_chromosome[individual_index][selected_position] = node_index * 10000 + AG_num_this_replication;
                        break;
                    }
                }
            }
            if (!distributed || !replication_mapped)
            {
                int already_AG_num = 0;
                bool AG_mapped = false;
                for (int j = (i + accumulated_replication_num) % (ChipH * ChipH); j < ((ChipH * ChipW)+(i+accumulated_replication_num)%(ChipH * ChipH)); ++j)
                {
                    int index_j = j;
                    if (index_j >= ChipH * ChipW)
                        index_j -= ChipH*ChipW;
                    for (int k = already_AG_num; k < AG_num_this_replication; ++k)
                    {
                        int selected_core = index_j;
                        if (rest_crossbar_per_core[index_j] >= crossbar_num_per_AG && core_AG_type_num[individual_index][selected_core] < max_AG_kind_per_core)
                        {
                            rest_crossbar_per_core[index_j] -= crossbar_num_per_AG;
                            already_AG_num++;

                            int selected_position = selected_core * max_AG_kind_per_core + core_AG_type_num[individual_index][selected_core];
                            core_AG_type_num[individual_index][selected_core]++;
                            core_crossbar_num[individual_index][selected_core] += crossbar_num_per_AG;
                            current_chromosome[individual_index][selected_position] = node_index * 10000 + 1;

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
        accumulated_replication_num += replication_num;
    }
    return;
}


void GeneticAlgorithm::InitMappingClustered(int individual_index)
{
    std::vector<int> rest_crossbar_per_core;
    for (int i = 0; i < core_num; ++i)
        rest_crossbar_per_core.push_back(CoreH * CoreW);
    int last_node_index = -1;
    for (auto node_index : CONV_FC_index)
    {
        int last_core_index = -1;
        int replication_num = node_replication_num[individual_index][node_index];
        int AG_num_this_replication = node_AG_num[node_index];
        int crossbar_num_per_AG = node_crossbar_num_per_AG[node_index];
        for (int j = 0; j < replication_num; ++j)
        {
            for (int k = 0; k < AG_num_this_replication; ++k)
            {
                for (int l = 0; l < ChipW * ChipH; ++l)
                {
                    if (crossbar_num_per_AG <= rest_crossbar_per_core[l])
                    {
                        int selected_core = l;
                        if (last_core_index == -1)
                            last_core_index = selected_core;
                        if (last_core_index != selected_core || last_node_index != node_index)
                        {
                            core_AG_type_num[individual_index][selected_core]++;
                            int selected_position = selected_core * max_AG_kind_per_core + core_AG_type_num[individual_index][selected_core];
                            core_crossbar_num[individual_index][selected_core] += crossbar_num_per_AG;
                            current_chromosome[individual_index][selected_position] = node_index * 10000 + 1;
                        }
                        else
                        {
                            int selected_position = selected_core * max_AG_kind_per_core + core_AG_type_num[individual_index][selected_core];
                            core_crossbar_num[individual_index][selected_core] += crossbar_num_per_AG;
                            current_chromosome[individual_index][selected_position] += 1;
                        }
                        rest_crossbar_per_core[l] -= crossbar_num_per_AG;
                        last_core_index = selected_core;
                        last_node_index = node_index;
                        break;
                    }
                }
            }
        }
    }
}

void GeneticAlgorithm::Select()
{
    std::map <double, std::vector<int>, std::less<double>> chromosome_map;
    std::map <double, std::vector<int>, std::less<double>> replication_map;
    std::map <double, std::vector<int>, std::less<double>> crossbar_num_map;
    std::map <double, std::vector<int>, std::less<double>> AG_type_num_map;
    for (int i = 0; i < population_num; ++i)
    {
        double fitness = Fitness(current_chromosome[i], node_replication_num[i]);
        while (chromosome_map.count(fitness) != 0) // 避免出现一样的fitness
        {
            fitness +=  (double)(rand()) / RAND_MAX / 10;
        }
        chromosome_map.insert(std::make_pair( fitness, current_chromosome[i]));
        replication_map.insert(std::make_pair(fitness, node_replication_num[i]));
        crossbar_num_map.insert(std::make_pair(fitness, core_crossbar_num[i]));
        AG_type_num_map.insert(std::make_pair(fitness, core_AG_type_num[i]));
    }

    int remain_len = ceil(population_num * remain_rate);
    auto iter1 = chromosome_map.begin();
    auto iter2 = replication_map.begin();
    auto iter3 = crossbar_num_map.begin();
    auto iter4 = AG_type_num_map.begin();
    current_chromosome.resize(0);
    node_replication_num.resize(0);
    core_crossbar_num.resize(0);
    core_AG_type_num.resize(0);
    for (int i = 0; i < remain_len; ++i)
    {
        if (i < 3)
        {
            std::cout << "    " << std::fixed <<  iter1->first << std::endl;
        }
        current_chromosome.push_back(iter1->second);
        node_replication_num.push_back(iter2->second);
        core_crossbar_num.push_back(iter3->second);
        core_AG_type_num.push_back(iter4->second);
        iter1++;
        iter2++;
        iter3++;
        iter4++;
    }
    for (int i = remain_len; i < population_num; ++i)
    {
        if( rand() % 100 < (random_select_rate * 100) )
        {
            current_chromosome.push_back(iter1->second);
            node_replication_num.push_back(iter2->second);
            core_crossbar_num.push_back(iter3->second);
            core_AG_type_num.push_back(iter4->second);
        }
        iter1++;
        iter2++;
        iter3++;
        iter4++;
    }
}

double GeneticAlgorithm::Fitness(std::vector<int> individual, std::vector<int> replication_num_vector)
{
    if (pipeline_granularity == "batch")
    {
        return FastEvaluationForBatch(individual, replication_num_vector);
    }
    else
    {
        return FastEvaluationForElement(individual, replication_num_vector);
    }
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
bool ** COMM_cycle_flag;
bool ** MEM_cycle_flag;
static int visited_single[MAX_CORE] = {0};
static double LATENCY_single[MAX_CORE] = {0};
static double LATENCY_first[MAX_CORE] = {0};
static std::map<int, int> comm_index_2_index_in_core;
static std::map<int, int> comm_index_2_core_index;
static long long last_MVMUL_exec_start_time[MAX_CORE] = {0};
static int MVMUL_instruction_group_core[MAX_CORE] = {0};
static long long last_synchronous_time[MAX_CORE] = {0};
static long long preparation_timeline[MAX_CORE] = {0};


double GeneticAlgorithm::FastEvaluationForBatch(std::vector<int> individual, std::vector<int> replication_num_vector)
{
    std::vector<int> fitness_individual;
    fitness_individual.assign(individual.begin(), individual.end());

    int appointed_instruction_group_num = 2;
    SimpleInstructionList.resize(appointed_instruction_group_num);
    SimpleWBInstructionList.resize(appointed_instruction_group_num);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        SimpleInstructionList[ig].resize(ChipH * ChipW);
        SimpleWBInstructionList[ig].resize(ChipH * ChipW);
    }

    int comm_index = 0;
    std::vector<int> core_IG_num;
    core_IG_num.resize(ChipH * ChipW);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        std::vector<int> node_AG_appearance_num;
        node_AG_appearance_num.resize(node_num);
        int max_IG_num = 0;
        int tmp_AG_index_in_total = 0;
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            int start_address = i * max_AG_kind_per_core;
            int end_address = (i+1) * max_AG_kind_per_core - 1;
            std::sort(fitness_individual.begin() + start_address, fitness_individual.begin() + end_address + 1);
            for (int j = start_address + 1; j <= end_address; ++j)
            {
                if (fitness_individual[j-1]/10000  == fitness_individual[j]/10000)
                {
                    fitness_individual[j] += fitness_individual[j-1] % 10000;
                    fitness_individual[j-1] = 0;
                }
            }
            for (int j = start_address; j <= end_address; ++j)
            {
                if (fitness_individual[j] == 0)
                    continue;
                int node_index = fitness_individual[j] / 10000;
                int AG_num_per_replication = node_AG_num[node_index];
                int AG_num = fitness_individual[j] % 10000;
                for (int k = 0; k < AG_num; ++k)
                {
                    int IG_num =  ceil(float(node_cycle_num[node_index])/float(replication_num_vector[node_index]));
                    if (IG_num > max_IG_num)
                        max_IG_num = IG_num;
                    int input_element_num = CrossbarH;
                    int output_element_num = PIMCOMP_node_list[node_index].output_dim[1];
                    int AG_index_in_replication = node_AG_appearance_num[node_index] % node_AG_num[node_index];
                    node_AG_appearance_num[node_index]++;

                    if (k / AG_num_per_replication == 0)
                    {
                        struct SimpleInst Instruction_ld;
                        Instruction_ld.operation = "LD";
                        Instruction_ld.element_num = input_element_num;
                        SimpleInstructionList[ig][i].push_back(Instruction_ld);
                    }
                    else
                    {
                        float new_load_rate = 1.0;
                        if (PIMCOMP_node_list[node_index].operation == "OP_CONV")
                            new_load_rate = float(PIMCOMP_node_list[node_index].param.stride_h * PIMCOMP_node_list[node_index].param.stride_w)
                                    / float(PIMCOMP_node_list[node_index].param.kernel_h * PIMCOMP_node_list[node_index].param.kernel_w);
                        struct SimpleInst Instruction_ld;
                        Instruction_ld.operation = "LD";
                        Instruction_ld.element_num = input_element_num * new_load_rate;
                        SimpleInstructionList[ig][i].push_back(Instruction_ld);
                    }

                    struct SimpleInst Instruction_mvmul;
                    Instruction_mvmul.operation = "MVMUL";
                    Instruction_mvmul.node_index = node_index;
                    Instruction_mvmul.AG_index = tmp_AG_index_in_total;
                    SimpleInstructionList[ig][i].push_back(Instruction_mvmul);

                    if (AG_index_in_replication == 0)
                    {
                        struct SimpleInst Instruction_st;
                        Instruction_st.operation = "ST";
                        Instruction_st.element_num = output_element_num;
                        SimpleWBInstructionList[ig][i].push_back(Instruction_st);
                    }

                    tmp_AG_index_in_total++;
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

    double max_latency = 0;
    double sum_latency = 0;
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
//        std::cout << "core:" << i << "  latency:" << std::fixed << std::setprecision(1)
//                  << (LATENCY_single[i] - LATENCY_first[i]) / (appointed_instruction_group_num-1) * (core_IG_num[i]-1) + LATENCY_first[i] << std::endl;
        if (max_latency < (LATENCY_single[i] - LATENCY_first[i]) / (appointed_instruction_group_num-1) * (core_IG_num[i]-1) + LATENCY_first[i])
            max_latency = (LATENCY_single[i] - LATENCY_first[i]) / (appointed_instruction_group_num-1) * (core_IG_num[i]-1) + LATENCY_first[i];
        sum_latency += (LATENCY_single[i] - LATENCY_first[i]) / (appointed_instruction_group_num-1) * (core_IG_num[i]-1) + LATENCY_first[i];
    }

    for (int i = 0; i < MAX_CHIP; ++i)
    {
        delete [] COMM_cycle_flag[i];
        delete [] MEM_cycle_flag[i];
    }
    delete [] COMM_cycle_flag;
    delete [] MEM_cycle_flag;
    SimpleInstructionList.clear();
    SimpleCOMMInstructionList.clear();
    SimpleWBInstructionList.clear();
    comm_index_2_index_in_core.clear();
    comm_index_2_core_index.clear();
    for (auto &v : visited_single) {v = 0;}
    for (auto &v : LATENCY_single) {v = 0;}
    for (auto &v : LATENCY_first) {v = 0;}
    for (auto &v : last_MVMUL_exec_start_time) {v = 0;}
    for (auto &v : MVMUL_instruction_group_core) {v = 0;}
    for (auto &v : last_synchronous_time) {v = 0;}
    for (auto &v : preparation_timeline) {v = 0;}
    return max_latency;
}

double GeneticAlgorithm::FastEvaluationForElement(std::vector<int> individual, std::vector<int> replication_num_vector)
{
    std::vector<int> fitness_individual;
    fitness_individual.assign(individual.begin(), individual.end());

    int appointed_instruction_group_num = 1;
    SimpleInstructionList.resize(appointed_instruction_group_num);
    SimpleCOMMInstructionList.resize(appointed_instruction_group_num);
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        SimpleInstructionList[ig].resize(ChipH * ChipW);
        SimpleCOMMInstructionList[ig].resize(ChipH * ChipW);
    }

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
        }
    }

    int comm_index = 0;
    for (int ig = 0; ig < appointed_instruction_group_num; ++ig)
    {
        int max_IG_num = 0;
        int tmp_AG_index_in_total = 0;
        std::vector<int> node_AG_appearance_num;
        node_AG_appearance_num.resize(node_num);
        for (int i = 0; i < ChipH * ChipW; ++i)
        {
            int start_address = i * max_AG_kind_per_core;
            int end_address = (i+1) * max_AG_kind_per_core - 1;
            std::sort(fitness_individual.begin() + start_address, fitness_individual.begin() + end_address + 1);
            for (int j = start_address + 1; j <= end_address; ++j)
            {
                if (fitness_individual[j-1]/10000  == fitness_individual[j]/10000)
                {
                    fitness_individual[j] += fitness_individual[j-1] % 10000;
                    fitness_individual[j-1] = 0;
                }
            }
            for (int j = start_address; j <= end_address; ++j)
            {
                if (fitness_individual[j] == 0)
                    continue;
                int node_index = fitness_individual[j] / 10000;
                int AG_num = fitness_individual[j] % 10000;
                for (int k = 0; k < AG_num; ++k)
                {
                    int IG_num =  ceil(float(node_cycle_num[node_index])/float(replication_num_vector[node_index]));
                    if (IG_num > max_IG_num)
                        max_IG_num = IG_num;
                    int input_element_num = CrossbarH;
                    int output_element_num = PIMCOMP_node_list[node_index].output_dim[1];
                    int AG_index_in_replication = node_AG_appearance_num[node_index] % node_AG_num[node_index];
                    node_AG_appearance_num[node_index]++;

                    struct SimpleInst Instruction_mvmul;
                    Instruction_mvmul.operation = "MVMUL";
                    Instruction_mvmul.node_index = node_index;
                    Instruction_mvmul.AG_index = tmp_AG_index_in_total;
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
                                Instruction_send.element_num = output_element_num;
                                Instruction_send.comm_index = comm_index;
                                SimpleCOMMInstructionList[ig][i].push_back(Instruction_send);

                                struct SimpleInst Instruction_recv;
                                Instruction_recv.operation = "RECV";
                                Instruction_recv.from_core = i;
                                Instruction_recv.element_num = output_element_num;
                                Instruction_recv.comm_index = comm_index;
                                SimpleCOMMInstructionList[ig][Instruction_send.to_core].push_back(Instruction_recv);

                                comm_index++;
                            }

                        }
                    }

                    tmp_AG_index_in_total++;
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

    double practical_time = 0.0;
    for (int i = 0; i < ChipH * ChipW; ++i)
    {
        if (practical_time < LATENCY_single[i])
            practical_time = LATENCY_single[i];
    }

    for (int i = 0; i < MAX_CHIP; ++i)
    {
        delete [] COMM_cycle_flag[i];
        delete [] MEM_cycle_flag[i];
    }
    delete [] COMM_cycle_flag;
    delete [] MEM_cycle_flag;
    SimpleInstructionList.clear();
    SimpleCOMMInstructionList.clear();
    SimpleWBInstructionList.clear();
    comm_index_2_index_in_core.clear();
    comm_index_2_core_index.clear();
    for (auto &v : visited_single) {v = 0;}
    for (auto &v : LATENCY_single) {v = 0;}
    for (auto &v : LATENCY_first) {v = 0;}
    for (auto &v : last_MVMUL_exec_start_time) {v = 0;}
    for (auto &v : MVMUL_instruction_group_core) {v = 0;}
    for (auto &v : last_synchronous_time) {v = 0;}
    for (auto &v : preparation_timeline) {v = 0;}

    return FastEvaluationInstructionGroupNum(replication_num_vector) * practical_time / appointed_instruction_group_num;
//    return practical_time / appointed_instruction_group_num;
}

 double GeneticAlgorithm::FastEvaluationInstructionGroupNum(std::vector<int> replication_num_vector)
{
    std::vector<int> tmp_replication_num_vector = replication_num_vector;
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
    return double(max_delay * effective_first_conv_replication * first_conv_instruction_group_num);
}


void GeneticAlgorithm::FastSingleInstructionGroupEvaluation(int instruction_group_index, int core_index, int index_in_core)
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
                int communication_needed_cycle = std::ceil((double(communication_bytes_num)) / (BUS_bandwidth / Frequency));
                int real_comm_latency = 0;

                // fully synchronized mechanism
                real_comm_latency = CheckBusBandwidth(0, LATENCY_single[corresponding_core_index], communication_needed_cycle);
                if (LATENCY_single[core_index] > LATENCY_single[corresponding_core_index])
                {
                    LATENCY_single[corresponding_core_index] = LATENCY_single[core_index];
                }
                else
                {
                    LATENCY_single[core_index] = LATENCY_single[corresponding_core_index];
                }
                last_synchronous_time[core_index] = LATENCY_single[core_index];
                last_synchronous_time[corresponding_core_index] = LATENCY_single[corresponding_core_index];
                LATENCY_single[corresponding_core_index] += real_comm_latency;
                LATENCY_single[core_index] += real_comm_latency;

                // Incomplete Synchronization Mechanism
//                if (tmpInstruction.operation == "RECV")  // Corresponding Core Send to Current Core
//                {
//                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[corresponding_core_index], communication_needed_cycle);
//                    LATENCY_single[corresponding_core_index] += real_comm_latency;  // Send Finish Time
//                    if (LATENCY_single[core_index] < LATENCY_single[corresponding_core_index])
//                    {
//                        LATENCY_single[core_index] = LATENCY_single[corresponding_core_index];
//                        last_synchronous_time[core_index] = LATENCY_single[core_index];
//                    }
//                }
//                else    // Current Core Send to Corresponding Core
//                {
//                    real_comm_latency = CheckBusBandwidth(0, LATENCY_single[core_index], communication_needed_cycle);
//                    LATENCY_single[core_index] += real_comm_latency;  // Send Finish Time
//                    if (LATENCY_single[corresponding_core_index] < LATENCY_single[core_index])
//                    {
//                        LATENCY_single[corresponding_core_index] = LATENCY_single[core_index];
//                        last_synchronous_time[corresponding_core_index] = LATENCY_single[corresponding_core_index];
//                    }
//                }

                // Shared
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
                int real_interval = GA_MVMUL_start_interval;
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
                if (preparation_timeline[core_index] < LATENCY_single[core_index])
                    last_MVMUL_exec_start_time[core_index] = LATENCY_single[core_index];
                else
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

int GeneticAlgorithm::CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle)
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

int GeneticAlgorithm::CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle)
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

int cmp2(const std::pair<int, int> &x, const std::pair<int, int> &y) { return x.second > y.second; }

void GeneticAlgorithm::Mutation()
{
    int mutation_num = population_num - current_chromosome.size();
    std::cout << "mutation_num:" <<  mutation_num << std::endl;
    std::vector<std::vector<int>> mutation_population;
    std::vector<std::vector<int>> mutation_replication_num_per_node;
    std::vector<std::vector<int>> mutation_crossbar_num_per_core;
    std::vector<std::vector<int>> mutation_AG_type_per_core;
    while (mutation_population.size() < mutation_num)
    {
        int current_population_num = current_chromosome.size();
        int select_individual = rand() % current_population_num;
        std::vector<int> mutation_individual = current_chromosome[select_individual];
        std::vector<int> mutation_individual_replication_num = node_replication_num[select_individual];
        std::vector<int> mutation_individual_crossbar_num = core_crossbar_num[select_individual];
        std::vector<int> mutation_individual_AG_type_num = core_AG_type_num[select_individual];
        int mutation_indicator = rand() % 100;

        //// 改变复制倍数
        if (mutation_indicator < increase_replication_num_rate * 100)
        {
            int conv_index = CONV_index[rand() % conv_num];
            int AG_num_per_replication = node_AG_num[conv_index];
            int crossbar_num_per_AG = node_crossbar_num_per_AG[conv_index];
            int crossbar_num_per_replication = AG_num_per_replication * crossbar_num_per_AG;

            std::vector<std::pair<int, int>> tmp_core_spare_crossbar_num;
            for (int i = 0; i < mutation_individual_crossbar_num.size(); ++i)
                if (CoreW * CoreH - mutation_individual_crossbar_num[i] >= crossbar_num_per_replication)
                    tmp_core_spare_crossbar_num.push_back(std::make_pair(i, CoreW * CoreH - mutation_individual_crossbar_num[i]));
//            std::sort(tmp_core_spare_crossbar_num.begin(), tmp_core_spare_crossbar_num.end(), cmp2); // Sort in descending order by rest crossbar resource

            if (tmp_core_spare_crossbar_num.size() == 0)
                continue;
            std::cout << "Increase  ";
            int random_core_index = rand() % tmp_core_spare_crossbar_num.size();
            int select_core_index = tmp_core_spare_crossbar_num[random_core_index].first;
            int rest_crossbar_num_in_core = tmp_core_spare_crossbar_num[random_core_index].second;

            mutation_individual_replication_num[conv_index] += 1;
            int selected_position =  select_core_index * max_AG_kind_per_core + mutation_individual_AG_type_num[select_core_index];
            mutation_individual_crossbar_num[select_core_index] += crossbar_num_per_replication;
            mutation_individual_AG_type_num[select_core_index]++;
            mutation_individual[selected_position] = conv_index * 10000 + AG_num_per_replication;
        }
        else if (mutation_indicator < (increase_replication_num_rate + decrease_replication_num_rate) * 100)
        {
            int selected_conv_node = rand() % conv_num;
            int selected_conv_index = CONV_index[selected_conv_node];
            if (mutation_individual_replication_num[selected_conv_index] == 1)
                continue;
            std::cout << "Decrease  ";
            mutation_individual_replication_num[selected_conv_index] -= 1;
            int reduce_AG_num = node_AG_num[selected_conv_index];
            int reduce_crossbar_num_per_AG = node_crossbar_num_per_AG[selected_conv_index];
            int start_index = rand() % chromosome_size;
            for (int i = start_index; i < chromosome_size + start_index; ++i)
            {
                int index = i % chromosome_size;
                if (mutation_individual[index] != 0)
                {
                    int core_index = index / max_AG_kind_per_core;
                    int node_index = mutation_individual[index] / 10000;
                    if (node_index != selected_conv_index)
                        continue;
                    int AG_num = mutation_individual[index] % 10000;
                    if (AG_num <= reduce_AG_num)
                    {
                        mutation_individual[index] = 0;
                        mutation_individual_AG_type_num[core_index]--;
                        mutation_individual_crossbar_num[core_index] -= AG_num * reduce_crossbar_num_per_AG;
                        reduce_AG_num -= AG_num;
                    }
                    else
                    {
                        AG_num -= reduce_AG_num;
                        mutation_individual_crossbar_num[core_index] -= reduce_AG_num * reduce_crossbar_num_per_AG;
                        mutation_individual[index] = selected_conv_index * 10000 + AG_num;
                        reduce_AG_num -= reduce_AG_num;
                    }
                    if (reduce_AG_num == 0)
                    {
                        break;
                    }
                }
            }
        }
        else if (mutation_indicator < (increase_replication_num_rate + decrease_replication_num_rate + split_rate) * 100)
        {
            std::cout << "Split  ";
            bool try_again = true;
            int tmp_max_iteration1 = 0;
            do
            {
                int selected_core = rand() % core_num;
                int start_address = selected_core * max_AG_kind_per_core;
                int end_address = (selected_core+1) * max_AG_kind_per_core - 1;
                for (int j = start_address; j <= end_address; ++j)
                {
                    if (mutation_individual[j] != 0)
                    {
                        int node_index = mutation_individual[j] /10000;
                        int AG_num = mutation_individual[j] % 10000;
                        if (AG_num == 1)
                            continue;
                        int crossbar_num = AG_num * node_crossbar_num_per_AG[node_index];
                        if (crossbar_num > CoreH * CoreW / 4)
                        {
                            int change_AG_num = rand() % (AG_num-1) + 1;
                            int change_crossbar_num = change_AG_num * node_crossbar_num_per_AG[node_index];
                            int tmp_max_iteration2 = 0;
                            int target_core_index = rand() % core_num;
                            while (mutation_individual_crossbar_num[target_core_index] + change_crossbar_num > CoreH * CoreW
                                   || mutation_individual_AG_type_num[target_core_index] >= max_AG_kind_per_core)
                            {
                                target_core_index = rand() % core_num;
                                tmp_max_iteration2++;
                                if (tmp_max_iteration2 == 30)
                                    break;
                            }
                            if (tmp_max_iteration2 == 30)
                                continue;
                            int target_position = target_core_index * max_AG_kind_per_core + mutation_individual_AG_type_num[target_core_index];
                            mutation_individual_crossbar_num[target_core_index] += change_crossbar_num;
                            mutation_individual_AG_type_num[target_core_index]++;
                            mutation_individual[target_position] = node_index * 10000 + change_AG_num;

                            int select_position = j;
                            mutation_individual_crossbar_num[selected_core] -= change_crossbar_num;
                            mutation_individual[select_position] = node_index * 10000 + (AG_num-change_AG_num);

                            if (CheckLegality(mutation_individual, mutation_individual_replication_num, mutation_individual_AG_type_num, mutation_individual_crossbar_num, "Mutation Split"))
                            {
                                try_again = false;
                                break;
                            }
                        }
                    }
                }
                tmp_max_iteration1++;
                if (tmp_max_iteration1 == 50)
                    break;
            } while(try_again);
        }
        else if (mutation_indicator < (increase_replication_num_rate + decrease_replication_num_rate + split_rate + gather_rate) * 100)
        {
            std::cout << "Gather  ";
            bool try_again = true;
            int tmp_max_iteration1 = 0;
            do
            {
                int selected_core = rand() % core_num;
                int start_address = selected_core * max_AG_kind_per_core;
                int end_address = (selected_core+1) * max_AG_kind_per_core - 1;
                for (int j = start_address; j <= end_address; ++j)
                {
                    if (mutation_individual[j] != 0)
                    {
                        int node_index = mutation_individual[j] /10000;
                        int AG_num = mutation_individual[j] % 10000;
                        if (AG_num > 2)
                            continue;
                        int crossbar_num = AG_num * node_crossbar_num_per_AG[node_index];
                        if (crossbar_num < CoreH * CoreW / 24)
                        {
                            int change_AG_num = AG_num;
                            int change_crossbar_num = crossbar_num;
                            int tmp_max_iteration2 = 0;
                            int target_core_index = rand() % core_num;
                            std::set<int> target_core_node;
                            for (int i = 0; i < mutation_individual_AG_type_num[target_core_index]; ++i)
                            {
                                int position = target_core_index * max_AG_kind_per_core + i;
                                int node_in_core = mutation_individual[position]/10000;
                                target_core_node.insert(node_in_core);
                            }
                            while (mutation_individual_crossbar_num[target_core_index] + change_crossbar_num > CoreH * CoreW
                                   || mutation_individual_AG_type_num[target_core_index] >= max_AG_kind_per_core
                                   || target_core_node.count(node_index) == 0)
                            {
                                target_core_index = rand() % core_num;
                                target_core_node.clear();
                                for (int i = 0; i < mutation_individual_AG_type_num[target_core_index]; ++i)
                                {
                                    int position = target_core_index * max_AG_kind_per_core + i;
                                    int node_in_core = mutation_individual[position]/10000;
                                    target_core_node.insert(node_in_core);
                                }
                                tmp_max_iteration2++;
                                if (tmp_max_iteration2 == 30)
                                    break;
                            }
                            if (tmp_max_iteration2 == 30)
                                continue;
                            int target_position = target_core_index * max_AG_kind_per_core + mutation_individual_AG_type_num[target_core_index];
                            mutation_individual_crossbar_num[target_core_index] += change_crossbar_num;
                            mutation_individual_AG_type_num[target_core_index]++;
                            mutation_individual[target_position] = node_index * 10000 + change_AG_num;

                            int select_position = j;
                            mutation_individual_crossbar_num[selected_core] -= change_crossbar_num;
                            mutation_individual_AG_type_num[selected_core]--;
                            mutation_individual[select_position] = 0;

                            if (CheckLegality(mutation_individual, mutation_individual_replication_num, mutation_individual_AG_type_num, mutation_individual_crossbar_num, "Mutation Gather"))
                            {
                                try_again = false;
                                break;
                            }
                        }
                    }
                }
                tmp_max_iteration1++;
                if (tmp_max_iteration1 == 50)
                    break;
            } while(try_again);
        }
        else if (mutation_indicator < (increase_replication_num_rate + decrease_replication_num_rate + split_rate + gather_rate + exchange_rate) * 100)
        {
            std::cout << "Exchange  ";
            bool try_again = true;
            int tmp_max_iteration = 0;
            do
            {
                int selected_position_1 = rand() % chromosome_size;
                while (mutation_individual[selected_position_1] == 0)
                    selected_position_1 = rand() % chromosome_size;
                int node_index_1 = mutation_individual[selected_position_1] / 10000;
                int AG_num_1 = mutation_individual[selected_position_1] % 10000;
                int crossbar_num_per_AG_1 = node_crossbar_num_per_AG[node_index_1];
                int core_index_1 = selected_position_1 / max_AG_kind_per_core;

                int selected_position_2 = rand() % chromosome_size;
                while (mutation_individual[selected_position_2] == 0)
                    selected_position_2 = rand() % chromosome_size;
                int node_index_2 = mutation_individual[selected_position_2] / 10000;
                int AG_num_2 = mutation_individual[selected_position_2] % 10000;
                int crossbar_num_per_AG_2 = node_crossbar_num_per_AG[node_index_2];
                int core_index_2 = selected_position_2 / max_AG_kind_per_core;

                if (AG_num_1 * crossbar_num_per_AG_1 < AG_num_2 * crossbar_num_per_AG_2)
                {
                    // exchange all AGs of node_1 on core_1 and part of AGs of node_2 on core_2
                    int exchange_AG_num_2 = std::floor(float(AG_num_1 * crossbar_num_per_AG_1) / float(crossbar_num_per_AG_2));
                    if (exchange_AG_num_2 == 0)
                    {
                        tmp_max_iteration++;
                        continue;
                    }
                    int exchange_AG_num_1 = std::floor(float(exchange_AG_num_2 * crossbar_num_per_AG_2) / float(crossbar_num_per_AG_1));
                    if (exchange_AG_num_1 == 0)
                    {
                        tmp_max_iteration++;
                        continue;
                    }

                    bool core_1_has_node_2 = false;
                    for (int i = core_index_1*max_AG_kind_per_core; i < (core_index_1+1)*max_AG_kind_per_core; ++i)
                    {
                        if (mutation_individual[i]/10000 == node_index_2)
                        {
                            core_1_has_node_2 = true;
                            mutation_individual[i] += exchange_AG_num_2;
                            mutation_individual_crossbar_num[core_index_1] += exchange_AG_num_2 * crossbar_num_per_AG_2;
                            break;
                        }
                    }
                    if (!core_1_has_node_2)
                    {
                        mutation_individual_AG_type_num[core_index_1]++;
                        int add_position = core_index_1 * max_AG_kind_per_core + mutation_individual_AG_type_num[core_index_1];
                        if (mutation_individual[add_position] != 0)
                            continue;
                        mutation_individual[add_position] = node_index_2 * 10000 + exchange_AG_num_2;
                        mutation_individual_crossbar_num[core_index_1] += exchange_AG_num_2 * crossbar_num_per_AG_2;
                    }

                    bool core_2_has_node_1 = false;
                    for (int i = core_index_2*max_AG_kind_per_core; i < (core_index_2+1)*max_AG_kind_per_core; ++i)
                    {
                        if (mutation_individual[i]/10000 == node_index_1)
                        {
                            core_2_has_node_1 = true;
                            mutation_individual[i] += exchange_AG_num_1;
                            mutation_individual_crossbar_num[core_index_2] += exchange_AG_num_1 * crossbar_num_per_AG_1;
                            break;
                        }
                    }
                    if (!core_2_has_node_1)
                    {
                        mutation_individual_AG_type_num[core_index_2]++;
                        int add_position = core_index_2 * max_AG_kind_per_core + mutation_individual_AG_type_num[core_index_2];
                        if (mutation_individual[add_position] != 0)
                            continue;
                        mutation_individual[add_position] = node_index_1 * 10000 + exchange_AG_num_1;
                        mutation_individual_crossbar_num[core_index_2] += exchange_AG_num_1 * crossbar_num_per_AG_1;
                    }

                    mutation_individual_crossbar_num[core_index_1] -= exchange_AG_num_1 * crossbar_num_per_AG_1;
                    mutation_individual[selected_position_1] -= exchange_AG_num_1;
                    if (exchange_AG_num_1 == AG_num_1)
                    {
                        mutation_individual_AG_type_num[core_index_1] --;
                        mutation_individual[selected_position_1] = 0;
                    }
                    mutation_individual_crossbar_num[core_index_2] -= exchange_AG_num_2 * crossbar_num_per_AG_2;
                    mutation_individual[selected_position_2] -= exchange_AG_num_2;
                    if (exchange_AG_num_2 == AG_num_2)
                    {
                        mutation_individual_AG_type_num[core_index_2] --;
                        mutation_individual[selected_position_2] = 0;
                    }

                    try_again = false;
                }
                else
                {
                    // exchange all AGs of node_2 on core_2 and part of AGs of node_1 on core_1
                    int exchange_AG_num_1 = std::floor(float(AG_num_2 * crossbar_num_per_AG_2) / float(crossbar_num_per_AG_1));
                    if (exchange_AG_num_1 == 0)
                    {
                        tmp_max_iteration++;
                        continue;
                    }
                    int exchange_AG_num_2 = std::floor(float(exchange_AG_num_1 * crossbar_num_per_AG_1) / float(crossbar_num_per_AG_2));
                    if (exchange_AG_num_2 == 0)
                    {
                        tmp_max_iteration++;
                        continue;
                    }

                    bool core_1_has_node_2 = false;
                    for (int i = core_index_1*max_AG_kind_per_core; i < (core_index_1+1)*max_AG_kind_per_core; ++i)
                    {
                        if (mutation_individual[i]/10000 == node_index_2)
                        {
                            core_1_has_node_2 = true;
                            mutation_individual[i] += exchange_AG_num_2;
                            mutation_individual_crossbar_num[core_index_1] += exchange_AG_num_2 * crossbar_num_per_AG_2;
                            break;
                        }
                    }
                    if (!core_1_has_node_2)
                    {
                        mutation_individual_AG_type_num[core_index_1]++;
                        int add_position = core_index_1 * max_AG_kind_per_core + mutation_individual_AG_type_num[core_index_1];
                        if (mutation_individual[add_position] != 0)
                            continue;
                        mutation_individual[add_position] = node_index_2 * 10000 + exchange_AG_num_2;
                        mutation_individual_crossbar_num[core_index_1] += exchange_AG_num_2 * crossbar_num_per_AG_2;
                    }

                    bool core_2_has_node_1 = false;
                    for (int i = core_index_2*max_AG_kind_per_core; i < (core_index_2+1)*max_AG_kind_per_core; ++i)
                    {
                        if (mutation_individual[i]/10000 == node_index_1)
                        {
                            core_2_has_node_1 = true;
                            mutation_individual[i] += exchange_AG_num_1;
                            mutation_individual_crossbar_num[core_index_2] += exchange_AG_num_1 * crossbar_num_per_AG_1;
                            break;
                        }
                    }
                    if (!core_2_has_node_1)
                    {
                        mutation_individual_AG_type_num[core_index_2]++;
                        int add_position = core_index_2 * max_AG_kind_per_core + mutation_individual_AG_type_num[core_index_2];
                        if (mutation_individual[add_position] != 0)
                            continue;
                        mutation_individual[add_position] = node_index_1 * 10000 + exchange_AG_num_1;
                        mutation_individual_crossbar_num[core_index_2] += exchange_AG_num_1 * crossbar_num_per_AG_1;
                    }

                    mutation_individual_crossbar_num[core_index_1] -= exchange_AG_num_1 * crossbar_num_per_AG_1;
                    mutation_individual[selected_position_1] -= exchange_AG_num_1;
                    if (exchange_AG_num_1 == AG_num_1)
                    {
                        mutation_individual_AG_type_num[core_index_1] --;
                        mutation_individual[selected_position_1] = 0;
                    }
                    mutation_individual_crossbar_num[core_index_2] -= exchange_AG_num_2 * crossbar_num_per_AG_2;
                    mutation_individual[selected_position_2] -= exchange_AG_num_2;
                    if (exchange_AG_num_2 == AG_num_2)
                    {
                        mutation_individual_AG_type_num[core_index_2] --;
                        mutation_individual[selected_position_2] = 0;
                    }

                    try_again = false;
                }
            } while (try_again && tmp_max_iteration < 30);
        }

        if (CheckLegality(mutation_individual, mutation_individual_replication_num, mutation_individual_AG_type_num, mutation_individual_crossbar_num, "Mutation Stage"))
        {
            std::cout << "success" << std::endl;
            mutation_population.push_back(mutation_individual);
            mutation_replication_num_per_node.push_back(mutation_individual_replication_num);
            mutation_crossbar_num_per_core.push_back(mutation_individual_crossbar_num);
            mutation_AG_type_per_core.push_back(mutation_individual_AG_type_num);
        }
        else
            std::cout << "fail" << std::endl;
    }
    current_chromosome.insert(current_chromosome.end(),mutation_population.begin(),mutation_population.end());
    node_replication_num.insert(node_replication_num.end(), mutation_replication_num_per_node.begin(), mutation_replication_num_per_node.end());
    core_crossbar_num.insert(core_crossbar_num.end(), mutation_crossbar_num_per_core.begin(), mutation_crossbar_num_per_core.end());
    core_AG_type_num.insert(core_AG_type_num.end(), mutation_AG_type_per_core.begin(), mutation_AG_type_per_core.end());
}

void GeneticAlgorithm::PostProcess()
{
    int candidate_num;
    if (appointed_candidate_num < population_num * remain_rate)
        candidate_num = appointed_candidate_num;
    else
        candidate_num = ceil(population_num * remain_rate);
    PIMCOMP_DSE_core_map_info.resize(candidate_num);
    PIMCOMP_DSE_result_info.resize(candidate_num);
    PIMCOMP_DSE_replication_num.assign(node_replication_num.begin(), node_replication_num.begin() + candidate_num);
    PIMCOMP_DSE_node_map_info.resize(candidate_num);
    int legal_candidate_num = 0;
    for (int i = 0; i < candidate_num; ++i)
    {
        // There are 2 problems in the generated result
        // ① node sequence is not increasing
        // ② the AG of the same node may appear in different positions
        // 因为生成的result中存在 "①node顺序不是递增 ②同一个node的AG出现在两个位置" 的问题，所以需要处理一下。
        std::vector<int> individual = current_chromosome[i];
        for (int j = 0; j < core_num; ++j)
        {
            int start_address = j * max_AG_kind_per_core;
            int end_address = (j+1) * max_AG_kind_per_core - 1;

            std::sort(individual.begin() + start_address, individual.begin() + end_address + 1);
            for (int k = start_address + 1; k <= end_address; ++k)
            {
                if (individual[k] == 0)
                    continue;
                if (individual[k-1]/10000  == individual[k]/10000)
                {
                    individual[k] += individual[k-1] % 10000;
                    individual[k-1] = 0;
                    core_AG_type_num[i][j]--;
                }
            }
        }

        // check its legality
        if (!CheckLegality(individual, node_replication_num[i], core_AG_type_num[i], core_crossbar_num[i], "Post"))
        {
            fprintf(stderr, "Illegal Result. \n");
            return;
        }

        // Save the data to PIMCOM_DSE_node_map_info (record which core each AG of each copy of each node of each candidate is mapped to)
        // 保存到PIMCOM_DSE_node_map_info（记录每个候选的每个节点的每个复制的每个AG映射到哪个核上）
        PIMCOMP_DSE_node_map_info[i].resize(node_num);
        for (int j = 0; j < node_num; ++j)
        {
            if (node_replication_num[i][j] != 0)
            {
                PIMCOMP_DSE_node_map_info[i][j].resize(node_replication_num[i][j]);
                for (int k = 0; k < node_replication_num[i][j]; ++k)
                {
                    PIMCOMP_DSE_node_map_info[i][j][k].resize(node_AG_num[j]);
                }
            }
        }

        // Save the data to PIMCOM_DSE_core_map_info (record which AGs are mapped on each core of each candidate)
        // 保存到PIMCOM_DSE_core_map_info（记录每个候选的每个核上都映射了哪些AG）
        int AG_num_total = 0;
        int crossbar_num_total = 0;
        int MVMUL_total = 0;
        PIMCOMP_DSE_core_map_info[i].resize(core_num);
        std::vector<int> tmp_node_crossbar_num;
        std::vector<int> tmp_node_replication_num;
        std::vector<int> tmp_node_AG_num;
        std::vector<int> tmp_core_mvmul_num;
        tmp_node_crossbar_num.resize(node_num);
        tmp_node_replication_num.resize(node_num);
        tmp_node_AG_num.resize(node_num);
        tmp_core_mvmul_num.resize(core_num);
        for (int j = 0; j < chromosome_size; ++j)
        {
            if (individual[j] != 0)
            {
                int core_index = j / max_AG_kind_per_core;
                int node_index = individual[j] / 10000;
                int AG_num = individual[j] % 10000;
                int crossbar_num = individual[j] % 10000 * node_crossbar_num_per_AG[node_index];
                int mvmul_num = node_cycle_num[node_index] / node_replication_num[i][node_index] * AG_num;
                AG_num_total += AG_num;
                crossbar_num_total += crossbar_num;
                tmp_core_mvmul_num[core_index] += mvmul_num;
                MVMUL_total += mvmul_num;
                while (AG_num > 0)
                {
                    if ( node_AG_num[node_index]*(tmp_node_replication_num[node_index]+1) - tmp_node_crossbar_num[node_index] <= AG_num )
                    {
//                        std::cout << "   core:" << core_index
//                                  << "   node:" << node_index
//                                  << "   replication:" << tmp_node_replication_num[node_index]
//                                  << "   AG_num:" << node_AG_num[node_index]*(tmp_node_replication_num[node_index]+1) - tmp_node_crossbar_num[node_index]
//                                  << "   crossbar_num:" << AG_num * node_crossbar_num_per_AG[individual[j]/10000] << std::endl;
                        int allocated_AG_num = node_AG_num[node_index]*(tmp_node_replication_num[node_index]+1) - tmp_node_crossbar_num[node_index];
                        for (int k = 0; k < allocated_AG_num; ++k)
                        {
                            struct DSE_AG_struct thisAG;
                            thisAG.node_index = node_index;
                            thisAG.core_index = core_index;
                            thisAG.replication_index = tmp_node_replication_num[node_index];
                            thisAG.crossbar_num_of_AG = node_crossbar_num_per_AG[node_index];
                            thisAG.input_cycle_num = mvmul_num;
                            thisAG.index_in_replication = (tmp_node_AG_num[node_index] + k ) % node_AG_num[node_index];
                            PIMCOMP_DSE_core_map_info[i][core_index].push_back(thisAG);
                            PIMCOMP_DSE_node_map_info[i][node_index][thisAG.replication_index][thisAG.index_in_replication] = core_index;
                        }
                        AG_num -= allocated_AG_num;
                        tmp_node_crossbar_num[node_index] += allocated_AG_num;
                        tmp_node_replication_num[node_index]++;
                        tmp_node_AG_num[node_index] += allocated_AG_num;
                    }
                    else
                    {
//                        std::cout << "   core:" << core_index
//                                  << "   node:" << node_index
//                                  << "   replication:" << tmp_node_replication_num[node_index]
//                                  << "   AG_num:" << AG_num
//                                  << "   crossbar_num:" << AG_num * node_crossbar_num_per_AG[individual[j]/10000] << std::endl;
                        int allocated_AG_num = AG_num;
                        for (int k = 0; k < allocated_AG_num; ++k)
                        {
                            struct DSE_AG_struct thisAG;
                            thisAG.node_index = node_index;
                            thisAG.core_index = core_index;
                            thisAG.replication_index = tmp_node_replication_num[node_index];
                            thisAG.crossbar_num_of_AG = node_crossbar_num_per_AG[node_index];
                            thisAG.input_cycle_num = mvmul_num;
                            thisAG.index_in_replication = (tmp_node_AG_num[node_index] + k ) % node_AG_num[node_index];
                            PIMCOMP_DSE_core_map_info[i][core_index].push_back(thisAG);
                            PIMCOMP_DSE_node_map_info[i][node_index][thisAG.replication_index][thisAG.index_in_replication] = core_index;
                        }
                        AG_num -= allocated_AG_num;
                        tmp_node_crossbar_num[node_index] += allocated_AG_num;
                        tmp_node_AG_num[node_index] += allocated_AG_num;
                    }
                }
            }
        }
        PIMCOMP_DSE_result_info[i].AG_num = AG_num_total;
        PIMCOMP_DSE_result_info[i].crossbar_num = crossbar_num_total;
        PIMCOMP_DSE_result_info[i].MVMUL_num = MVMUL_total;
        int tmp_max_mvmul_num = 0;
        for (int j = 0; j < core_num; ++j)
        {
            if (tmp_core_mvmul_num[j] > tmp_max_mvmul_num)
                tmp_max_mvmul_num = tmp_core_mvmul_num[j];
        }
        PIMCOMP_DSE_result_info[i].max_MVMUL_num = tmp_max_mvmul_num;
    }
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "            GA Successfully           " << std::endl;
    std::cout << "------------- DSE Result -------------" << std::endl
              << "[RRAMs_num:" << PIMCOMP_DSE_result_info[0].crossbar_num << "]" << std::endl
              << "[usage:"  << float(PIMCOMP_DSE_result_info[0].crossbar_num)/float(ChipW * ChipH * CoreW * CoreH)*100 << "%]" << std::endl
              << "[AG_num:" << PIMCOMP_DSE_result_info[0].AG_num << "]" << std::endl;
}

bool GeneticAlgorithm::CheckLegality(std::vector<int> & individual, std::vector<int> & replication_num_vector, std::vector<int> & AG_type_num_vector, std::vector<int> & crossbar_num_vector, std::string Stage)
{
    std::vector<int> node_AG_check;
    std::vector<int> core_AG_type_check;
    std::vector<int> core_crossbar_num_check;
    node_AG_check.resize(node_num);
    core_AG_type_check.resize(core_num);
    core_crossbar_num_check.resize(core_num);
    for (int i = 0; i < ChipW * ChipH; ++i)
    {
        int start_address = i * max_AG_kind_per_core;
        int end_address = (i+1) * max_AG_kind_per_core - 1;
        int crossbar_num_check = 0;
        for (int j = start_address; j <= end_address; ++j)
        {
            if (individual[j] != 0)
            {
                int node_index = individual[j] /10000;
                int AG_num = individual[j] % 10000;
                crossbar_num_check += node_crossbar_num_per_AG[node_index] * AG_num;
                node_AG_check[node_index] += AG_num;
                core_AG_type_check[i]++;
                core_crossbar_num_check[i] += node_crossbar_num_per_AG[node_index] * AG_num;
            }
        }
        if (crossbar_num_check > CoreW * CoreH)
        {
//            std::cout << Stage  << "Core" << i << "'s Crossbar Num is " << crossbar_num_check << ", Exceeds Constraint" << std::endl;
            return false;
        }
    }
    for (int i = 0; i < node_num; ++i)
    {
        if (node_AG_num[i] != 0)
        {
            if (node_AG_check[i] != node_AG_num[i] * replication_num_vector[i])
            {
//                std::cout << Stage << "   Node " << i << "'s total AG num is " << node_AG_check[i] << " , Replication Num is " <<  replication_num_vector[i] << ", AG per node is " << node_AG_num[i] << std::endl;
                return false;
            }
        }
    }

    for (int i = 0; i < core_num; ++i)
    {
        if (core_AG_type_check[i] != AG_type_num_vector[i])
            return false;
        if (core_crossbar_num_check[i] != crossbar_num_vector[i])
        {
//            std::cout << Stage << "   Core " << i << "'s total crossbar num is " << core_crossbar_num_check[i] << " , But given num is " <<  crossbar_num_vector[i] << std::endl;
            return false;
        }
    }
    return true;
}

void GeneticAlgorithm::SaveIntermediateInfo()
{
    Json::Value JsonGA;
    JsonGA["conv_num"] = conv_num;
    JsonGA["fc_num"] = fc_num;
    for (int i = 0; i < node_num; ++i)
    {
        if (node_AG_num[i] != 0)
        {
            JsonGA["node_MVMUL_num"][i] = node_MVMUL_num[i];
            JsonGA["node_cycle_num"][i] = node_cycle_num[i];
            JsonGA["node_crossbar_num_per_AG"][i] = node_crossbar_num_per_AG[i];
            JsonGA["node_AG_num"][i] = node_AG_num[i];
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

    JsonGA["node_list"] = DNNInfo["node_list"];

    std::string strJson = JsonGA.toStyledString();
    std::ofstream fob("../output/GeneticAlgorithm.json", std::ios::trunc | std::ios::out);
    if (fob.is_open())
    {
        fob.write(strJson.c_str(), strJson.length());
        fob.close();
    }
}
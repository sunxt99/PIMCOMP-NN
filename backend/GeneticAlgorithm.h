#ifndef PIMCOMP_OPEN_GENETICALGORITHM_H
#define PIMCOMP_OPEN_GENETICALGORITHM_H
#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"
#include "../evaluation/EvaluationConfig.h"

class GeneticAlgorithm
{
public:
    GeneticAlgorithm(std::string model_name_, std::string pipeline_granularity_);
    void GeneticAlgorithmOptimizing();
private:
    std::string model_name;
    std::string pipeline_granularity;
    double GA_MVMUL_start_interval;
    int core_num = ChipH * ChipW;
    int node_num = PIMCOMP_node_list_origin.size();

    int population_num = 200;
    int max_iteration = 3;

    int max_AG_kind_per_core = 32;
    int chromosome_size = max_AG_kind_per_core * core_num;

    double remain_rate = 0.3;
    double random_select_rate = 0.2;
    double increase_replication_num_rate = 0.2;
    double decrease_replication_num_rate = 0.2;
    double split_rate = 0.2;
    double gather_rate = 0.2;
    double exchange_rate = 0.2;

    int conv_num;
    int fc_num;
    std::vector<int> CONV_FC_index;
    std::vector<int> CONV_index;
    std::vector<int> FC_index;
    std::vector<int> node_crossbar_num_per_AG;
    std::vector<int> node_AG_num;
    std::vector<int> node_cycle_num;
    std::vector<int> node_MVMUL_num;
    std::vector<std::vector<int>> current_chromosome;
    std::vector<std::vector<int>> node_replication_num;
    std::vector<std::vector<int>> core_AG_type_num;
    std::vector<std::vector<int>> core_crossbar_num;
    // Init
    void Init();
    void InitReplicating(float margin_factor, int individual_index);
    void InitMapping(int individual_index);
    void InitMappingClustered(int individual_index);
    void InitMappingDistributed(int individual_index);
    // Select
    void Select();
    double Fitness(std::vector<int> individual, std::vector<int> replication_num_vector);
    double FastEvaluationForBatch(std::vector<int> individual, std::vector<int> replication_num_vector);
    double FastEvaluationForElement(std::vector<int> individual, std::vector<int> replication_num_vector);
    double FastEvaluationInstructionGroupNum(std::vector<int> replication_num_vector);
    void FastSingleInstructionGroupEvaluation(int instruction_group_index, int core_index, int index_in_core);
    int CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle);
    int CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle);
    // Mutation
    void Mutation();
    bool CheckLegality(std::vector<int> & individual, std::vector<int> & replication_num_vector, std::vector<int> & AG_type_num_vector, std::vector<int> & crossbar_num_vector, std::string Stage);
    // Post Process
    void PostProcess();
    void SaveIntermediateInfo();
};


#endif //PIMCOMP_OPEN_GENETICALGORITHM_H

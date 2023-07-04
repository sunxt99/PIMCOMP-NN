//
// Created by SXT on 2022/8/18.
//

#include "common.h"
#include "configure.h"
#include <stdlib.h>
#include "backend/Preparation.h"
#include "backend/WeightReplication.h"
#include "backend/CrossbarPartition.h"
#include "backend/HierarchyMapping.h"
#include "backend/GeneticAlgorithm.h"
#include "backend/ElementPipelineSchedule.h"
#include "backend/BatchPipelineSchedule.h"
#include "backend/MemoryAllocation.h"
#include "backend/DataReload.h"
#include "backend/InstructionOptimization.h"
#include "backend/ElementPlacement.h"
#include "evaluation/ModelEvaluation.h"

void PIMCOMP(std::string model_name, std::string replicating_method, std::string pipeline_granularity, std::string output_need, std::string verification_need, std::string intermediate_need, std::string simulation_need)
{
    clock_t start_time = clock();

    std::cout << "=========================== LOADING ===========================" << std::endl;
    Json::Reader jsonReader;
    Json::Value DNNInfo;
    std::ifstream jsonFile("../models/JSON/" + model_name + ".json");
    if (!jsonReader.parse(jsonFile, DNNInfo, true))
    {
        std::cout << "Open Json File Error" << std::endl;
        return;
    }

    std::cout << "=========================== PreProcessing ===========================" << std::endl;
    LoadHardwareConfiguration();
    GetStructNodeListFromJson(DNNInfo);
    GetConvPoolInputOutputInfo();
    GetConvPoolInputOutputInfoForInputPreparation();
    GetTopologyInformation();
    CopyFromOriginNodeList();
    GetPriorInfoForElementPipeline();

    std::cout << "=========================== MODEL INFO ===========================" << std::endl;
    ShowModelInfo();

    if (replicating_method == "GA")
    {
        std::cout << "====================== DESIGN SPACE EXPLORATION ======================" << std::endl;
        GeneticAlgorithm GA(model_name, pipeline_granularity);
        GA.GeneticAlgorithmOptimizing();
    }

    std::cout << "=========================== REPLICATING ===========================" << std::endl;
    WeightReplication replication;
    replication.ReplicateWeight(replicating_method);

    std::cout << "=========================== PARTITIONING ===========================" << std::endl;
    CrossbarPartition partition;
    partition.PartitionCrossbar();

    std::cout << "============================= MAPPING =============================" << std::endl;
    HierarchyMapping mapping;
    mapping.MapHierarchy(replicating_method);
    mapping.SaveMappingResult();

    if (pipeline_granularity == "element")
    {
        std::cout << "========================= ELEMENT PIPELINE SCHEDULING =========================" << std::endl;
        ElementPipelineSchedule schedule(model_name);
        schedule.ScheduleExecution();
    }
    else
    {
        std::cout << "========================= BATCH PIPELINE SCHEDULING =========================" << std::endl;
        std::cout << "------------- INSTRUCTION GENERATION -------------" << std::endl;
        BatchPipelineSchedule schedule(model_name);
        schedule.ScheduleExecution();

        std::cout << "------------- MEMORY ALLOCATION -------------" << std::endl;
        MemoryAllocation allocation;
        allocation.AllocateMemory();

        std::cout << "------------- DATA RELOAD -------------" << std::endl;
        DataReload reload;
        reload.ReloadData();
//        reload.BatchInstruction();
    }

    std::cout << "=========================== OPTIMIZATION ===========================" << std::endl;
    InstructionOptimization optimization;
    optimization.OptimizeInstruction();

    std::cout << "============================= PLACING =============================" << std::endl;
    ElementPlacement placement(0);
    placement.PlaceElement();

    if (verification_need == "YES")
    {
        std::cout << "========================= SAVING VERIFICATION =========================" << std::endl;
        placement.SaveVerificationInfo(DNNInfo);
    }
    if (simulation_need == "YES")
    {
        std::cout << "========================= SAVING SIMULATION =========================" << std::endl;
        placement.SaveSimulationInfo();
    }


    std::cout << "============================= EVALUATING =============================" << std::endl;
    ModelEvaluation evaluation(0, MVMUL_start_interval);
    evaluation.EvaluateCompute();

    if (output_need == "YES")
    {
        std::cout << "========================= SAVING EVALUATION =========================" << std::endl;
        evaluation.SaveEvaluation();
    }

    if (intermediate_need == "YES")
    {
        std::cout << "========================= SAVING INTERMEDIATE =========================" << std::endl;
        SaveIntermediateInfo(DNNInfo);
    }

    std::cout << "============================= OVERALL TIME =============================" << std::endl;
    clock_t end_time = clock();
    std::cout << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
}

int main(int argc, char* argv[])
{
    std::string model_name = "resnet18";
    std::string replicating_method = "balance";
    std::string pipeline_granularity = "batch";
    std::string output_need = "NO";
    std::string verification_need = "NO";
    std::string intermediate_need = "NO";
    std::string simulation_need = "YES";

    int nOptionIndex = 1;
    while (nOptionIndex < argc)
    {
        if (strncmp(argv[nOptionIndex], "-m=", 3) == 0)
        {
            model_name = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-r=", 3) == 0)
        {
            replicating_method = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-p=", 3) == 0)
        {
            pipeline_granularity = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-o=", 3) == 0)
        {
            output_need = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-v=", 3) == 0)
        {
            verification_need = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-i=", 3) == 0)
        {
            intermediate_need = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-s=", 3) == 0)
        {
            simulation_need = &argv[nOptionIndex][3];
        }
        else if (strncmp(argv[nOptionIndex], "-h", 2) == 0)
        {
            std::cout << "Options:" << std::endl;
            std::cout << "  -m=model_name              This parameter MUST be given." << std::endl;
            std::cout << "  -r=replicating_method      [options]:balance/W0H0/uniform/GA   [default]:balance" << std::endl;
            std::cout << "  -p=pipeline_granularity    [options]:batch/element             [default]:batch" << std::endl;
            std::cout << "  -o=output_need             [options]:Yes/NO                    [default]:NO" << std::endl;
            std::cout << "  -v=verification_need       [options]:Yes/NO                    [default]:NO" << std::endl;
            std::cout << "  -i=intermediate_need       [options]:Yes/NO                    [default]:NO" << std::endl;
            std::cout << "  -s=simulation_need         [options]:Yes/NO                    [default]:NO" << std::endl;
            std::cout << "  -h=help" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Options '" << argv[nOptionIndex] << "' not valid. Run '" << argv[0] << "' for details." << std::endl;
            return -1;
        }
        nOptionIndex++;
    }
    if (model_name == "")
    {
        std::cout << "Model Name Must Be Given." << std::endl;
        return -1;
    }

    std::set<std::string> replication_method_set = {"balance","W0H0","uniform","GA"};
    std::set<std::string> pipeline_granularity_set = {"element","batch"};
    std::set<std::string> yes_no_set = {"YES","NO"};
    if (replication_method_set.count(replicating_method) == 0 ||
        pipeline_granularity_set.count(pipeline_granularity) == 0 ||
        yes_no_set.count(output_need) == 0 ||
        yes_no_set.count(verification_need) == 0 ||
        yes_no_set.count(intermediate_need) == 0 ||
        yes_no_set.count(simulation_need) == 0
        )
    {
        fprintf(stderr, "Please Make Sure The Parameters Is Legal.\n");
        abort();
    }

    std::cout << "===================== Compiling Parameters =====================" << std::endl;
    std::cout << "model name:" << model_name << std::endl;
    std::cout << "replicating method:" << replicating_method << std::endl;
    std::cout << "pipeline granularity:" << pipeline_granularity << std::endl;
    std::cout << "output need:" << output_need << std::endl;
    std::cout << "verification need:" << verification_need << std::endl;
    std::cout << "intermediate need:" << intermediate_need << std::endl;
    std::cout << "simulation need:" << simulation_need << std::endl;

    PIMCOMP(model_name, replicating_method, pipeline_granularity, output_need, verification_need, intermediate_need, simulation_need);
}
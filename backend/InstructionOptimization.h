//
// Created by SXT on 2023/6/28.
//

#ifndef PIMCOMP_OPEN_INSTRUCTIONOPTIMIZATION_H
#define PIMCOMP_OPEN_INSTRUCTIONOPTIMIZATION_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class InstructionOptimization
{
public:
    InstructionOptimization();
    void OptimizeInstruction();
private:
    std::vector<struct PIMCOMP_4_instruction_ir> intermediate_instruction_ir;
    void MergePreparation();
    void MergeLoadOperation();

};


#endif //PIMCOMP_OPEN_INSTRUCTIONOPTIMIZATION_H

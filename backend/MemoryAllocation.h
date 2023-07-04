//
// Created by SXT on 2023/2/23.
//

#ifndef PIMCOMP_TC_MEMORYALLOCATION_H
#define PIMCOMP_TC_MEMORYALLOCATION_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class MemoryAllocation
{
public:
    void AllocateMemory();
private:
    void GetAGBaseInfo();
    void GetAGReuseInfo();
    void BaseAllocate();
    void GenerateAddress();
    void Clear();

};


#endif //PIMCOMP_TC_MEMORYALLOCATION_H

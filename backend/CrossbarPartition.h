//
// Created by SXT on 2022/8/19.
//

#ifndef PIMCOMP_CROSSBARPARTITION_H
#define PIMCOMP_CROSSBARPARTITION_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class CrossbarPartition {
public:
    void PartitionCrossbar();
private:
    void Clear();
    void Check();
    void Partition();
};


#endif //PIMCOMP_CROSSBARPARTITION_H

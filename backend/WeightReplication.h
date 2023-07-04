//
// Created by SXT on 2022/8/19.
//

#ifndef PIMCOMP_WEIGHTREPLICATION_H
#define PIMCOMP_WEIGHTREPLICATION_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class WeightReplication
{
public:
    void ReplicateWeight(std::string replicating_method);
private:
    int node_num;
    void ReplicateUniformly();
    void ReplicationByW0H0();
    void ReplicationByBalance();
    void LoadGAReplicationResult(int candidate_index);
};


#endif //PIMCOMP_WEIGHTREPLICATION_H

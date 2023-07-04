//
// Created by SXT on 2023/3/20.
//

#ifndef PIMCOMP_TC_MEMORYMANAGER_H
#define PIMCOMP_TC_MEMORYMANAGER_H

#include <iostream>
#include <list>
#include <vector>

class MemoryManager
{
private:
    int core_num;
    int max_element_num;
public:
    MemoryManager(){};
    void SetParameter(int core_num, int max_size);
    long long MemoryAllocate(int core_index, int require_memory_element_num);
    bool MemoryFree(int core_index, long long free_start_element_address, int free_element_num);
    bool MemoryReallocate();
    void PrintMemoryInfo(int core_index);
    int GetCoreElementNum(int core_index);
    int GetMaxElementNum();
    int FindElementLength(int core_index, int start_address);
};


#endif //PIMCOMP_TC_MEMORYMANAGER_H

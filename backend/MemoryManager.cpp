//
// Created by SXT on 2023/3/20.
//

#include "MemoryManager.h"

struct MemoryBlock
{
    long long StartElementAddress;
    long long EndElementAddress;
    int ElementNum;
    MemoryBlock(long long Start, long long End)
            : StartElementAddress(Start)
            , EndElementAddress(End)
            , ElementNum(End - Start + 1) {}
};

std::vector<std::list<struct MemoryBlock>> CoreMemoryRecord;
std::vector<int> CoreElementNum;
std::vector<std::list<struct MemoryBlock>::iterator> MemoryAccessAddress;

void MemoryManager::SetParameter(int core_num_, int max_elt_num_)
{
    core_num = core_num_;
    max_element_num = max_elt_num_;
    CoreMemoryRecord.resize(core_num);
    CoreElementNum.resize(core_num);
    MemoryAccessAddress.resize(core_num);
    for (int i = 0; i < core_num; ++i)
    {
        CoreMemoryRecord[i].push_back(MemoryBlock(-1,-1));
        CoreMemoryRecord[i].push_back(MemoryBlock(max_element_num,max_element_num));
        MemoryAccessAddress[i] = CoreMemoryRecord[i].begin();
    }
}


void MemoryManager::PrintMemoryInfo(int core_index)
{
    std::cout << "core:" << core_index;
    for (auto iter = CoreMemoryRecord[core_index].begin(); iter != CoreMemoryRecord[core_index].end(); iter++)
    {
         std::cout << "    " << iter->StartElementAddress  << "-" << iter->EndElementAddress;
    }
    std::cout << std::endl;
}

long long MemoryManager::MemoryAllocate(int core_index, int require_memory_element_num)
{
    long long last_end = 0;
    long long this_start = 0;
    for (auto iter = CoreMemoryRecord[core_index].begin(); iter != CoreMemoryRecord[core_index].end(); iter++)
    {
        if (iter == CoreMemoryRecord[core_index].begin())
        {
            last_end = iter->EndElementAddress;
            continue;
        }
        else
        {
            this_start = iter->StartElementAddress;
            if (this_start - last_end -1 >= require_memory_element_num)
            {
                CoreMemoryRecord[core_index].insert(iter, MemoryBlock(last_end+1,last_end + require_memory_element_num));
                CoreElementNum[core_index] += require_memory_element_num;
                return last_end+1;
            }
            else
            {
                last_end = iter->EndElementAddress;
            }
        }
    }
    return -1;
}

int MemoryManager::GetCoreElementNum(int core_index)
{
    return CoreElementNum[core_index];
}

int MemoryManager::GetMaxElementNum()
{
    int max_core_element_num = -1;
    int max_core_index = -1;
    for (int i = 0; i < core_num; ++i)
        if (CoreElementNum[i] > max_core_element_num)
        {
            max_core_element_num = CoreElementNum[i];
            max_core_index = i;
        }
    std::cout << "max_memory_core_index:" << max_core_index << "   " ;
    return max_core_element_num;
}

bool MemoryManager::MemoryFree(int core_index, long long free_start_element_address, int free_element_num)
{
    for (auto iter = CoreMemoryRecord[core_index].begin(); iter != CoreMemoryRecord[core_index].end(); iter++)
    {
        long long block_start_element_address = iter->StartElementAddress;
        int block_element_num = iter->ElementNum;
        if (free_start_element_address == block_start_element_address && free_element_num == block_element_num)
        {
            CoreMemoryRecord[core_index].erase(iter);
            CoreElementNum[core_index] -= free_element_num;
            return true;
        }
    }
    return false;
}

int MemoryManager::FindElementLength(int core_index, int start_address)
{
    for (auto iter = CoreMemoryRecord[core_index].begin(); iter != CoreMemoryRecord[core_index].end(); iter++)
    {
        long long block_start_element_address = iter->StartElementAddress;
        int block_element_num = iter->ElementNum;
        if (start_address == block_start_element_address)
            return block_element_num;
    }
    return -1;
}


bool MemoryManager::MemoryReallocate()
{
    return false;
}




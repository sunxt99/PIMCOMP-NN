//
// Created by SXT on 2023/2/22.
//

#ifndef PIMCOMP_TC_DATARELOAD_H
#define PIMCOMP_TC_DATARELOAD_H

#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"

class DataReload
{
public:
    void ReloadData();
    void BatchInstruction();
    void SaveInstruction();
private:
    void GetAGInputInfo(); // For Reload
    void ReloadInput();
    void StoreOutput();
    void Clear();
};


#endif //PIMCOMP_TC_DATARELOAD_H

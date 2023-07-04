//
// Created by SXT on 2022/8/19.
//

#ifndef PIMCOMP_CONFIGURE_H
#define PIMCOMP_CONFIGURE_H
#include "./backend/json/json.h"

//const int ArithmeticPrecision = 16;
//const int CellPrecision = 2;
//const int CrossbarW = 128;
//const int CrossbarH = 128;
//const int CoreW = 8;  // #Crossbars every row in Core (Logical)
//const int CoreH = 8;  // #Crossbars every column in Core (Logical)
//const int ChipW = 12;  // #Cores every row in Chip
//const int ChipH = 12;  // #Cores every column in Chip

const int OneChipWidth = 12;
const int OneChipHeight = 12;

const int MAX_CHIP = 16;
const int MAX_AG = 20000;
const int MAX_CORE = 1000;
const int MAX_NODE = 1000;

const int batch_start = 100;
const int batch_end = 100;

const int appointed_candidate_num = 1; // GA candidate num (now only 1 is supported)
const int user_given_instruction_group_num = 50000;
const int operation_cycle_before_comm = 1;

#endif //PIMCOMP_CONFIGURE_H
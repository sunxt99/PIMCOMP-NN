//
// Created by SXT on 2022/10/21.
//

#ifndef PIMCOMP_EVALUATIONCONFIG_H
#define PIMCOMP_EVALUATIONCONFIG_H

//////////////////////////////////////////////// Compute ////////////////////////////////////////////////
const double Frequency = 1000000000.0; // 1GHz

const double MVMUL_read_latency = 200.0;
const double MVMUL_process_latency_per_bit = 200.0;
const double MVMUL_ADC_latency = 200.0;
const double MVMUL_sa_latency = 200.0;
const double MVMUL_write_latency = 200.0;
const double MVMUL_latency = MVMUL_read_latency + ArithmeticPrecision * MVMUL_process_latency_per_bit + MVMUL_ADC_latency + MVMUL_sa_latency + MVMUL_write_latency;
const double MVMUL_start_interval = 200.0;

const double VECTOR_process_bytes = 64; // 64B
const double VECTOR_unit_latency = 10.0; // (ns); read(10ns，64B) + write(10ns, 64B)

const double LOCAL_MEMORY_bandwidth = 25600000000.0; // 25.6 GB/s
const double LINK_bandwidth = 8000000000; // 8.0 GB/s

const double GLOBAL_MEMORY_bandwidth = 64000000000.0;  // 64.0 GB/s, shared by all cores
const double Hyper_Transport_bandwidth = 6400000000.0; // 6.4 GB/s

const double BUS_bandwidth = 25600000000; // 25.6 GB/s

#endif //PIMCOMP_EVALUATIONCONFIG_H

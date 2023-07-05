# PIMCOMP-NN

PIMCOMP-NN is a compilation optimization framework designed for crossbar-based PIM DNN accelerators. The project comprises three submodules: frontend, backend, and verification program. With PIMCOMP-NN, we can conveniently compile DNN models to get the instruction flow that can be executed on PIM accelerators.

We design a frontend to load and parse DNN models in [ONNX](https://github.com/onnx/onnx) format and generate the input file for the backend.

The backend is the core part of PIMCOMP-NN. The overview of backend is shown in the following figure. The entire backend includes 4 key stages. Node Partition describes the rules for partitioning weight data to fit for the PIM crossbars. Weight Replicating determines the replication numbers of different nodes. Core Mapping decides the mapping relationship between crossbars and cores and Dataflow Scheduling performs scheduling and optimization according to user’s requirement to generate control flow or instruction flow. In addition to these four stages, PIMCOMP-NN also has some pre-processing and post-optimization. In order to expand the scope of application of the PIM accelerators, we provide two compilation modes for users to choose from: High Throughput (batch pipeline) and Low Latency (element pipeline), which are suitable for scenarios with continuous input data of large batches and intermittent input of a small amount, respectively. 

![overview](https://typora-ict.oss-cn-beijing.aliyuncs.com/img/202307022208317.png)

We also design a verification program to verify the correctness of the instruction stream. We simulate the calculation results instruction by instruction and compare them with results obtained by ONNX runtime.

PIMCOMP-NN has an associated simulator, [pimsim-nn](https://github.com/wangxy-2000/pimsim-nn). The produced instruction stream can be simulated by the simulator to estimate the performance (inference latency and/or throughput), power dissipation, and energy consumption, using the same architecture configuration.

# Usability

To describe the operation of DNN network in detail and retain generality, we propose a general and representative NVM crossbar based accelerator architecture as a hardware abstraction. At a high level, the accelerator consists of multiple cores interconnected through NoC or busses. The weights of the neural network are stored in the cores, while the inputs, outputs and intermediate results are stored in the global memory or local memory. Each core has **PIM matrix unit** to compute matrix-vector-multiplication, and **vector functional unit** to perform vector operations.

![abstraction](https://typora-ict.oss-cn-beijing.aliyuncs.com/img/202307022258469.png)

Our proposed abstract architecture is compatible with the Crossbar/IMA/Tile/Chip structure adopted in previous work.

| Paper            | #Core | #Crossbar in one core | Crossbar Size |
| ---------------- | ----- | --------------------- | ------------- |
| ISAAC (ISCA'16)  | 168   | 12*8                  | 128*128       |
| PUMA (ASPLOS'19) | 138   | 8*16                  | 128*128       |
| SRE (ISCA'19)    | 168   | 12*8                  | 128*128       |
| 15.4 (ISSCC'20)  | 1     | 8                     | 512*512       |
| 16.1 (ISSCC'21)  | 1     | 8                     | 1024*512      |
| FORMS (ISCA'21)  | 168   | 12*8                  | 128*128       |
| 11.4 (ISSCC'22)  | 1     | 32                    | 1024*256      |
| 16.6 (ISSCC'23)  | 4     | 8                     | 512*1024      |

# Usage

## Requirements

In order to run the frontend and verification programs, you need to install the following python package.

- onnx
- onnxruntime
- numpy
- cv2

To compile the backend using cmake, you need to meet the following requirements.

- cmake >= 3.6
- g++ >= 4.8.5

## Frontend

We can convert ONNX model to the input file of backend in JSON format containing node parameters and topological connections of the DNN network. The backend reads this file for subsequent processes.

Users can download ONNX model from [ONNX Model Zoo](https://github.com/onnx/models) and save the model to `PIMCOMP-NN/models/ONNX/`. The output file will be saved in `PIMCOMP-NN/models/JSON/`. In order to facilitate users to use PIMCOMP-NN, we have converted some commonly used network models into json files in advance and users can directly use the backend to generate instructions.

```shell
cd PIMCOMP-NN/frontend/
python frontend.py --model_path ../models/ONNX/resnet18.onnx --save_path ../models/JSON/resnet18.json
```

## Backend

### Hardware configuration

The hardware information is defined in `Preparation.cpp` and the value of these parameters are configured in `config.json`. Other unused parameters in `config.json` are used for pimsim-nn for simulation.

| Parameters          | Description                                     | Definition in config.json |
| ------------------- | ----------------------------------------------- | ------------------------- |
| ArithmeticPrecision | precision of input, weight and output           | adc_count                 |
| CellPrecision       | precision of crossbar cell                      | cell_precision            |
| CrossbarW           | width of an NVM crossbar                        | xbar_size[1]              |
| CrossbarH           | height of an NVM crossbar                       | xbar_size[0]              |
| CoreW               | \#logical crossbars in every row of one core    | 1                         |
| CoreH               | \#logical crossbars in every column of one core | xbar_array_count          |
| ChipW               | \#cores in every row of one chip                | layout[1]                 |
| ChipH               | \#cores in every column of one chip             | layout[0]                 |

### Building

```shell
cd PIMCOMP-NN
mkdir build
cd build
cmake ..
make
```

### Parameter

| Parameters | Description                              | Options                 | Default       |
| ---------- | ---------------------------------------- | ----------------------- | ------------- |
| -m         | model name                               |                         | MUST be given |
| -r         | replicating method                       | balance/W0H0/uniform/GA | balance       |
| -p         | pipeline granularity                     | element/batch           | batch         |
| -o         | save evaluation output or not            | YES/NO                  | NO            |
| -v         | save instruction for verification or not | YES/NO                  | NO            |
| -i         | save intermediate information or not     | YES/NO                  | NO            |
| -s         | save instruction for simulation or not   | YES/NO                  | NO            |

For example, if you have `resnet18.json` in `PIMCOMP-NN/models/JSON/` and want to compile it using balance replication method and element pipeline (low latency mode). 

```shell
./PIMCOMP-NN –m=resnet18 –r=balance –p=element –o=NO –v=NO –i=NO -s=NO
```

The model name is a required parameter. If you set `-m=MODEL_NAME`, then the backend will search and load  `MODEL_NAME.json` in `PIMCOMP-NN/models/JSON/`.

Balance, W0H0, uniform and GA are different replicating strategies from the perspectives of calculation balance, layer size balance, unified replication and performance, respectively.

If you want to save the evaluation result of the whole instruction stream, please set `-o=YES` and the result will be saved in `PIMCOMP-NN/output/EvaluationResult.txt`.

If you want to save the instruction and other information for verification, please set `-v=YES` and the result will be saved in `PIMCOMP-NN/output/VerificationInfo.json`.

If you want to save the intermediate information during compiling, please set `-i=YES` and the result will be saved in `PIMCOMP-NN/output/IntermediateInfo.json`.

If you want to save the instruction for more detailed simulation using pimsim-nn, please set `-s=YES` and the result will be saved in `PIMCOMP-NN/output/SimulationInfo.gz`.

## Verification

The verification program is used to verify the generated instruction stream. Before verifying, please set `-v=YES` when calling the PIMCOMP-NN backend and the instruction for verification will be saved in `PIMCOMP-NN/output/VerificationInfo.json`.

```shell
cd PIMCOMP-NN/verification/
python verification.py --model_path ../models/ONNX/resnet18.onnx --pipeline_type element
```

We have compiled and validated these models below.

- AlexNet
- VGG16
- ResNet18/34/50
- GoogleNet
- SqueezeNet
- CaffeNet
- ShuffleNet-v1
- MobileNetv2

# Development

## To compile new model

Please note the following information if you want to compile a new model using PIMCOMP-NN.

1. Currently operators such as CONV/Group CONV/FC/POOL/ADD/CONCAT/ReLU are supported.
2. Specify last_node_index and last_node_output_channel_num if you choose element pipeline in ElementPipelineSchedule.cpp
3. Make sure the data_type of your ONNX model is 1, i.e. float data.
4. The node immediately adjacent to the input needs to be a convolution node.

# Citing PIMCOMP-NN

**Citation Information**: Xiaotian Sun, Xinyu Wang, Wanqian Li, Lei Wang, Yinhe Han, Xiaoming Chen, ["PIMCOMP: A Universal Compilation Framework for Crossbar-based PIM DNN Accelerators"](https://typora-ict.oss-cn-beijing.aliyuncs.com/paper/PIMCOMP.pdf), in Design Automation Conference (DAC'23), 2023.

# Code Author

[Xiaotian Sun](sunxiaotian21s@ict.ac.cn) (Institute of Computing Technology, Chinese Academy of Sciences)

## Project PI

[Xiaoming Chen](https://people.ucas.edu.cn/~chenxm)

# License

[Apache License v2.0](https://github.com/sunxt99/PIMCOMP-NN/blob/master/LICENSE)

# Acknowledgement

[onnx](https://github.com/onnx/onnx)

[jsoncpp](https://github.com/open-source-parsers/jsoncpp)

[zlib](https://github.com/madler/zlib)

[zstr](https://github.com/mateidavid/zstr)

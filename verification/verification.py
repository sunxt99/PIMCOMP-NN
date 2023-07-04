import onnx
from onnx import numpy_helper
import onnxruntime
import numpy as np
import json
import cv2
import argparse
import sys
sys.setrecursionlimit(500000)


class ModelInfo:
    def __init__(self, model_path, pipeline_type):
        self.onnx_model = onnx.load(model_path)
        self.pipeline_type = pipeline_type

    def load_weight(self):
        print("==================== Load Weight ====================")
        # load weight to OriginalWeightDict
        weights, names = [], []
        for t in self.onnx_model.graph.initializer:
            weights.append(numpy_helper.to_array(t))
            names.append(t.name)
        self.OriginalWeightDict = dict(zip(names, weights))
        for k, v in self.OriginalWeightDict.items():
            print(k, "  ", v.shape)

        # OutputToWeightDict : node_name → weight_name
        # OutputToBiasDict: node_name → bias_name
        # So node_index → node_name → weight_name → weight_data
        self.OutputToWeightDict = {}
        self.OutputToBiasDict = {}
        for node in self.onnx_model.graph.node:
            if node.op_type == "Conv" or node.op_type == "Gemm":
                if len(node.input) == 2:
                    self.OutputToWeightDict[node.output[0]] = node.input[1]
                elif len(node.input) == 3:
                    self.OutputToWeightDict[node.output[0]] = node.input[1]
                    self.OutputToBiasDict[node.output[0]] = node.input[2]

        # for group_conv:
        self.group_weight_2_conv_name = {}
        for i, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "Conv":
                for attribute in node.attribute:
                    if attribute.name == "group" and attribute.i != 1:
                        self.group_weight_2_conv_name[node.input[1]] = node.output[0]

        # convert original weight (4-d) to weight matrix (2-d, height=K*K*Cin, Width=Cout)
        self.GEMMWeightDict = {}
        for k, v in self.OriginalWeightDict.items():
            if k in self.group_weight_2_conv_name: # Group CONV Weight
                self.GEMMWeightDict[k] = v.transpose((2,3,1,0))
            elif len(v.shape) == 4:  # CONV Weight
                self.GEMMWeightDict[k] = v.transpose( (0, 2, 3, 1)).reshape((v.shape[0], v[0].size)).transpose()
            else:  # FC Weight and Bias
                self.GEMMWeightDict[k] = v.transpose()

        # if self.pipeline_type == "element":
        #     # node_name = "vgg0_dense0_fwd"
        #     # if node_name in self.OutputToWeightDict:
        #     #     weight_name = self.OutputToWeightDict[node_name]
        #     #     weight_matrix = self.GEMMWeightDict[weight_name]
        #     #     weight_matrix = weight_matrix.reshape((512, 7, 7, 4096))
        #     #     weight_matrix = weight_matrix.transpose(1, 2, 0, 3)
        #     #     weight_shape = weight_matrix.shape
        #     #     self.GEMMWeightDict[weight_name] = weight_matrix.reshape(weight_matrix[:, :, :, 0].size, weight_shape[3])

    def eliminate_no_consider_OP(self):
        # eliminate LRN OP
        for i, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "LRN":
                self.onnx_model.graph.node[i].attribute[1].f = 1e-8  # "alpha"
                self.onnx_model.graph.node[i].attribute[2].f = 1e-8  # "beta"
                self.onnx_model.graph.node[i].attribute[3].f = 1e-8  # "bias"
            if node.op_type == "Clip":
                self.onnx_model.graph.node[i].attribute[1].f = -1e8
                self.onnx_model.graph.node[i].attribute[0].f = 1e8

        # for idx,tensor in enumerate(self.onnx_model.graph.initializer):
        #     if len(tensor.dims) == 1:
        #         # for float_data:
        #         if len(tensor.float_data) != 0:
        #             if not "batchnorm" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 0.0 #element by element
        #             elif "beta" in tensor.name or "mean" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 0.0
        #             elif "gamma" in tensor.name or "var" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 1.0
        #         # for raw_data:
        #         else:
        #             raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
        #             raw_data_float32 = np.zeros(raw_data_float32.shape, dtype=np.float32)
        #             self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()

        BN_zero_params = []
        BN_one_params = []
        for node in self.onnx_model.graph.node:
            if node.op_type == "BatchNormalization":
                BN_zero_params.append(node.input[2])
                BN_zero_params.append(node.input[3])
                BN_one_params.append(node.input[1])
                BN_one_params.append(node.input[4])
        for idx,tensor in enumerate(self.onnx_model.graph.initializer):
            if tensor.name in BN_zero_params:
                if len(tensor.dims) == 1 and len(tensor.float_data) != 0:
                    for l in range(len(tensor.float_data)):
                        self.onnx_model.graph.initializer[idx].float_data[l] = 0.0
                else:
                    raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
                    raw_data_float32 = np.zeros(raw_data_float32.shape, dtype=np.float32)
                    self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()
            if tensor.name in BN_one_params:
                if len(tensor.dims) == 1 and len(tensor.float_data) != 0:
                    for l in range(len(tensor.float_data)):
                        self.onnx_model.graph.initializer[idx].float_data[l] = 1.0
                else:
                    raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
                    raw_data_float32 = np.ones(raw_data_float32.shape, dtype=np.float32)
                    self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()


    def load_input(self):
        print("==================== Load Input ====================")
        img_path = "./input/cat.png"
        self.img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        self.img = cv2.resize(self.img, (224, 224))  # (224, 224, 3)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # from BGR to RGB
        self.img = np.transpose(self.img, (2, 0, 1))  # (1, 3, 224, 224)
        self.img = np.expand_dims(self.img, 0)
        self.img = self.img.astype(np.float32)
        self.img /= 255
        self.nn_input = self.img.transpose((0, 2, 3, 1)).flatten()

    def get_ground_truth(self):
        print("==================== Get GroundTruth ====================")
        # make every node be output node
        for node in self.onnx_model.graph.node:
            for output in node.output:
                self.onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        ort_session = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: self.img}
        ort_outputs = ort_session.run(None, ort_inputs)
        # get all output node name
        node_name = [x.name for x in ort_session.get_outputs()]
        self.onnx_runtime_outs = dict(zip(node_name, ort_outputs))

        self.intermediate_result = {}
        for k,v in self.onnx_runtime_outs.items():
            if len(v.shape) == 4:
                self.intermediate_result[k] = v.transpose(0,2,3,1).flatten()
            elif len(v.shape) == 2:
                self.intermediate_result[k] = v.transpose().flatten()

class Memory:
    def __init__(self, core_num):
        self.core_num = core_num
        self.local_memory_max_size = 262144*4 # this size is element_num. 512kB * 4 / 16bit = 262144 * 4
        self.local_memory = np.zeros((self.core_num, self.local_memory_max_size))
        self.global_memory_max_size = 536870912*4 # this size is element_num. 1GB * 4/ 16bit = 536870912 * 4
        self.global_memory = np.zeros(self.global_memory_max_size)


class Verification(ModelInfo):
    def __init__(self, model_path, pipeline_type):
        ModelInfo.__init__(self, model_path, pipeline_type)
        self.pipeline_type = pipeline_type


    def load_compilation(self):
        print("==================== Load Compilation ====================")
        with open("../output/VerificationInfo.json", "r", encoding="utf-8") as f:
            self.FinalInfo = json.load(f)

        self.comm_pair_total_num = self.FinalInfo["comm_pair_total_num"]
        self.AG_height_start = []
        self.AG_height_end = []
        self.AG_num = len(self.FinalInfo["AG_info"])
        for AG in self.FinalInfo["AG_info"]:
            node_name = AG["node_name"]
            weight_name = self.OutputToWeightDict[node_name]
            height_start = AG["height_start"]
            height_end = AG["height_end"]
            self.AG_height_start.append(height_start)
            self.AG_height_end.append(height_end)

        self.max_output_element_num = 0
        # It should be noted that the node index in FinalInfo is not exactly the same as the serial number of the ONNX model
        # This is due to the addition of input nodes and preprocessing during compilation.
        # But the node_name of both is the same.
        # Actually the node_index here is not that important as long as it is self-consistent.
        # When it comes to getting data from weight or bias, you need to pass node_name, which is unique.
        # 这里需要注意的是FinalInfo中的节点序号和ONNX模型的序号不完全一致
        # 这是由于编译时会添加输入节点以及进行预处理。但是两者的node_name是一致的。
        # 其实这里的node_index不是那么重要。只要前后自洽即可。涉及到从weight或bias拿数据时需要通过node_name，这是唯一确定的。
        self.node_name_2_index = {}
        self.node_name_2_output_element_num = {}
        for node in self.FinalInfo["node_list"]:
            output_element_num = 1
            output_dim_num = node["output_dim_num"]
            for i in range(output_dim_num):
                output_element_num = output_element_num * node["output_dim"][i]
            if self.max_output_element_num < output_element_num:
                self.max_output_element_num = output_element_num
            self.node_name_2_output_element_num[node["name"]] = output_element_num
            self.node_name_2_index[node["name"]] = node["new_node_index"]

        # For Group Conv
        for k,v in self.GEMMWeightDict.items():
            if k in self.group_weight_2_conv_name:
                OriWeight = self.GEMMWeightDict[k]
                conv_node_name = self.group_weight_2_conv_name[k]
                conv_node_index = self.node_name_2_index[conv_node_name]
                group = self.FinalInfo["node_list"][conv_node_index]["param"]["group"]
                kernel_h = self.FinalInfo["node_list"][conv_node_index]["param"]["kernel_h"]
                kernel_w = self.FinalInfo["node_list"][conv_node_index]["param"]["kernel_w"]
                full_input_channel = self.FinalInfo["node_list"][conv_node_index]["param"]["input_channel"]
                output_channel = self.FinalInfo["node_list"][conv_node_index]["param"]["output_channel"]
                group_input_channel = full_input_channel // group
                
                New_GEMM_Weight = np.zeros((kernel_h*kernel_w*full_input_channel, output_channel))
                for i in range(output_channel):
                    same_group_kernel_num = output_channel//group
                    Zero_Weight = np.zeros((kernel_h,kernel_w,full_input_channel))
                    Zero_Weight[:,:,(i//same_group_kernel_num)*group_input_channel:(i//same_group_kernel_num+1)*group_input_channel] = OriWeight[:,:,:,i]
                    Zero_Weight = Zero_Weight.flatten()
                    New_GEMM_Weight[:,i] = Zero_Weight
                self.GEMMWeightDict[k] = New_GEMM_Weight

        # Reshape Weight for FC node
        if self.pipeline_type == "element":
            if self.FinalInfo["reshape_info"] != None and "name" in self.FinalInfo["reshape_info"].keys():
                reshape_node_name = self.FinalInfo["reshape_info"]["name"]
                input_dim = self.FinalInfo["reshape_info"]["input_dim"]
                if reshape_node_name in self.OutputToWeightDict:
                    weight_name = self.OutputToWeightDict[reshape_node_name]
                    weight_matrix = self.GEMMWeightDict[weight_name]
                    new_shape = (input_dim[1], input_dim[2], input_dim[3], weight_matrix.shape[1])
                    # weight_matrix = weight_matrix.reshape((256, 6, 6, 4096))
                    weight_matrix = weight_matrix.reshape(new_shape)
                    weight_matrix = weight_matrix.transpose(1, 2, 0, 3)
                    weight_shape = weight_matrix.shape
                    self.GEMMWeightDict[weight_name] = weight_matrix.reshape(weight_matrix[:, :, :, 0].size, weight_shape[3])

        self.core_num = len(self.FinalInfo["instruction"]["core_list"])
        self.visited_single = np.zeros((1000000), dtype=np.int16)
        self.comm_index_2_index_in_core = {}
        self.comm_index_2_core_index = {}
        self.inst_num_traversal = np.zeros((self.core_num), dtype=np.int16)
        self.CoreMemory = Memory(self.core_num)
        for core_idx in range(self.core_num):
            if self.FinalInfo["instruction"]["core_list"][core_idx] != None:
                for inst_idx, instruction in enumerate(self.FinalInfo["instruction"]["core_list"][core_idx]):
                    if (instruction["operation"] == "SEND" or instruction["operation"] == "RECV"):
                        self.FinalInfo["instruction"]["core_list"][core_idx][inst_idx]["instruction_index_in_core"] = inst_idx



    def start_simulation(self,core_index, index_in_core):
        if core_index >= self.core_num:
            return
        if self.FinalInfo["instruction"]["core_list"][core_index] == None:
            next_core_index = core_index + 1
            while (not self.visited_single[next_core_index] == 0):
                next_core_index = next_core_index + 1
            self.start_simulation(next_core_index, 0)
            return

        instruction_num = len(self.FinalInfo["instruction"]["core_list"][core_index])
        self.visited_single[core_index] = 1
        for k in range(index_in_core, instruction_num):
            instruction = self.FinalInfo["instruction"]["core_list"][core_index][k]
            if instruction["operation"] == "SEND" or instruction["operation"] == "RECV":
                comm_index = instruction["comm_index"]
                instruction_index_in_core = instruction["instruction_index_in_core"]
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                if not comm_index in self.comm_index_2_index_in_core:
                    if len(self.comm_index_2_core_index) % (self.comm_pair_total_num // 20) == 0 or len(self.comm_index_2_core_index) == self.comm_pair_total_num - 1:
                        print("{:.2f}%".format(len(self.comm_index_2_core_index) / self.comm_pair_total_num * 100))
                    self.comm_index_2_index_in_core[comm_index] = instruction_index_in_core
                    self.comm_index_2_core_index[comm_index] = core_index
                    next_core_index = core_index + 1
                    while (not self.visited_single[next_core_index] == 0):
                        next_core_index = next_core_index + 1
                    self.start_simulation(next_core_index, 0)
                else:
                    corresponding_core_index = self.comm_index_2_core_index[comm_index]
                    corresponding_instruction_index_in_core = self.comm_index_2_index_in_core[comm_index]
                    element_num = instruction["element_num"]
                    if (instruction["operation"] == "RECV"):
                        destination_address = instruction["destination_address"]
                        source_address = self.FinalInfo["instruction"]["core_list"][corresponding_core_index][corresponding_instruction_index_in_core]["source_address"]
                        self.CoreMemory.local_memory[core_index, destination_address:destination_address + element_num] = \
                            self.CoreMemory.local_memory[corresponding_core_index, source_address:source_address + element_num]
                    else:
                        source_address = instruction["source_address"]
                        destination_address = self.FinalInfo["instruction"]["core_list"][corresponding_core_index][corresponding_instruction_index_in_core]["destination_address"]
                        self.CoreMemory.local_memory[corresponding_core_index, destination_address:destination_address + element_num] = \
                            self.CoreMemory.local_memory[core_index, source_address:source_address + element_num]
                    self.start_simulation(core_index, instruction_index_in_core + 1)
                    self.start_simulation(corresponding_core_index, corresponding_instruction_index_in_core + 1)
                return
            elif instruction["operation"] == "LD":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                node_name = self.FinalInfo["node_list"][node_index]["name"]
                stage = instruction["stage"]
                if self.pipeline_type == "batch":
                    if stage == "INPUT" or stage == "POST":
                        if node_index == 1:
                            self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                                self.nn_input[source_offset:source_offset + element_num]
                        else:
                            provider_index = -1 * instruction["source_address"]
                            provider = self.FinalInfo["node_list"][provider_index]["name"]
                            input_data = self.intermediate_result[provider][source_offset:source_offset + element_num]
                            if self.FinalInfo["node_list"][provider_index]["operation"] == "OP_CONV" or self.FinalInfo["node_list"][provider_index]["operation"] == "OP_FC":
                                if self.FinalInfo["node_list"][provider_index]["with_act"] == 1:
                                    input_data = (input_data + np.abs(input_data) ) / 2
                            self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = input_data
                elif self.pipeline_type == "element":
                    if stage == "INPUT":
                        self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                            self.nn_input[source_offset:source_offset + element_num]
                if stage == "BIAS":
                    bias_name = self.OutputToBiasDict[node_name]
                    self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                        self.OriginalWeightDict[bias_name][0:element_num]
            elif instruction["operation"] == "ST":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                node_index = instruction["node_index"]
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = self.max_output_element_num * node_index
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.global_memory[
                destination_address + destination_offset: destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[core_index, source_address + source_offset: source_address + source_offset + element_num]
            elif instruction["operation"] == "MVMUL":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                AG_index = instruction["source"]
                node_name = self.FinalInfo["AG_info"][AG_index]["node_name"]
                weight_name = self.OutputToWeightDict[node_name]
                height_start = self.AG_height_start[AG_index]
                height_end = self.AG_height_end[AG_index]
                weight_matrix = self.GEMMWeightDict[weight_name][height_start:height_end + 1, :]
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                input_element_num = instruction["input_element_num"]
                input_vector = self.CoreMemory.local_memory[core_index][source_address+source_offset:source_address+source_offset+input_element_num]
                result = np.matmul(input_vector, weight_matrix)
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                output_element_num = instruction["output_element_num"]
                self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + output_element_num] = result
            elif instruction["operation"] == "LLDI":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                imm_val = instruction["imm_value"]
                self.CoreMemory.local_memory[core_index, destination_address + destination_offset: destination_address + destination_offset + element_num] = imm_val
            elif instruction["operation"] == "VVADD":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.local_memory[core_index,
                destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num] + \
                    self.CoreMemory.local_memory[core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
            elif instruction["operation"] == "VVMUL":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                source_1 = self.CoreMemory.local_memory[core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num]
                source_2 = self.CoreMemory.local_memory[core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
                self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = source_1 * source_2
            elif instruction["operation"] == "LMV":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[core_index, source_address + source_offset:source_address + source_offset + element_num]
            elif instruction["operation"] == "VVMAX":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                source_1 = self.CoreMemory.local_memory[core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num]
                source_2 = self.CoreMemory.local_memory[core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
                self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    np.where(source_1 > source_2, source_1, source_2)
            elif instruction["operation"] == "VRELU":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                self.CoreMemory.local_memory[core_index,
                destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    (np.abs(self.CoreMemory.local_memory[core_index, source_address + source_offset:source_address + source_offset + element_num]) +
                     self.CoreMemory.local_memory[core_index,source_address + source_offset:source_address + source_offset + element_num]) / 2
            # elif instruction["operation"] == "VRSU":
            #     self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
            #     source_address = instruction["source_address"]
            #     source_offset = instruction["source_offset"]
            #     destination_address = instruction["destination_address"]
            #     destination_offset = instruction["destination_offset"]
            #     element_num = instruction["element_num"]
            #     node_index = instruction["node_index"]
            #     imm_val = instruction["imm_val"]
            #     vec1 = self.CoreMemory.local_memory[core_index, source_address + source_offset:source_address + source_offset + element_num]
            #     vec2 = np.ones(vec1.shape) * imm_val
            #     vec3 = np.zeros(vec1.shape)
            #     vec3[vec1 >= vec2] = vec2[vec1 >= vec2]
            #     vec3[vec1 < vec2] = vec1[vec1 < vec2]
            #     self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = vec3
            # elif instruction["operation"] == "VRSL":
            #     self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
            #     source_address = instruction["source_address"]
            #     source_offset = instruction["source_offset"]
            #     destination_address = instruction["destination_address"]
            #     destination_offset = instruction["destination_offset"]
            #     element_num = instruction["element_num"]
            #     node_index = instruction["node_index"]
            #     imm_val = instruction["imm_val"]
            #     vec1 = self.CoreMemory.local_memory[core_index, source_address + source_offset:source_address + source_offset + element_num]
            #     vec2 = np.ones(vec1.shape) * imm_val
            #     vec3 = np.zeros(vec1.shape)
            #     vec3[vec1 >= vec2] = vec1[vec1 >= vec2]
            #     vec3[vec1 < vec2] = vec2[vec1 < vec2]
            #     self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = vec3
            elif instruction["operation"] == "VER":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                store_address = self.max_output_element_num * node_index
                input_cycle_index = instruction["input_cycle_index"]
                output_channel_element_num = self.FinalInfo["node_list"][node_index]["output_dim"][1]
                store_offset = input_cycle_index * output_channel_element_num
                self.CoreMemory.global_memory[store_address + store_offset:store_address + store_offset + element_num] = \
                    self.CoreMemory.local_memory[core_index, source_address + source_offset:source_address + source_offset + element_num]
            else:
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
        next_core_index = core_index + 1
        while (not self.visited_single[next_core_index] == 0):
            next_core_index = next_core_index + 1
        self.start_simulation(next_core_index, 0)

    def simulation(self):
        print("==================== Start Simulation ====================")
        self.start_simulation(0,0)

    def comparing(self):
        print("==================== Show Comparison  ====================")
        # verification_node_set = {"OP_CONV", "OP_FC", "OP_POOL", "OP_ELTWISE", "OP_CONCAT"}
        verification_node_set = {"OP_CONV", "OP_FC"}
        for k in self.onnx_runtime_outs.keys():
            verify_node_name = k
            if verify_node_name not in self.node_name_2_index:
                continue
            verify_node_index = self.node_name_2_index[verify_node_name]
            verify_node_operation = self.FinalInfo["node_list"][verify_node_index]["operation"]
            if not verify_node_operation in verification_node_set:
                continue
            if self.FinalInfo["node_list"][verify_node_index]["output_dim_num"] == 4:
                print(verify_node_name)
                start_element_address = verify_node_index * self.max_output_element_num
                output_channel_num = self.FinalInfo["node_list"][verify_node_index]["output_dim"][2] * \
                                     self.FinalInfo["node_list"][verify_node_index]["output_dim"][3]
                output_channel_element_num = self.FinalInfo["node_list"][verify_node_index]["output_dim"][1]
                print(output_channel_num, "*", output_channel_element_num)
                s = 0                     # start_output_channel_index
                e = output_channel_num    # end_output_channel_index
                verify_result = self.CoreMemory.global_memory[start_element_address + output_channel_element_num * s:start_element_address + output_channel_element_num * e]
                ground_truth = (self.onnx_runtime_outs[verify_node_name].transpose(0, 2, 3, 1)).flatten()[output_channel_element_num * s:output_channel_element_num * e]
                if self.FinalInfo["node_list"][verify_node_index]["operation"] == "OP_CONV":
                    if self.FinalInfo["node_list"][verify_node_index]["with_act"] == 1:
                        ground_truth = (abs(ground_truth) + ground_truth) / 2
                verify_result = verify_result + 1e-2
                ground_truth = ground_truth + 1e-2
                print(np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) * 100, "%")
                if np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) > 1:
                    # for channel_x in range(e):
                    #     verify_result_channel = verify_result[output_channel_element_num*channel_x:output_channel_element_num*(channel_x+1)]
                    #     ground_truth_channel = ground_truth[output_channel_element_num*channel_x:output_channel_element_num*(channel_x+1)]
                    #     if np.mean(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))) > 1:
                    #         print(channel_x)
                    #         print(np.max(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(np.argmax(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(np.mean(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(verify_result_channel)
                    #         print(ground_truth_channel)
                    #         break
                    print(verify_result)
                    print(ground_truth)
                print("\n")
            elif self.FinalInfo["node_list"][verify_node_index]["output_dim_num"] == 2:
                print(verify_node_name)
                start_element_address = verify_node_index * self.max_output_element_num
                output_element_num = self.node_name_2_output_element_num[verify_node_name]
                print(output_element_num)
                verify_result = self.CoreMemory.global_memory[start_element_address:start_element_address + output_element_num]
                ground_truth = (self.onnx_runtime_outs[verify_node_name].transpose(0, 1)).flatten()[0:output_element_num]
                if self.FinalInfo["node_list"][verify_node_index]["operation"] == "OP_FC":
                    if self.FinalInfo["node_list"][verify_node_index]["with_act"] == 1:
                        ground_truth = (np.abs(ground_truth) + ground_truth) / 2
                verify_result = verify_result + 1e-2
                ground_truth = ground_truth + 1e-2
                print(np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) * 100, "%")
                if np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) > 1:
                    print(verify_result)
                    print(ground_truth)
                print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIMCOMP Verification Module')
    parser.add_argument("-ModelPath", "--model_path", default="../models/ONNX/alexnet.onnx", help="onnx model path")
    parser.add_argument("-Pipeline", "--pipeline_type", default="element", help="[element] or [batch]")
    args = parser.parse_args()
    # create verification
    verification = Verification(args.model_path, args.pipeline_type)
    # load model
    verification.load_weight()
    verification.load_input()
    verification.eliminate_no_consider_OP()
    verification.get_ground_truth()
    # loading
    verification.load_compilation()
    # simulation
    verification.simulation()
    # comparison
    verification.comparing()

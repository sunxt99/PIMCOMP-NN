import onnx
import json
import numpy as np
import argparse

class FrontEnd:
    def __init__(self, model_path, save_path):
        self.ONNX_op_2_PIMCOMP_op = {
                "Conv": "OP_CONV",
                "Relu": "OP_RELU",
                "Tanh": "OP_TANH",
                "Sigmoid": "OP_SIGMOID",
                "MaxPool": "OP_POOL",
                "Flatten": "OP_FLATTEN",
                "Gemm": "OP_FC",
                "Dropout": "OP_DROPOUT",
                "LRN": "OP_LRN",
                "Concat": "OP_CONCAT",
                "AveragePool": "OP_POOL",
                "GlobalAveragePool": "OP_POOL",
                "Reshape": "OP_RESHAPE",
                "Transpose": "OP_TRANSPOSE",
                "Softmax": "OP_SOFTMAX",
                "BatchNormalization": "OP_BN",
                "Sum": "OP_ELTWISE",
                "Add": "OP_ELTWISE",
                "Sub": "OP_ELTWISE",
                "Mul": "OP_ELTWISE",
                "Pad": "OP_PAD",
                "Clip": "OP_CLIP",
                "Squeeze": "OP_SQUEEZE",
                "MatMul": "OP_MATMUL",
                "Shape": "OP_SHAPE",
                "Gather": "OP_GATHER",
                "Unsqueeze": "OP_UNSQUEEZE"}
        self.model_path = model_path
        self.save_path = save_path
        self.model_name = model_path.split("/")[-1].split(".")[0]

    def load_model(self):
        self.model = onnx.load_model(self.model_path)
        #将输入[N,3,224,224]变为[1,3,224,224]
        # for input in self.model.graph.input:
        #     print(input.name)
        self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        # self.model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
        # self.model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
        # self.model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224
        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.save(self.model, self.model_path)
        self.node_num = len(self.model.graph.node)
        # print(self.model.graph.value_info)
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                print(node.name)

    def parse_model(self):
        # 对节点进行重命名
        for i in range(self.node_num):
            node = self.model.graph.node[i]
            self.model.graph.node[i].name = node.output[0]
        # 根据tensor的name获取其dim和dim_num
        self.get_dim_num_from_tensor_name = {}
        self.get_dim_from_tensor_name = {}
        for ioput in self.model.graph.value_info:
            name = ioput.name
            dim_num = len(ioput.type.tensor_type.shape.dim)
            dim = [ioput.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            print("name=%r dim_num=%r dim=%r" % (name, dim_num, dim))
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim
        # Pad节点预处理：得到constant节点对应的参数值
        self.pad_constant_node = {}
        for node in self.model.graph.node:
            if node.op_type == "Constant" and node.attribute[0].t.dims == [8]:
                self.pad_constant_node[node.name] = np.frombuffer(node.attribute[0].t.raw_data, dtype = np.int64)
        # 获取所有initializer_tensor的名称
        self.initializer_tensor = [tensor.name for tensor in self.model.graph.initializer]
        # 根据initializer_tensor的名称获取其index
        self.initializer_name_to_index = dict(zip(self.initializer_tensor,[i for i in range(len(self.initializer_tensor))]))
        self.constant_node = [node.name for node in self.model.graph.node if node.op_type == "Constant"]
        # 获取每个节点的生产者消费者信息（最终的output没有消费者，最初的data没有生产者）
        self.node_provider = {}
        self.node_consumer = {}
        for node in self.model.graph.node:
            for input_tensor in node.input:
                if not input_tensor in self.initializer_tensor and not input_tensor in self.pad_constant_node and not input_tensor in self.constant_node:
                    if node.name in self.node_provider:
                        self.node_provider[node.name] += [input_tensor]
                        if (self.node_consumer.get(input_tensor)):
                            self.node_consumer[input_tensor] += [node.name]
                        else:
                            self.node_consumer[input_tensor] = [node.name]
                    else:
                        self.node_provider[node.name] = [input_tensor]
                        if (self.node_consumer.get(input_tensor)):
                            self.node_consumer[input_tensor] += [node.name]
                        else:
                            self.node_consumer[input_tensor] = [node.name]
        # 根据tensor的name获取其dim和dim_num
        self.get_dim_num_from_tensor_name = {}
        self.get_dim_from_tensor_name = {}
        for tensor in self.model.graph.value_info:
            name = tensor.name
            dim_num = len(tensor.type.tensor_type.shape.dim)
            dim = [tensor.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

        # 增加input对应的tensor的dim_num和dim
        self.input_num = len(self.model.graph.input)
        self.input_names = [x.name for x in self.model.graph.input]
        for i in range(self.input_num):
            input_i = self.model.graph.input[i]
            name = input_i.name
            dim_num = len(input_i.type.tensor_type.shape.dim)
            dim = [input_i.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

        # 增加output对应的tensor的dim_num和dim
        self.output_num = len(self.model.graph.output)
        self.output_names = [x.name for x in self.model.graph.output]
        for i in range(self.output_num):
            output_i = self.model.graph.output[i]
            name = output_i.name
            dim_num = len(output_i.type.tensor_type.shape.dim)
            dim = [output_i.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

    def produce_info(self):
        self.node_list = []

        for i in range(self.input_num):
            input_node = self.model.graph.input[i]
            input_name = input_node.name
            if (input_name in self.initializer_tensor):
                continue
            input_node_info = {"bitwidth": 16,
                               "index": len(self.node_list),
                               "name": input_name,
                               "operation": "OP_INPUT",
                               "provider_num": 0,
                               "consumer_num": len(self.node_consumer[input_name]),
                               "consumer": self.node_consumer[input_name],
                               "output_dim_num": self.get_dim_num_from_tensor_name[input_name],
                               "output_dim": self.get_dim_from_tensor_name[input_name]
                               }
            input_node_info = dict(sorted(input_node_info.items()))
            self.node_list.append(input_node_info)

        # 首先将每个节点命名为其output tensor的名字
        effective_input_num = len(self.node_list)
        for i in range(self.node_num):
            node = self.model.graph.node[i]

            # output_name (node_name)
            output_name = node.output[0]

            # operation
            operation = self.ONNX_op_2_PIMCOMP_op.get(node.op_type)
            if node.op_type == "Constant":
                continue
            elif operation == None:
                print("operation: ", node.op_type, " not considered")
                break

            # output_dim_num
            output_dim_num = self.get_dim_num_from_tensor_name[output_name]
            output_dim = self.get_dim_from_tensor_name[output_name]

            node_info = {"bitwidth": 16,
                         "index": len(self.node_list),
                         "name": output_name,
                         "operation": operation,
                         "output_dim_num": output_dim_num,
                         "output_dim": output_dim
                         }

            # params
            params = {}
            attribute = node.attribute
            if node.op_type == "Conv":
                node_info["with_bn"] = 0
                node_info["with_act"] = 0
                node_info["act_type"] = -1
                node_info["with_clip"] = 0
                node_info["clip_min"] = -10000000
                node_info["clip_max"] = 10000000
                param_num = len(attribute)
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "group"):
                        params["group"] = one_param.i
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                    if (one_param.name == "dilations"):
                        params["dilation_h"] = one_param.ints[0]
                        params["dilation_w"] = one_param.ints[1]
                params["input_channel"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][1]
                params["output_channel"] = output_dim[1]
                if len(node.input) == 2:
                    params["with_bias"] = 0
                elif len(node.input) == 3:
                    params["with_bias"] = 1
            elif node.op_type == "MaxPool":
                param_num = len(attribute)
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                params["pool_method"] = 0
                params["global"] = 0
            elif node.op_type == "AveragePool":
                param_num = len(attribute)
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                params["pool_method"] = 1
                params["global"] = 0
            elif node.op_type == "GlobalAveragePool":
                params["stride_h"] = 1
                params["stride_w"] = 1
                params["pad_h0"] = 0
                params["pad_h1"] = 0
                params["pad_w0"] = 0
                params["pad_w1"] = 0
                params["kernel_h"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][2]
                params["kernel_w"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][3]
                params["pool_method"] = 1
                params["global"] = 1
            elif node.op_type == "Pad":
                # Pad在ONNX中可能有两种表示。所以分情况处理。
                if len(node.input) == 1:  # 把参数直接写进attribute中
                    params["pad_0_h"] = attribute[1].ints[0]
                    params["pad_0_w"] = attribute[1].ints[4]
                    params["pad_1_h"] = attribute[1].ints[1]
                    params["pad_1_w"] = attribute[1].ints[5]
                    params["pad_2_h"] = attribute[1].ints[2]
                    params["pad_2_w"] = attribute[1].ints[6]
                    params["pad_3_h"] = attribute[1].ints[3]
                    params["pad_3_w"] = attribute[1].ints[7]
                else:  # 把参数直接写进常量中
                    pad_constant_name = node.input[1]
                    # 这里的0_h和0_w其实是第0个维度上begin和end的意思。这个表述应该是和tengine一致。
                    params["pad_0_h"] = int(self.pad_constant_node[pad_constant_name][0])
                    params["pad_0_w"] = int(self.pad_constant_node[pad_constant_name][4])
                    params["pad_1_h"] = int(self.pad_constant_node[pad_constant_name][1])
                    params["pad_1_w"] = int(self.pad_constant_node[pad_constant_name][5])
                    params["pad_2_h"] = int(self.pad_constant_node[pad_constant_name][2])
                    params["pad_2_w"] = int(self.pad_constant_node[pad_constant_name][6])
                    params["pad_3_h"] = int(self.pad_constant_node[pad_constant_name][3])
                    params["pad_3_w"] = int(self.pad_constant_node[pad_constant_name][7])
                params["value"] = 0
            elif node.op_type == "Gemm":
                node_info["with_bn"] = 0
                node_info["with_act"] = 0
                node_info["act_type"] = -1
                node_info["with_clip"] = 0
                node_info["clip_min"] = -10000000
                node_info["clip_max"] = 10000000
                if len(self.get_dim_from_tensor_name[self.node_provider[output_name][0]]) > 1:
                    params["num_input"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][1]
                else:
                    weight_index = self.initializer_name_to_index[node.input[1]]
                    params["num_input"] = self.model.graph.initializer[weight_index].dims[1]
                params["num_output"] = output_dim[1]
                if len(node.input) == 2:
                    params["with_bias"] = 0
                elif len(node.input) == 3:
                    params["with_bias"] = 1
            elif node.op_type == "Add" or node.op_type == "Sum" or node.op_type == "Mul":
                params["eletype"] = 2
            elif node.op_type == "Sub":
                params["eletype"] = 4
            elif node.op_type == "Clip":
                # 假设data_type都是1，也就是float
                if len(node.input) > 1 and node.input[1] in self.initializer_name_to_index and node.input[2] in self.initializer_name_to_index:
                    min_index = self.initializer_name_to_index[node.input[1]]
                    min_value = float(np.frombuffer(self.model.graph.initializer[min_index].raw_data, dtype=np.float32))
                    max_index = self.initializer_name_to_index[node.input[2]]
                    max_value = float(np.frombuffer(self.model.graph.initializer[max_index].raw_data, dtype=np.float32))
                    params["min"] = min_value
                    params["max"] = max_value
                else:
                    min_value = node.attribute[1].f
                    max_value = node.attribute[0].f
                    params["min"] = min_value
                    params["max"] = max_value
            params = dict(sorted(params.items()))
            if params != {}:
                node_info["param"] = params

            # consumer and provider
            if self.node_consumer.get(output_name):
                consumer_num = len(self.node_consumer[output_name])
                node_info["consumer_num"] = consumer_num
                node_info["consumer"] = self.node_consumer[output_name]
            else:
                node_info["consumer_num"] = 0

            if self.node_provider.get(output_name):
                provider_num = len(self.node_provider[output_name])
                node_info["provider_num"] = provider_num
                node_info["provider"] = self.node_provider[output_name]
            else:
                node_info["provider_num"] = 0

            node_info = dict(sorted(node_info.items()))
            self.node_list.append(node_info)

        # Get Input Dim
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        for idx in range(node_num):
            node = self.node_list[idx]
            if "provider" in node.keys():
                provider_index = name_2_index_map[node["provider"][0]]
                input_dim_num = self.node_list[provider_index]["output_dim_num"]
                self.node_list[idx]["input_dim_num"] = input_dim_num
                self.node_list[idx]["input_dim"] = []
                for input_idx in range(input_dim_num):
                    self.node_list[idx]["input_dim"].append(self.node_list[provider_index]["output_dim"][input_idx])

    def optimize_model(self):
        self.manual_fix()
        self.clear_unused_struct()
        self.optimize_for_shuffle()
        self.merge_padding()
        self.fuse_operators()
        self.final_process()

    def manual_fix(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list], [i for i in range(node_num)]))
        # fix onnx info for mobilenetv2
        if self.model_name == "mobilenetv2":
            reshape_node_index = name_2_index_map["472"]
            reshape_provider_index = name_2_index_map["464"]
            self.node_list[reshape_node_index]["output_dim_num"] = 2
            self.node_list[reshape_node_index]["output_dim"] = self.node_list[reshape_provider_index]["output_dim"][0:2]

            gemm_node_index = name_2_index_map["output"]
            self.node_list[gemm_node_index]["input_dim_num"] = self.node_list[reshape_node_index]["output_dim_num"]
            self.node_list[gemm_node_index]["input_dim"] = self.node_list[reshape_node_index]["output_dim"]
            self.node_list[gemm_node_index]["output_dim_num"] = 2
            self.node_list[gemm_node_index]["output_dim"][0] = 1

        # record reshape or flatten node before FC node
        self.reshape_info = {}
        for node in self.node_list:
            if node["operation"] == "OP_FLATTEN" or node["operation"] == "OP_RESHAPE":
                if node["consumer_num"] == 1:
                    consumer_index = name_2_index_map[node["consumer"][0]]
                    consumer_node = self.node_list[consumer_index]
                    if consumer_node["operation"] == "OP_FC":
                        if node["input_dim_num"] != 2:
                            print(node)
                            print(node["input_dim"])
                            print(node["output_dim"])
                            self.reshape_info = {"name": consumer_node["name"],
                                                 "input_dim": node["input_dim"],
                                                 "output_dim": node["output_dim"]
                                                 }


    def clear_unused_struct(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        delete_index_list = []
        # Shape - Gather - Unsqueeze - Concat
        for idx in range(node_num):
            node = self.node_list[idx]
            if node["operation"] == "OP_SHAPE":
                consumer_1st_order_index = name_2_index_map[node["consumer"][0]]
                consumer_1st_order_node = self.node_list[consumer_1st_order_index]
                if consumer_1st_order_node["operation"] == "OP_GATHER":
                    consumer_2nd_order_index = name_2_index_map[consumer_1st_order_node["consumer"][0]]
                    consumer_2nd_order_node = self.node_list[consumer_2nd_order_index]
                    if consumer_2nd_order_node["operation"] == "OP_UNSQUEEZE":
                        consumer_3rd_order_index = name_2_index_map[consumer_2nd_order_node["consumer"][0]]
                        consumer_3rd_order_node = self.node_list[consumer_3rd_order_index]
                        if consumer_3rd_order_node["operation"] == "OP_CONCAT":
                            delete_index_list.append(idx)
                            delete_index_list.append(consumer_1st_order_index)
                            delete_index_list.append(consumer_2nd_order_index)
                            delete_index_list.append(consumer_3rd_order_index)
                            # Shape's provider's consumer
                            shape_provider_index = name_2_index_map[node["provider"][0]]
                            new_consumer_list = []
                            for consumer_name in self.node_list[shape_provider_index]["consumer"]:
                                if consumer_name != node["name"]:
                                    new_consumer_list.append(consumer_name)
                            self.node_list[shape_provider_index]["consumer"] = new_consumer_list
                            self.node_list[shape_provider_index]["consumer_num"] -= 1
                            print(self.node_list[shape_provider_index]["consumer"])
                            print(self.node_list[shape_provider_index]["consumer_num"])
                            # Concat's consumer's provider
                            concat_consumer_index = name_2_index_map[consumer_3rd_order_node["consumer"][0]]
                            new_provider_list = []
                            for provider_name in self.node_list[concat_consumer_index]["provider"]:
                                if provider_name != consumer_3rd_order_node["name"]:
                                    new_provider_list.append(provider_name)
                            self.node_list[concat_consumer_index]["provider"] = new_provider_list
                            self.node_list[concat_consumer_index]["provider_num"] -= 1
                            print(self.node_list[concat_consumer_index]["provider"])
                            print(self.node_list[concat_consumer_index]["provider_num"])
        # delete unused node
        delete_index_list = sorted(delete_index_list)
        for del_idx, del_node_idx in enumerate(delete_index_list):
            print("delete node",self.node_list[del_node_idx - del_idx]["name"], "unused")
            del self.node_list[del_node_idx - del_idx]


    def optimize_for_shuffle(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        delete_index_list = []
        shuffle_num = 0
        # Reshape - Transpose - Reshape
        for idx in range(node_num):
            node = self.node_list[idx]
            if node["operation"] == "OP_RESHAPE":
                consumer_1st_order_index = name_2_index_map[node["consumer"][0]]
                consumer_1st_order_node = self.node_list[consumer_1st_order_index]
                if consumer_1st_order_node["operation"] == "OP_TRANSPOSE":
                    consumer_2nd_order_index = name_2_index_map[consumer_1st_order_node["consumer"][0]]
                    consumer_2nd_order_node = self.node_list[consumer_2nd_order_index]
                    if consumer_2nd_order_node["operation"] == "OP_RESHAPE":
                        shuffle_node_info = {"bitwidth": 16,
                                     "index": len(self.node_list),
                                     "name": consumer_2nd_order_node["name"] , # facilitate verification
                                     "operation": "OP_SHUFFLE",
                                     "output_dim_num": self.node_list[consumer_2nd_order_index]["output_dim_num"],
                                     "output_dim": self.node_list[consumer_2nd_order_index]["output_dim"],
                                     "input_dim_num": node["input_dim_num"],
                                     "input_dim": node["input_dim"],
                                     "provider_num": node["provider_num"],
                                     "provider": node["provider"],
                                     "consumer_num": consumer_2nd_order_node["consumer_num"],
                                     "consumer": consumer_2nd_order_node["consumer"],
                                     "param": {"input_channel":node["output_dim"][1] * node["output_dim"][2],
                                               "split_factor":node["output_dim"][1]} }
                        # the first reshape's provider
                        reshape1_provider_index = name_2_index_map[node["provider"][0]]
                        for c_idx,consumer_name in enumerate(self.node_list[reshape1_provider_index]["consumer"]):
                            if consumer_name == node["name"]:
                                self.node_list[reshape1_provider_index]["consumer"][c_idx] = shuffle_node_info["name"]
                        print(self.node_list[reshape1_provider_index]["consumer"])
                        # the second reshape's consumer
                        reshape2_consumer_index = name_2_index_map[consumer_2nd_order_node["consumer"][0]]
                        for p_idx, provider_name in enumerate(self.node_list[reshape2_consumer_index]["provider"]):
                            if provider_name == consumer_2nd_order_node["name"]:
                                self.node_list[reshape2_consumer_index]["provider"][p_idx] = shuffle_node_info["name"]

                        self.node_list[idx] = shuffle_node_info
                        delete_index_list.append(consumer_1st_order_index)
                        delete_index_list.append(consumer_2nd_order_index)
                        shuffle_num += 1
        # delete unused node
        delete_index_list = sorted(delete_index_list)
        for del_idx, del_node_idx in enumerate(delete_index_list):
            print("delete node",self.node_list[del_node_idx - del_idx]["name"])
            del self.node_list[del_node_idx - del_idx]

    def merge_padding(self):
        delete_node_index = []
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list], [i for i in range(node_num)]))
        for idx, pad_node in enumerate(self.node_list):
            if pad_node["operation"] == "OP_PAD":
                if pad_node["consumer_num"] == 1:
                    consumer_index = name_2_index_map[pad_node["consumer"][0]]
                    consumer_node = self.node_list[consumer_index]
                    if consumer_node["provider_num"] == 1 and (consumer_node["operation"] == "OP_CONV" or consumer_node["operation"] == "OP_POOL"):
                        self.node_list[consumer_index]["provider_num"] = pad_node["provider_num"]
                        self.node_list[consumer_index]["provider"] = []
                        for pad_provider in pad_node["provider"]:
                            self.node_list[consumer_index]["provider"].append(pad_provider)
                            pad_provider_index = name_2_index_map[pad_provider]
                            pad_provider_node = self.node_list[pad_provider_index]
                            for ppc_idx, pad_provider_consumer in enumerate(pad_provider_node["consumer"]):
                                if pad_provider_consumer == pad_node["name"]:
                                    self.node_list[pad_provider_index]["consumer"][ppc_idx] = consumer_node["name"]

                        pad_0_h = self.node_list[idx]["param"]["pad_0_h"]
                        pad_0_w = self.node_list[idx]["param"]["pad_0_w"]
                        pad_1_h = self.node_list[idx]["param"]["pad_1_h"]
                        pad_1_w = self.node_list[idx]["param"]["pad_1_w"]
                        assert pad_0_h == 0 and pad_0_w == 0 and pad_1_h == 0 and pad_1_w == 0

                        pad_2_h = self.node_list[idx]["param"]["pad_2_h"]
                        pad_2_w = self.node_list[idx]["param"]["pad_2_w"]
                        pad_3_h = self.node_list[idx]["param"]["pad_3_h"]
                        pad_3_w = self.node_list[idx]["param"]["pad_3_w"]

                        self.node_list[consumer_index]["param"]["pad_h0"] += pad_2_h
                        self.node_list[consumer_index]["param"]["pad_h1"] += pad_2_w
                        self.node_list[consumer_index]["param"]["pad_w0"] += pad_3_h
                        self.node_list[consumer_index]["param"]["pad_w1"] += pad_3_w

                        self.node_list[consumer_index]["input_dim_num"] = pad_node["input_dim_num"]
                        self.node_list[consumer_index]["input_dim"] = pad_node["input_dim"]
                        delete_node_index.append(idx)

        for del_idx, del_node_idx in enumerate(delete_node_index):
            print("delete node",self.node_list[del_node_idx - del_idx]["name"], self.node_list[del_node_idx - del_idx]["operation"])
            del self.node_list[del_node_idx - del_idx]





    def fuse_operators(self):
        # fuse_operator_list = ["OP_BN", "OP_RELU", "OP_TANH", "OP_SIGMOID", "OP_CLIP"]
        fuse_operator_list = ["OP_BN", "OP_RELU", "OP_TANH", "OP_SIGMOID"]
        for fuse_idx, fuse_operator in enumerate(fuse_operator_list):
            delete_node_index = []
            node_num = len(self.node_list)
            name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
            for idx, fuse_node in enumerate(self.node_list):
                if fuse_node["operation"] == fuse_operator:
                    # the fuse operator has only one provider
                    if fuse_node["provider_num"] == 1:
                        provider_index = name_2_index_map[fuse_node["provider"][0]]
                        provider_node = self.node_list[provider_index]
                        # the provider of the fuse operator has only one consumer
                        if provider_node["consumer_num"] == 1 and (provider_node["operation"] == "OP_CONV" or provider_node["operation"] == "OP_FC"):
                            if fuse_idx == 0:
                                self.node_list[provider_index]["with_bn"] = 1
                            elif fuse_idx == 1 or fuse_idx == 2 or fuse_idx == 3:
                                self.node_list[provider_index]["with_act"] = 1
                                self.node_list[provider_index]["act_type"] = fuse_idx - 1
                            # elif fuse_idx == 4:
                            #     self.node_list[provider_index]["with_clip"] = 1
                            #     self.node_list[provider_index]["clip_min"] = fuse_node["param"]["min"]
                            #     self.node_list[provider_index]["clip_max"] = fuse_node["param"]["max"]
                            self.node_list[provider_index]["consumer_num"] = fuse_node["consumer_num"]
                            self.node_list[provider_index]["consumer"] = []
                            for fuse_consumer in fuse_node["consumer"]:
                                self.node_list[provider_index]["consumer"].append(fuse_consumer)
                                fuse_consumer_index = name_2_index_map[fuse_consumer]
                                fuse_consumer_node = self.node_list[fuse_consumer_index]
                                for fcp_idx, fuse_consumer_provider in enumerate(fuse_consumer_node["provider"]):
                                    if fuse_consumer_provider == fuse_node["name"]:
                                        self.node_list[fuse_consumer_index]["provider"][fcp_idx] = provider_node["name"]
                            delete_node_index.append(idx)

            for del_idx, del_node_idx in enumerate(delete_node_index):
                print("delete node",self.node_list[del_node_idx - del_idx]["name"], self.node_list[del_node_idx - del_idx]["operation"])
                del self.node_list[del_node_idx - del_idx]



    def final_process(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        # Reorder the index
        for idx in range(node_num):
            self.node_list[idx]["index"] = idx
            self.node_list[idx]["new_node_index"] = idx
        # Get the Provider_Index and Consumer_Index
        for idx in range(node_num):
            node = self.node_list[idx]
            if "consumer_num" in node.keys():
                consumer_num = node["consumer_num"]
                self.node_list[idx]["consumer_index"] = []
                for j in range(consumer_num):
                    consumer_name = node["consumer"][j]
                    consumer_index = name_2_index_map[consumer_name]
                    self.node_list[idx]["consumer_index"].append(consumer_index)
            if "provider_num" in node.keys():
                provider_num = node["provider_num"]
                self.node_list[idx]["provider_index"] = []
                for j in range(provider_num):
                    provider_name = node["provider"][j]
                    provider_index = name_2_index_map[provider_name]
                    self.node_list[idx]["provider_index"].append(provider_index)


    def check_info(self):
        pass

    def save_info(self):
        node_list_wrapper = {"node_list": self.node_list,
                             "reshape_info": self.reshape_info}
        with open(self.save_path, "w", encoding='utf-8') as file:
            json.dump(node_list_wrapper, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIMCOMPP FrontEnd Module')
    parser.add_argument("-ModelPath", "--model_path", default="../models/ONNX/shufflenet.onnx", help="onnx model path")
    parser.add_argument("-SavePath", "--save_path", default="../models/JSON/shufflenet.json", help="json file save path")
    args = parser.parse_args()

    frontend = FrontEnd(args.model_path, args.save_path)
    frontend.load_model()
    frontend.parse_model()
    frontend.produce_info()
    frontend.optimize_model()
    frontend.check_info()
    frontend.save_info()

# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from model_compression_toolkit.common import Graph
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.similarity_analyzer import compute_cs


def _get_compare_points(input_graph: Graph) -> Tuple[List[BaseNode], List[str]]:
    """
    Create a list of nodes in a graph where we collect their output statistics, and a corresponding list
    of their names for tensors comparison purposes.
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph, and a list of the their names.
    """
    compare_points = []
    compare_points_name = []
    for n in input_graph.nodes():
        tensors = input_graph.get_out_stats_collector(n)
        if (not isinstance(tensors, list)) and tensors.require_collection():
            compare_points.append(n)
            compare_points_name.append(n.name)
    return compare_points, compare_points_name


class NNVisualizer:
    """
    Class to build two models from two graph: a float and a quantized version.
    NNVisualizer can compare the two models outputs after each layer.
    The results can be displayed using plot_cs_graph by passing it an input image to use
    for inference of the models.
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quantized: Graph,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo):
        """
        Initialize an NNVisualizer object.
        Args:
            graph_float: Float version of the graph.
            graph_quantized: Quantized version of the graph.
            fw_impl: FrameworkImplementation object of the framework of the model.
            fw_info: FrameworkInfo object of the framework of the model.

        """

        self.graph_float = graph_float
        self.graph_quantized = graph_quantized
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # Get compare points of two graphs.
        self.compare_points, self.compare_points_name = _get_compare_points(self.graph_quantized)
        self.compare_points_float, self.compare_points_name_float = _get_compare_points(self.graph_float)

        self.quantized_model, _ = self.fw_impl.model_builder(self.graph_quantized,
                                                             mode=ModelBuilderMode.QUANTIZED,
                                                             append2output=self.compare_points,
                                                             fw_info=self.fw_info)

        self.float_model, _ = self.fw_impl.model_builder(self.graph_float,
                                                         mode=ModelBuilderMode.FLOAT,
                                                         append2output=self.compare_points_float,
                                                         fw_info=self.fw_info)

    def plot_cs_graph(self,
                      input_image: np.ndarray,
                      y_limits: Tuple[float, float] = (-1.2, 1.2)) -> Figure:
        """
        Compare and plot the outputs of the quantized and the float versions
        of a neural network that KerasNNVisualizer has.

        Args:
            input_image: Image to use as input to the networks.
            y_limits: Limits for y axis of the plot.

        Returns:
            Figure of the cosine similarity per layer.
        """

        # To compare cosine similarity, we use a single image as input (per input),
        # to make the difference more noticeable when exists
        new_inputs = []
        for single_input in input_image:
            img = single_input[0]
            new_inputs.append(np.expand_dims(img, axis=0))

        # Get outputs
        tensors_float = self.float_model(new_inputs)
        tensors_fxp = self.quantized_model(new_inputs)

        # Compute cosine similarities between couples of outputs.
        cs_array = np.asarray(
            [compute_cs(t_float.numpy(), t_fxp.numpy()) for t_float, t_fxp in zip(tensors_float, tensors_fxp)])

        cs_array = 1 - 2 * cs_array  # Convert cs to range of [-1, 1] as compute_cs range is [0, 1] (smaller -> more similar)
        # Display the result: cosine similarity at every layer's output.
        fig = plt.figure()
        plt.plot(cs_array)
        plt.ylim(y_limits)
        plt.grid()
        plt.xlabel('Layer')
        plt.ylabel('Cosine Similarity')
        return fig

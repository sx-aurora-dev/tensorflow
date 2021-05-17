# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_ve.
  """
  test_configs = [("NCHW", True), ("NHWC", False)]

  return test_configs


class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_ve):
      return [dtypes.float32]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, dilations,
                            strides, padding, data_format, dtype, use_ve):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_ve: True if the operations should be run on VE
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with test_util.device(use_ve=use_ve,use_gpu=False):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
      conv = nn_ops.conv2d(
          t1,
          t2,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, strides, padding,
                    expected):
    tensors = []
    dilations = [1, 1]
    for (data_format, use_ve) in GetTestConfigs():
      for dtype in self._DtypesToTest(use_ve):
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            dilations,
            strides,
            padding,
            data_format,
            dtype,
            use_ve=use_ve)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tf_logging.info("expected = ", expected)
        tf_logging.info("actual = ", value)
        tol = 1e-5
        if value.dtype == np.float16:
          tol = 1e-3
        self.assertAllClose(expected, np.ravel(value), atol=tol, rtol=tol)
        self.assertShapeEqual(value, conv)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  '''
  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2Same(self):
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)
  '''

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride1x2(self):
    expected_output = [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 6, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 2],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSmallerThanStrideValid(self):
    expected_output = [65, 95, 275, 305]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  # Testing for backprops
  def _RunAndVerifyBackpropFilter(self, input_sizes, filter_sizes, output_sizes,
                                  strides, padding, expected, data_format,
                                  use_ve):
    total_input_size = 1
    total_output_size = 1
    for s in input_sizes:
      total_input_size *= s
    for s in output_sizes:
      total_output_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x0 = [f * 1.0 for f in range(1, total_input_size + 1)]
    x2 = [f * 1.0 for f in range(1, total_output_size + 1)]
    for dtype in self._DtypesToTest(use_ve=use_ve):
      with test_util.device(use_ve=use_ve, use_gpu=False):
        t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        explicit_strides = [1] + strides + [1]
        if data_format == "NCHW":
          t0 = test_util.NHWCToNCHW(t0)
          t2 = test_util.NHWCToNCHW(t2)
          explicit_strides = test_util.NHWCToNCHW(explicit_strides)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=explicit_strides,
            padding=padding,
            data_format=data_format)
        value = self.evaluate(conv)
        self.assertShapeEqual(value, conv)
      tf_logging.info("expected = ", expected)
      tf_logging.info("actual = ", value)
      self.assertArrayNear(expected, value.flatten(), 1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Depth1ValidBackpropFilter(self):
    expected = [5.0, 8.0, 14.0, 17.0]
    for (data_format, use_ve) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_ve=use_ve)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DBackpropFilterWithEmptyInput(self):
    expected = [0, 0, 0, 0]
    for (data_format, use_ve) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[0, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[0, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_ve=use_ve)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Depth3ValidBackpropFilter(self):
    expected = [
        17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0, 32.0, 43.0, 54.0,
        37.0, 50.0, 63.0, 42.0, 57.0, 72.0, 62.0, 85.0, 108.0, 67.0, 92.0,
        117.0, 72.0, 99.0, 126.0, 77.0, 106.0, 135.0, 82.0, 113.0, 144.0, 87.0,
        120.0, 153.0
    ]
    for (data_format, use_ve) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 3, 3],
          filter_sizes=[2, 2, 3, 3],
          output_sizes=[1, 1, 2, 3],
          strides=[1, 1],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_ve=use_ve)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSizeMatchesInputSizeBackpropFilter(self):
    expected_output = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]
    for (data_format, use_ve) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 2, 1],
          filter_sizes=[2, 2, 1, 2],
          output_sizes=[1, 1, 1, 2],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_ve=use_ve)

if __name__ == "__main__":
  test.main()

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/kernels/conv_grad_input_ops.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/protobuf/autotuning.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/framework/ve_ops_common.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_VE
typedef Eigen::VeDevice VEDevice;
#endif

// To be used inside depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropInputOp<CPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropInputOp<CPUDevice, float>;
template struct LaunchConv2DBackpropInputOp<CPUDevice, double>;

// GPU definitions.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// The slow version (but compiles for GPU)

// A dummy type to group forward backward data autotune results together.
struct ConvBackwardDataAutoTuneGroup {
  static string name() { return "ConvBwdData"; }
};

typedef AutoTuneSingleton<ConvBackwardDataAutoTuneGroup, ConvParameters,
                          se::dnn::AlgorithmConfig>
    AutoTuneConvBwdData;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on GPU
// for int32 inputs.
template <>
struct LaunchConv2DBackpropInputOp<GPUDevice, int32> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    LaunchConv2DBackpropInputOpImpl<GPUDevice, int32> launcher;
    launcher(ctx, use_cudnn, cudnn_use_autotune, out_backprop, filter,
             row_dilation, col_dilation, row_stride, col_stride, padding,
             explicit_paddings, in_backprop, data_format);
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void LaunchConv2DBackpropInputOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64>& explicit_paddings, Tensor* in_backprop,
    TensorFormat data_format) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;

  std::vector<int32> strides(4, 1);
  std::vector<int32> dilations(4, 1);
  auto input_h = GetTensorDimIndex(data_format, 'H');
  auto input_w = GetTensorDimIndex(data_format, 'W');
  strides[input_h] = row_stride;
  strides[input_w] = col_stride;
  dilations[input_h] = row_dilation;
  dilations[input_w] = col_dilation;
  TensorShape input_shape = in_backprop->shape();

  const TensorShape& filter_shape = filter.shape();
  ConvBackpropDimensions dims;
  OP_REQUIRES_OK(
      ctx, ConvBackpropComputeDimensionsV2(
               "Conv2DSlowBackpropInput", /*num_spatial_dims=*/2, input_shape,
               filter_shape, out_backprop.shape(), dilations, strides, padding,
               explicit_paddings, data_format, &dims));

  int64 padding_top = -1, padding_bottom = -1;
  int64 padding_left = -1, padding_right = -1;
  if (padding == EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &padding_top,
                             &padding_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &padding_left,
                             &padding_right);
  }
  int64 expected_out_rows, expected_out_cols;
  // The function is guaranteed to succeed because we checked the output and
  // padding was valid earlier.
  TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
      dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
      row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
      &padding_bottom));
  DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);
  TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
      dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
      col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
      &padding_right));
  DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  if (!use_cudnn) {
    ctx->SetStatus(errors::Unimplemented(
        "Conv2DBackpropInput for GPU is not currently supported "
        "without cudnn"));
    return;
  }

  // If the filter in-depth (filter_shape.dim_size(2)) is 1 and smaller than the
  // input depth, it's a depthwise convolution. More generally, if the filter
  // in-depth divides but is smaller than the input depth, it is a grouped
  // convolution.
  bool is_grouped_convolution = filter_shape.dim_size(2) != dims.in_depth;
  if (dims.spatial_dims[0].filter_size == 1 &&
      dims.spatial_dims[1].filter_size == 1 && !is_grouped_convolution &&
      dims.spatial_dims[0].stride == 1 && dims.spatial_dims[1].stride == 1 &&
      data_format == FORMAT_NHWC && (padding == VALID || padding == SAME)) {
    // 1x1 filter, so call cublas directly.
    const uint64 m = dims.batch_size * dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size;
    const uint64 k = dims.out_depth;
    const uint64 n = dims.in_depth;

    auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                in_backprop->template flat<T>().size());

    auto transpose = se::blas::Transpose::kTranspose;
    auto no_transpose = se::blas::Transpose::kNoTranspose;

    bool blas_launch_status =
        stream
            ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                           a_ptr, k, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  } else if (dims.spatial_dims[0].filter_size ==
                 dims.spatial_dims[0].input_size &&
             dims.spatial_dims[1].filter_size ==
                 dims.spatial_dims[1].input_size &&
             !is_grouped_convolution && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same height/width, and we are not
    // using grouped convolution, so call cublas directly.
    const uint64 m = dims.batch_size;
    const uint64 k = dims.out_depth;
    const uint64 n = dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size * dims.in_depth;

    auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                in_backprop->template flat<T>().size());

    auto transpose = se::blas::Transpose::kTranspose;
    auto no_transpose = se::blas::Transpose::kNoTranspose;

    bool blas_launch_status =
        stream
            ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                           a_ptr, k, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  }

  const int64 common_padding_rows = std::min(padding_top, padding_bottom);
  const int64 common_padding_cols = std::min(padding_left, padding_right);
  TensorShape compatible_input_shape;
  if (padding_top != padding_bottom || padding_left != padding_right) {
    // Pad the input in the same way we did during the forward pass, so that
    // cuDNN or MIOpen receives the same input during the backward pass function
    // as it did during the forward pass function.
    const int64 padding_rows_diff = std::abs(padding_bottom - padding_top);
    const int64 padding_cols_diff = std::abs(padding_right - padding_left);
    const int64 new_in_rows =
        dims.spatial_dims[0].input_size + padding_rows_diff;
    const int64 new_in_cols =
        dims.spatial_dims[1].input_size + padding_cols_diff;
    compatible_input_shape = ShapeFromFormat(
        data_format, dims.batch_size, new_in_rows, new_in_cols, dims.in_depth);
  } else {
    compatible_input_shape = input_shape;
  }

  CHECK(common_padding_rows >= 0 && common_padding_cols >= 0)  // Crash OK
      << "Negative row or col paddings: (" << common_padding_rows << ", "
      << common_padding_cols << ")";

  // The Tensor Core in NVIDIA Volta+ GPUs supports efficient convolution with
  // fp16 in NHWC data layout. In all other configurations it's more efficient
  // to run computation in NCHW data format.
  const bool compute_in_nhwc =
      DataTypeToEnum<T>::value == DT_HALF && IsVoltaOrLater(*stream->parent());

  // We only do one directional conversion: NHWC->NCHW. We never convert in the
  // other direction. Grappler layout optimizer selects the preferred layout and
  // adds necessary annotations to the graph.
  const TensorFormat compute_data_format =
      (compute_in_nhwc && data_format == FORMAT_NHWC) ? FORMAT_NHWC
                                                      : FORMAT_NCHW;

  VLOG(3) << "Compute Conv2DBackpropInput with cuDNN:"
          << " data_format=" << ToString(data_format)
          << " compute_data_format=" << ToString(compute_data_format);

  constexpr auto kComputeInNHWC =
      std::make_tuple(se::dnn::DataLayout::kBatchYXDepth,
                      se::dnn::FilterLayout::kOutputYXInput);
  constexpr auto kComputeInNCHW =
      std::make_tuple(se::dnn::DataLayout::kBatchDepthYX,
                      se::dnn::FilterLayout::kOutputInputYX);

  se::dnn::DataLayout compute_data_layout;
  se::dnn::FilterLayout filter_layout;

  std::tie(compute_data_layout, filter_layout) =
      compute_data_format == FORMAT_NHWC ? kComputeInNHWC : kComputeInNCHW;

  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(dims.batch_size)
      .set_height(GetTensorDim(compatible_input_shape, data_format, 'H'))
      .set_width(GetTensorDim(compatible_input_shape, data_format, 'W'))
      .set_feature_map_count(dims.in_depth)
      .set_layout(compute_data_layout);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(dims.batch_size)
      .set_height(dims.spatial_dims[0].output_size)
      .set_width(dims.spatial_dims[1].output_size)
      .set_feature_map_count(dims.out_depth)
      .set_layout(compute_data_layout);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(dims.spatial_dims[0].filter_size)
      .set_input_filter_width(dims.spatial_dims[1].filter_size)
      .set_input_feature_map_count(filter_shape.dim_size(2))
      .set_output_feature_map_count(filter_shape.dim_size(3))
      .set_layout(filter_layout);
  se::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_dilation_rate(dims.spatial_dims[0].dilation)
      .set_horizontal_dilation_rate(dims.spatial_dims[1].dilation)
      .set_vertical_filter_stride(dims.spatial_dims[0].stride)
      .set_horizontal_filter_stride(dims.spatial_dims[1].stride)
      .set_zero_padding_height(common_padding_rows)
      .set_zero_padding_width(common_padding_cols)
      .set_group_count(dims.in_depth / filter_shape.dim_size(2));

  // Tensorflow filter format: HWIO
  // cuDNN filter formats: (data format) -> (filter format)
  //   (1) NCHW -> OIHW
  //   (2) NHWC -> OHWI

  Tensor transformed_filter;
  const auto transform_filter = [&](FilterTensorFormat dst_format) -> Status {
    VLOG(4) << "Transform filter tensor from " << ToString(FORMAT_HWIO)
            << " to " << ToString(dst_format);

    TensorShape dst_shape =
        dst_format == FORMAT_OIHW
            ? TensorShape({filter.dim_size(3), filter.dim_size(2),
                           filter.dim_size(0), filter.dim_size(1)})
            : TensorShape({filter.dim_size(3), filter.dim_size(0),
                           filter.dim_size(1), filter.dim_size(2)});

    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value, dst_shape,
                                          &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        ctx->eigen_device<GPUDevice>(), dst_format,
        To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    return Status::OK();
  };

  if (compute_data_format == FORMAT_NCHW) {
    OP_REQUIRES_OK(ctx, transform_filter(FORMAT_OIHW));
  } else if (compute_data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(ctx, transform_filter(FORMAT_OHWI));
  } else {
    ctx->SetStatus(errors::InvalidArgument("Invalid compute data format: ",
                                           ToString(compute_data_format)));
    return;
  }

  Tensor transformed_out_backprop;
  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the `out_backprop` tensor from NHWC to NCHW.";
    TensorShape compute_shape = ShapeFromFormat(
        compute_data_format, dims.batch_size, dims.spatial_dims[0].output_size,
        dims.spatial_dims[1].output_size, dims.out_depth);
    if (dims.out_depth > 1) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<T>::value, compute_shape,
                                        &transformed_out_backprop));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(), out_backprop.tensor<T, 4>(),
          transformed_out_backprop.tensor<T, 4>());
    } else {
      // If depth <= 1, then just reshape.
      CHECK(transformed_out_backprop.CopyFrom(out_backprop, compute_shape));
    }
  } else {
    transformed_out_backprop = out_backprop;
  }

  Tensor pre_transformed_in_backprop;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DataTypeToEnum<T>::value,
               ShapeFromFormat(
                   compute_data_format,
                   GetTensorDim(compatible_input_shape, data_format, 'N'),
                   GetTensorDim(compatible_input_shape, data_format, 'H'),
                   GetTensorDim(compatible_input_shape, data_format, 'W'),
                   GetTensorDim(compatible_input_shape, data_format, 'C')),
               &pre_transformed_in_backprop));

  auto out_backprop_ptr =
      AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                     transformed_out_backprop.template flat<T>().size());
  auto filter_ptr =
      AsDeviceMemory(transformed_filter.template flat<T>().data(),
                     transformed_filter.template flat<T>().size());
  auto in_backprop_ptr =
      AsDeviceMemory(pre_transformed_in_backprop.template flat<T>().data(),
                     pre_transformed_in_backprop.template flat<T>().size());

  int64 workspace_bytes = 1LL << 32;  // 4GB by default.
  // CuDNN frontend will expose more engines some of which might use too much
  // workspace. This would increase the overall demand of memory when training
  // models.
  if (CudnnUseFrontend()) {
    workspace_bytes = 1LL << 30;  // 1GB by default.
  }
  static int64 ConvolveBackwardDataScratchSize =
      GetDnnWorkspaceLimit("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", workspace_bytes);
  DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize, ctx);
  int device_id = stream->parent()->device_ordinal();
  DataType dtype = out_backprop.dtype();
  ConvParameters conv_parameters = {
      dims.batch_size,                     // batch
      dims.in_depth,                       // in_depths
      {{input_desc.height(),               // in_rows
        input_desc.width()}},              // in_cols
      compute_data_format,                 // compute_data_format
      dims.out_depth,                      // out_depths
      {{dims.spatial_dims[0].filter_size,  // filter_rows
        dims.spatial_dims[1].filter_size,  // filter_cols
        filter_shape.dim_size(2)}},        // filter_depths
      {{dims.spatial_dims[0].dilation,     // dilation_rows
        dims.spatial_dims[1].dilation}},   // dilation_cols
      {{dims.spatial_dims[0].stride,       // stride_rows
        dims.spatial_dims[1].stride}},     // stride_cols
      {{common_padding_rows,               // padding_rows
        common_padding_cols}},             // padding_cols
      dtype,                               // tensor data type
      device_id,                           // device_id
      conv_desc.group_count()              // group_count
  };
#if TENSORFLOW_USE_ROCM
  // cudnn_use_autotune is applicable only the CUDA flow
  // for ROCm/MIOpen, we need to call GetMIOpenConvolveAlgorithms explicitly
  // if we do not have a cached algorithm_config for this conv_parameters
  cudnn_use_autotune = true;
#endif
  AlgorithmConfig algorithm_config;

  if (cudnn_use_autotune && !AutoTuneConvBwdData::GetInstance()->Find(
                                conv_parameters, &algorithm_config)) {
    std::vector<std::unique_ptr<se::dnn::ConvolveExecutionPlan>> plans;
#if GOOGLE_CUDA
    std::vector<AlgorithmDesc> algorithms;
    std::vector<AlgorithmConfig> configs;
    if (CudnnUseFrontend()) {
      OP_REQUIRES(
          ctx,
          stream->parent()->GetConvolveExecutionPlans(
              se::dnn::ConvolutionKind::BACKWARD_DATA,
              se::dnn::ToDataType<T>::value, stream, input_desc, filter_desc,
              output_desc, conv_desc, &plans),
          errors::Unknown("Failed to get convolution execution plan. This is "
                          "probably because cuDNN failed to initialize, so try "
                          "looking to see if a warning log message was printed "
                          "above."));
      for (const auto& plan : plans) {
        configs.push_back(
            AlgorithmConfig(AlgorithmDesc{plan->getTag(), plan->get_raw_desc()},
                            plan->getWorkspaceSize()));
      }
    } else {
      OP_REQUIRES(
          ctx,
          stream->parent()->GetConvolveBackwardDataAlgorithms(
              conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
                  stream->parent()),
              &algorithms),
          errors::Unknown("Failed to get convolution execution plan. This is "
                          "probably because cuDNN failed to initialize, so try "
                          "looking to see if a warning log message was printed "
                          "above."));
      for (const auto& algorithm : algorithms) {
        configs.push_back(AlgorithmConfig(algorithm));
      }
    }

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);

    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());

    se::DeviceMemory<T> in_backprop_ptr_rz(
        WrapRedzoneBestEffort(&rz_allocator, in_backprop_ptr));

    std::vector<tensorflow::AutotuneResult> results;
    for (auto& profile_config : configs) {
      // TODO(zhengxq): profile each algorithm multiple times to better
      // accuracy.
      DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                            ctx);
      se::RedzoneAllocator rz_scratch_allocator(
          stream, &tf_allocator_adapter, se::GpuAsmOpts(),
          /*memory_limit=*/ConvolveBackwardDataScratchSize);
      se::ScratchAllocator* allocator_used =
          !RedzoneCheckDisabled()
              ? static_cast<se::ScratchAllocator*>(&rz_scratch_allocator)
              : static_cast<se::ScratchAllocator*>(&scratch_allocator);
      ProfileResult profile_result;
      Status cudnn_launch_status;
      if (CudnnUseFrontend()) {
        cudnn_launch_status = stream->ConvolveBackwardDataWithExecutionPlan(
            filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
            input_desc, &in_backprop_ptr_rz, allocator_used, profile_config,
            &profile_result);
      } else {
        cudnn_launch_status = stream->ConvolveBackwardDataWithAlgorithm(
            filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
            input_desc, &in_backprop_ptr_rz, allocator_used, profile_config,
            &profile_result);
      }

      if (cudnn_launch_status.ok() && profile_result.is_valid()) {
        results.emplace_back();
        auto& result = results.back();
        if (CudnnUseFrontend()) {
          result.mutable_cuda_conv_plan()->set_exec_plan_id(
              profile_config.algorithm()->exec_plan_id());
        } else {
          result.mutable_conv()->set_algorithm(
              profile_config.algorithm()->algo_id());
          result.mutable_conv()->set_tensor_ops_enabled(
              profile_config.algorithm()->tensor_ops_enabled());
        }

        result.set_scratch_bytes(
            !RedzoneCheckDisabled()
                ? rz_scratch_allocator.TotalAllocatedBytesExcludingRedzones()
                : scratch_allocator.TotalByteSize());
        *result.mutable_run_time() = proto_utils::ToDurationProto(
            absl::Milliseconds(profile_result.elapsed_time_in_ms()));

        CheckRedzones(rz_scratch_allocator, &result);
        CheckRedzones(rz_allocator, &result);
      } else if (CudnnUseFrontend()) {
        // When CuDNN frontend APIs are used, we need to make sure the profiling
        // results are one-to-one mapping of the "plans". So, we insert dummy
        // results when the excution fails.
        results.emplace_back();
        auto& result = results.back();
        result.mutable_failure()->set_kind(AutotuneResult::UNKNOWN);
        result.mutable_failure()->set_msg(
            absl::StrCat("Profiling failure on CUDNN engine: ",
                         profile_config.algorithm()->exec_plan_id()));
      }
    }
#elif TENSORFLOW_USE_ROCM
    DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize, ctx);
    std::vector<ProfileResult> algorithms;
    OP_REQUIRES(
        ctx,
        stream->parent()->GetMIOpenConvolveAlgorithms(
            se::dnn::ConvolutionKind::BACKWARD_DATA,
            se::dnn::ToDataType<T>::value, stream, input_desc, in_backprop_ptr,
            filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
            &scratch_allocator, &algorithms),
        errors::Unknown(
            "Failed to get convolution algorithm. This is probably "
            "because MIOpen failed to initialize, so try looking to "
            "see if a warning log message was printed above."));

    std::vector<tensorflow::AutotuneResult> results;
    if (algorithms.size() == 1) {
      auto profile_result = algorithms[0];
      results.emplace_back();
      auto& result = results.back();
      result.mutable_conv()->set_algorithm(
          profile_result.algorithm().algo_id());
      result.mutable_conv()->set_tensor_ops_enabled(
          profile_result.algorithm().tensor_ops_enabled());

      result.set_scratch_bytes(profile_result.scratch_size());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    } else {
      for (auto miopen_algorithm : algorithms) {
        auto profile_algorithm = miopen_algorithm.algorithm();
        ProfileResult profile_result;
        auto miopen_launch_status = stream->ConvolveBackwardDataWithAlgorithm(
            filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
            input_desc, &in_backprop_ptr, &scratch_allocator,
            AlgorithmConfig(profile_algorithm, miopen_algorithm.scratch_size()),
            &profile_result);

        if (miopen_launch_status.ok() && profile_result.is_valid()) {
          results.emplace_back();
          auto& result = results.back();
          result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
          result.mutable_conv()->set_tensor_ops_enabled(
              profile_algorithm.tensor_ops_enabled());
          result.set_scratch_bytes(scratch_allocator.TotalByteSize());
          *result.mutable_run_time() = proto_utils::ToDurationProto(
              absl::Milliseconds(profile_result.elapsed_time_in_ms()));
        }
      }
    }
#endif
    LogConvAutotuneResults(
        se::dnn::ConvolutionKind::BACKWARD_DATA, se::dnn::ToDataType<T>::value,
        in_backprop_ptr, filter_ptr, out_backprop_ptr, input_desc, filter_desc,
        output_desc, conv_desc, stream->parent(), results);
    if (CudnnUseFrontend()) {
      OP_REQUIRES_OK(
          ctx, BestCudnnConvAlgorithm(results, &plans, &algorithm_config));

    } else {
      OP_REQUIRES_OK(
          ctx, BestCudnnConvAlgorithm(results, nullptr, &algorithm_config));
    }
    AutoTuneConvBwdData::GetInstance()->Insert(conv_parameters,
                                               algorithm_config);
  }

  Status cudnn_launch_status;
  if (CudnnUseFrontend()) {
    if (algorithm_config.algorithm().has_value()) {
      VLOG(4) << "Conv2DBackpropInput Execution Plan: "
              << algorithm_config.algorithm()->exec_plan_id();
    } else {
      VLOG(4) << "Convolution AutoTune has been turned off";
    }
    cudnn_launch_status = stream->ConvolveBackwardDataWithExecutionPlan(
        filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
        input_desc, &in_backprop_ptr, &scratch_allocator, algorithm_config,
        nullptr);
  } else {
    cudnn_launch_status = stream->ConvolveBackwardDataWithAlgorithm(
        filter_desc, filter_ptr, output_desc, out_backprop_ptr, conv_desc,
        input_desc, &in_backprop_ptr, &scratch_allocator, algorithm_config,
        nullptr);
  }

  if (!cudnn_launch_status.ok()) {
    ctx->SetStatus(cudnn_launch_status);
    return;
  }

  if (padding_top != padding_bottom || padding_left != padding_right) {
    Tensor in_backprop_remove_padding;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 ShapeFromFormat(compute_data_format,
                                 GetTensorDim(input_shape, data_format, 'N'),
                                 GetTensorDim(input_shape, data_format, 'H'),
                                 GetTensorDim(input_shape, data_format, 'W'),
                                 GetTensorDim(input_shape, data_format, 'C')),
                 &in_backprop_remove_padding));

    // Remove the padding that was added to the input shape above.
    const int64 input_pad_top = padding_top - common_padding_rows;
    const int64 input_pad_bottom = padding_bottom - common_padding_rows;
    const int64 input_pad_left = padding_left - common_padding_cols;
    const int64 input_pad_right = padding_right - common_padding_cols;
    functor::PadInput<GPUDevice, T, int, 4>()(
        ctx->template eigen_device<GPUDevice>(),
        To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                    .tensor<T, 4>()),
        {{static_cast<int>(-input_pad_top), static_cast<int>(-input_pad_left)}},
        {{static_cast<int>(-input_pad_bottom),
          static_cast<int>(-input_pad_right)}},
        To32Bit(in_backprop_remove_padding.tensor<T, 4>()), compute_data_format,
        T{});

    pre_transformed_in_backprop = in_backprop_remove_padding;
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the output tensor back from NCHW to NHWC.";
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        toConstTensor(pre_transformed_in_backprop).template tensor<T, 4>(),
        in_backprop->tensor<T, 4>());
  } else {
    *in_backprop = pre_transformed_in_backprop;
  }
}

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(               \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,         \
      typename TTypes<T, 4, int>::ConstTensor in,                       \
      typename TTypes<T, 4, int>::Tensor out);                          \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;         \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 4>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,   \
      const std::array<int, 2>& padding_left,                           \
      const std::array<int, 2>& padding_right,                          \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format, \
      const T& padding_value);                                          \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

template <>
void SpatialConvolutionBackwardInputFunc<GPUDevice, int32>::operator()(
    const GPUDevice&, typename TTypes<int32, 4>::Tensor,
    typename TTypes<int32, 4>::ConstTensor,
    typename TTypes<int32, 4>::ConstTensor, Eigen::DenseIndex,
    Eigen::DenseIndex, Eigen::DenseIndex, Eigen::DenseIndex);
extern template struct SpatialConvolutionBackwardInputFunc<GPUDevice, int32>;

template <>
void SpatialConvolutionBackwardInputWithExplicitPaddingFunc<
    GPUDevice, int32>::operator()(const GPUDevice&,
                                  typename TTypes<int32, 4>::Tensor,
                                  typename TTypes<int32, 4>::ConstTensor,
                                  typename TTypes<int32, 4>::ConstTensor,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex);
extern template struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc<
    GPUDevice, int32>;

}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, int32>);

// To be used inside depthwise_conv_grad_op.cc.
// TODO(reedwm): Move this and the definition to depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropInputOp<GPUDevice, float>;
template struct LaunchConv2DBackpropInputOp<GPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropInputOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_VE
template <typename T>
struct LaunchConv2DBackpropInputOp<VEDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    VLOG(2) << "LaunchConv2DBackpropInputOp<VEDevice, T>";
    VLOG(2) << "LaunchConv2DBackpropInputOp<VEDevice, T>: DeviceContext=" << ctx->op_device_context();

    if (padding == EXPLICIT) {
      ctx->SetStatus(errors::Unimplemented(
          "VE conv implementation only supports SAME/VALID padding."
	  "Explicit padding is specified."));
      return;
    }

    const int64 batch = GetTensorDim(*in_backprop, data_format, 'N');

    const int64 out_rows = GetTensorDim(out_backprop, data_format, 'H');
    const int64 out_cols = GetTensorDim(out_backprop, data_format, 'W');
    const int64 out_depths = GetTensorDim(out_backprop, data_format, 'C');

    const int64 f_rows = GetTensorDim(filter, FORMAT_HWCN, 'H');
    const int64 f_cols = GetTensorDim(filter, FORMAT_HWCN, 'W');
    const int64 f_ic  = GetTensorDim(filter, FORMAT_HWCN, 'C');
    const int64 f_oc  = GetTensorDim(filter, FORMAT_HWCN, 'N');

    const int64 in_rows = GetTensorDim(*in_backprop, data_format, 'H');
    const int64 in_cols = GetTensorDim(*in_backprop, data_format, 'W');
    const int64 in_depths = GetTensorDim(*in_backprop, data_format, 'C');

    int row_padding = 0;
    int col_padding = 0;
    int data_type = in_backprop->dtype();

    if (padding == SAME) {
      const int row_pad_all = std::max<int>(0,
					    (out_rows - 1) * row_stride +
					    (f_rows - 1) * row_dilation + 1 - in_rows );
      const int col_pad_all = std::max<int>(0,
					    (out_cols - 1) * col_stride +
					    (f_cols - 1) * col_dilation + 1 - in_cols );

      row_padding = row_pad_all - row_pad_all/2 ;
      col_padding = col_pad_all - col_pad_all/2 ;
    }

    VEOpKernelHelper::ArgsImpl<> args;

    /* Filter */
    Tensor filter_transposed ;
    if( f_ic * f_oc == 1 ) {
      OP_REQUIRES(ctx, filter_transposed.CopyFrom(filter, ShapeFromFormat(FORMAT_NCHW, f_oc, f_rows, f_cols, f_ic)),
                  errors::Internal("Error during reshape and copy."));
    }
    else {
      OP_REQUIRES_OK(ctx,
		     ctx->allocate_temp(
			 reinterpret_cast<DataType>(filter.dtype()),
			 ShapeFromFormat(FORMAT_NCHW, f_oc, f_rows, f_cols, f_ic),
			 &filter_transposed));

      OP_REQUIRES_OK( ctx, VEDoTranspose(ctx, filter, {3,2,0,1}, &filter_transposed));
    }

    /* Output_backprop, In_backprop */
    Tensor in_bp_transposed, out_bp_transposed;
    if ( data_format == FORMAT_NHWC ) {
      // if data_format is NHWC, then allocate in/out tensor and transpose out tensor

      OP_REQUIRES_OK(ctx,
		     ctx->allocate_temp(
			 reinterpret_cast<DataType>(in_backprop->dtype()),
			 ShapeFromFormat(FORMAT_NCHW, batch, out_rows, out_cols, out_depths),
                         &out_bp_transposed));

      OP_REQUIRES_OK(ctx,
		     ctx->allocate_temp(
			 reinterpret_cast<DataType>(in_backprop->dtype()),
			 ShapeFromFormat(FORMAT_NCHW, batch, in_rows, in_cols, in_depths),
                         &in_bp_transposed)
		     );

      OP_REQUIRES_OK( ctx, VEDoTranspose(ctx, out_backprop, {0,3,1,2}, &out_bp_transposed));

      args.addArg<Tensor>(out_bp_transposed) ;	// 0
      args.addArg(filter_transposed) ;		// 1
      args.addArg<Tensor>(in_bp_transposed) ;	// 2
    }
    else {
      args.addArg<Tensor>(out_backprop) ;	// 0
      args.addArg(filter_transposed) ;		// 1
      args.addArg<Tensor>(*in_backprop) ;	// 2
    }

    /* conv params */
    args.addArg<int>(row_stride);               // 3
    args.addArg<int>(col_stride);               // 4
    args.addArg<int>(row_dilation);             // 5
    args.addArg<int>(col_dilation);             // 6
    args.addArg<int>(row_padding);              // 7
    args.addArg<int>(col_padding);              // 8

    VEOpKernelHelper::Call(ctx, "Conv2DBackpropInput", args);

    if ( data_format == FORMAT_NHWC ) {
      // if data_format is NHWC, then transpose in tensor
      OP_REQUIRES_OK( ctx, VEDoTranspose(ctx, in_bp_transposed, {0,2,3,1}, in_backprop));
    }

  }
};

template <typename Device, class T>
class Conv2DVEBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DVEBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    int stride_h = GetTensorDim(strides_, data_format_, 'H');
    int stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));


    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv2DFastBackpropInput", /*num_spatial_dims=*/2,
                       input_shape, filter.shape(), out_backprop.shape(),
                       strides_, padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));


    LaunchConv2DBackpropInputOp<Device, T>()(
        context, false, false, out_backprop, filter,
        /*row_dilation=*/1, /*col_dilation=*/1, dims.spatial_dims[0].stride,
        dims.spatial_dims[1].stride, padding_, 
        explicit_paddings_, in_backprop, data_format_);
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  std::vector<int64> explicit_paddings_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                        .Device(DEVICE_VE)
                        .TypeConstraint<float>("T")
                        .HostMemory("input_sizes"),
                        Conv2DVEBackpropInputOp<VEDevice, float>);

template struct LaunchConv2DBackpropInputOp<VEDevice, float>;
#endif // TENSORFLOW_USE_VE

}  // namespace tensorflow

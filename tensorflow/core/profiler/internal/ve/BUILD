load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "device_tracer",
    srcs = [
        "ve_tracer.cc"
        ],
    deps = [
        #"//tensorflow/core/common_runtime:core_cpu_impl",
        #"//tensorflow/core/common_runtime:ve_runtime",
        #"//third_party/veoffload",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",

    ],
    alwayslink = 1,
)

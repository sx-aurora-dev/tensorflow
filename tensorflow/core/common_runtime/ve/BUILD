load("//third_party/veoffload:build_defs.bzl", "if_ve")

cc_library(
    name = "ve_runtime_impl",
    srcs = [
        "ve_device.cc",
        ],
    deps = [
        "//tensorflow/core:core_cpu_impl",
        "//third_party/veoffload",
        ":ve_runtime",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 0,
)

cc_library(
    name = "ve_runtime",
    hdrs = [
        "ve_device.h",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 0,
)


licenses(["unencumbered"])

config_setting(
    name = "using_ve",
    values = {
      "define": "using_ve=true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
  name = "veoffload",
  #srcs = ["libveo.so"],
  includes = ["include"],
  linkopts = ["-Wl,-rpath=/opt/nec/ve/veos/lib64", "-L/opt/nec/ve/veos/lib64", "-lveo"],
  visibility = ["//visibility:public"],
)

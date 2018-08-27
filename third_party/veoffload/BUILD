licenses(["unencumbered"])

cc_library(
  name = "veoffload",
  #srcs = ["libveo.so"],
  includes = ["include"],
  linkopts = ["-L/opt/nec/ve/veos/lib64", "-lveo -lveo_mt"],
  visibility = ["//visibility:public"],
)

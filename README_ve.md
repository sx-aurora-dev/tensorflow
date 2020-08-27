# TensroFlow with VE support

You can use prebuilt packages if you do not need to modify tensorflow.

## Using prebuilt packages

We are providing a whl package on github. See [releases](https://github.com/sx-aurora-dev/tensorflow/releases) page.

We have tested on CentOS 7.7 and:

- veos: 2.5.0
- veoffload: 2.5.0
- python: 3.6

### Enable Huge Page for DMA

If huge page is enabled on VH, data is transfered using [VE
DMA](https://veos-sxarr-nec.github.io/libsysve/group__vedma.html).  Here is an
example to enable huge pages.

    % cat /etc/sysctl.d/90-hugepage.conf
    vm.nr_hugepages=1024

### Install required packages

```
% yum install centos-release-scl
% yum install rh-python36 veoffload veoffload-veorun
```

### Create virtualenv with python 3.6

Create virtualenv and update packages after enabling scl, then install prebuilt
packages.

```
$ scl enable rh-python36 bash
$ virtualenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)$ pip install -U six numpy wheel setuptools
(tmp)% pip install -U tensorflow_ve-2.1.0-cp36-cp36m-linux_x86_64.whl
```

Now you can run your scripts.

Important note: Some kernels for VE such as conv2d are optimized for NCHW data
format, while default format of TF is NHWC.  You may need to rewrite your TF
program to support NCHW format.


## Building TensorFlow

We have tested on above envirionment with:

- bazel 3.1.0
- gcc 8.3.1 (devtoolset-8)
- git 2.9.3 (rh-git29)


### Setup

Install required packages and create virtualenv as described above. In
addition, you have to install some packages.

```
$ yum install devtoolset-8 rh-git29 veoffload-devel veoffload-veorun-devel
```

Install bazel.

```
% cd /etc/yum.repos.d
% wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
% yum install bazel
```

If you can not find the specific version of bazel, see https://github.com/vbatts/copr-build-bazel.

### Build tensorflow

Build tensorflow with scl and virtualenv.

```
$ scl enable rh-python36 devtoolset-8 rh-git29 bash
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)% pip install keras-preprocessing
(tmp)% ./configure # answer N for all questions. You can probably ignore an error on getsitepackages.
(tmp)% BAZEL_LINKLIBS="-lstdc++" BAZEL_LINKOPTS="" bazel build --jobs 24 --config=ve --config=opt $* //tensorflow/tools/pip_package:build_pip_package
(tmp)% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --project_name tensorflow_ve .
```

You can see a tensorflow package in current direcotry.

We need BAZEL_LINKLIBS and BAZEL_LINKOPTS. See https://github.com/bazelbuild/bazel/issues/10327.

## (option) Build veorun_tf

`veorun_tf` is an executable for VE and includes kernel implementaions that are
called from tf running on CPU through veoffload.

Prebuilt veorun_tf is included in source tree of tf and whl packages. If you
want to add new kernels or write more efficient kernels, you can build
veorun_tf by yourself.

llvm-ve is required to build veorun_tf because intrinsic functions provided by
llvm-ve are used to write efficient kernels.

You can install the llvm-ve rpm package. See
https://github.com/sx-aurora-dev/llvm-project.

```
(tmp)% cd <working directory>
(tmp)% git clone https://github.com/sx-aurora-dev/vetfkernel vetfkernel
(tmp)% cd vetfkernel
(tmp)% (mkdir build && cd build && cmake3 -DUSE_PREBUILT_VEDNN=ON .. && make)
```

You can specify version of ncc/nc++.

```
(tmp)% (cd build && cmake3 \
        -DNCC=/opt/nec/ve/bin/ncc-3.0.1 \
        -DNCXX=/opt/nec/ve/bin/nc++-3.0.1 .. && make)
```

Your veorun_tf can be used by setting VEORUN_BIN.

```
(tmp)% VEORUN_BIN=<path to your veorun_tf> python ...
```

We have tested on above envirionment with:

- llvm-ve 1.11.0
- ncc/nc++ 3.0.1


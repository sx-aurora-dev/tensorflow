# Tensroflow for VE

Prebuild packages will be available soon.

## Building TensorFlow

We have tested on CentOS 7.5 and:

- ncc/nc++: 2.1.27
- llvm/clang: 1.1.0
- veos: 2.0.3
- veoffload: 2.0.3
- bazel: 0.25.2
- python: 3.6

We have installed VEOS and veoffload using [VEOS yum Repository on the
Web](https://sx-aurora.github.io/posts/VEOS-yum-repository/).

### Setup

#### Enable Huge Page for DMA

If huge page is enabled on VH, data is transfered using [VE
DMA](https://veos-sxarr-nec.github.io/libsysve/group__vedma.html).  Here is an
example to enable huge pages.

    % echo 1024 > /proc/sys/vm/nr_hugepages

#### Install required packages

```
% yum install centos-releas-scl
% yum install rh-python36 devtoolset-8 rh-git29 veoffload-devel veoffload-veorun-devel
```

Install java-11-openjdk that is required by bazel.

- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-11.0.3.7-0.el7_6.x86_64.rpm
- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-devel-11.0.3.7-0.el7_6.x86_64.rpm
- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-headless-11.0.3.7-0.el7_6.x86_64.rpm

Install bazel. 0.24.1 or higher is required.

```
% cd /etc/yum.repos.d
% wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
% yum install bazel
```

#### Create virtualenv with python 3.6

Create virtualenv and update packages after enabling scl.

```
$ scl enable rh-python36 devtoolset-8 rh-git29 bash
$ virtualenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)$ pip install -U six numpy wheel Keras-Preprocessing setuptools
```

### Build source code

Build tensorflow with scl and virtualenv.

```
$ scl enable rh-python36 devtoolset-8 rh-git29 bash
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)% ./configure # answer N for all questions. You can probably ignore an error on getsitepackages.
(tmp)% bazel build --config=ve --config=opt //tensorflow/tools/pip_package:build_pip_package
(tmp)% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --project_name tensorflow_ve .
```

You can see a tensorflow package in current direcotry.

Build keras.

```
(tmp)% python setup.py bdist_wheel
```

You can find a package in `dist` directory.

### Install packages 

```
(tmp)% pip install -U tensorflow_ve-1.13.1-cp36-cp36m-linux_x86_64.whl
(tmp)% pip install -U Keras-2.2.4-py3-none-any.whl
```

### Run samples

Work in the virtualenv.

```
(tmp)% cd <working directory>
(tmp)% git clone <url of sampels> samples
(tmp)% samples
(tmp)% python mnist_cnn.py
```


## (option) Build veorun_tf

`veorun_tf` is an executable for VE and includes kernel implementaions that are
called from tf running on CPU through veoffload.

Prebuild veorun_tf is included in source tree of tf and whl packages. If you
want to add new kernels or write more efficient kernels, you can build
veorun_tf by yourself.

llvm-ve is required to build veorun_tf because intrinsic functions provided by
llvm-ve are used to write efficient kernels.

You can install the rpm package for llvm-ve. See [our
post](https://sxauroratsubasaresearch.github.io/blog/post/2019-05-22-llvm-rpm/).

```
(tmp)% cd <working directory>
(tmp)% git clone <url of vetfkernel> vetfkernel
(tmp)% git clone <url of vednn> vetfkernel/libs/vednn
(tmp)% cd vetfkernel
(tmp)% (mkdir build && cd build && cmake3 -DLLVM_DIR=/opt/nec/nosupport/llvm-ve/lib/cmake/llvm .. && make)
```

You can use the llvm that is installed into different directory by setting
LLVM_DIR when you build vetfkernel. You can also specify version of ncc/nc++.

```
(tmp)% (cd build && cmake3 \
        -DLLVM_DIR=/opt/nec/nosupport/llvm-ve/lib/cmake/llvm \
        -DNCC=/opt/nec/ve/bin/ncc-2.0.8 \
        -DNCXX=/opt/nec/ve/bin/nc++-2.0.8 .. && make)
```

# TensroFlow with VE support

You can use prebuild packages if you are interested in using TF on SX-Aurora.
You can also modifiy the source code and build it.

## Using prebuild packages

We are providing two whl files on github.

- tensorflow_ve-1.13.1-cp36-cp36m-linux_x86_64.whl
- Keras-2.2.4-py3-none-any.whl

We have tested on CentOS 7.5 and:

- veos: 2.1.0
- veoffload: 2.1.0
- python: 3.6

We have installed VEOS and veoffload using [VEOS yum Repository on the
Web](https://sx-aurora.github.io/posts/VEOS-yum-repository/).

### Enable Huge Page for DMA

If huge page is enabled on VH, data is transfered using [VE
DMA](https://veos-sxarr-nec.github.io/libsysve/group__vedma.html).  Here is an
example to enable huge pages.

    % echo 1024 > /proc/sys/vm/nr_hugepages

### Install required packages

```
% yum install centos-release-scl
% yum install rh-python36 veoffload veoffload-veorun
```

### Create virtualenv with python 3.6

Create virtualenv and update packages after enabling scl, then install prebuild
packages.

```
$ scl enable rh-python36 bash
$ virtualenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)$ pip install -U six numpy wheel Keras-Preprocessing setuptools
(tmp)% pip install -U tensorflow_ve-1.13.1-cp36-cp36m-linux_x86_64.whl
(tmp)% pip install -U Keras-2.2.4-py3-none-any.whl
```

Now you can run your scripts.


## Building TensorFlow

We have tested on above envirionment with:

- bazel 0.25.2
- gcc 8.2.1 (devtoolset-8)
- git 2.9.3 (rh-git29)


### Setup

Install required packages and create virtualenv as described above. In
addition, you have to install some packages.

```
$ yum install devtoolset-8 rh-git29 veoffload-devel veoffload-veorun-devel
```

Install java-11-openjdk that is required by bazel.

- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-11.0.3.7-0.el7_6.x86_64.rpm
- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-devel-11.0.3.7-0.el7_6.x86_64.rpm
- http://mirror.centos.org/centos/7/updates/x86_64/Packages/java-11-openjdk-headless-11.0.3.7-0.el7_6.x86_64.rpm

Install bazel.

```
% cd /etc/yum.repos.d
% wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
% yum install bazel
```

### Build tensorflow

Build tensorflow with scl and virtualenv.

```
$ scl enable rh-python36 devtoolset-8 rh-git29 bash
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)% ./configure # answer N for all questions. You can probably ignore an error on getsitepackages.
(tmp)% bazel build --config=ve --config=opt //tensorflow/tools/pip_package:build_pip_package
(tmp)% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --project_name tensorflow_ve .
```

You can see a tensorflow package in current direcotry.

## (option) Build keras

Clone https://github.com/sx-aurora-dev/keras.

```
(tmp)% python setup.py bdist_wheel
```

You can find a package in `dist` directory.

## (option) Build veorun_tf

`veorun_tf` is an executable for VE and includes kernel implementaions that are
called from tf running on CPU through veoffload.

Prebuild veorun_tf is included in source tree of tf and whl packages. If you
want to add new kernels or write more efficient kernels, you can build
veorun_tf by yourself.

llvm-ve is required to build veorun_tf because intrinsic functions provided by
llvm-ve are used to write efficient kernels.

You can install the rpm package for llvm-ve. See [our
post](https://sx-aurora-dev.github.io/blog/post/2019-05-22-llvm-rpm/).

```
(tmp)% cd <working directory>
(tmp)% git clone <url of vetfkernel> vetfkernel
(tmp)% git clone <url of vednn> vetfkernel/libs/vednn
(tmp)% cd vetfkernel
(tmp)% (mkdir build && cd build && cmake3 -DLLVM_DIR=/opt/nec/nosupport/llvm-ve/lib/cmake/llvm .. && make)
```

You can specify version of ncc/nc++.

```
(tmp)% (cd build && cmake3 \
        -DNCC=/opt/nec/ve/bin/ncc-2.2.2 \
        -DNCXX=/opt/nec/ve/bin/nc++-2.2.2 .. && make)
```

Your veorun_tf can be used by setting VEORUN_BIN.

```
(tmp)% VEORUN_BIN=<path to your veorun_tf> python ...
```

We have tested on above envirionment with:

- llvm-ve 1.1.0
- ncc/nc++ 2.2.2


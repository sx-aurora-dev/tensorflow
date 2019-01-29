# Tensroflow for VE

We have tested on CentOS 7.5 and:

- ncc/nc++: 2.0.8
- llvm: a47a7dea63d3d654795d28daa2fdbf1d885ceca6
- clang: 580c9d41f8b253c1cb2011461683c91d1e4d3d2e
- veos: 2.0.1 (installed from https://sx-aurora.github.io/posts/VEOS-yum-repository)
- veoffload: Included in veos
- bazel: 0.19.2

## Setup

### Install veoffload

```
% yum install veoffload-devel veoffload-veorun-devel
```

### Install bazel

Download and install repo.

https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo

Install bazel.

```
% yum install bazel
```

If you can not find bazel-0.19.2, you can build it.

```
% wget https://vbatts.fedorapeople.org/bazel-0.19.2-1.fc28.src.rpm
% rpmbuild --rebuild bazel-0.19.2-1.fc28.src.rpm
% rpm -Uvh --oldpackage bazel-0.19.2-1.el7.centos.x86_64.rpm
```

### Create virtualenv with python 3.5

Add SCL repository.

```
% yum install centos-releas-scl
```

If this does not work, you can download and install:
- http://mirror.centos.org/centos/7/extras/x86_64/Packages/centos-release-scl-2-2.el7.centos.noarch.rpm
- http://mirror.centos.org/centos/7/extras/x86_64/Packages/centos-release-scl-rh-2-2.el7.centos.noarch.rpm

Install python 3.5.

```
% yum install \
  rh-python35-python-pip-7.1.0-2.el7.noarch \
  rh-python35-python-libs-3.5.1-11.el7.x86_64 \
  rh-python35-python-setuptools-18.0.1-2.el7.noarch \
  rh-python35-python-devel-3.5.1-11.el7.x86_64 \
  rh-python35-runtime-2.0-2.el7.x86_64 \
  rh-python35-python-3.5.1-11.el7.x86_64 \
  rh-python35-python-virtualenv-13.1.2-2.el7.noarch
```

Create virtualenv and update packages.

```
$ scl enable rh-python35 bash
$ virtualenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)$ pip install -U six numpy wheel Keras-Preprocessing
```

### Build source code

Install devtoolset.

```
% yum install devtoolset-8
```

Work in the virtualenv as below

```
$ scl enable rh-python35 devtoolset-8 bash
$ source ~/.virtualenvs/tmp/bin/activate
```

You can check version as below.

```
(tmp)$ python --version
Python 3.5.1
(tmp)$ gcc --version
gcc (GCC) 8.2.1 20180905 (Red Hat 8.2.1-3)
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Build tensorflow.

```
(tmp)% ./configure # answer N for all questions. You can probably ignore an error on getsitepackages.
(tmp)% bazel build --config=ve --config=opt //tensorflow/tools/pip_package:build_pip_package
(tmp)% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --project_name tensorflow_ve .
```

You can see tensorflow package in current direcotry.

Build keras.

```
(tmp)% python setup.py bdist_wheel
```

You can find package in `dist` directory.

### Install packages 

```
(tmp)% pip install -U tensorflow_ve-1.12.0-cp35-cp35m-linux_x86_64.whl
(tmp)% pip install -U Keras-2.2.4-py3-none-any.whl
```

### Install llvm

llvm is used to build vetfkenel and vednn.
Install llvm for VE from [[https://github.com/SXAuroraTSUBASAResearch/llvm]].

## Run samples

Work in the virtualenv.

```
(tmp)% cd <working directory>
(tmp)% git clone <url of vetfkernel> vetfkernel
(tmp)% git clone <url of vednn> vetfkernel/libs/vednn
(tmp)% git clone <url of sampels> samples
(tmp)% cd vetfkernel
(tmp)% (mkdir build && cd build && cmake3 .. && make)
(tmp)% cd ../samples
(tmp)% ln -s ../vetfkernel/build/veorun_tf veorun_tf
(tmp)% ./run.sh keras.sample.py
```

You can use the llvm that is installed into non-standard directory by setting
LLVM_DIR when you build vetfkernel. You can also specify version of ncc/nc++.

```
(tmp)% (cd build && cmake3 \
        -DLLVM_DIR=<install prefix>/lib/cmake/llvm \
        -DNCC=/opt/nec/ve/bin/ncc-2.0.8 \
        -ENCXX=/opt/nec/ve/bin/nc++-2.0.8 .. && make)
```

# TensroFlow with VE support

You can use prebuilt packages if you do not need to modify tensorflow.

[tf-samples](https://github.com/sx-aurora-dev/tf-samples) repository includes sample programs we have tested with tensorflow-ve.

## Using prebuilt packages

We are providing a whl package on github. See [releases](https://github.com/sx-aurora-dev/tensorflow/releases) page.

We have tested on CentOS8.1:

- veos: 2.7.5
- python: 3.8.3

### Enable Huge Page for DMA

If huge page is enabled on VH, data is transfered using [VE
DMA](https://veos-sxarr-nec.github.io/libsysve/group__vedma.html).  Here is an
example to enable huge pages.

    % cat /etc/sysctl.d/90-hugepage.conf
    vm.nr_hugepages=1024

### Install required packages

CentOS8

```
% yum install python38 python38-devel
```


### Create virtualenv with python 3.8

Create virtualenv and update package, then install prebuilt packages.

```
$ python3.8 -mvenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)% pip install -U tensorflow_ve-2.5.0-cp38-cp38-linux_x86_64.whl
```

Now you can run your scripts.

Important note: Some kernels for VE such as conv2d are optimized for NCHW data
format, while default format of TF is NHWC.  You may need to rewrite your TF
program to support NCHW format.


## Building TensorFlow

We have tested on above envirionment with:

- bazel 3.7.2
- gcc 8.3.1
- git 2.27.0


### Setup

Download bazel 3.7.2 from <https://github.com/bazelbuild/bazel/releases/tag/3.7.2> and install.

### Build tensorflow

Build tensorflow with virtualenv.

```
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)% pip install numpy==1.19.2 keras-preprocessing wheel
(tmp)% ./configure # answer N for all questions. You can probably ignore an error on getsitepackages.
(tmp)% BAZEL_LINKLIBS="-lstdc++" BAZEL_LINKOPTS="" bazel build --jobs 24 --config=ve --config=opt $* //tensorflow/tools/pip_package:build_pip_package
(tmp)% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --project_name tensorflow_ve .
```

You can see a tensorflow package in current direcotry.

If you have problem on http proxy, try bazel option:
`--host_jvm_args=-Djavax.net.ssl.trustStore='/etc/pki/ca-trust/extracted/java/cacerts'
--host_jvm_args=-Djavax.net.ssl.trustStorePassword='changeit'`.

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
(tmp)% git clone https://github.com/sx-aurora-dev/vml vml
(tmp)% cd vml
(tmp)% (mkdir build && cd build && cmake3 -DUSE_PREBUILT_VEDNN=ON .. && make)
```

You can specify version of ncc/nc++.

```
(tmp)% (cd build && cmake3 \
        -DNCC=/opt/nec/ve/bin/ncc-3.0.6 \
        -DNCXX=/opt/nec/ve/bin/nc++-3.0.6 .. && make)
```

Your veorun_tf can be used by setting VEORUN_BIN.

```
(tmp)% VEORUN_BIN=<path to your veorun_tf> python ...
```

We have tested on above envirionment with:

- llvm-ve 1.16.0
- ncc/nc++ 3.0.6


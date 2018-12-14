## Setup

Install veo.

~~~
% yum localinstall \
    veoffload-1.2.2e-1.el7.centos.x86_64.rpm \
    veoffload-veorun-1.2.2a-1.el7.centos.x86_64.rpm
~~~

Add SCL repository.

~~~
% yum install centos-releas-scl
~~~

If this does not work, you can download and install:
- http://mirror.centos.org/centos/7/extras/x86_64/Packages/centos-release-scl-2-2.el7.centos.noarch.rpm
- http://mirror.centos.org/centos/7/extras/x86_64/Packages/centos-release-scl-rh-2-2.el7.centos.noarch.rpm

Install python 3.5.

~~~
% yum install \
  rh-python35-python-pip-7.1.0-2.el7.noarch \
  rh-python35-python-libs-3.5.1-11.el7.x86_64 \
  rh-python35-python-setuptools-18.0.1-2.el7.noarch \
  rh-python35-python-devel-3.5.1-11.el7.x86_64 \
  rh-python35-runtime-2.0-2.el7.x86_64 \
  rh-python35-python-3.5.1-11.el7.x86_64 \
  rh-python35-python-virtualenv-13.1.2-2.el7.noarch
~~~

Create virtualenv and update packages.

~~~
$ scl enable rh-python35 bash
$ virtualenv ~/.virtualenvs/tmp
$ source ~/.virtualenvs/tmp/bin/activate
(tmp)$ pip install -U pip
(tmp)$ pip install -U six numpy wheel
~~~

Install tensorflow and keras

~~~
(tmp)% pip install -U tensorflow_ve-1.12.0rc0-cp35-cp35m-linux_x86_64.whl
(tmp)% pip install -U Keras-2.2.4-py3-none-any.whl
~~~

## Run with veorun_tf

Place veorun_tf in current directory when you run your script.


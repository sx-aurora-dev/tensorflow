# Contributing guidelines of Tensorflow-VE

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Add [Unit Tests](http://socsv218.svp.cl.nec.co.jp:3002/ve-tensorflow/tensorflow/src/branch/develop/CONTRIBUTING_ve.md#adding-unit-tests).
- Run [Unit Tests](http://socsv218.svp.cl.nec.co.jp:3002/ve-tensorflow/tensorflow/src/branch/develop/CONTRIBUTING_ve.md#running-unit-tests).

## How to become a contributor and submit your own code

### Contribution guidelines and standards

#### Adding unit tests

If you add Tensorflow Kernel, add a unit test code based on //tensorflow/python/kernel_tests to
//tensorflow/python/kernel_tests/test_ve. And modify the BUILD file in 'test_ve' directory.

#### Running unit tests

There are a way to run TensorFlow unit tests.

1. Using tools and libraries installed on your 'virtualenv' system.
   To see [Setup](http://socsv218.svp.cl.nec.co.jp:3002/ve-tensorflow/tensorflow/src/branch/develop/README_ve.md)
   before beginning TensorFlow unit tests.

   Once you have the packages installed on your 'virtualenv' system, you can run a specific unit test in
   bazel by doing as follows:

   If the tests are to be run on VE, add the `ve` option flag and add 'VEORUN_BIN' environment value
   with'--test_env' option.
   You may add '--local_test_jobs 1' option or 'VE_NODE_NUMBER' environment value to prevent conflicting of
   the test processes on VE.

   ```bash
   export flags="--config=opt --config=ve -k --test_env VEORUN_BIN=<path_to_'veorun_tf'> --local_test_jobs 1"
   ```

   For example, to run all tests under tensorflow/python/kernel_tests/test_ve, do:

   ```bash
   bazel test ${flags} //tensorflow/python/kernel_tests/test_ve/...
   ```

   If the tests are to be run on CPU, set '-1' to 'VE_NODE_NUMBER' environment value with '--test_env' option.

   For example, to run all tests under tensorflow/python do:

   ```bash
   bazel test --config=opt --config=ve --test_env VE_NODE_NUMBER=-1 //tensorflow/python/...
   ```


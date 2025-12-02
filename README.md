// ...existing code...
# MNIST - Pure C++ and Python MLP examples

This folder contains a minimal MNIST multilayer perceptron (MLP) implementation in C++ and a matching implementation in Python. Use the C++ solution for a standalone, high-performance reference and the Python code for easier experimentation and saved model exports.

Contents
- C++ train & eval:
  - [MNIST/cpp/CMakeLists.txt](MNIST/cpp/CMakeLists.txt)
  - [MNIST/cpp/mnist_mlp.cpp](MNIST/cpp/mnist_mlp.cpp)
  - Trained/evaluation outputs: [MNIST/cpp/build/epoch_accuracy.csv](MNIST/cpp/build/epoch_accuracy.csv), [MNIST/cpp/build/step_times.csv](MNIST/cpp/build/step_times.csv)
  - MNIST dataset files: [MNIST/cpp/train-images-idx3-ubyte](MNIST/cpp/train-images-idx3-ubyte), [MNIST/cpp/train-labels-idx1-ubyte](MNIST/cpp/train-labels-idx1-ubyte), [MNIST/cpp/t10k-images-idx3-ubyte](MNIST/cpp/t10k-images-idx3-ubyte), [MNIST/cpp/t10k-labels-idx1-ubyte](MNIST/cpp/t10k-labels-idx1-ubyte)

- Python training & inference:
  - [MNIST/python/mnist_mlp.py](MNIST/python/mnist_mlp.py)
  - Pretrained weights: [MNIST/python/best_weights.weights.h5](MNIST/python/best_weights.weights.h5)
  - Saved TF models: [MNIST/python/saved_mlp/](MNIST/python/saved_mlp/), [MNIST/python/saved_mlp_best/](MNIST/python/saved_mlp_best/), [MNIST/python/saved_mlp_last/](MNIST/python/saved_mlp_last/)
  - TensorBoard logs: [MNIST/python/tb_log/](MNIST/python/tb_log/)

- Third-party licensing:
  - TensorFlow C third-party licenses at [third_party/tensorflow_c/THIRD_PARTY_TF_C_LICENSES](third_party/tensorflow_c/THIRD_PARTY_TF_C_LICENSES) and [third_party/tensorflow_c/LICENSE](third_party/tensorflow_c/LICENSE). Review these if you redistribute.

Quickstart

1) Build and run the C++ trainer
- From the repository root:
  - cd MNIST/cpp
  - mkdir -p build && cd build
  - cmake ..
  - make
  - ./mnist_train
  The compiled binary is at [MNIST/cpp/build/mnist_train](MNIST/cpp/build/mnist_train). Training outputs will be written to the `build` directory (CSV logs).

2) Run the Python trainer/evaluator
- Ensure Python 3 and required packages (TensorFlow, numpy, etc.) are installed.
  - cd MNIST/python
  - python3 mnist_mlp.py --help  # shows flags/options, training, evaluate, export
  - python3 mnist_mlp.py --train  # example — check flags in script

Notes
- The C++ implementation reads data from the raw MNIST files in the `MNIST/cpp` directory; those files are present here (no download step included).
- For the C++ project, see [MNIST/cpp/CMakeLists.txt](MNIST/cpp/CMakeLists.txt) for build options.
- For the Python project, saved model assets and logs are available in [MNIST/python/saved_mlp/](MNIST/python/saved_mlp/) and [MNIST/python/tb_log/](MNIST/python/tb_log/).
- If you need to re-run experiments, check and remove existing `MNIST/cpp/build` and `MNIST/python/saved_mlp` folders as needed.

Contact
- Check the code and comments in [MNIST/cpp/mnist_mlp.cpp](MNIST/cpp/mnist_mlp.cpp) and [MNIST/python/mnist_mlp.py](MNIST/python/mnist_mlp.py) for implementation details and function-level behavior.

License
- This repository includes third-party software; see the license files under `third_party/` for details — in particular [third_party/tensorflow_c/THIRD_PARTY_TF_C_LICENSES](third_party/tensorflow_c/THIRD_PARTY_TF_C_LICENSES).
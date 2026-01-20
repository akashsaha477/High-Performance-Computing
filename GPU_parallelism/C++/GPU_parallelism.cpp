#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/platform/env.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace tensorflow;
using namespace tensorflow::ops;

/* ============================================================
   PROFILING
   ============================================================ */
struct ProfileEntry {
    std::string name;
    double time;
};
std::vector<ProfileEntry> PROFILE;

double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}
void tick(const std::string& name, double t0) {
    PROFILE.push_back({name, now() - t0});
}

/* ============================================================
   CIFAR-10 LOADER (binary format)
   ============================================================ */
void LoadCIFAR10(
    const std::string& filename,
    Tensor& images,
    Tensor& labels,
    int num_samples
) {
    auto img = images.tensor<float, 4>();
    auto lbl = labels.tensor<int64, 1>();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open CIFAR file");
    }

    for (int i = 0; i < num_samples; ++i) {
        unsigned char lab;
        file.read(reinterpret_cast<char*>(&lab), 1);
        lbl(i) = lab;

        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) {
                    unsigned char pixel;
                    file.read(reinterpret_cast<char*>(&pixel), 1);
                    img(i, y, x, c) = pixel / 255.0f;
                }
    }
}

/* ============================================================
   RESNET BLOCK
   ============================================================ */
Output ResNetBlock(
    Scope scope,
    Output input,
    int filters,
    int stride,
    bool projection,
    const std::string& name
) {
    int in_ch = input.shape().dim_size(3);

    auto w1 = Variable(scope.WithOpName(name + "_w1"),
        {3,3,in_ch,filters}, DT_FLOAT);
    auto conv1 = Conv2D(scope, input, w1,
        {1,stride,stride,1}, "SAME");
    auto bn1 = BatchNormWithGlobalNormalization(
        scope, conv1,
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        0.001f, true);
    auto r1 = Relu(scope, bn1);

    auto w2 = Variable(scope.WithOpName(name + "_w2"),
        {3,3,filters,filters}, DT_FLOAT);
    auto conv2 = Conv2D(scope, r1, w2,
        {1,1,1,1}, "SAME");
    auto bn2 = BatchNormWithGlobalNormalization(
        scope, conv2,
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        Variable(scope, {filters}, DT_FLOAT),
        0.001f, true);

    Output shortcut = input;
    if (projection || in_ch != filters) {
        auto ws = Variable(scope.WithOpName(name + "_ws"),
            {1,1,in_ch,filters}, DT_FLOAT);
        shortcut = Conv2D(scope, input, ws,
            {1,stride,stride,1}, "SAME");
    }

    return Relu(scope, Add(scope, bn2, shortcut));
}

/* ============================================================
   RESNET-18 CIFAR
   ============================================================ */
Output BuildResNet18(Scope scope, Output input, Output* logits_out) {
    auto w0 = Variable(scope, {3,3,3,64}, DT_FLOAT);
    auto x = Relu(scope, Conv2D(scope, input, w0,
        {1,1,1,1}, "SAME"));

    x = ResNetBlock(scope, x, 64, 1, false, "b1_1");
    x = ResNetBlock(scope, x, 64, 1, false, "b1_2");

    x = ResNetBlock(scope, x, 128, 2, true, "b2_1");
    x = ResNetBlock(scope, x, 128, 1, false, "b2_2");

    x = ResNetBlock(scope, x, 256, 2, true, "b3_1");
    x = ResNetBlock(scope, x, 256, 1, false, "b3_2");

    x = ResNetBlock(scope, x, 512, 2, true, "b4_1");
    x = ResNetBlock(scope, x, 512, 1, false, "b4_2");

    auto gap = Mean(scope, x, {1,2});
    auto wfc = Variable(scope, {512,10}, DT_FLOAT);
    auto logits = MatMul(scope, gap, wfc);

    *logits_out = logits;
    return Softmax(scope, logits);
}

/* ============================================================
   MAIN
   ============================================================ */
int main() {
    double t0 = now();

    Scope root = Scope::NewRootScope();

    Tensor x_train(DT_FLOAT, TensorShape({50000,32,32,3}));
    Tensor y_train(DT_INT64, TensorShape({50000}));
    Tensor x_test(DT_FLOAT, TensorShape({10000,32,32,3}));
    Tensor y_test(DT_INT64, TensorShape({10000}));

    LoadCIFAR10("data_batch_1.bin", x_train, y_train, 50000);
    LoadCIFAR10("test_batch.bin", x_test, y_test, 10000);

    tick("dataset_load", t0);

    auto X = Placeholder(root, DT_FLOAT);
    auto Y = Placeholder(root, DT_INT64);

    Output logits;
    auto preds = BuildResNet18(root, X, &logits);

    auto loss = Mean(root,
        SparseSoftmaxCrossEntropyWithLogits(root, Y, logits).loss,
        {0});

    std::vector<Output> vars;
    TF_CHECK_OK(root.graph()->GetCollection("variables", &vars));

    std::vector<Output> grads;
    TF_CHECK_OK(AddSymbolicGradients(root, {loss}, vars, &grads));

    float lr = 0.1f;
    float momentum = 0.9f;

    std::vector<Operation> train_ops;
    for (size_t i = 0; i < vars.size(); ++i) {
        train_ops.push_back(
            ApplyMomentum(root, vars[i],
                Variable(root, vars[i].shape(), DT_FLOAT),
                lr, grads[i], momentum).operation);
    }

    ClientSession session(root);
    TF_CHECK_OK(session.Run({}, nullptr));

    const int BATCH = 128;
    const int EPOCHS = 100;
    const int STEPS = 50000 / BATCH;

    for (int e = 0; e < EPOCHS; ++e) {
        for (int s = 0; s < STEPS; ++s) {
            Tensor bx(DT_FLOAT, TensorShape({BATCH,32,32,3}));
            Tensor by(DT_INT64, TensorShape({BATCH}));

            std::copy_n(
                x_train.flat<float>().data() + s*BATCH*32*32*3,
                BATCH*32*32*3,
                bx.flat<float>().data()
            );
            std::copy_n(
                y_train.flat<int64>().data() + s*BATCH,
                BATCH,
                by.flat<int64>().data()
            );

            TF_CHECK_OK(session.Run(
                {{X,bx},{Y,by}},
                {}, train_ops, nullptr));
        }
        std::cout << "Epoch " << e << " done\n";
    }

    tick("training_time", t0);

    std::cout << "\nPROFILING SUMMARY\n";
    for (auto& p : PROFILE)
        std::cout << p.name << ": " << p.time << " sec\n";

    return 0;
}
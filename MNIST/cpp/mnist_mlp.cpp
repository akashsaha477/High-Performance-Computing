/*How I organized this:

Headers & aliases

Big-endian integer reader

MNIST image loaders

Small RNG helper

MLP class (constructor, forward, softmax, backward/update, predict)

CSV logging

training loop (data shuffling, batching, logging, eval)


*/




/* lReferences I used
https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c?utm_source=chatgpt.com
https://en.cppreference.com/w/cpp/numeric/random/normal_distribution.html
https://github.com/gbiro/Cpp_MLP/tree/master


*/


#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

//IDX header integers are big-endian
int readBigEndianInt(std::ifstream &f) {

    // Read 4 bytes and convert from big-endian to host-endian 
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    
    //big-endian (MSB first)
    v = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
    return static_cast<int>(v);
}


//explicit magic checks, count  reads, single-byte reads, and normalization.

void loadMnistImages(const std::string &path, std::vector<std::vector<float>> &images) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    int magic = readBigEndianInt(f);
    int count = readBigEndianInt(f);
    int rows = readBigEndianInt(f);
    int cols = readBigEndianInt(f);
    images.resize(count, std::vector<float>(rows * cols));
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
}

void loadMnistLabels(const std::string &path, std::vector<uint8_t> &labels) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    int magic = readBigEndianInt(f);
    int count = readBigEndianInt(f);
    labels.resize(count);
    for (int i = 0; i < count; ++i) {
        unsigned char lab;
        f.read(reinterpret_cast<char*>(&lab), 1);
        labels[i] = lab;
    }
}


//distribution on the fly

float randNormal(std::mt19937 &rng, float mean = 0.0f, float stddev = 0.05f) {
    static std::normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}

struct MLP {
    int inputDim;
    int hiddenDim;
    int outputDim;
    std::vector<float> W1; // flattened: W1[i * hiddenDim + h]
    std::vector<float> b1; // biases for hidden layer (size hiddenDim)
    std::vector<float> W2; // flattened: W2 [h * outputDim + o]

    std::vector<float> b2; // biases for output layer
    float learningRate;

    //allocate and initialize weights (small Gaussian)
    MLP(int inD, int hidD, int outD, float lr, std::mt19937 &rng)
    : inputDim(inD), hiddenDim(hidD), outputDim(outD), learningRate(lr) {
        W1.resize(inputDim * hiddenDim);
        b1.resize(hiddenDim);
        W2.resize(hiddenDim * outputDim);
        b2.resize(outputDim);
        for (auto &x : W1) x = randNormal(rng);
        for (auto &x : W2) x = randNormal(rng);
        for (auto &x : b1) x = 0.0f;
        for (auto &x : b2) x = 0.0f;
    }


    // Forward: compute hidden = ReLU(x * W1 + b1)
    void forward(const std::vector<float> &x, std::vector<float> &hidden, std::vector<float> &logits) {
        hidden.assign(hiddenDim, 0.0f);
        for (int h = 0; h < hiddenDim; ++h) {
            float sum = b1[h];
            int base = h;  // W1 layout uses i*hiddenDim + h
            for (int i = 0; i < inputDim; ++i) {
                sum += x[i] * W1[i * hiddenDim + h];
            }
            // ReLU activation
            hidden[h] = std::fmax(0.0f, sum);
        }
        logits.assign(outputDim, 0.0f);
        for (int o = 0; o < outputDim; ++o) {
            float sum = b2[o];
            for (int h = 0; h < hiddenDim; ++h) {
                sum += hidden[h] * W2[h * outputDim + o];
            }
            logits[o] = sum;
        }
    }

    static void softmax_inplace(std::vector<float> &logits) {
        float maxv = logits[0];
        for (float v : logits) if (v > maxv) maxv = v;
        float s = 0.0f;
        for (auto &v : logits) { v = std::exp(v - maxv); s += v; }
        for (auto &v : logits) v /= s;
    }
    

    // Returns the scalar loss
    float backward_and_update(const std::vector<float> &x, const std::vector<float> &hidden,
                              std::vector<float> &logits, int label) {
        softmax_inplace(logits);
        float loss = -std::log(std::max(1e-8f, logits[label]));


        // gradient of loss wrt logits using oftmax derivative
        std::vector<float> dlogits(outputDim);
        for (int o = 0; o < outputDim; ++o) dlogits[o] = logits[o];
        dlogits[label] -= 1.0f;
        for (int h = 0; h < hiddenDim; ++h) {
            for (int o = 0; o < outputDim; ++o) {
                float grad = dlogits[o] * hidden[h];
                W2[h * outputDim + o] -= learningRate * grad;
            }
        }
        for (int o = 0; o < outputDim; ++o) {
            b2[o] -= learningRate * dlogits[o];
        }
        std::vector<float> dhidden(hiddenDim, 0.0f);
        for (int h = 0; h < hiddenDim; ++h) {
            float sum = 0.0f;
            for (int o = 0; o < outputDim; ++o) sum += dlogits[o] * W2[h * outputDim + o];
            //  pass gradient only if hidden>0
            dhidden[h] = (hidden[h] > 0.0f) ? sum : 0.0f;
        }
        for (int i = 0; i < inputDim; ++i) {
            for (int h = 0; h < hiddenDim; ++h) {
                float grad = dhidden[h] * x[i];
                W1[i * hiddenDim + h] -= learningRate * grad;
            }
        }
        for (int h = 0; h < hiddenDim; ++h) {
            b1[h] -= learningRate * dhidden[h];
        }
        return loss;
    }


    //returns argmax

    int predict_label(const std::vector<float> &x) {
        std::vector<float> hidden(hiddenDim);
        std::vector<float> logits(outputDim);
        forward(x, hidden, logits);
        softmax_inplace(logits);
        int best = 0;
        float bv = logits[0];
        for (int o = 1; o < outputDim; ++o) if (logits[o] > bv) { bv = logits[o]; best = o; }
        return best;
    }
};

void writeStepHeader() {
    std::ofstream f("step_times.csv");
    f << "epoch,step,data_load_ms,forward_ms,backward_ms,other_ms,total_ms\n";
}

void writeEpochHeader() {
    std::ofstream f("epoch_accuracy.csv");
    f << "epoch,train_acc,val_acc\n";
}

int main(int argc, char** argv) {
    std::string trainImages = "train-images-idx3-ubyte";
    std::string trainLabels = "train-labels-idx1-ubyte";
    std::string testImages = "t10k-images-idx3-ubyte";
    std::string testLabels = "t10k-labels-idx1-ubyte";
    int epochs = 100;
    int batchSize = 64;
    int hiddenDim = 128;
    float learningRate = 0.01f;

    if (argc >= 2) trainImages = argv[1];
    if (argc >= 3) trainLabels = argv[2];
    if (argc >= 4) testImages = argv[3];
    if (argc >= 5) testLabels = argv[4];
    if (argc >= 6) epochs = std::stoi(argv[5]);
    if (argc >= 7) batchSize = std::stoi(argv[6]);

    std::vector<std::vector<float>> trainX;
    std::vector<uint8_t> trainY;
    std::vector<std::vector<float>> testX;
    std::vector<uint8_t> testY;

    loadMnistImages(trainImages, trainX);
    loadMnistLabels(trainLabels, trainY);
    loadMnistImages(testImages, testX);
    loadMnistLabels(testLabels, testY);

    int inputDim = static_cast<int>(trainX[0].size());
    int outputDim = 10;

    std::random_device rd;
    std::mt19937 rng(rd());
    MLP model(inputDim, hiddenDim, outputDim, learningRate, rng);

    writeStepHeader();
    writeEpochHeader();

    std::ofstream stepFile("step_times.csv", std::ios::app);
    std::ofstream epochFile("epoch_accuracy.csv", std::ios::app);

    int stepsPerEpoch = static_cast<int>(trainX.size()) / batchSize;
    std::vector<int> indices(trainX.size());
    for (int i = 0; i < indices.size(); ++i) indices[i] = i;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int step = 0; step < stepsPerEpoch; ++step) {
            auto t0 = Clock::now();
            auto tData0 = Clock::now();

            std::vector<std::vector<float>> batchX(batchSize, std::vector<float>(inputDim));
            std::vector<int> batchY(batchSize);
            int start = step * batchSize;
            for (int i = 0; i < batchSize; ++i) {
                int idx = indices[start + i];
                batchX[i] = trainX[idx];
                batchY[i] = static_cast<int>(trainY[idx]);
            }

            auto tData1 = Clock::now();
            auto tFwd0 = Clock::now();

            std::vector<float> hidden(model.hiddenDim);
            std::vector<float> logits(model.outputDim);
            float batchLoss = 0.0f;
            for (int i = 0; i < batchSize; ++i) {
                model.forward(batchX[i], hidden, logits);
                batchLoss += model.backward_and_update(batchX[i], hidden, logits, batchY[i]);
            }
            auto tFwd1 = Clock::now();
            auto tBwd0 = Clock::now();
            auto tBwd1 = Clock::now();

            auto t1 = Clock::now();

            double data_ms = ms(tData1 - tData0).count();
            double forward_ms = ms(tFwd1 - tFwd0).count();
            double backward_ms = ms(tBwd1 - tBwd0).count();
            double total_ms = ms(t1 - t0).count();
            double other_ms = total_ms - (data_ms + forward_ms + backward_ms);

            stepFile << epoch << "," << step+1 << ","
                     << std::fixed << std::setprecision(3)
                     << data_ms << "," << forward_ms << "," << backward_ms << "," << other_ms << "," << total_ms << "\n";

            if ((step+1) % 100 == 0) {
                std::cout << "epoch " << epoch << " step " << (step+1) << " total_ms=" << total_ms << "\n";
            }
        }

        int correctTrain = 0;
        int sampleTrain = std::min(1000, static_cast<int>(trainX.size()));
        for (int i = 0; i < sampleTrain; ++i) {
            int pred = model.predict_label(trainX[i]);
            if (pred == trainY[i]) ++correctTrain;
        }
        double trainAcc = static_cast<double>(correctTrain) / sampleTrain;

        int correctVal = 0;
        int sampleVal = std::min(1000, static_cast<int>(testX.size()));
        for (int i = 0; i < sampleVal; ++i) {
            int pred = model.predict_label(testX[i]);
            if (pred == testY[i]) ++correctVal;
        }
        double valAcc = static_cast<double>(correctVal) / sampleVal;

        epochFile << epoch << "," << std::fixed << std::setprecision(6) << trainAcc << "," << valAcc << "\n";
        std::cout << "Epoch " << epoch << " train_acc=" << trainAcc << " val_acc=" << valAcc << "\n";
    }

    stepFile.close();
    epochFile.close();
    return 0;
}
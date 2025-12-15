#include <tensorflow/c/c_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdint>

using namespace std;
using Clock = chrono::high_resolution_clock;



const int EPOCHS = 100;
const int H = 28, W = 28, C = 1;
const int NUM_CLASSES = 10;
const int N_EVAL = 1000;



double now() {
    return chrono::duration<double>(Clock::now().time_since_epoch()).count();
}



void NoOpDeallocator(void*, size_t, void*) {}



int readBigEndianInt(ifstream &f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    v = ((v & 0xFF) << 24) |
        ((v & 0xFF00) << 8) |
        ((v & 0xFF0000) >> 8) |
        ((v & 0xFF000000) >> 24);
    return static_cast<int>(v);
}

void loadMnistImages(const string &path, vector<vector<float>> &images) {
    ifstream f(path, ios::binary);
    if (!f) { cerr << "Cannot open " << path << endl; exit(1); }

    readBigEndianInt(f);
    int n = readBigEndianInt(f);
    int r = readBigEndianInt(f);
    int c = readBigEndianInt(f);

    images.resize(n, vector<float>(r * c));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < r * c; j++) {
            unsigned char px;
            f.read((char*)&px, 1);
            images[i][j] = px / 255.0f;
        }
}

void loadMnistLabels(const string &path, vector<int> &labels) {
    ifstream f(path, ios::binary);
    if (!f) { cerr << "Cannot open " << path << endl; exit(1); }

    readBigEndianInt(f);
    int n = readBigEndianInt(f);
    labels.resize(n);

    for (int i = 0; i < n; i++) {
        unsigned char v;
        f.read((char*)&v, 1);
        labels[i] = v;
    }
}



int main() {

  

    const string MODEL_DIR =
        "/Users/akashsaha/Desktop/High-Performance-Computing/CNN_Profiling/Cpp/cnn_capi_export";

    const string TEST_IMAGES = "data/t10k-images-idx3-ubyte";
    const string TEST_LABELS = "data/t10k-labels-idx1-ubyte";

   

    const char* INPUT_OP  = "serve_keras_tensor";
    const char* OUTPUT_OP = "StatefulPartitionedCall";

    

    vector<int> BATCH_SIZES = {32, 64, 128, 256, 512, 1024};

   

    vector<vector<float>> images;
    vector<int> labels;

    double t0 = now();
    loadMnistImages(TEST_IMAGES, images);
    loadMnistLabels(TEST_LABELS, labels);
    double t1 = now();

    

    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();

    const char* tags[] = {"serve"};

    TF_Session* session = TF_LoadSessionFromSavedModel(
        opts, nullptr,
        MODEL_DIR.c_str(),
        tags, 1,
        graph, nullptr,
        status
    );

    if (TF_GetCode(status) != TF_OK) {
        cerr << "Model load failed: " << TF_Message(status) << endl;
        return 1;
    }

    TF_Operation* input_op  = TF_GraphOperationByName(graph, INPUT_OP);
    TF_Operation* output_op = TF_GraphOperationByName(graph, OUTPUT_OP);

    if (!input_op || !output_op) {
        cerr << "ERROR: Input/Output op not found" << endl;
        return 1;
    }

    

    ofstream ecsv("C++_eval_data.csv");
    ecsv << "label";
    for (int i = 0; i < H * W; i++) ecsv << ",p" << i;
    ecsv << "\n";

    int evalN = min(N_EVAL, (int)images.size());
    for (int i = 0; i < evalN; i++) {
        ecsv << labels[i];
        for (float px : images[i])
            ecsv << "," << fixed << setprecision(6) << px;
        ecsv << "\n";
    }
    ecsv.close();

    

    for (int BS : BATCH_SIZES) {

        cout << "\n===== Batch Size = " << BS << " =====\n";

        ofstream acc_csv("C++_accuracy_bs_" + to_string(BS) + ".csv");
        acc_csv << "epoch,train_accuracy,val_accuracy,epoch_time_s\n";

        ofstream epoch_csv("C++_epoch_timings_bs_" + to_string(BS) + ".csv");
        epoch_csv << "epoch,epoch_time_s\n";

        vector<pair<string,double>> timings;
        timings.push_back({"data load", t1 - t0});
        timings.push_back({"model load", 0.0});

        

        double inf0 = now();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {

            int correct = 0;
            int total = labels.size();
            double e0 = now();

            for (int i = 0; i < total; i += BS) {

                int bs = min(BS, total - i);
                vector<float> batch(bs * H * W * C);

                for (int b = 0; b < bs; b++)
                    for (int p = 0; p < H * W; p++)
                        batch[b * H * W + p] = images[i + b][p];

                int64_t dims[4] = {bs, H, W, C};

                TF_Tensor* input_tensor = TF_NewTensor(
                    TF_FLOAT, dims, 4,
                    batch.data(),
                    batch.size() * sizeof(float),
                    &NoOpDeallocator, nullptr
                );

                TF_Output inputs[]  = {{input_op, 0}};
                TF_Output outputs[] = {{output_op, 0}};
                TF_Tensor* output_tensor = nullptr;

                TF_SessionRun(
                    session, nullptr,
                    inputs, &input_tensor, 1,
                    outputs, &output_tensor, 1,
                    nullptr, 0, nullptr,
                    status
                );

                float* out = static_cast<float*>(TF_TensorData(output_tensor));

                for (int b = 0; b < bs; b++) {
                    int pred = max_element(
                        out + b * NUM_CLASSES,
                        out + (b + 1) * NUM_CLASSES
                    ) - (out + b * NUM_CLASSES);

                    if (pred == labels[i + b]) correct++;
                }

                TF_DeleteTensor(input_tensor);
                TF_DeleteTensor(output_tensor);
            }

            double e1 = now();
            double acc = (double)correct / total;
            double et = e1 - e0;

            acc_csv << epoch << ",,"
                    << fixed << setprecision(6)
                    << acc << "," << et << "\n";

            epoch_csv << epoch << ","
                      << fixed << setprecision(6)
                      << et << "\n";

            cout << "Epoch " << epoch
                 << " val_acc=" << acc
                 << " time=" << et << "s\n";
        }

        double inf1 = now();
        timings.push_back({"inference_total", inf1 - inf0});

        acc_csv.close();
        epoch_csv.close();

        ofstream tcsv("C++_timings_bs_" + to_string(BS) + ".csv");
        tcsv << "step,time_s\n";
        for (auto &r : timings)
            tcsv << r.first << "," << fixed << setprecision(6) << r.second << "\n";
        tcsv.close();
    }

    

    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    cout << "\nAll batch-size experiments completed.\n";
    return 0;
}
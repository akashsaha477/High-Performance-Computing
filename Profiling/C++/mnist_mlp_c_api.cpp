/* References I used for coding
https://github.com/probablygab/nano-nn
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h

*/


/* reads MNIST IDX test files

loads a TensorFlow SavedModel using the C API

runs batched inference

computes accuracy and saves to CSV

*/

#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdint>

using namespace std;

// Update these paths to match your local setup
const string MODEL_DIR = "./saved_mlp"; 
const string TEST_IMAGES_PATH = "./t10k-images-idx3-ubyte"; 
const string TEST_LABELS_PATH = "./t10k-labels-idx1-ubyte";

const int BATCH_SIZE = 100;
const int IMAGE_SIZE = 784;
const int NUM_CLASSES = 10;
const int NUM_TEST_SAMPLES = 10000;

//CSV file and writes: Total samples processed Number of correct predictions and Final accuracy
void write_accuracy_csv(float acc, int total, int correct) {
    ofstream file("mnist_accuracy.csv");
    if (file) {
        file << "Metric,Value\n";
        file << "Total Samples," << total << "\n";
        file << "Correct Predictions," << correct << "\n";
        file << "Accuracy," << fixed << setprecision(4) << acc << "\n";
        cout << "Saved accuracy data to mnist_accuracy.csv" << endl;
    }
}

//records inference speed
void write_profiler_csv(const vector<double>& latencies) {
    ofstream file("inference_profiling_cpp.csv");
    if (file) {
        // Headers matched to Python script for comparison
        file << "batch_id,batch_size,latency_ms\n";
        for (size_t i = 0; i < latencies.size(); ++i) {
            file << (i + 1) << "," << BATCH_SIZE << "," << latencies[i] << "\n";
        }
        cout << "Saved profiler data to inference_profiling_cpp.csv" << endl;
    }
}

//converts big-endian format numbers to little-endian
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val << 8)  & 0x00ff0000) |
           ((val >> 8)  & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

//TensorFlow C API uses raw float arrays , so it converts pixel values to binary amd saves it to  vectors
struct MNISTData {
    vector<vector<float>> images;
    vector<int> labels;
};

MNISTData load_mnist_data() {
    MNISTData data;
    ifstream image_file(TEST_IMAGES_PATH, ios::binary);
    ifstream label_file(TEST_LABELS_PATH, ios::binary);

    if (!image_file || !label_file) {
        cerr << "Error: Dataset files not found." << endl;
        exit(1);
    }

    uint32_t magic, num, rows, cols, l_magic, l_num;
    image_file.read((char*)&magic, 4); image_file.read((char*)&num, 4);
    image_file.read((char*)&rows, 4);  image_file.read((char*)&cols, 4);
    label_file.read((char*)&l_magic, 4); label_file.read((char*)&l_num, 4);

    num = swap_endian(num); rows = swap_endian(rows); cols = swap_endian(cols);

    data.images.resize(num, vector<float>(rows * cols));
    data.labels.resize(num);

    cout << "Loading " << num << " images..." << endl;

    for (uint32_t i = 0; i < num; ++i) {
        unsigned char label;
        label_file.read((char*)&label, 1);
        data.labels[i] = (int)label;

        // Read whole row at once for speed
        vector<unsigned char> buf(rows * cols);
        image_file.read((char*)buf.data(), rows * cols);
        for (size_t p = 0; p < buf.size(); ++p) {
            data.images[i][p] = buf[p] / 255.0f;
        }
    }
    return data;
}

//TensorFlow C API requires a deallocator function for tensors.
//But I am passing memory owned by C++ containers, so TF not free it.
void NoOpDeallocator(void* data, size_t a, void* b) {}

int main() {
    // Loading SavedModel
    MNISTData mnist = load_mnist_data();
    
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Graph* graph = TF_NewGraph();
    const char* tags[] = {"serve"};

    TF_Session* session = TF_LoadSessionFromSavedModel(
        sess_opts, NULL, MODEL_DIR.c_str(), tags, 1, graph, NULL, status
    );

    if (TF_GetCode(status) != TF_OK) {
        cerr << "Error loading model: " << TF_Message(status) << endl;
        return 1;
    }
    cout << "Model loaded successfully." << endl;

    // Try standard TF2.x input/output names
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_keras_tensor"), 0};
    if (!input_op.oper) input_op = {TF_GraphOperationByName(graph, "serve_keras_tensor"), 0};
    
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    if (!input_op.oper || !output_op.oper) {
        cerr << "Error: Could not find operations." << endl; 
        return 1;
    }

    int correct_predictions = 0;
    int total_processed = 0;
    vector<double> batch_times;
    
    // Pre-allocate batch buffer
    vector<float> batch_input_data(BATCH_SIZE * IMAGE_SIZE);
    int64_t dims[] = {BATCH_SIZE, IMAGE_SIZE};

    cout << "Starting Inference on " << NUM_TEST_SAMPLES << " samples..." << endl;

    for (int i = 0; i < NUM_TEST_SAMPLES; i += BATCH_SIZE) {
        
        /*inference:

        Convert MNIST batch into a flat float array

        Wrap it into a TF tensor

        Run the model using TF_SessionRun

        Collect predictions

        Update accuracy counters

        Measure time for profiling*/

        // 1. Prepare Batch
        for (int b = 0; b < BATCH_SIZE; ++b) {
            if (i + b < NUM_TEST_SAMPLES) {
                copy(mnist.images[i+b].begin(), mnist.images[i+b].end(), 
                     batch_input_data.begin() + (b * IMAGE_SIZE));
            } else {
                fill(batch_input_data.begin() + (b * IMAGE_SIZE), 
                     batch_input_data.begin() + ((b+1) * IMAGE_SIZE), 0.0f);
            }
        }

        TF_Tensor* input_tensor = TF_NewTensor(
            TF_FLOAT, dims, 2, batch_input_data.data(), 
            batch_input_data.size() * sizeof(float), &NoOpDeallocator, 0
        );

        TF_Output inputs[] = {input_op};
        TF_Tensor* input_values[] = {input_tensor};
        TF_Output outputs[] = {output_op};
        TF_Tensor* output_values[] = {nullptr};

        auto start = chrono::high_resolution_clock::now();

        TF_SessionRun(session, NULL, inputs, input_values, 1, outputs, output_values, 1, NULL, 0, NULL, status);

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        batch_times.push_back(ms);

        if (TF_GetCode(status) != TF_OK || !output_values[0]) {
            cerr << "Inference error: " << TF_Message(status) << endl;
            TF_DeleteTensor(input_tensor);
            continue;
        }

        // Check Predictions
        float* out_data = static_cast<float*>(TF_TensorData(output_values[0]));
        for (int b = 0; b < BATCH_SIZE; ++b) {
            if (i + b >= NUM_TEST_SAMPLES) break;
            
            float* logits = out_data + (b * NUM_CLASSES);
            int pred = distance(logits, max_element(logits, logits + NUM_CLASSES));
            
            if (pred == mnist.labels[i + b]) correct_predictions++;
            total_processed++;
        }

        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(output_values[0]);

        if ((i / BATCH_SIZE) % 10 == 0) cout << "Processed batch " << (i / BATCH_SIZE) << "..." << endl;
    }

    // Accuracy Output + Cleanup
    float acc = total_processed ? (float)correct_predictions / total_processed : 0.0f;
    
    cout << "\nResults:\nAccuracy: " << acc * 100.0f << "%\n";
    
    write_accuracy_csv(acc, total_processed, correct_predictions);
    write_profiler_csv(batch_times);

    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdint>

using namespace std;

const string MODEL_DIR = "mnist_saved_model";
const string TEST_IMAGES_PATH = "/Users/akashsaha/Desktop/High-Performance-Computing/MNIST/cpp/data/t10k-images-idx3-ubyte";
const string TEST_LABELS_PATH = "/Users/akashsaha/Desktop/High-Performance-Computing/MNIST/cpp/data/t10k-labels-idx1-ubyte";
const char* INPUT_OP_NAME = "serving_default_mnist_input";
const char* OUTPUT_OP_NAME = "StatefulPartitionedCall";
const int BATCH_SIZE = 100;
const int NUM_TEST_SAMPLES = 10000;
const int IMAGE_SIZE = 784;
const int NUM_CLASSES = 10;

void write_accuracy_csv(float accuracy, int total_samples, int correct_predictions) {
    ofstream file("mnist_accuracy.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open mnist_accuracy.csv" << endl;
        return;
    }
    file << "Metric,Value\n";
    file << "Total Samples," << total_samples << "\n";
    file << "Correct Predictions," << correct_predictions << "\n";
    file << "Accuracy," << fixed << setprecision(4) << accuracy << "\n";
    file.close();
    cout << "Saved accuracy data to mnist_accuracy.csv" << endl;
}

void write_profiler_csv(const vector<double>& batch_times) {
    ofstream file("mnist_profiler.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open mnist_profiler.csv" << endl;
        return;
    }
    file << "Batch_ID,Time_ms,Samples_per_Batch\n";
    double total_time = 0;
    for (size_t i = 0; i < batch_times.size(); ++i) {
        file << i + 1 << "," << batch_times[i] << "," << BATCH_SIZE << "\n";
        total_time += batch_times[i];
    }
    file << "Total," << total_time << "," << batch_times.size() * BATCH_SIZE << "\n";
    file << "Average," << total_time / batch_times.size() << "," << BATCH_SIZE << "\n";
    file.close();
    cout << "Saved profiler data to mnist_profiler.csv" << endl;
}

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

struct MNISTData {
    vector<vector<float>> images;
    vector<int> labels;
};

MNISTData load_mnist_data() {
    MNISTData data;
    ifstream image_file(TEST_IMAGES_PATH, ios::binary);
    ifstream label_file(TEST_LABELS_PATH, ios::binary);

    if (!image_file || !label_file) {
        cerr << "Error: Failed to open MNIST dataset files at 'data/'." << endl;
        cerr << "Please ensure 't10k-images-idx3-ubyte' and 't10k-labels-idx1-ubyte' exist." << endl;
        exit(1);
    }

    uint32_t magic, num_items, rows, cols;
    image_file.read((char*)&magic, 4);
    image_file.read((char*)&num_items, 4);
    image_file.read((char*)&rows, 4);
    image_file.read((char*)&cols, 4);

    num_items = swap_endian(num_items);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    uint32_t label_magic, label_num_items;
    label_file.read((char*)&label_magic, 4);
    label_file.read((char*)&label_num_items, 4);
    label_num_items = swap_endian(label_num_items);

    if (num_items != label_num_items) {
        cerr << "Mismatch in image/label counts!" << endl;
        exit(1);
    }

    cout << "Loading " << num_items << " images (" << rows << "x" << cols << ")..." << endl;

    data.images.resize(num_items, vector<float>(rows * cols));
    data.labels.resize(num_items);

    for (uint32_t i = 0; i < num_items; ++i) {
        unsigned char label;
        label_file.read((char*)&label, 1);
        data.labels[i] = (int)label;

        for (uint32_t p = 0; p < rows * cols; ++p) {
            unsigned char pixel;
            image_file.read((char*)&pixel, 1);
            data.images[i][p] = static_cast<float>(pixel) / 255.0f;
        }
    }
    return data;
}

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main() {
    MNISTData mnist = load_mnist_data();

    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = NULL;

    const char* tags[] = {"serve"};
    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = TF_LoadSessionFromSavedModel(
        sess_opts, run_opts, MODEL_DIR.c_str(), tags, 1, graph, NULL, status
    );

    if (TF_GetCode(status) != TF_OK) {
        cerr << "Error loading model: " << TF_Message(status) << endl;
        cerr << "Make sure you ran the Python script to create 'mnist_saved_model'" << endl;
        return 1;
    }
    cout << "Model loaded successfully." << endl;

    TF_Output input_op = {TF_GraphOperationByName(graph, INPUT_OP_NAME), 0};
    if (!input_op.oper) {
        cout << "Warning: input op '" << INPUT_OP_NAME << "' not found. Trying 'serving_default_input_1'..." << endl;
        input_op = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
        if(!input_op.oper) {
            cerr << "Error: Failed to find Input Operation. Use saved_model_cli to check tag names." << endl;
            return 1;
        }
    }

    TF_Output output_op = {TF_GraphOperationByName(graph, OUTPUT_OP_NAME), 0};
    if (!output_op.oper) {
        cerr << "Error: Failed to find Output Operation." << endl;
        return 1;
    }

    int correct_predictions = 0;
    vector<double> batch_times;
    int total_processed = 0;

    vector<float> batch_input_data(BATCH_SIZE * IMAGE_SIZE);
    int64_t input_dims[] = {BATCH_SIZE, 28, 28, 1};

    cout << "Starting Inference on " << NUM_TEST_SAMPLES << " samples..." << endl;

    for (int i = 0; i < NUM_TEST_SAMPLES; i += BATCH_SIZE) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
            if (i + b < NUM_TEST_SAMPLES) {
                copy(mnist.images[i+b].begin(), mnist.images[i+b].end(), batch_input_data.begin() + (b * IMAGE_SIZE));
            }
        }

        TF_Tensor* input_tensor = TF_NewTensor(
            TF_FLOAT, input_dims, 4,
            batch_input_data.data(), BATCH_SIZE * IMAGE_SIZE * sizeof(float),
            &NoOpDeallocator, 0
        );

        TF_Output inputs[] = {input_op};
        TF_Tensor* input_values[] = {input_tensor};
        TF_Output outputs[] = {output_op};
        TF_Tensor* output_values[] = {nullptr};

        auto start = chrono::high_resolution_clock::now();

        TF_SessionRun(
            session, NULL,
            inputs, input_values, 1,
            outputs, output_values, 1,
            NULL, 0, NULL, status
        );

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        batch_times.push_back(duration.count());

        if (TF_GetCode(status) != TF_OK) {
            cerr << "Error during inference: " << TF_Message(status) << endl;
            TF_DeleteTensor(input_tensor);
            continue;
        }

        float* output_data = static_cast<float*>(TF_TensorData(output_values[0]));

        for (int b = 0; b < BATCH_SIZE; ++b) {
            if (i + b >= NUM_TEST_SAMPLES) break;

            int predicted_label = 0;
            float max_prob = output_data[b * NUM_CLASSES];
            for (int c = 1; c < NUM_CLASSES; ++c) {
                if (output_data[b * NUM_CLASSES + c] > max_prob) {
                    max_prob = output_data[b * NUM_CLASSES + c];
                    predicted_label = c;
                }
            }

            if (predicted_label == mnist.labels[i + b]) {
                correct_predictions++;
            }
            total_processed++;
        }

        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(output_values[0]);

        if ((i / BATCH_SIZE) % 10 == 0) {
            cout << "Processed batch " << (i / BATCH_SIZE) << "..." << endl;
        }
    }

    float final_accuracy = (float)correct_predictions / total_processed;

    cout << "\n--- Results ---" << endl;
    cout << "Total Samples: " << total_processed << endl;
    cout << "Correct: " << correct_predictions << endl;
    cout << "Accuracy: " << final_accuracy * 100.0f << "%" << endl;

    write_accuracy_csv(final_accuracy, total_processed, correct_predictions);
    write_profiler_csv(batch_times);

    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return 0;
}
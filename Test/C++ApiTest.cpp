#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

int main() {
    // Create a new session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);

    if (!status.ok()) {
        std::cerr << "Error creating session: " << status.ToString() << std::endl;
        return 1;
    }

    std::cout << "Successfully initialized TensorFlow C++ Session!" << std::endl;

    // Close and delete the session
    session->Close();
    delete session;

    return 0;
}
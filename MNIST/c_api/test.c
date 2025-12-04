#include <stdio.h>
#include "tensorflow/c/c_api.h"

int main() {
    printf("TF version: %s\n", TF_Version());
}
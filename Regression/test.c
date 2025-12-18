#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>




TF_Tensor* ScalarFloatTensor(float v) {
    float* buf = malloc(sizeof(float));
    *buf = v;
    return TF_NewTensor(
        TF_FLOAT, NULL, 0, buf, sizeof(float),
        TensorDeallocator, NULL);
}





TF_Operation* Placeholder(TF_Graph* g, TF_Status* s, const char* name) {
    TF_OperationDescription* d =
        TF_NewOperation(g, "Placeholder", name);
    TF_SetAttrType(d, "dtype", TF_FLOAT);
    return TF_FinishOperation(d, s);
}




TF_Operation* Assign(TF_Graph* g, TF_Status* s,
                     const char* name,
                     TF_Operation* var,
                     TF_Operation* value) {
    TF_OperationDescription* d =
        TF_NewOperation(g, "Assign", name);
    TF_AddInput(d, (TF_Output){var, 0});
    TF_AddInput(d, (TF_Output){value, 0});
    return TF_FinishOperation(d, s);
}



int main() {

    FILE* loss_csv = fopen("loss.csv", "w");
    FILE* prof_csv = fopen("profile.csv", "w");
    fprintf(loss_csv, "step,loss\n");
    fprintf(prof_csv, "step,time_ms\n");

    struct timespec t0, t1;


    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();


    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    fprintf(prof_csv, "graph creation,%f\n", time_diff_ms(t0, t1));




   
    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Operation* x = Placeholder(graph, status, "input");
    TF_Operation* y = Placeholder(graph, status, "target");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "placeholders,%f\n", time_diff_ms(t0, t1));


 
   
    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Operation* W = Variable(graph, status, "W");
    TF_Operation* b = Variable(graph, status, "b");

    TF_Operation* zero = ConstFloat(graph, status, "zero", 0.0f);
    TF_Operation* initW = Assign(graph, status, "initW", W, zero);
    TF_Operation* initb = Assign(graph, status, "initb", b, zero);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "variables,%f\n", time_diff_ms(t0, t1));




    clock_gettime(CLOCK_MONOTONIC, &t0);


    TF_OperationDescription* mul_d =
        TF_NewOperation(graph, "Mul", "mul");
    TF_AddInput(mul_d, (TF_Output){W, 0});
    TF_AddInput(mul_d, (TF_Output){x, 0});
    TF_Operation* Wx = TF_FinishOperation(mul_d, status);




   
    TF_OperationDescription* add_d =
        TF_NewOperation(graph, "Add", "add");
    TF_AddInput(add_d, (TF_Output){Wx, 0});
    TF_AddInput(add_d, (TF_Output){b, 0});
    TF_Operation* y_pred = TF_FinishOperation(add_d, status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "forward graph,%f\n", time_diff_ms(t0, t1));


    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_OperationDescription* out_d =
        TF_NewOperation(graph, "Identity", "output");
    TF_AddInput(out_d, (TF_Output){y_pred, 0});
    TF_Operation* output = TF_FinishOperation(out_d, status);

  



    TF_OperationDescription* sub_d =
        TF_NewOperation(graph, "Sub", "sub");
    TF_AddInput(sub_d, (TF_Output){output, 0});
    TF_AddInput(sub_d, (TF_Output){y, 0});
    TF_Operation* diff = TF_FinishOperation(sub_d, status);

    TF_OperationDescription* sq_d =
        TF_NewOperation(graph, "Square", "square");
    TF_AddInput(sq_d, (TF_Output){diff, 0});
    TF_Operation* sq = TF_FinishOperation(sq_d, status);

    TF_Operation* axis = ConstInt(graph, status, "axis", 0);

    TF_OperationDescription* mean_d =
        TF_NewOperation(graph, "Mean", "loss");
    TF_AddInput(mean_d, (TF_Output){sq, 0});
    TF_AddInput(mean_d, (TF_Output){axis, 0});
    TF_Operation* loss = TF_FinishOperation(mean_d, status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "loss graph,%f\n", time_diff_ms(t0, t1));



    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Output ys[1] = {{loss, 0}};
    TF_Output xs[2] = {{W, 0}, {b, 0}};
    TF_Output grads[2];
    TF_AddGradients(graph, ys, 1, xs, 2, NULL, status, grads);
    Check(status);


}
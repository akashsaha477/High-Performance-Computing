#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*checks whether the TensorFlow API call
 was successful or not. */

void Check(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "TF ERROR: %s\n", TF_Message(status));
        exit(1);
    }
}

double time_diff_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 +
           (b.tv_nsec - a.tv_nsec) / 1e6;
}

void TensorDeallocator(void* data, size_t len, void* arg) {
    free(data);
}


/* Creates a scalar float tensor.*/

TF_Tensor* ScalarFloatTensor(float v) {
    float* buf = malloc(sizeof(float));
    *buf = v;
    return TF_NewTensor(
        TF_FLOAT, NULL, 0, buf, sizeof(float),
        TensorDeallocator, NULL);
}


/*Creates a scalar integer tensor*/

TF_Tensor* ScalarIntTensor(int v) {
    int* buf = malloc(sizeof(int));
    *buf = v;
    return TF_NewTensor(
        TF_INT32, NULL, 0, buf, sizeof(int),
        TensorDeallocator, NULL);
}


/*placeholder operation in the graph. */

TF_Operation* Placeholder(TF_Graph* g, TF_Status* s, const char* name) {
    TF_OperationDescription* d =
        TF_NewOperation(g, "Placeholder", name);
    TF_SetAttrType(d, "dtype", TF_FLOAT);
    return TF_FinishOperation(d, s);
}



/*initializing variables and learning rate. (constant float) */

TF_Operation* ConstFloat(TF_Graph* g, TF_Status* s, const char* name, float v) {
    TF_Tensor* t = ScalarFloatTensor(v);
    TF_OperationDescription* d =
        TF_NewOperation(g, "Const", name);
    TF_SetAttrTensor(d, "value", t, s);
    TF_SetAttrType(d, "dtype", TF_FLOAT);
    TF_Operation* op = TF_FinishOperation(d, s);
    TF_DeleteTensor(t);
    return op;
}



/*constant integer tensor */

TF_Operation* ConstInt(TF_Graph* g, TF_Status* s, const char* name, int v) {
    TF_Tensor* t = ScalarIntTensor(v);
    TF_OperationDescription* d =
        TF_NewOperation(g, "Const", name);
    TF_SetAttrTensor(d, "value", t, s);
    TF_SetAttrType(d, "dtype", TF_INT32);
    TF_Operation* op = TF_FinishOperation(d, s);
    TF_DeleteTensor(t);
    return op;
}


/*
 Creates a trainable variable. W and b are created using this function.

*/


TF_Operation* Variable(TF_Graph* g, TF_Status* s, const char* name) {
    TF_OperationDescription* d =
        TF_NewOperation(g, "VariableV2", name);
    TF_SetAttrType(d, "dtype", TF_FLOAT);
    TF_SetAttrShape(d, "shape", NULL, 0);
    return TF_FinishOperation(d, s);
}



/*Used for initializing W and b to zero.*/


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


    /* Placeholders for input x and target y */
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    fprintf(prof_csv, "graph creation,%f\n", time_diff_ms(t0, t1));




    /* Trainable parameters of the linear model:
     y = W*x + b */
    
    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Operation* x = Placeholder(graph, status, "input");
    TF_Operation* y = Placeholder(graph, status, "target");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "placeholders,%f\n", time_diff_ms(t0, t1));


     /* Initialize W and b to zero*/
   
    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Operation* W = Variable(graph, status, "W");
    TF_Operation* b = Variable(graph, status, "b");

    TF_Operation* zero = ConstFloat(graph, status, "zero", 0.0f);
    TF_Operation* initW = Assign(graph, status, "initW", W, zero);
    TF_Operation* initb = Assign(graph, status, "initb", b, zero);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "variables,%f\n", time_diff_ms(t0, t1));



    /* Compute W * x */
    clock_gettime(CLOCK_MONOTONIC, &t0);


    TF_OperationDescription* mul_d =
        TF_NewOperation(graph, "Mul", "mul");
    TF_AddInput(mul_d, (TF_Output){W, 0});
    TF_AddInput(mul_d, (TF_Output){x, 0});
    TF_Operation* Wx = TF_FinishOperation(mul_d, status);




    /*
     Compute (W * x) + b */

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

  


     /* loss = mean((y_pred - y)^2) */

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


    /*gradients of loss with respect to W and b*/

    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Output ys[1] = {{loss, 0}};
    TF_Output xs[2] = {{W, 0}, {b, 0}};
    TF_Output grads[2];
    TF_AddGradients(graph, ys, 1, xs, 2, NULL, status, grads);
    Check(status);
  
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "gradient_graph,%f\n", time_diff_ms(t0, t1));



     /* Apply gradient descent update */
    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_Operation* lr = ConstFloat(graph, status, "lr", 0.01f);

    TF_OperationDescription* applyW_d =
        TF_NewOperation(graph, "ApplyGradientDescent", "applyW");
    TF_AddInput(applyW_d, (TF_Output){W, 0});
    TF_AddInput(applyW_d, (TF_Output){lr, 0});
    TF_AddInput(applyW_d, grads[0]);
    TF_Operation* applyW = TF_FinishOperation(applyW_d, status);

    TF_OperationDescription* applyb_d =
        TF_NewOperation(graph, "ApplyGradientDescent", "applyb");
    TF_AddInput(applyb_d, (TF_Output){b, 0});
    TF_AddInput(applyb_d, (TF_Output){lr, 0});
    TF_AddInput(applyb_d, grads[1]);
    TF_Operation* applyb = TF_FinishOperation(applyb_d, status);

    TF_OperationDescription* train_d =
        TF_NewOperation(graph, "NoOp", "train");
    TF_AddControlInput(train_d, applyW);
    TF_AddControlInput(train_d, applyb);
    TF_Operation* train = TF_FinishOperation(train_d, status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "optimizer_graph,%f\n", time_diff_ms(t0, t1));


    clock_gettime(CLOCK_MONOTONIC, &t0);

    TF_SessionOptions* opts = TF_NewSessionOptions();
    TF_Session* sess = TF_NewSession(graph, opts, status);
    Check(status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "session_creation,%f\n", time_diff_ms(t0, t1));


 
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const TF_Operation* init_ops[] = {initW, initb};
    TF_SessionRun(sess, NULL,
                  NULL, NULL, 0,
                  NULL, NULL, 0,
                  init_ops, 2,
                  NULL, status);
    Check(status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(prof_csv, "variable_initialization,%f\n", time_diff_ms(t0, t1));


    
    int64_t dims[] = {1};

    /*. Training loop*/

    for (int step = 0; step < 200; step++) {
        float xv = (float)rand() / RAND_MAX;
        float yv = 3.0f * xv + 2.0f;

        float* xb = malloc(sizeof(float));
        float* yb = malloc(sizeof(float));
        *xb = xv;
        *yb = yv;

        TF_Tensor* tx = TF_NewTensor(
            TF_FLOAT, dims, 1, xb, sizeof(float),
            TensorDeallocator, NULL);
        TF_Tensor* ty = TF_NewTensor(
            TF_FLOAT, dims, 1, yb, sizeof(float),
            TensorDeallocator, NULL);

        TF_Output inputs[] = {{x, 0}, {y, 0}};
        TF_Tensor* in_vals[] = {tx, ty};
        TF_Output out_ops[] = {{loss, 0}};
        TF_Tensor* out_vals[1];
        const TF_Operation* train_ops[] = {train};

        clock_gettime(CLOCK_MONOTONIC, &t0);

        TF_SessionRun(sess, NULL,
                      inputs, in_vals, 2,
                      out_ops, out_vals, 1,
                      train_ops, 1,
                      NULL, status);
        Check(status);

        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = time_diff_ms(t0, t1);
        float loss_val = *(float*)TF_TensorData(out_vals[0]);

        fprintf(loss_csv, "%d,%f\n", step, loss_val);
        fprintf(prof_csv, "%d,%f\n", step, ms);

        TF_DeleteTensor(tx);
        TF_DeleteTensor(ty);
        TF_DeleteTensor(out_vals[0]);
    }

    fclose(loss_csv);
    fclose(prof_csv);

    printf("Training finished. Loss and profiling CSV written.\n");

    TF_DeleteSession(sess, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

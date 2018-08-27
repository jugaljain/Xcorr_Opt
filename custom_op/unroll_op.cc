#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("NormXCorr")
    .Input("input1: float")
    .Input("input2: float")
    .Output("output: float")
    .Doc(R"doc(
Takes all the 444 5x5 neighborhoods from the 25 37x12 feature maps and outputs them as a 25x444x5x5 tensor.
Don't pad beforehand, this op takes care of it.
)doc");

void NormXCorrKernelLauncher(const float* in1, const float* in2, float* out);

class NormXCorrOp : public OpKernel {
 public:
  explicit NormXCorrOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    auto input1 = input_tensor1.flat<float>();

    const Tensor& input_tensor2 = context->input(1);
    auto input2 = input_tensor2.flat<float>();

    TensorShape outShape;
    outShape.AddDim(25);
    outShape.AddDim(444);
    outShape.AddDim(60);

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, outShape,
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Call the cuda kernel launcher
    NormXCorrKernelLauncher(input1.data(), input2.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("NormXCorr").Device(DEVICE_GPU), NormXCorrOp);
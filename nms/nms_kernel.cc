#include <math.h>
#include "tensorflow/core/framework/op_kernel.h"



#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

using namespace tensorflow;
void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id);
void _overlaps_batch(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int batch, int device_id);
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class Rbox_iou : public OpKernel {
public:
	explicit Rbox_iou(OpKernelConstruction* context) : OpKernel(context) {}
   
   void Compute(OpKernelContext* context) override
   {
	//std::cout << "test" << std::endl;
	const Tensor& quadboxes = context->input(0);
	const Tensor& query_quadboxes = context->input(1);


	auto input = quadboxes.tensor<float, 3>();
	auto input2 = query_quadboxes.tensor<float, 3>();



	auto batch = quadboxes.shape().dim_size(0);
	auto n = quadboxes.shape().dim_size(1);
	auto k = query_quadboxes.shape().dim_size(1);

	//std::cout <<"batch " << batch << " n " << n << " k " << k <<std::endl;


	Tensor* output_tensor = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch, n, k}) , &output_tensor));
	//auto output_idx_tensor = (*output_tensor).tensor<float, 3>();

	//std::vector<float> iou_matrix(n*k);
    _overlaps_batch(output_tensor->tensor<float, 3>().data(), input.data(), input2.data() , n, k, batch, -1);



   }
   

};

REGISTER_KERNEL_BUILDER(Name("Rboxiou").Device(DEVICE_GPU), Rbox_iou);

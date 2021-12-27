#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Rboxiou")
	.Input("boxes: float32")
	.Input("quaryboxes: float32")
	.Output("output: float32");

REGISTER_OP("Rboxnms")
	.Input("boxes: float32")
	.Input("iou: float32")
	.Output("output: float32");

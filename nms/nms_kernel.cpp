
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);
void _rbox_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);
void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id);
namespace py = pybind11;


 std::vector<int> nms(py::array_t<float, py::array::c_style | py::array::forcecast> quad_n9, float iou_threshold)
 {
    auto pbuf = quad_n9.request();
    if (pbuf.ndim != 2 || pbuf.shape[1] != 5)
    throw std::runtime_error("quadrangles must have a shape of (n, 5)");
    auto n = pbuf.shape[0];
    auto boxes_dim = pbuf.shape[1];
    int keep[n] = {0};
    //std::vector<int> keep(n);
    int num_out = 0;
    int device_id = -1;
    std::vector<int> keep_vector;
    auto ptr = static_cast<float *>(pbuf.ptr);
    _nms(keep, &num_out, ptr, n,boxes_dim, iou_threshold, device_id);
    for(int i = 0 ; i < num_out ; i++)
    {
     keep_vector.push_back(keep[i]);
    }
    return keep_vector;
 }
  std::vector<int> rbox_nms(py::array_t<float, py::array::c_style | py::array::forcecast> quad_n9, float iou_threshold)
 {
    auto pbuf = quad_n9.request();
    if (pbuf.ndim != 2 || pbuf.shape[1] != 9)
    throw std::runtime_error("quadrangles must have a shape of (n, 9)");
    auto n = pbuf.shape[0];
    auto boxes_dim = pbuf.shape[1];
    int keep[n] = {0};
    //std::vector<int> keep(n);
    int num_out = 0;
    int device_id = -1;
    std::vector<int> keep_vector;
    auto ptr = static_cast<float *>(pbuf.ptr);
    _rbox_nms(keep, &num_out, ptr, n,boxes_dim, iou_threshold, device_id);
    for(int i = 0 ; i < num_out ; i++)
    {
     keep_vector.push_back(keep[i]);
    }
    return keep_vector;
 }
   std::tuple<std::vector<float>, std::vector<int>> rbox_overlap(py::array_t<float, py::array::c_style | py::array::forcecast> quad_boxes, py::array_t<float, py::array::c_style | py::array::forcecast> query_boxes)
 {
    std::cout <<"rbox_overlap" << std::endl;
    auto pbuf_quad_boxes = quad_boxes.request();
    auto pbuf_query_boxes = query_boxes.request();
    if (pbuf_quad_boxes.ndim != 2 || pbuf_quad_boxes.shape[1] != 8)
    throw std::runtime_error("quadrangles must have a shape of (n, 8)");
    if (pbuf_query_boxes.ndim != 2 || pbuf_query_boxes.shape[1] != 8)
    throw std::runtime_error("quadrangles must have a shape of (n, 8)");
    std::cout <<"rbox_overlap check" << std::endl;
    auto n = pbuf_quad_boxes.shape[0];
    auto k = pbuf_query_boxes.shape[0];
    std::cout <<"rbox_overlap check2" << std::endl;
    std::vector<float> iou(n*k, 0);
    std::cout <<"rbox_overlap check3" << std::endl;
    int device_id = -1;

    auto ptr_quad_boxes = static_cast<float *>(pbuf_quad_boxes.ptr);
    auto ptr_query_boxes = static_cast<float *>(pbuf_query_boxes.ptr);
    std::cout <<"in cal rbox iou" << std::endl;
    _overlaps(iou.data(), ptr_quad_boxes, ptr_query_boxes, n, k, device_id);
    std::cout <<"in cal rbox iou done!" << std::endl;

    std::vector<int> keep;
    std::vector<float> iou_result;
    for(int i = 0; i < n*k; i++)
    {
        if(iou[i] > 0.0)
        {
            keep.push_back(i);
            iou_result.push_back(iou[i]);
        }
    }
    std::cout <<"in cal rbox iou copy done!" << std::endl;
    std::cout <<iou_result.size() << std::endl;

    return std::make_tuple(iou_result, keep);
 }

PYBIND11_MODULE(nms_kernel, m)
{
  m.def("nms", nms);
  m.def("rbox_nms", rbox_nms);
  m.def("rbox_overlap", rbox_overlap);
}

/*
PYBIND11_PLUGIN(nms_kernel) {
	py::module m("adaptor", "NMS");
	m.def("nms", &nms,
			"nms");
	return m.ptr();
}
*/
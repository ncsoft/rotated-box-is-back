// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
// ------
// https://github.com/mjq11302010044/RRPN/blob/master/lib/rotation/rbbox_overlaps_kernel.cu from cal rbox iou overlap
// -----
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;



__device__ inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}
__device__ inline float quad_area(float const * const pts)
{

    float area = 0;
    for(int i = 0 ; i < 4 ; i ++)
    {   int first_idx = i % 4;
        int sencod_idx = (i+1) % 4;
        float x1 = pts[2*first_idx];
        float y1 = pts[2*first_idx+1];
        float x2 = pts[2*(sencod_idx)];
        float y2 = pts[2*(sencod_idx)+1];
        area += x1 * y2 - x2 * y1;

    }
    return fabs(0.5*area);
}

__device__ inline float area(float * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}



__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {



  if(num_of_inter > 0) {

    float center[2];

    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

__device__ inline bool in_rect(float pt_x, float pt_y, float * pts) {

  float ab[2];
  float ad[2];
  float ap[2];

  float abab;
  float abap;
  float adad;
  float adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap and abap >= 0 and adad >= adap and adap >= 0;
}
__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}


__device__ inline float inter(float * pts1, float * pts2) {
  float int_pts[16];
  int num_of_inter;
  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);

}


__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[4], b[4]);
  float top = max(a[1], b[1]), bottom = min(a[5], b[5]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[4] - a[0] + 1) * (a[5] - a[1] + 1);
  float Sb = (b[4] - b[0] + 1) * (b[5] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__device__ inline float devRotateIoU(float const * const pts1, float const * const pts2) {
  float pts1_copy[8]; // = (float*)pts1;
  float pts2_copy[8]; //= (float*)pts2;
  for(int i = 0 ; i < 8 ; i++)
  {
    //std::cout << i << " " << pts1[i] << std::endl;
    //std::cout << i << " " << pts2[i] << std::endl;
    pts1_copy[i] = pts1[i];
    pts2_copy[i] = pts2[i];
  }
  float area1 = quad_area(pts1_copy); // (pts1_copy[4] - pts1_copy[0] + 1) * (pts1_copy[5] - pts1_copy[1] + 1);//
  float area2 = quad_area(pts2_copy); // (pts2_copy[4] - pts2_copy[0] + 1) * (pts2_copy[5] - pts2_copy[1] + 1);//
  if(area1 == 0 | area2 == 0)
  {
    return 0;
  }

  float area_inter = inter(pts1_copy, pts2_copy);
  //area_inter = std::max(area_inter, (float)1.0);
  //area1 = std::max(area1, (float)1.0);
  //area2 = std::max(area2, (float)1.0);
  if(area_inter > area1)
  {
    area_inter = area1;
  }
  if(area_inter > area2)
  {
    area_inter = area2;
  }

  float result = area_inter / (area1 + area2 - area_inter);

  return result;


}


__global__ void rbox_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 9];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 9 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
    block_boxes[threadIdx.x * 9 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
    block_boxes[threadIdx.x * 9 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
    block_boxes[threadIdx.x * 9 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
    block_boxes[threadIdx.x * 9 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
    block_boxes[threadIdx.x * 9 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
    block_boxes[threadIdx.x * 9 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
    block_boxes[threadIdx.x * 9 + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    block_boxes[threadIdx.x * 9 + 8] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 9;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }

    for (i = start; i < col_size; i++) {
      if (devRotateIoU(cur_box, block_boxes + i * 9) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id);

void _rbox_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rbox_nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}


void _rbox_nms_batch(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int batch, int device_id) {
  _set_device(device_id);

  const float* boxes_dev = boxes_host;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  /*
  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  */
  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  for(int i = 0 ; i < batch ; i++)
  {
      rbox_nms_kernel<<<blocks, threads>>>(boxes_num,
                                      nms_overlap_thresh,
                                      boxes_dev + i *(boxes_num) * 9,
                                      mask_dev);

      std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
      CUDA_CHECK(cudaMemcpy(&mask_host[0],
                            mask_dev,
                            sizeof(unsigned long long) * boxes_num * col_blocks,
                            cudaMemcpyDeviceToHost));

      std::vector<unsigned long long> remv(col_blocks);
      memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

      int num_to_keep = 0;
      for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
          keep_out[num_to_keep++] = i;
          unsigned long long *p = &mask_host[0] + i * col_blocks;
          for (int j = nblock; j < col_blocks; j++) {
            remv[j] |= p[j];
          }
        }
      }
      *num_out = num_to_keep;
  }


  //CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {

  const int col_start = blockIdx.y;
  const int row_start = blockIdx.x;

  const int row_size =
        min(N - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(K - col_start * threadsPerBlock, threadsPerBlock);


  __shared__ float block_boxes[threadsPerBlock * 8];
  __shared__ float block_query_boxes[threadsPerBlock * 8];
  if (threadIdx.x < col_size) {
    block_query_boxes[threadIdx.x * 8 + 0] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 0];
    block_query_boxes[threadIdx.x * 8 + 1] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 1];
    block_query_boxes[threadIdx.x * 8 + 2] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 2];
    block_query_boxes[threadIdx.x * 8 + 3] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 3];
    block_query_boxes[threadIdx.x * 8 + 4] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 4];
    block_query_boxes[threadIdx.x * 8 + 5] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 5];
    block_query_boxes[threadIdx.x * 8 + 6] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 6];
    block_query_boxes[threadIdx.x * 8 + 7] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 7];
  }

  if (threadIdx.x < row_size) {
    block_boxes[threadIdx.x * 8 + 0] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 0];
    block_boxes[threadIdx.x * 8 + 1] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 1];
    block_boxes[threadIdx.x * 8 + 2] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 2];
    block_boxes[threadIdx.x * 8 + 3] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 3];
    block_boxes[threadIdx.x * 8 + 4] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 4];
    block_boxes[threadIdx.x * 8 + 5] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 5];
    block_boxes[threadIdx.x * 8 + 6] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 6];
    block_boxes[threadIdx.x * 8 + 7] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 8 + 7];
  }

  __syncthreads();

  if (threadIdx.x < row_size) {

    for(int i = 0;i < col_size; i++) {
      int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i ;
      dev_overlaps[offset] = devRotateIoU(block_boxes + threadIdx.x * 8, block_query_boxes + i * 8);
    }

  }
}

void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id) {

  _set_device(device_id);

  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;


  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        n * 8 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        n * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));



  CUDA_CHECK(cudaMalloc(&query_boxes_dev,
                        k * 8 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        k * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&overlaps_dev,
                        n * k * sizeof(float)));


  if (true){}


  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(k, threadsPerBlock));

  dim3 threads(threadsPerBlock);

  overlaps_kernel<<<blocks, threads>>>(n, k,
                                    boxes_dev,
                                    query_boxes_dev,
                                    overlaps_dev);


  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));





  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));




}



void _overlaps_batch(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int batch, int device_id) {

  _set_device(device_id);
  /*
  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;


  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        batch * n * 8 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        batch * n * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));



  CUDA_CHECK(cudaMalloc(&query_boxes_dev,
                        batch * k * 8 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        batch * k * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&overlaps_dev,
                        batch * n * k * sizeof(float)));
  */

  if (true){}


  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(k, threadsPerBlock));

  dim3 threads(threadsPerBlock);
  for(int i = 0 ; i < batch ; i++)
  {
  overlaps_kernel<<<blocks, threads>>>(n, k,
                                    boxes + i * (n*8),
                                    query_boxes + i * (k*8),
                                    overlaps + i * n * k);
  }
  /*
  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        batch * n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));





  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));
  */



}
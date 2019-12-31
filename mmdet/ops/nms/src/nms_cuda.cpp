// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor nms(const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");

}


/////////////////////////////////// for jit ///////////////////////
at::Tensor nms_jit(const at::Tensor& dets, double threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, float(threshold) );
}

#include <torch/script.h>
static auto registry = torch::jit::RegisterOperators("my_ops::nms_jit", &nms_jit);



#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/unlabeled_matching_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UnlabeledMatchingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = queue_size_;
  // compute the matching scores
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
      (Dtype)1., bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(),
      (Dtype)0., top[0]->mutable_gpu_data());
}

template <typename Dtype>
void UnlabeledMatchingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = queue_size_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N,
        (Dtype)1., top[0]->gpu_diff(), this->blobs_[0]->gpu_data(),
        (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[0]) {
    // Make sure the bottom diff is already computed
    CUDA_CHECK(cudaDeviceSynchronize());
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_gpu_data();
    // update instance feature
    for (int i = 0; i < M; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (label_value != -1) continue;
      caffe_gpu_memcpy(K, bottom_data + i * K, weight + queue_tail_ * K);
      if (++queue_tail_ >= queue_size_) queue_tail_ = 0;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(UnlabeledMatchingLayer);

}  // namespace caffe

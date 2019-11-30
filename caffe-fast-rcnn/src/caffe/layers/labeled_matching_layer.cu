#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/labeled_matching_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabeledMatchingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = num_classes_;
  // compute the matching scores
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
      (Dtype)1., bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(),
      (Dtype)0., top[0]->mutable_gpu_data());
  // assign the labels
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  caffe_copy(M, bottom_label, top_label);
  for (int i = 0; i < M; ++i) {
    const int label_value = static_cast<int>(bottom_label[i]);
    if (label_value < 0 || label_value >= num_classes_)
      top_label[i] = (Dtype)(-1.);
  }
}

template <typename Dtype>
void LabeledMatchingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = num_classes_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N,
        (Dtype)1., top[0]->gpu_diff(), this->blobs_[0]->gpu_data(),
        (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
  CHECK_EQ(this->param_propagate_down_[0], this->param_propagate_down_[1])
      << "Instance matrix and timestamp vector should be both updated or not";
  if (this->param_propagate_down_[0]) {
    // Make sure the bottom diff is already computed
    CUDA_CHECK(cudaDeviceSynchronize());
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_gpu_data();
    // increase all timestamp
    Dtype* tstamp = this->blobs_[1]->mutable_cpu_data();
    caffe_add_scalar(this->blobs_[1]->count(), (Dtype)1., tstamp);
    // update instance feature
    for (int i = 0; i < M; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (label_value < 0 || label_value >= N) continue;
      // w <- momentum * w + (1-momentum) * x
      caffe_gpu_axpby(K, (Dtype)1. - momentum_, bottom_data + i * K,
                      momentum_, weight + label_value * K);
      // reset the timestamp
      tstamp[label_value] = (Dtype)0.;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LabeledMatchingLayer);

}  // namespace caffe

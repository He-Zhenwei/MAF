#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/background_matching_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BackgroundMatchingLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  queue_size_ = this->layer_param_.background_matching_param().queue_size();
  queue_tail_ = 0;
  CHECK_GT(queue_size_, 0) << "Unlabeled queue size must be positive";

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the matrix for saving instance features
    vector<int> shape(2);
    shape[0] = queue_size_;
    shape[1] = bottom[0]->count(1);
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    // Reset the blobs data to zero
    caffe_set(this->blobs_[0]->count(), (Dtype)0,
              this->blobs_[0]->mutable_cpu_data());
  }

  // "Parameters" will be updated, but not by standard backprop with gradients.
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BackgroundMatchingLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(this->blobs_[0]->count(1), bottom[0]->count(1))
      << "Input size incompatible with initialization.";
  vector<int> shape(2);
  shape[0] = bottom[0]->shape(0);
  shape[1] = queue_size_;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void BackgroundMatchingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = queue_size_;
  // compute the matching scores
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
      (Dtype)1., bottom[0]->cpu_data(), this->blobs_[0]->cpu_data(),
      (Dtype)0., top[0]->mutable_cpu_data());
}

template <typename Dtype>
void BackgroundMatchingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int M = bottom[0]->shape(0);
  const int K = bottom[0]->count(1);
  const int N = queue_size_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N,
        (Dtype)1., top[0]->cpu_diff(), this->blobs_[0]->cpu_data(),
        (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    // update instance feature
    for (int i = 0; i < M; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (label_value != 483) continue;
      caffe_copy(K, bottom_data + i * K, weight + queue_tail_ * K);
      if (++queue_tail_ >= queue_size_) queue_tail_ = 0;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BackgroundMatchingLayer);
#endif

INSTANTIATE_CLASS(BackgroundMatchingLayer);
REGISTER_LAYER_CLASS(BackgroundMatching);

}  // namespace caffe

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_d_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
    		weight_data, bottom_data, (Dtype)0., top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight_data, (Dtype)0., top_data);
  }
  caffe_copy(this->blobs_[0]->count(), weight_data, top[1]->mutable_gpu_data());
}

template <typename Dtype>
void InnerProductDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	Blob<Dtype> tmp1;
	vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
	tmp1.Reshape(weight_shape);
//    BlobProto tmp_out;
//    top[1]->ToProto(&tmp_out, true);
//    WriteProtoToTextFile(tmp_out, "tmp1.blob");

    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)0., tmp1.mutable_gpu_data());

    caffe_gpu_add(this->blobs_[0]->count(), tmp1.gpu_data(), top[1]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff());
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, K_, N_,
        (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
        (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductDLayer);

}  // namespace caffe

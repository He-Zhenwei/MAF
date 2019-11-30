#include <algorithm>
#include <vector>
#include "caffe/layers/enlarge_layer.hpp"


namespace caffe {

template <typename Dtype>
__global__ void enlargeforward(const int n, const int sp_ns, const int sp_os, int in_channel, int out_channel, int scale, int b_w, int t_w, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
  	int batch = index / out_channel / sp_ns;
    int channel = index / sp_ns % out_channel;
	int h = index % sp_ns / t_w;
	int w = index % sp_ns % t_w;

	int tmp = channel% (scale* scale);
    int bottom_index = batch* in_channel* sp_os + (channel/ (scale* scale))* sp_os + (h* scale + tmp/ scale)* b_w + (w* scale + tmp% scale);
	top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = top[0]->count();

	const int sp_os = bottom[0]->count(2);
	const int sp_ns = top[0]->count(2); 

	enlargeforward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sp_ns, sp_os, ch_ori_, group_, scale_, w_ori_, top_w_, bottom_data, top_data);
}

template <typename Dtype>
__global__ void enlargebackward(const int n, const int sp_ns, const int sp_os, int in_channel, int out_channel, int scale, int b_w, int t_w, Dtype* bottom_diff, const Dtype* top_diff) {
  CUDA_KERNEL_LOOP(index, n) {
  	int batch = index / out_channel / sp_ns;
    int channel = index / sp_ns % out_channel;
	int h = index % sp_ns / t_w;
	int w = index % sp_ns % t_w;

	int tmp = channel% (scale* scale);
    int bottom_index = batch* in_channel* sp_os + (channel/ (scale* scale))* sp_os + (h* scale + tmp/ scale)* b_w + (w* scale + tmp% scale);
	bottom_diff[bottom_index] = top_diff[index];
  }
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if(propagate_down[0])
	{
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int count = top[0]->count();

		const int sp_os = bottom[0]->count(2);
		const int sp_ns = top[0]->count(2); 

		enlargebackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sp_ns, sp_os, ch_ori_, group_, scale_, w_ori_, top_w_, bottom_diff, top_diff);
	}
}
INSTANTIATE_LAYER_GPU_FUNCS(EnlargeLayer);
}

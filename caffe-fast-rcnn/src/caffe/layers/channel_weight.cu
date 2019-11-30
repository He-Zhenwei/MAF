#include <vector>

#include "caffe/layers/channel_weight.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void wforward(const int nthreads, const Dtype* inblob, const Dtype* weight, Dtype* outblob, const int channel, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int weight_index = index / (height* width) % channel;
		outblob[index] = inblob[index]* weight[weight_index];
	}
}

template <typename Dtype>
void ChannelweightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();
	const Dtype* in_data = bottom[0]->gpu_data();
	const Dtype* weights = bottom[1]->gpu_data();
	Dtype* out_data = top[0]->mutable_gpu_data();

	const int height = bottom[0]->shape(2);
	const int width = bottom[0]->shape(3);
	wforward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, in_data, weights, out_data, _num_channel, height, width);
}

template <typename Dtype>
__global__ void wbackward(const int nthreads, const Dtype* top_diff, const Dtype* bottom_data, Dtype* weight_diff, const int height, const int width, const int batch) {
    const int index_dim = height* width* nthreads;
	CUDA_KERNEL_LOOP(index, nthreads) {
		int data_index;
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
               for(int n=0;n<batch;n++){
				 data_index = index_dim* n + index* (height* width) + i* width + j;
				 weight_diff[index] = weight_diff[index] + top_diff[data_index]* bottom_data[data_index];
               }
			}
		}
//		weight_diff[index] = weight_diff[index] / (height* width);
	}
}

template <typename Dtype>
__global__ void bbackward(const int nthreads, const Dtype* top_diff, const Dtype* weight, Dtype* bottom_diff, const int channel, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int weight_index = index / (height* width) % channel;
		bottom_diff[index] = top_diff[index]* weight[weight_index];
	}
}

template <typename Dtype>
void ChannelweightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int count = bottom[0]->count();
	const Dtype* in_data = bottom[0]->gpu_data();
	const Dtype* weights = bottom[1]->gpu_data();

	const int height = bottom[0]->shape(2);
	const int width = bottom[0]->shape(3);

	const Dtype* top_diff = top[0]->gpu_diff();
	if(propagate_down[0]) {
		Dtype* b1_diff = bottom[0]->mutable_gpu_diff();
		bbackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, weights, b1_diff, _num_channel, height, width);
	}

	if(propagate_down[1]) {
		Dtype* b2_diff = bottom[1]->mutable_gpu_diff();
		caffe_gpu_set(bottom[1]->count(), Dtype(0), b2_diff);
		wbackward<Dtype><<<CAFFE_GET_BLOCKS(_num_channel), CAFFE_CUDA_NUM_THREADS>>>(_num_channel, top_diff, in_data, b2_diff, height, width, _num_sample);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelweightLayer);

}

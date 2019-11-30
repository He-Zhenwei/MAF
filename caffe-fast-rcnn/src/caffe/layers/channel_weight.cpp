#include <vector>

#include "caffe/layers/channel_weight.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
	void ChannelweightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		_num_channel = bottom[0]->shape(1);
        _num_sample = bottom[0]->shape(0);

		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void ChannelweightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		_num_channel = bottom[0]->shape(1);
        _num_sample = bottom[0]->shape(0);

		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void ChannelweightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		NOT_IMPLEMENTED;
	}

	template <typename Dtype>
	void ChannelweightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		NOT_IMPLEMENTED;
	}

	#ifdef CPU_ONLY
	STUB_GPU(ChannelweightLayer);
	#endif

	INSTANTIATE_CLASS(ChannelweightLayer);
	REGISTER_LAYER_CLASS(Channelweight);
}

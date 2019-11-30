#ifndef COS_DIS_LAYER_
#define COS_DIS_LAYER_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template <typename Dtype>
	class CosineDisLayer : public Layer<Dtype> {
	public:
		explicit CosineDisLayer(const LayerParameter& param): Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "cosinedis"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		// blob used to save the dictionary for each pair
		Blob<Dtype> useful_dic_;
		// 0 for each pairs
		Blob<Dtype> labels_;
		// num of samples
		int num_sample_;
		// length of the dic
		int len_dic_;
		int len_feat_;
		// blob for saving the feat
		Blob<Dtype> feat1;
		Blob<Dtype> feat2;
		Blob<Dtype> tmp_diff;
	};
}

#endif

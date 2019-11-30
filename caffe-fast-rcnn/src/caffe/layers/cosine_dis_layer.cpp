#include <vector>

#include "caffe/layers/cosine_dis_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void CosineDisLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "number of the two feat must the same!";
		CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "dimensions of the two feature must same!";
		CHECK_EQ(bottom[0]->channels(), bottom[2]->channels()) << "dimensions of the feature and dic must same!";

		num_sample_ = bottom[0]->num();
		len_dic_ = bottom[2]->num();
		len_feat_ = bottom[0]->channels();

		vector<int> shape_blob(2);
		shape_blob[0] = num_sample_;
		shape_blob[1] = len_dic_ + 1;
		top[0]->Reshape(shape_blob);

		shape_blob.resize(1);
		shape_blob[0] = num_sample_;
		top[1]->Reshape(shape_blob);
	}

	template <typename Dtype>
	void CosineDisLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "number of the two feat must the same!";
		CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "dimensions of the two feature must same!";
		CHECK_EQ(bottom[0]->channels(), bottom[2]->channels()) << "dimensions of the feature and dic must same!";

		num_sample_ = bottom[0]->num();
		len_dic_ = bottom[2]->num();
		len_feat_ = bottom[0]->channels();

		vector<int> shape_blob(2);
		shape_blob[0] = num_sample_* 2;
		shape_blob[1] = len_dic_ + 1;
		top[0]->Reshape(shape_blob);

		shape_blob.resize(1);
		shape_blob[0] = num_sample_* 2;
		top[1]->Reshape(shape_blob);

		shape_blob[0] = len_feat_;
		feat1.Reshape(shape_blob);
		feat2.Reshape(shape_blob);

		shape_blob[0] = len_dic_ + 1;
		tmp_diff.Reshape(shape_blob);
	}

	template <typename Dtype>
	void CosineDisLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		NOT_IMPLEMENTED;
	}

	template <typename Dtype>
	void CosineDisLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		NOT_IMPLEMENTED;
	}

	#ifdef CPU_ONLY
	STUB_GPU(CosineDisLayer);
	#endif

	INSTANTIATE_CLASS(CosineDisLayer);
	REGISTER_LAYER_CLASS(CosineDis);
}

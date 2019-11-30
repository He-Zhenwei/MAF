#include <vector>

#include "caffe/layers/cosine_dis_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
	template <typename Dtype>
	void CosineDisLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		// initize the dic for the compute
		Dtype* top1_data = top[0]->mutable_gpu_data();
		const Dtype* feat1_data = bottom[0]->gpu_data();
		const Dtype* feat2_data = bottom[1]->gpu_data();

		vector<int> shape_blob(2);
		shape_blob[0] = len_dic_ + 1;
		shape_blob[1] = len_feat_;
		useful_dic_.Reshape(shape_blob);
//		LOG(INFO) << bottom[2]->shape_string();

		caffe_copy(len_dic_* len_feat_, bottom[2]->gpu_data(), useful_dic_.mutable_gpu_data()+len_feat_);
//		BlobProto tmp_out;
//		useful_dic_.ToProto(&tmp_out, true);
//		WriteProtoToTextFile(tmp_out, "useful_dict.blob");

		for(int i=0; i<num_sample_; i++){
			// prepare data for every feature
			caffe_copy(len_feat_, feat1_data+i*len_feat_, feat1.mutable_gpu_data());
			caffe_copy(len_feat_, feat2_data+i*len_feat_, feat2.mutable_gpu_data());

			// compute for every dic
			caffe_copy(len_feat_, feat2.gpu_data(), useful_dic_.mutable_gpu_data());
			caffe_gpu_gemv(CblasNoTrans,
					len_dic_+1, len_feat_, (Dtype)1,
					useful_dic_.gpu_data(), feat1.gpu_data(), (Dtype)0, top1_data+i*(len_dic_+1));

			caffe_copy(len_feat_, feat1.gpu_data(), useful_dic_.mutable_gpu_data());
			caffe_gpu_gemv(CblasNoTrans,
					len_dic_+1, len_feat_, (Dtype)1,
					useful_dic_.gpu_data(), feat2.gpu_data(), (Dtype)0, top1_data+(i+num_sample_)*(len_dic_+1));
		}
//		top[0]->ToProto(&tmp_out, true);
//		WriteProtoToTextFile(tmp_out, "top.blob");

		// set_label for each sample
		caffe_gpu_set(num_sample_* 2, (Dtype)0, top[1]->mutable_gpu_data());
	}

	template <typename Dtype>
	void CosineDisLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* feat1_data = bottom[0]->gpu_data();
		const Dtype* feat2_data = bottom[1]->gpu_data();
		// initize the dic for the compute
		caffe_copy(len_dic_* len_feat_, bottom[2]->gpu_data(), useful_dic_.mutable_gpu_data()+len_feat_);
		BlobProto tmp_out;

		if(propagate_down[0]){
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			for(int i=0; i<num_sample_; i++){
				// get needed diff and dic
				caffe_copy(len_feat_, feat2_data+i*len_feat_, useful_dic_.mutable_gpu_data());
				caffe_copy(len_dic_ + 1, top_diff+i*(len_dic_+1), tmp_diff.mutable_gpu_data());

//				tmp_diff.ToProto(&tmp_out, true);
//				WriteProtoToTextFile(tmp_out, "tmp_diff.blob");
				caffe_gpu_gemv(CblasTrans,
						len_dic_+1, len_feat_, (Dtype)1,
						useful_dic_.gpu_data(), tmp_diff.gpu_data(), (Dtype)0, bottom_diff+i*len_feat_);
			}
		}
		if(propagate_down[1]){
			Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
			for(int i=0; i<num_sample_; i++){
				// get needed diff and dic
				caffe_copy(len_feat_, feat1_data+i*len_feat_, useful_dic_.mutable_gpu_data());
				caffe_copy(len_dic_ + 1, top_diff+(i+num_sample_)*(len_dic_+1), tmp_diff.mutable_gpu_data());
				caffe_gpu_gemv(CblasTrans,
						len_dic_+1, len_feat_, (Dtype)1,
						useful_dic_.gpu_data(), tmp_diff.gpu_data(), (Dtype)0, bottom_diff+i*len_feat_);
			}
		}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(CosineDisLayer);
}

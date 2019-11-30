#include <algorithm>
#include <vector>
#include "caffe/layers/enlarge_layer.hpp"

namespace caffe {

template <typename Dtype>
void EnlargeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    scale_ = this->layer_param_.enlarge_param().scale();
    CHECK_GT(scale_, 1)<<"scale must be greater than 1";

    batch_ = bottom[0]->num();
    ch_ori_ = bottom[0]->channels();
    h_ori_ = bottom[0]->height();
    w_ori_ = bottom[0]->width();

    group_ = int(ch_ori_ * (scale_*scale_)); //channels after enlarge

    top_h_ = h_ori_/ scale_;
    top_w_ = w_ori_/ scale_;
    top[0]->Reshape(batch_,group_,top_h_,top_w_);
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    batch_ = bottom[0]->num();
    ch_ori_ = bottom[0]->channels();
    h_ori_ = bottom[0]->height();
    w_ori_ = bottom[0]->width();

    group_ = int(ch_ori_ * (scale_*scale_));

    top_h_ = h_ori_/ scale_;
    top_w_ = w_ori_/ scale_;
    top[0]->Reshape(batch_,group_,top_h_,top_w_);
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	int pos_top, pos_bottom;
	int b_h, b_w, b_c;

	for (int m = 0; m < batch_; ++m)
	{
		for (int n = 0; n < top_h_; ++n)
		{
			for (int i = 0; m < top_w_; ++m)
			{
				for (int j = 0; j < group_; ++j)
				{
					pos_top = m*(top_h_* top_w_* group_) + j* (top_h_* top_w_) + n* (top_w_) + i;
					b_h = n* scale_ + j % (scale_* scale_) / scale_;
					b_w = i* scale_ + j % (scale_* scale_) % scale_;
					b_c = j / (scale_* scale_);
					pos_bottom = m*(h_ori_* w_ori_* ch_ori_) + b_c* (h_ori_* w_ori_) + b_h* (w_ori_) + b_w;
					top_data[pos_top] = bottom_data[pos_bottom];
				}
			}
		}
	}	
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if(propagate_down[0])
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		int pos_top, pos_bottom;
		int b_h, b_w, b_c;

		for (int m = 0; m < batch_; ++m)
		{
			for (int n = 0; n < group_; ++n)
			{
				for (int h = 0; h < top_h_; ++h)
				{
					for (int w = 0; w < top_w_; ++w)
					{
						pos_top = m*(top_h_* top_w_* group_) + n* (top_h_* top_w_) + h* (top_w_) + w;
						b_h = h* scale_ + n % (scale_* scale_) / scale_;
						b_w = w* scale_ + n % (scale_* scale_) % scale_;
						b_c = n / (scale_* scale_);
						pos_bottom = m*(h_ori_* w_ori_* ch_ori_) + b_c* (h_ori_* w_ori_) + b_h* (w_ori_) + b_w;
						bottom_diff[pos_bottom] = top_diff[pos_top];
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(EnlargeLayer);
#endif

INSTANTIATE_CLASS(EnlargeLayer);
REGISTER_LAYER_CLASS(Enlarge);
	
}

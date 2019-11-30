import caffe
import yaml
import numpy as np

class Gradient_Scale(caffe.Layer):
	def setup(self, bottom, top):
		params = yaml.load(self.param_str)
		self.scale = params['scale']

		num_sample = bottom[0].shape[0]

		top[0].reshape(*bottom[0].shape)
		top[1].reshape(*(num_sample, 2))

	def forward(self, bottom, top):
		num_sample = bottom[0].shape[0]
		top[0].reshape(*bottom[0].shape)
		top[0].data[...] = bottom[0].data

		zero_data = np.zeros((num_sample, 2), dtype='float32')
		top[1].reshape(*zero_data.shape)
		top[1].data[...] = zero_data

	def backward(self, top, propagate_down, bottom):
		top_diff = top[0].diff.copy()
		soft_score = top[1].data.copy()
		dc_label = bottom[1].data.copy().squeeze()[0]

		true_score = soft_score[:, dc_label]
		dim = top_diff.shape[1]
		true_weights = np.tile(true_score[:, np.newaxis], (1, dim))

		bottom_diff = true_weights* top_diff* self.scale
		bottom[0].diff[...] = bottom_diff

	def reshape(self, bottom, top):
		pass

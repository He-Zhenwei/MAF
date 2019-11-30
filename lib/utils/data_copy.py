import caffe
import numpy as np

class Data_Copy(caffe.Layer):
	def setup(self, bottom, top):
		shape1 = bottom[0].shape
		shape2 = bottom[1].shape
#		print shape1[0], shape2[0], shape1[1], shape2[1]

		assert shape1[0] == shape2[0]
		assert shape1[1] == shape2[1]

	def forward(self, bottom, top):
		data1 = bottom[0].data.copy()
		bottom[1].data[...] = data1

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass

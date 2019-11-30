# MAF
The implementation of paper "Multi-adversarial Faster-RCNN for Unrestricted Object Detection"


This code is the implementation of our ICCV2019 work "Multi-adversarial Faster-RCNN for Unrestricted Object Detection". The paper can be found in "https://arxiv.org/abs/1907.10343".

Our code is based on the implementation of DA-Faster-RCNN (https://github.com/yuhuayc/da-faster-rcnn)

#Usage

1. Build caffe and pycaffe.

2. Build Cython modules.

    cd $FRCN_ROOT/lib
    
    make

3. Prepare the training data following the da-faster-rcnn.

4. Training and testing the model.

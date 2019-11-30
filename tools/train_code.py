#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import cv2
import random
from scipy.sparse import csr_matrix

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name, flag=0):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        for rd in roidb:
            rd['dc_label'] = flag
        return roidb, imdb

    imdbs = imdb_names.split('+')
    if len(imdbs) == 2:
        t_source, imdb_s = get_roidb(imdbs[0], 0)
        t_target, imdb_t = get_roidb(imdbs[1], 1)
        source_classes = imdb_s._classes
        source_ind_class = imdb_s._class_to_ind
        target_ind_class = dict(zip(range(imdb_t.num_classes), imdb_t._classes))
        
        source = []
        for s in t_source:
            s['dc_label'] = 1
            source.append(s)

        target = []
        for t in t_target:
            gt_classes = t['gt_classes'].copy()
            max_classes = t['max_classes'].copy()
            boxes = t['boxes'].copy()
            max_overlaps = t['max_overlaps'].copy()
            gt_overlaps = t['gt_overlaps'].copy()
            gt_overlaps = gt_overlaps.toarray()

            t_gt_classes = []
            t_max_classes = []
            t_boxes = []
            t_max_overlaps = []
            t_gt_overlaps = []

            for i in range(len(gt_classes)):
                cls_name = target_ind_class[gt_classes[i]]
                if cls_name in source_classes:
                    t_gt_classes.append(source_ind_class[cls_name])
                    t_max_classes.append(source_ind_class[cls_name])
                    t_boxes.append(boxes[i, :])
                    t_max_overlaps.append(max_overlaps[i])
                    t_gt_overlaps.append(gt_overlaps[i, :])

            if len(t_gt_classes) > 0:
                t['gt_classes'] = np.asarray(t_gt_classes)
                t['boxes'] = np.asarray(t_boxes)
                t['max_classes'] = np.asarray(t_max_classes)
                t['gt_overlaps'] = csr_matrix(np.asarray(t_max_overlaps))
                t['max_overlaps'] = np.asarray(t_max_overlaps)
                t['dc_label'] = 0

                target.append(t)

        roidbs = [source, target, target, target]
        roidb = source
    else:
        roidbs = []
        for s in imdbs:
            t, _ = get_roidb(s)
            roidbs.append(t)
        roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb, roidb = combined_roidb(args.imdb_name)
#    print roidb
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    print len(roidb)

    '''
    random.shuffle(roidb)
    for db in roidb:
        im_path = db['image']
#        filped = db['filped']
        bbox = db['boxes']
        print db['flipped']

        im = cv2.imread(im_path)
        if db['flipped']:
            t = im.copy()
            im = t[:, ::-1, :].copy()

        for bb in bbox:
            cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255))
        cv2.imshow('t', im)
        cv2.waitKey(0)
    '''

    '''
    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
    '''

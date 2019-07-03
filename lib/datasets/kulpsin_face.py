"""
Modified from pascal_voc.py to work with KulpsinAnnotations.
"""
from __future__ import print_function
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import json
import logging
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .kulpsin_annotation import KulpsinAnnotations
from .kulpsin_eval import kulpsin_eval
from fast_rcnn.config import cfg
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)


class kulpsin_face(imdb):

    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'kulpsin_face_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = ''
        self._image_index = self._load_image_set_ids()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        """
        # Specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}
        """

        self._annotations = KulpsinAnnotations(
                os.path.join(self._data_path, "Images", self._image_set, "kulpsin_annotations.json")
        )

        assert os.path.exists(self._devkit_path), \
                'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        try:
            image_id = self._image_index[i]
        except IndexError as e:
            logger.error("Attempted to use index %i. when there's only %i images."%(
                i, len(self._image_index)
            ))
            raise

        image_path = self._annotations.get_image_path(image_id)

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_ids(self):
        """
        Load the ids listed in this dataset's image set file.
        In this case those are hashes.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        logger.debug("Number of images: %i"%len(image_index))
        return image_index

    def _get_default_path(self):
        """
        Return the default path where Kulpsin Face is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'kulpsin_face_devkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        #raise NotImplementedError
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #"""
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        #"""
        gt_roidb = [self._load_kulpsin_annotation(image_id)
                    for image_id in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        raise NotImplementedError

    def rpn_roidb(self):
        raise NotImplementedError

    def _load_rpn_roidb(self, gt_roidb):
        raise NotImplementedError

    def _load_selective_search_roidb(self, gt_roidb):
        raise NotImplementedError

    def _load_kulpsin_annotation(self, image_id):
        """
        Load image and bounding boxes info from txt file in the kulpsin
        format.
        """
        filename = os.path.join(self._data_path, 'ImageSets',
                                self._image_set + '.txt')
        with open(filename) as f:
            image_ids = f.readlines()
        image_data = self._annotations.load_image_data(image_id=image_id)

        num_objs = len(image_data["bounding_boxes"])

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, bbox in enumerate(image_data["bounding_boxes"]):
            x1 = bbox["pt1"][0]
            y1 = bbox["pt1"][1]
            x2 = bbox["pt2"][0]
            y2 = bbox["pt2"][1]
            cls = self._class_to_ind[bbox["entity"]]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        #raise NotImplementedError
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id


    def _get_kulpsin_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        #raise NotImplementedError
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_kulpsin_results_file(self, all_boxes):
        #raise NotImplementedError
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Kulpsin Face results file'.format(cls))
            filename = self._get_kulpsin_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def _do_python_eval(self, output_dir = 'output'):
        #raise NotImplementedError
        imagesetfile = os.path.join(
            self._devkit_path,
            'data',
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print("At _do_python_eval")
        res = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_kulpsin_results_file_template().format(cls)
            rec, prec, ap = kulpsin_eval(
                filename, self._annotations, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            res += [(cls, rec, prec, ap,)]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        #for ap in aps:
        #    print('{:.3f}'.format(ap))
        for cls, rec, prec, ap in res:
             print('cls: {:20}rec: {:.5f},  prec: {:.5f},  ap: {:.5f}'.format(str(cls), float(np.mean(rec)), float(np.mean(prec)), float(ap)))
        print('{:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        #raise NotImplementedError
        self._write_kulpsin_results_file(all_boxes)
        self._do_python_eval(output_dir)

        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_kulpsin_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        #raise NotImplementedError
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    raise NotImplementedError
    #d = kulpsin_face('trainval')
    #res = d.roidb
    #from IPython import embed; embed()

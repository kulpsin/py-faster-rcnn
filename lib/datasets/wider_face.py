"""
Modified from pascal_voc.py to work with wider_face dataset.
"""
from __future__ import print_function
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .wider_eval import wider_eval
from fast_rcnn.config import cfg

class wider_face(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'wider_face_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
        #                 'aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')
                         'person')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = ''
        self._image_index = self._load_image_set_index()
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
        assert os.path.exists(self._devkit_path), \
                'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._data_path, 'Images', self._image_set,
                                  self._image_index[i])
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where Wider Face is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'wider_face_devkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        """
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        """
        gt_roidb = list(self._load_wider_annotation())

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        raise NotImplementedError
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb


        #"""
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        #"""
        #roidb = self.gt_roidb()
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        raise NotImplementedError
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_wider_annotation(self):
        """
        Load image and bounding boxes info from txt file in the WIDER FACE
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations',
                                self._image_set + '.txt')
        with open(filename) as f:
            data = f.readlines()
        annotations = dict()
        stats = {
            "blur": 0,
            "occlusion": 0,
            "invalid_image": 0,
            "face_size": 0,
            "atypical_pose": 0,
        }
        remaining_faces = 0
        while data:
            #image_path = os.path.join(self._data_path, 'Images', self._image_set,
            #                          data.pop(0).strip("\n"))
            image_name = data.pop(0).strip("\n")
            num_objs = int(data.pop(0))

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            face_counter = 0

            # Load object bounding boxes into a data frame.
            for ix in range(num_objs):
                """Format:
                0  1  2 3 4    5          6            7       8         9
                x1 y1 w h blur expression illumination invalid occlusion pose
                """
                face = data.pop(0).strip("\n").split(" ")

                x1 = int(face[0])
                assert x1 >= 0, "x1 must be positive ({})".format(x1)

                y1 = int(face[1])
                assert y1 >= 0, "y1 must be positive ({})".format(y1)

                x2 = x1 + int(face[2])
                assert x2 >= x1, "x2 ({}) must be larger than x1 ({}), {}".format(x1, x2, image_name)

                y2 = y1 + int(face[3])
                assert y2 >= y1, "y2 ({}) must be larger than y1 ({}), {}".format(y1, y2, image_name)

                cls = self._class_to_ind['person']
                face_size = (x2 - x1 + 1) * (y2 - y1 + 1)


                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                seg_areas[ix] = face_size

                # Setting overlaps to -1 will cause them to be excluded
                # during training phase.
                if int(face[8]) > 0: # "2":
                    stats["occlusion"] += 1
                    overlaps[ix, :] = -1.0
                elif face[4] == "2":
                    stats["blur"] += 1
                    overlaps[ix, :] = -1.0
                elif face[7] == "1":
                    stats["invalid_image"] += 1
                    overlaps[ix, :] = -1.0
                elif face[9] == "1":
                    stats["atypical_pose"] += 1
                    overlaps[ix, :] = -1.0
                elif face_size < 400:
                    stats["face_size"] += 1
                    overlaps[ix, :] = -1.0
                else:
                    remaining_faces += 1
                    face_counter += 1
                    overlaps[ix, cls] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)
            annotations[image_name] = {'boxes' : boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps' : overlaps,
                   'flipped' : False,
                   'seg_areas' : seg_areas}

        for reason, amount in stats.items():
            print("Ignoring {} faces due to {}".format(amount, reason))
        print("Total {} faces were preserved".format(remaining_faces))

        for image_name in self.image_index:
            if image_name in annotations:
                yield annotations[image_name]


    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_wider_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_wider_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Wider Face results file'.format(cls))
            filename = self._get_wider_results_file_template().format(cls)
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
        annofilepath = os.path.join(
            self._devkit_path,
            'data',
            'Annotations',
            self._image_set + '.txt')
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
        exit()
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_wider_results_file_template().format(cls)
            rec, prec, ap = wider_eval(
                filename, annofilepath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_wider_results_file(all_boxes)
        self._do_python_eval(output_dir)

        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_wider_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    # from datasets.wider_face import wider_face
    d = wider_face('test')
    res = d.roidb
    from IPython import embed; embed()

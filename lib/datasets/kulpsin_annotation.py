"""
Contains KulpsinAnnotations-class which hopefully makes interacting with my
annotations easier.

This is W.I.P. (work in progress) though, I advise not to use it just yet.

Code: Python 3
"""

import os
import json
import hashlib
import imghdr
import logging
logger = logging.getLogger(__file__)

CONFIG = {
    'min_box_side_length': 5,
}

class KulpsinAnnotations:

    def __init__(self, save_path=None):
        """
        save_path: default saving file name"""
        self._annotations = None
        self._annotation_json = save_path
        if save_path is not None:
            self.load_annotations()


    def load_annotations(self, annotation_json=None):
        """
        Load annotations from disk at `annotation_json`.
        If `annotation_json` is None, u            if hash not in annotations:
                annotations[hash] = {
                    "file_paths": [file_name],
                    "bounding_boxes": []
                }se `save_path` given
        when object was initialized.
        """
        if self._annotations is not None:
            raise RuntimeError("Annotations loaded already")
        # Load existing annotations:
        if annotation_json is None:
            annotation_json = self._annotation_json
        if annotation_json is None:
            raise RuntimeError("Save path not defined.")
        try:
            with open(annotation_json, "r") as json_file:
                self._annotations = json.load(json_file)
        except FileNotFoundError as e_msg:
            logging.error("Annotation file not found: %s"%annotation_json)
            self._annotations = {"last_file_index":0, "images": {}}


    def save_annotations(self, annotation_json=None, last_file_index=None, save_sets=False):
        """Save annotations to disk at `annotation_json`.
        If `annotation_json` is None, use `save_path` given
        when object was initialized."""
        if last_file_index is not None:
            self._annotations["last_file_index"] = last_file_index
        if annotation_json is None:
            annotation_json = self._annotation_json
        if annotation_json is None:
            raise RuntimeError("Save path not defined.")

        with open(annotation_json, 'w') as json_file:
            json.dump(self._annotations, json_file)
        if save_sets:
            test_path = os.path.join(os.path.dirname(self._annotation_json), "test.txt")
            trainval_path = os.path.join(os.path.dirname(self._annotation_json), "trainval.txt")
            logger.debug("Saving %s and %s"%(test_path, trainval_path))
            with open(test_path, "w") as test_f, \
                 open(trainval_path, "w") as trainval_f:
                for image_id in self._annotations["images"]:
                    if self.get_image_path(image_id) is None:
                        continue
                    if "image_set" in self._annotations["images"][image_id]:
                        if self._annotations["images"][image_id]["image_set"] == "test":
                            test_f.write(image_id + "\n")
                        elif self._annotations["images"][image_id]["image_set"] == "trainval":
                            trainval_f.write(image_id + "\n")

    def load_image_data(self, file_name=None, image_id=None):
        """Loads existing annotations for the image.
        Adds it if it doesn't exist
        image_id: if id is known already, skip hash calculation
        returns image id and annotations
        """
        file_type = None
        if image_id is None:

            # First check that it's image/photo
            file_type = imghdr.what(file_name)
            if file_type is None:
                return None

            # We use hash as image identifier because duplicate images
            with open(file_name, 'rb') as f:
                image_id = hashlib.sha256(f.read()).hexdigest()
            logger.debug("Image shaim_boxes256 hash: %s"%hash)
        else:
            assert image_id in self._annotations["images"], "Image_id not found"

        new_entry = False
        if file_name is not None:
            if image_id in self._annotations["images"]:
                # Duplicate image paths are both saved (in case the other
                # is removed)
                if file_name not in self._annotations["images"][image_id]["file_paths"]:
                    self._annotations["images"][image_id]["file_paths"].append(file_name)

                # Earlier version didn't have "image_format"-field:
                if "image_format" not in self._annotations["images"][image_id]:
                    if file_type is None:
                        file_type = imghdr.what(file_name)
                    assert file_type is not None, "File type must not be None"
                    self._annotations["images"][image_id]["image_format"] = file_type
            else:
                # Adding image data to database
                self._annotations["images"][image_id] = {
                    "file_paths": [file_name],
                    "bounding_boxes": [],
                    "image_format": file_type,
                }
                self.save_annotations(self._annotation_json + ".tmp")
                new_entry = True
        return {
            "image_id": image_id,
            "bounding_boxes": self._annotations["images"][image_id]["bounding_boxes"],
            "new_entry": new_entry,
        }

    def clear_removed_files(self):
        """Checks "File_paths" and removes files that have been
        removed"""
        raise NotImplementedError

    def get_last_file_index(self):
        if "last_file_index" in self._annotations:
            return self._annotations["last_file_index"]
        else:
            return 0

    def get_image_path(self, image_id):
        """Getter for image path"""
        for p in self._annotations["images"][image_id]["file_paths"]:
            if os.path.exists(p):
                return p

    def get_image_paths(self, image_id):
        return self._annotations["images"][image_id]["file_paths"]

    def annotations(self):
        for ann in self._annotations["images"]:
            yield ann

    def fix_and_remove_invalid_entries(self, image_id=None):
        """Checks for:
        - Removed images
        - Negative coordinates
        - Too small bounding boxes"""
        from PIL import Image
        def fix_and_remove_single_entry(image_id, value):
            # Check for removed files:
            index = 0
            while index < len(value["file_paths"]):
                p = value["file_paths"][index]
                if not os.path.isfile(p):
                    value["file_paths"].pop(index)
                else:
                    index += 1
            # TODO: Fix too large coordinates...
            first_file = self.get_image_path(image_id)
            size = None
            if first_file is not None:
                size = Image.open(self.get_image_path(image_id)).size

            # Check bounding boxes
            index = 0
            while index < len(value["bounding_boxes"]):
                b = value["bounding_boxes"][index]

                # TODO: Missing entity?
                #if "entity" not in b:
                #    logger.info("Entity fixed...")
                #    value["bounding_boxes"][index]["entity"] = "person"

                # Fix negative coordinates to 0
                if (b["pt1"][0] < 0 or
                    b["pt1"][1] < 0 or
                    b["pt2"][0] < 0 or
                    b["pt2"][1] < 0):
                    logger.info("Fixed negative coordinate (%s)!"%str(b))
                    value["bounding_boxes"][index]["pt1"] = \
                        (max(b["pt1"][0], 0), max(b["pt1"][1], 0))
                    value["bounding_boxes"][index]["pt2"] = \
                        (max(b["pt2"][0], 0), max(b["pt2"][1], 0))

                # Fix too large coordinates to width-1 or height-1
                if size is not None and \
                     (b["pt1"][0] >= size[0] or
                     b["pt1"][1] >= size[1] or
                     b["pt2"][0] >= size[0] or
                     b["pt2"][1] >= size[1]):
                    logger.info("Fixed too large coordinate (%s)!"%str(b))
                    value["bounding_boxes"][index]["pt1"] = \
                        (min(b["pt1"][0], size[0]-1), min(b["pt1"][1], size[1]-1))
                    value["bounding_boxes"][index]["pt2"] = \
                        (min(b["pt2"][0], size[0]-1), min(b["pt2"][1], size[1]-1))

                # Remove boxes too small
                if (b["pt2"][0] - b["pt1"][0] < CONFIG['min_box_side_length'] or
                    b["pt2"][1] - b["pt1"][1] < CONFIG['min_box_side_length']):
                    logger.info("Removed too small bounding box!")
                    value["bounding_boxes"].pop(index)
                else:
                    index += 1
        if image_id is None:
            for _image_id, value in self._annotations["images"].items():
                fix_and_remove_single_entry(_image_id, value)

        else:
            fix_and_remove_single_entry(image_id, self._annotations["images"][image_id])

        # Save temporary copy:
        self.save_annotations(self._annotation_json + ".tmp")
    def remove_path(self, image_id, removable):
        """Removes specified path from specific image and then saves annotations
        to drive
        """
        self._annotations["images"][image_id]["file_paths"].remove(removable)
        self.save_annotations()

    def add_annotation(self, image_id, points, entity):
        """Adds annotation into image
        image_id: id from load_image_data()-method
        points: tuple format: ((x1, y1), (x2, y2))
        entity: string e.g. "person"
        """
        assert type(points) is tuple, "Use tuple for points"
        assert len(points) == 2, "There must be exactly 2 points"
        assert type(points[0]) is tuple, "Use tuple for point coordinates"
        assert type(points[1]) is tuple, "Use tuple for point coordinates"
        assert image_id in self._annotations["images"], "Image_id not found"

        self._annotations["images"][image_id]["bounding_boxes"].append({
            "entity": entity,
            "pt1": points[0],
            "pt2": points[1],
        })
        # Save temporary copy:
        self.save_annotations(self._annotation_json + ".tmp")

    def select_bounding_box(self, image_id, point, index=0):
        """Find annotation bounding box that is around the given `point`.
        index: with overlapping boxes, this allows to choose specific one
        returns coordinates as tuple format ((x1, y1), (x2, y2))"""

        assert image_id in self._annotations["images"], "Image_id not found"
        box_list = []
        for d in self._annotations["images"][image_id]["bounding_boxes"]:
            if (d["pt1"][0] <= point[0] <= d["pt2"][0] and
                d["pt1"][1] <= point[1] <= d["pt2"][1]):
                 box_list.append((d["pt1"], d["pt2"]))
        if len(box_list) == 0:
            return None
        return box_list[index % len(box_list)]


    def remove_annotation(self, image_id, points, entity=None):
        """Remove specific bounding box from annotation list."""
        assert image_id in self._annotations["images"], "Image_id not found"
        index = 0
        while index < len(self._annotations["images"][image_id]["bounding_boxes"]):
            d = self._annotations["images"][image_id]["bounding_boxes"][index]
            if (d["pt1"] == points[0] and
                d["pt2"] == points[1] and
                (entity is None or entity == d["entity"])):
                removed = self._annotations["images"][image_id]["bounding_boxes"].pop(index)
                logger.debug("Removed bounding box: ({}, {}), ({},{})".format(
                    removed["pt1"][0], removed["pt1"][1],
                    removed["pt2"][0], removed["pt2"][1]))
            else:
                index += 1

    def set_trainval(self, image_id):
        """Assign image to trainval-set"""
        assert image_id in self._annotations["images"], "Image_id not found"
        self._annotations["images"][image_id]["image_set"] = "trainval"

    def set_test(self, image_id):
        """Assign image to test-set"""
        assert image_id in self._annotations["images"], "Image_id not found"
        self._annotations["images"][image_id]["image_set"] = "test"


if __name__ == "__main__":
    raise NotImplementedError

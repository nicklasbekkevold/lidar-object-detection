import os

import pandas as pd
from pycocotools.coco import COCO


def build_coco_wrappers(annotations_path='./annotations'):
    coco_files = os.listdir(annotations_path)
    coco_file_paths = [f'{annotations_path}/{coco_file}' for coco_file in coco_files]
    return [COCO(coco_file_path) for coco_file_path in coco_file_paths]


def get_annotations_from_coco_wrapper(coco_wrapper):
    for image_id in coco_wrapper.imgs:
        annotation_ids = coco_wrapper.getAnnIds(imgIds=image_id)
        annotations = coco_wrapper.loadAnns(annotation_ids)
        yield annotations


def convert_coco_wrapper_to_data_frame(coco_wrapper):
    annotation_data = []
    for annotations in get_annotations_from_coco_wrapper(coco_wrapper):
        for annotation in annotations:
            annotation_data.append({
                'image_id': annotation['image_id'],
                
                'category_id': annotation['category_id'],
                'bbox': annotation['bbox'],
                'area': annotation['attributes'],

                'occluded': annotation['attributes']['occluded'],
                'track_id': annotation['attributes']['track_id'],
                'keyframe': annotation['attributes']['keyframe'],

                'is_crowd': annotation['iscrowd'],
                'segmentation': annotation['segmentation'],
            })

    annotations_data_frame = pd.DataFrame(annotation_data)
    annotations_data_frame.set_index('image_id', inplace=True)
    return annotations_data_frame


def main():
    coco_wrappers = build_coco_wrappers()
    print(convert_coco_wrapper_to_data_frame(coco_wrappers[0]).head())


if __name__ == '__main__':
    main()

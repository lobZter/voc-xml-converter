# -------------------------------------------------------------------------------
#   Created by ben_cheng-610 2021-05-05
#   Extract only person and vehicle category and convert to VOC xml format
# -------------------------------------------------------------------------------
# %%
import os
import shutil
from pathlib import Path

from joblib import Parallel, delayed
from pycocotools.coco import COCO
from tqdm import tqdm

import pascal_voc_writer

# %%
# trainval = 'train'
trainval = 'val'
prefix_folder = 'coco2017/'

dataset_dir = '/data/datasets/COCO'
src_ann_file = f'{dataset_dir}/annotations/instances_{trainval}2017.json'
src_img_dir = f'{dataset_dir}/images/{trainval}2017'
dst_ann_dir = f'~/workspace/annotations/{trainval}/{prefix_folder}'  # YOUR_XML_OUTPUT_PATH
dst_img_dir = f'{dataset_dir}/vehicle+person/{trainval}'             # FILTERED_IMG_OUTPUT_PATH

cat_names = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']


# %%
def gen_xml(img_id):
    img_desc = coco.loadImgs(img_id)[0]
    fname = img_desc["file_name"]

    # copy image
    shutil.copy(f'{src_img_dir}/{fname}', f'{dst_img_dir}/{fname}')

    # process annotations
    bboxs = []
    ann_ids = coco.getAnnIds(imgIds=img_desc['id'], catIds=cat_ids)
    ann_descs = coco.loadAnns(ann_ids)
    for ann_desc in ann_descs:
        cat = COCO_CAT_NAMES[ann_desc['category_id']]
        if cat in cat_names:
            bbox = ann_desc['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2]) + int(bbox[0])
            ymax = int(bbox[3]) + int(bbox[1])
            bboxs.append((cat, xmin, ymin, xmax, ymax))

    # get image info
    img_path = f'{prefix_folder}{fname}'
    img_w = img_desc["width"]
    img_h = img_desc["height"]
    img_d = 3

    # save annotation info to xml file
    dst_ann_path = f'{dst_ann_dir}/{fname[:-4]}.xml'
    writer = pascal_voc_writer.Writer(img_path, img_w, img_h, img_d)
    for bbox in bboxs:
        writer.addObject(*bbox)
    writer.save(dst_ann_path)


# %%
if __name__ == '__main__':

    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    if not os.path.exists(dst_ann_dir):
        os.makedirs(dst_ann_dir)

    # load coco json annotation file
    coco = COCO(src_ann_file)

    # show all categories in coco
    COCO_CAT_NAMES = dict()
    for cat in coco.dataset['categories']:
        COCO_CAT_NAMES[cat['id']] = cat['name']
    print(COCO_CAT_NAMES)

    # filter images by categories
    cat_ids = coco.getCatIds(catNms=cat_names)
    img_ids = set()
    for cat_id in cat_ids:
        img_ids |= set(coco.getImgIds(catIds=cat_id))

    # single thread
    for img_id in tqdm(img_ids):
        gen_xml(img_id)
# %%

# -------------------------------------------------------------------------------
#   Created by ben_cheng-610 2021-05-06
#   Convert to VOC xml format
#   Note: After convertion, following categories are re-classified or renamed.
#       pedestrian  -> person
#       people      -> person
#       van         -> car
#       motor       -> motorcycle
# -------------------------------------------------------------------------------
# %%
import os
from pathlib import Path

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

import pascal_voc_writer

# %%
trainval = 'train'
# trainval = 'val'
prefix_folder = 'visdrone2018/'

dataset_dir = '/data/datasets/VisDrone2018'
src_ann_dir = f'{dataset_dir}/VisDrone2018-DET-{trainval}/annotations'
src_img_dir = f'{dataset_dir}/VisDrone2018-DET-{trainval}/images'
dst_ann_dir = f'~/workspace/annotations/{trainval}/{prefix_folder}'  # YOUR_XML_OUTPUT_PATH

# ignored(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), others(11)
cat_dict = {
    "1": "person",
    "2": "person",
    "3": "bicycle",
    "4": "car",
    "5": "car",
    "6": "truck",
    "9": "bus",
    "10": "motorcycle",
}


# %%
def gen_xml(ann_path):
    fname = ann_path.stem

    with open(ann_path, 'r') as f:
        lines = f.readlines()

    # calc bbox
    bboxs = []
    for line in lines:
        obj = line.strip('\n').split(',')
        if obj[5] in cat_dict:
            xmin = int(obj[0])
            ymin = int(obj[1])
            xmax = int(obj[2]) + int(obj[0])
            ymax = int(obj[3]) + int(obj[1])
            bboxs.append((cat_dict[obj[5]], xmin, ymin, xmax, ymax))

    # get image info
    img = cv2.imread(f'{src_img_dir}/{fname}.jpg')
    img_path = f'{prefix_folder}{fname}.jpg'
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_d = img.shape[2]

    # save annotation info to xml file
    dst_ann_path = f'{dst_ann_dir}/{fname}.xml'
    writer = pascal_voc_writer.Writer(img_path, img_w, img_h, img_d)
    for bbox in bboxs:
        writer.addObject(*bbox)
    writer.save(dst_ann_path)


#%%
if __name__ == '__main__':

    if not os.path.exists(dst_ann_dir):
        os.makedirs(dst_ann_dir)

    ann_paths = list(Path(src_ann_dir).glob('**/*.txt'))

    # multi thread
    result = Parallel(n_jobs=32)(delayed(gen_xml)(ann_path)
                                for ann_path in tqdm(ann_paths))

    # single thread
    # for ann_path in tqdm(ann_paths)
    #     gen_xml(ann_path)

# %%

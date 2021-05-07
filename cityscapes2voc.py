# -------------------------------------------------------------------------------
#   Created by ben_cheng-610 2021-05-05
#   Convert to VOC xml format
#   Note: Rider category are classified into person category after convertion.
# -------------------------------------------------------------------------------
# %%
import json
import os
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

import pascal_voc_writer

# %%
# trainval = 'train'
trainval = 'val'
prefix_folder = 'cityscapes/'

dataset_dir = '/data/datasets/cityscapes'
src_ann_dir = f'{dataset_dir}/gtFine/{trainval}'
src_img_dir = f'{dataset_dir}/leftImg8bit/{trainval}'
dst_ann_dir = f'~/workspace/annotations/{trainval}/{prefix_folder}'  # YOUR_XML_OUTPUT_PATH

cat_names = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']



# %%
def gen_xml(json_path):
    fname = f'{json_path.parent.stem}/{json_path.name[:-21]}_leftImg8bit'

    with open(json_path, 'r') as f:
        data = json.load(f)

    # calc bbox
    bboxs = []
    for obj in data["objects"]:
        label = 'person' if obj["label"] == 'rider' else obj["label"]
        polygon = obj["polygon"]
        if label in cat_names:
            x_coords, y_coords = zip(*polygon)
            xmin = min(x_coords)
            ymin = min(y_coords)
            xmax = max(x_coords)
            ymax = max(y_coords)
            bboxs.append((label, xmin, ymin, xmax, ymax))

    # get image info
    img_path = f'{prefix_folder}{fname}.png'
    img_w = data['imgWidth']
    img_h = data['imgHeight']
    img_d = 3

    # save annotation info to xml file
    dst_ann_path = f'{dst_ann_dir}/{fname}.xml'
    writer = pascal_voc_writer.Writer(img_path, img_w, img_h, img_d)
    for bbox in bboxs:
        writer.addObject(*bbox)
    writer.save(dst_ann_path)

# %%
if __name__ == '__main__':

    for d in list(Path(src_ann_dir).glob('**/')):
        dir_path = d.as_posix().replace(src_ann_dir, dst_ann_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    ann_paths = list(Path(src_ann_dir).glob('**/*.json'))

    # multi thread
    result = Parallel(n_jobs=32)(delayed(gen_xml)(ann_path) for ann_path in tqdm(ann_paths))

    # single thread
    # for ann_path in tqdm(ann_paths)
    #     gen_xml(ann_path)

# %%

# -------------------------------------------------------------------------------
#   Created by ben_cheng-610 2021-05-05
#   Extract vbox (visible body bbox) and convert to VOC xml format
# -------------------------------------------------------------------------------
# %%
import os

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

import pascal_voc_writer

# %%
trainval = 'train'
# trainval = 'val'
prefix_folder = 'crowdhuman/'

dataset_dir = '/data/datasets/CrowdHuman'
src_ann_file = f'{dataset_dir}/annotation_{trainval}.odgt'
src_img_dir = f'{dataset_dir}/Images/{trainval}'
dst_ann_dir = f'~/workspace/annotations/{trainval}/{prefix_folder}'  # YOUR_XML_OUTPUT_PATH


# %%
def gen_xml(json_line):
    data = eval(json_line)

    # calc bboxs
    bboxs = []
    img_id = data['ID']
    for gtbox in data['gtboxes']:
        if gtbox['tag'] == 'person':
            bbox = gtbox['vbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2]) + int(bbox[0])
            ymax = int(bbox[3]) + int(bbox[1])
            bboxs.append(('person', xmin, ymin, xmax, ymax))

    # get image info
    img = cv2.imread(f'{src_img_dir}/{img_id}.jpg')
    img_path = f'{prefix_folder}{img_id}.jpg'
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_d = img.shape[2]

    # save annotation info to xml file
    dst_ann_path = f'{dst_ann_dir}/{img_id}.xml'
    writer = pascal_voc_writer.Writer(img_path, img_w, img_h, img_d)
    for bbox in bboxs:
        writer.addObject(*bbox)
    writer.save(dst_ann_path)


# %%
if __name__ == '__main__':

    if not os.path.exists(dst_ann_dir):
        os.makedirs(dst_ann_dir)

    with open(src_ann_file, 'r') as f:
        lines = f.readlines()

    # multi thread
    result = Parallel(n_jobs=32)(delayed(gen_xml)(line) for line in tqdm(lines))

    # single thread
    # for line in tqdm(lines):
    #     gen_xml(line)

# %%

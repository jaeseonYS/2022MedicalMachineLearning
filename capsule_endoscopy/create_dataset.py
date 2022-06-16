import numpy as np
import pandas as pd
import cv2
import json
import base64
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


IMG_SIZE = 320
base_path = Path('../../data/capsule_endoscopy/')
train_path = list((base_path /'train').glob('train*'))
test_path = list((base_path / 'test').glob('test*'))

label_info = pd.read_csv((base_path /'class_id_info.csv'))
categories = {i[0]: i[1]-1 for i in label_info.to_numpy()}
names = {i[1]-1: i[0] for i in label_info.to_numpy()}

def xyxy2coco(xyxy):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


def xyxy2yolo(xyxy):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    xc = x1 + int(np.round(w / 2))
    yc = y1 + int(np.round(h / 2))
    return [xc / IMG_SIZE, yc / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE]


def scale_bbox(img, xyxy):
    scale_x = IMG_SIZE / img.shape[1]
    scale_y = IMG_SIZE / img.shape[0]
    x1, y1, x2, y2 = xyxy
    x1 = int(np.round(x1 * scale_x, 4))
    y1 = int(np.round(y1 * scale_y, 4))
    x2 = int(np.round(x2 * scale_x, 4))
    y2 = int(np.round(y2 * scale_y, 4))
    return [x1, y1, x2, y2]


def save_image_label(json_file, mode):
    with open(json_file, 'r') as f:
        json_file = json.load(f)

    image_id = json_file['file_name'].replace('.json', '')

    # decode image data
    image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(str(new_image_path / (image_id + '.png')), image)

    # extract bbox
    origin_bbox = []
    if mode == 'train':
        with open(new_label_path / (image_id + '.txt'), 'w') as f:
            for i in json_file['shapes']:
                bbox = i['points'][0] + i['points'][2]
                origin_bbox.append(bbox + [categories[i['label']]])
                bbox = scale_bbox(image, bbox)
                bbox = xyxy2yolo(bbox)
                labels = [categories[i['label']]] + bbox
                f.writelines([f'{i} ' for i in labels] + ['\n'])
    return origin_bbox


# [train]
# 저장할 파일 경로
save_path = Path('../datasets/train')
new_image_path = save_path / 'images'
new_label_path = save_path / 'labels'
new_image_path.mkdir(parents=True, exist_ok=True)
new_label_path.mkdir(parents=True, exist_ok=True)

# 데이터 생성
tmp = Parallel(n_jobs=mp.cpu_count(), prefer="threads")(delayed(save_image_label)(str(train_json), 'train') for train_json in tqdm(train_path))

# 데이터 시각화
for n in range(20):
    filename = train_path[n].name.replace('.json', '.png')
    sample = cv2.imread(f'../datasets/images/{filename}')[:, :, ::-1].astype(np.uint8)
    for i in tmp[n]:
        bbox = list(map(int, i))
        label = names[i[4]]
        sample = cv2.rectangle(sample, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
        sample = cv2.putText(sample, label,  (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    plt.imsave(f'../outputs/{filename}', sample)

# 학습, 검증 데이터 분리
images_path = list(new_image_path.glob('*'))
train_path_list, valid_path_list = train_test_split(images_path, test_size=0.1, random_state=42)
with open('../datasets/train/train_dataset.txt', 'w') as f:
    f.writelines([f'{str(i)[3:]}\n' for i in train_path_list])
with open('../datasets/train/valid_dataset.txt', 'w') as f:
    f.writelines([f'{str(i)[3:]}\n' for i in valid_path_list])

# [test]
# 저장할 파일 경로
save_path = Path('../datasets/test').resolve()
new_image_path = save_path / 'images'
new_image_path.mkdir(parents=True, exist_ok=True)
test_path_list = list(new_image_path.glob('*'))

# 데이터 생성
tmp = Parallel(n_jobs=2, prefer="threads")(delayed(save_image_label)(str(test_json), 'test') for test_json in tqdm(test_path))

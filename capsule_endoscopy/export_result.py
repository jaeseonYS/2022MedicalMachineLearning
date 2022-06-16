import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def xywh2xyxy(x):
    y = x.copy()
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


total_list = []
results = {
    'file_name': [], 'class_id': [], 'confidence': [],
    'point1_x': [], 'point1_y': [],
    'point2_x': [], 'point2_y': [],
    'point3_x': [], 'point3_y': [],
    'point4_x': [], 'point4_y': []
}

result_path = Path('../test/endoscopy/exp')
result_img = list(result_path.glob('*.png'))
result_label = list(result_path.glob('labels/*.txt'))

for i in result_label:
    with open(str(i), 'r') as f:
        file_name = i.name.replace('.txt', '.json')
        img_name = file_name.replace('.json', '.png')
        ow, oh, _ = cv2.imread(str(result_path / img_name))[:, :, ::-1].shape
        for line in f.readlines():
            corrdi = line[:-1].split(' ')
            label, xc, yc, w, h, score = corrdi
            xc, yc, w, h, score = list(map(float, [xc, yc, w, h, score]))
            xc, w = np.array([xc, w]) * ow
            yc, h = np.array([yc, h]) * oh

            refine_cordi = xywh2xyxy([xc, yc, w, h])
            refine_cordi = np.array(refine_cordi).astype(int)
            x_min, y_min, x_max, y_max = refine_cordi

            results['file_name'].append(file_name)
            results['class_id'].append(label)
            results['confidence'].append(score)
            results['point1_x'].append(x_min)
            results['point1_y'].append(y_min)
            results['point2_x'].append(x_max)
            results['point2_y'].append(y_min)
            results['point3_x'].append(x_max)
            results['point3_y'].append(y_max)
            results['point4_x'].append(x_min)
            results['point4_y'].append(y_max)


df = pd.DataFrame(results)
df['class_id'] = df['class_id'].apply(lambda x: int(x)+1)
pd.DataFrame(df).to_csv('../outputs/final_submission.csv', index=False)

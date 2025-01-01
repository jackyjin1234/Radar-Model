import cv2
import os

if __name__ == '__main__':
    name1 = os.listdir('./pred_img')
    pred = [f"./pred_img/{file_name}" for file_name in name1]
    pred = [cv2.imread(file) for file in pred]

    name2 = os.listdir('./test/label_img')
    origin = [f"./test/label_img/{file_name}" for file_name in name2]
    origin = [cv2.imread(file) for file in origin]

    for i in range(105):
        img_v = cv2.vconcat([pred[i], origin[i]])
        cv2.imwrite(f'./compare/{name2[i]}.jpg', img_v)  
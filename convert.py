import os
import cv2
import numpy as np
from torchvision import transforms

def out2img4(out, name, dir, d=200):  # 输出反演介电质伪彩色图像
    img = np.log((abs(out) * 300) + 1)

    # v img = img * (64/6)
    img = ((img + 3) % 6) * (255 / 6)

    img = np.reshape(img, (70, d))
    img = img.astype(np.uint8).reshape((70, d))

    # 生成RGB颜色渐变表
    gradient = []
    for i in range(0, 64):
        gradient.append((255, i * 4, 0))
    for i in range(0, 64):
        gradient.append((255 - i * 4, 255, 0))
    for i in range(0, 64):
        gradient.append((0, 255, i * 4))
    for i in range(0, 64):
        gradient.append((0, 255 - i * 4, 255))

    s0, s1 = img.shape
    c = []
    for i in range(0, s0):
        for j in range(0, s1):
            c.append(gradient[img[i, j]])
    img2 = np.array(c, dtype=np.uint8).reshape((70, d, 3))

    cv2.imwrite(f'./{dir}/{name}.jpg', img2)

if __name__ == '__main__':
    list = os.listdir('./train/labels')
    for file in list:
        matrix = np.load(f"train/labels/{file}")
        matrix = transforms.Compose([transforms.ToTensor()])(matrix)
        matrix = matrix.cpu().detach().numpy()
        out2img4(matrix,file, 'train/label_img')
        print(file)
import cv2
import os

if __name__ == '__main__':
    folder = 'Tunneling/shanxi-xishi/fy_imgs/ygy'
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        images.append(img)
    result = cv2.hconcat(images)
    cv2.imwrite('7.1.2024/shanxi_xishi/shanxi_xishi_ygy_scan.jpg', result)

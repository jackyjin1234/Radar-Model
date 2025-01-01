import torch
import cv2
import numpy as np
from torchvision import transforms
from train1 import MyDataset, GPRLoss
from train1 import GPRNet
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from PIL import Image


def out2img3(out, name):  # 输出反演介电质伪彩色图像
    img = out * 255
    # img = img**2
    img = img.astype(np.uint8).reshape((70, 200))
    # print(img.shape)
    img2=cv2.applyColorMap(img, 9)
    # print(img2.shape)
    cv2.imwrite(f'pred_img/{name}.jpg', img2)

def out2img4(out, name, d=200):  # 输出反演介电质伪彩色图像
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

    cv2.imwrite(f'pred_img/{name}.jpg', img2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToPILImage()

test_dataset = MyDataset('test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

gpr = GPRNet()
gpr.to(device)

model_weights_path = r'Mymodel/01.pth'
gpr.load_state_dict(torch.load(model_weights_path, map_location=device))

ssim_loss = GPRLoss()
gpr.eval()
loss = 0.0

with torch.no_grad():
    for i, (inputs, targets, idx) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = gpr(inputs)
        outputs = outputs.to(dtype=torch.float)

        targets[targets < 8] = 1
        targets[targets > 10] = 1
        targets[targets > 2] = 0
        targets = targets.to(dtype=torch.float)
        
        loss += torch.mean(ssim(outputs, targets, data_range=1, win_size=5, size_average=True, nonnegative_ssim=True)).item()

        for j in range(outputs.shape[0]):
            model_output_sample = outputs[j, 0, :, :]
            model_output_sample_cpu = model_output_sample.cpu().detach().numpy()

            image = Image.fromarray(np.uint8(model_output_sample_cpu * 255), 'L')
            image.save(f'./pred_img/{idx[j]}.png')
            # out2img4(model_output_sample.cpu().detach().numpy(), idx[j])
    
loss /= len(test_loader)
print(f"Testing SSIM: {loss:.4f}")


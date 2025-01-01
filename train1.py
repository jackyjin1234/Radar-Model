import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim, ssim, MS_SSIM
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import os

class MyDataset():
    def __init__(self, root_dir, transform=True):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'images'))  # Folder containing your input images
        self.labels = os.listdir(os.path.join(root_dir, 'labels'))    # Folder containing your segmentation masks
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, 'images', self.images[idx])
        label_name = os.path.join(self.root_dir, 'labels', self.labels[idx])

        image = Image.open(img_name).convert('L')
        label = np.load(label_name)

        image = transforms.Compose([transforms.ToTensor()])(image)
        label = transforms.Compose([transforms.ToTensor()])(label)

        return image, label, self.labels[idx]

class GPRNet(torch.nn.Module):
    def __init__(self):
        super(GPRNet, self).__init__()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=5, padding="same")
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=5, padding="same")
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=5, padding="same")
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.conv4 = torch.nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv5 = torch.nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.bn5 = torch.nn.BatchNorm2d(64)

        self.dropout = torch.nn.Dropout2d(0.2)

        self.permute = lambda x: x.permute(0, 1, 3, 2)

        self.fc1 = torch.nn.Linear(500, 512)
        self.bn6 = torch.nn.BatchNorm2d(64)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn7 = torch.nn.BatchNorm2d(64)
        self.fc3 = torch.nn.Linear(256, 256)
        self.bn8 = torch.nn.BatchNorm2d(64)
        self.fc4 = torch.nn.Linear(256, 45)
        self.bn9 = torch.nn.BatchNorm2d(64)

        self.permute_back = lambda x: x.permute(0, 1, 3, 2)

        self.convT1 = torch.nn.ConvTranspose2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn11 = torch.nn.BatchNorm2d(128)

        self.upsample = torch.nn.Upsample(size=[70, 200], mode='bilinear', align_corners=False)
        self.conv7 = torch.nn.Conv2d(128, 64, kernel_size=3, padding="same")
        self.bn12 = torch.nn.BatchNorm2d(64)

        self.conv8 = torch.nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn13 = torch.nn.BatchNorm2d(64)
        self.conv9 = torch.nn.Conv2d(64, 32, kernel_size=3, padding="same")
        self.bn14 = torch.nn.BatchNorm2d(32)
        self.conv10 = torch.nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn15 = torch.nn.BatchNorm2d(32)

        self.dropout2 = torch.nn.Dropout2d(0.2)

        self.output_layer = torch.nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))  #(batch_size,4,500,200])
        x = self.bn1(x)
        x = F.relu(self.conv2(x)) #(batch_size,8,500,200)
        x = self.bn2(x)
        x = F.relu(self.conv3(x)) #(batch_size,16,500,200)
        x = self.bn3(x)
        x = F.relu(self.conv4(x)) #(batch_size,32,500,200)
        x = self.bn4(x)
        x = F.relu(self.conv5(x)) #(batch_size,64,500,200)
        x = self.bn5(x)
        x = self.dropout(x)

        x = self.permute(x)

        x = F.relu(self.fc1(x))
        x = self.bn6(x)
        x = F.relu(self.fc2(x))
        x = self.bn7(x)
        x = F.relu(self.fc3(x))
        x = self.bn8(x)
        x = F.relu(self.fc4(x))
        x = self.bn9(x)
        x = self.dropout(x)

        x = self.permute_back(x)

        x = self.convT1(x)
        x = F.relu(self.conv6(x))
        x = self.bn11(x)

        x = self.upsample(x)

        x = F.relu(self.conv7(x))
        x = self.bn12(x)
        x = F.relu(self.conv8(x))
        x = self.bn13(x)
        x = F.relu(self.conv9(x))
        x = self.bn14(x)
        x = F.relu(self.conv10(x))
        x = self.bn15(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x

class GPRLoss(torch.nn.Module):
    def __init__(self, k = 0.5, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GPRLoss, self).__init__()
        self.k = k
        self.device = device
        self.msssim_loss = MS_SSIM(win_size=5, data_range=1, size_average=True, channel=1).to(self.device)
    def forward(self, predict, target):
        mse = F.mse_loss(predict, target)
        mssim = 1 - self.msssim_loss(predict, target)
        # print(f"Loss: {mse}(mse) + {mssim}(ms_ssim)")
        return mse + mssim

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset('train')
    val_dataset = MyDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

    gpr = GPRNet()
    gpr.to(device)
    loss = GPRLoss()

    optimizer = optim.Adam(gpr.parameters(), lr=0.0002)

    for epoch in range(100):
        gpr.train()
        train_loss = 0.0
        ssim_val = 0.0
        count = 0
        for i, (inputs, targets, idx) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = gpr(inputs)
            outputs = outputs.to(dtype=torch.float)

            targets[targets < 8] = 1
            targets[targets > 10] = 1
            targets[targets > 2] = 0
            targets = targets.to(dtype=torch.float)
            # targets = gpr.bn0(targets)

            result_loss = loss(outputs, targets)  # Compute loss
            result_loss.requires_grad_(True)

            # Backward pass and optimization
            result_loss.backward()  # Compute gradients of all variables wrt loss
            optimizer.step()

            train_loss += result_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        gpr.eval()
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for j, (val_inputs, val_targets, idx) in enumerate(val_loader):
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = gpr(val_inputs)
                val_outputs = val_outputs.to(dtype=torch.float)
                
                val_targets[val_targets < 8] = 1
                val_targets[val_targets > 10] = 1
                val_targets[val_targets > 2] = 0
                val_targets = val_targets.to(dtype=torch.float)
                # val_targets = gpr.bn0(val_targets)
                
                val_result_loss = loss(val_outputs, val_targets)
                val_loss += val_result_loss.item()

                ssim_val += torch.mean(ssim(val_outputs, val_targets, data_range=1, win_size=5, size_average=True, nonnegative_ssim=True)).item()
                count = j

        ssim_val /= len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{100}], Ave Loss: {avg_train_loss:.4f}, Ave Val Loss: {avg_val_loss:.4f}, Val_ssim: {ssim_val:.4f}')

    torch.save(gpr.state_dict(), './Mymodel/01.pth')

    
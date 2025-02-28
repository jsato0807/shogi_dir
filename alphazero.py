import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):  #https://tech-blog.optim.co.jp/entry/2021/12/02/100000#%E6%AC%A1%E3%81%AE%E4%B8%80%E6%89%8B%E3%82%92%E4%BA%88%E6%B8%AC%E3%81%99%E3%82%8BPolicyNetwork%E4%BD%9C%E6%88%90
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 104, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv10 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv12 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv13 = nn.Conv2d(in_channels = 256, out_channels = 14, kernel_size = 1, padding = 1)
        self.fc1 = nn.Linear(14 * 11 * 11, 14 * 9 * 9)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x
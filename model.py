import torch.nn as nn
import torch.nn.functional as F


class MnistNet_1(nn.Module):
    def __init__(self):
        super(MnistNet_1, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),    # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            # nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),   # 24 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 22
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),


            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),   # 22 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 22
            nn.MaxPool2d(kernel_size=2, stride=2)                                   # 22 / 2 = 11
        )   

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),   # 11 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 9
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),  # 9 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 7
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),  # 7 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 5
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False),  # 5 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 3
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, bias=False),  # 3 + 2*0 - 1*(2 - 1) - 1 / 1 + 1 = 2
            # # nn.BatchNorm2d(16),
            # # nn.Dropout(0.1),
            # nn.ReLU(),

            # nn.Conv2d(in_channels=16, out_channels=10, kernel_size=2, bias=False),  # 2 + 2*0 - 1*(2 - 1) - 1 / 1 + 1 = 1
            # # nn.BatchNorm2d(16),
            # # nn.Dropout(0.1),
            # nn.ReLU(),
        )
        
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, bias=False)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        # x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)


class MnistNet_2(nn.Module):
    def __init__(self):
        super(MnistNet_2, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),    # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),   # 24 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 24
            nn.MaxPool2d(kernel_size=2, stride=2)                                   # 24 / 2 = 12
        )   

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),   # 12 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 10
            nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),  # 10 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 8
            nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),  # 8 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 6
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False),  # 6 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 6
            nn.ReLU(),

        )
        
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=6, bias=False)   # 6 + 2*0 - 1*(6 - 1) - 1 / 1 + 1 = 1
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        # x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)


class MnistNet_3(nn.Module):
    def __init__(self):
        super(MnistNet_3, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),    # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            nn.BatchNorm2d(8),
            # nn.Dropout(0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),   # 24 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 24
            nn.MaxPool2d(kernel_size=2, stride=2)                                   # 24 / 2 = 12
        )   

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, bias=False),   # 12 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 10
            nn.BatchNorm2d(12),
            nn.Dropout(0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, bias=False),  # 10 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 8
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),  # 8 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 6
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.ReLU(),

            # nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False),  # 6 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 6
            # nn.ReLU(),

        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)   # 1 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 6
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)
    

class MnistNet_4(nn.Module):
    def __init__(self):
        super(MnistNet_4, self).__init__()
        drop_rate = 0.05
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),    # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(drop_rate),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),   # 24 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 24
            nn.MaxPool2d(kernel_size=2, stride=2)                                    # 24 / 2 = 12
        )   

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),   # 12 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),  # 10 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),  # 8 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                    # 6 + 2*0 - 1*(6 - 1) - 1 / 1 + 1 = 1

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)   # 1 + 2*0 - 1*(1 - 1) - 1 / 1 + 1 = 1
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)
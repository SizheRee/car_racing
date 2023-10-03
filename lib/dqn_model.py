import torch
import torch.nn as nn
import numpy as np




class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(  # input shape 4 x 96 x 96
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # 32 x 23 x 23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 64 x 10 x 10
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=2, stride=2), # 64 x 5 x 5
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), # 64 x 4 x 4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), # 64 x 3 x 3
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

if __name__ == "__main__":
    net = DQN(input_shape=(4, 84, 84), n_actions=14)
    print(net)
    # generate random input
    input = torch.randn(32, 4, 84, 84)
    print("Input size:")
    print(input.size())
    
    output = net(input)
    print("Output size:")
    print(output.size())

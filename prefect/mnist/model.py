import torch


class MnistNet(torch.nn.Module):
    def __init__(self, l1):
        super(MnistNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(7 * 7 * 64, l1, bias=True)
        self.fc2 = torch.nn.Linear(l1, 32, bias=True)
        self.last_layer = torch.nn.Linear(32, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.last_layer(out)
        return out

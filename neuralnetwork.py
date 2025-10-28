import torch.nn as nn

# This is the multi layer perceptron neural network which has good capabilities
# when it comes to pattern recognition which is perfect for stock prediction
# as it can develop patterns that will most likely make sense to the stock market.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Sigmoid()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, output_count: int, ):
        super(NeuralNetwork, self).__init__()

        if not type(output_count) == int or output_count < 1:
            raise Exception(f'Failed to init: <output_count: {output_count} {type(output_count)}>')

        self.linear_stack = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_count),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

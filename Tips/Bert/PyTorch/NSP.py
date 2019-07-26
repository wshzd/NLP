class NextSentencePrediction(nn.Module):

    def __init__(self, hidden):

        super(NextSentencePrediction, self).__init__()

        self.linear = nn.Linear(hidden, 2)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):

    def __init__(self, hidden, vocab_size):

        super(MaskedLanguageModel, self).__init__()

        self.linear = nn.Linear(hidden, vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x))

class BERT(PreTrainedBERTModel):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):

        config = BertConfig(vocab_size, hidden_size=hidden, num_hidden_layers=n_layers,num_attention_heads=attn_heads, hidden_dropout_prob=dropout)

        super(BERT, self).__init__(config)

        self.hidden = hidden

        self.n_layers = n_layers

        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:

            x = transformer.forward(x, mask)

        return x

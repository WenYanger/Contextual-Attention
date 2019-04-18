class Attention(nn.Module):
    def __init__(self, input_shape):
        super(Attention, self).__init__()

        self.max_len = input_shape[1]
        self.emb_size = input_shape[2]

        self.weight = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.bias = nn.Parameter(torch.Tensor(self.max_len, 1))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'max_len={}, emb_size={}'.format(
            self.max_len, self.emb_size
        )

    def forward(self, x, mask=None):
        # Here    x should be [batch_size, time_step, emb_size]
        #      mask should be [batch_size, time_step, 1]

        W_bs = self.weight.unsqueeze(0).repeat(x.size()[0], 1, 1)  # Copy the Attention Matrix for batch_size times
        scores = torch.bmm(x, W_bs)  # Dot product between input and attention matrix
        scores = torch.tanh(scores)

        # scores = Cal_Attention()(x, self.weight, self.bias)

        if mask is not None:
            mask = mask.long()
            scores = scores.masked_fill(mask == 0, -1e9)

        a_ = F.softmax(scores.squeeze(-1), dim=-1)
        a = a_.unsqueeze(-1).repeat(1, 1, x.size()[2])

        weighted_input = x * a

        output = torch.sum(weighted_input, dim=1)

        return output, a_

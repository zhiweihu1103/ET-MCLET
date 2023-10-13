import torch
import torch.nn as nn

class CSRAWithMoE(nn.Module):
    def __init__(self, T, lam):
        super(CSRAWithMoE, self).__init__()
        self.T = T
        self.lam = lam
        self.softmax = nn.Softmax(dim=1)

    def forward(self, score, weight):
        score = score * weight
        base_logit = torch.mean(score, dim=1)

        if self.T == 99:
            att_logit = torch.max(score, dim=1)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=1)

        return base_logit + self.lam * att_logit


class MHAwithMoE(nn.Module):
    temp_settings = {
        1: [3],
        2: [3, 99],
        3: [2, 4, 99],
        4: [2, 3, 4, 99],
        5: [2, 2.5, 3.5, 4.5, 99],
        6: [2, 3, 4, 5, 6, 99],
        7: [0.5, 2.5, 3.5, 4.5, 5.5, 6.5, 99],
        8: [0.5, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, embedding_dim, weight=False, gating_units=32, gating_activation='softmax'):
        super(MHAwithMoE, self).__init__()
        self.num_heads = num_heads
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRAWithMoE(self.temp_list[i], lam)
            for i in range(num_heads)
        ])
        self.weight = nn.Parameter(torch.ones(num_heads, 1))
        if weight:
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

        self.gating = nn.Sequential(
            nn.Linear(embedding_dim, gating_units),
            nn.ModuleDict([
                ['softmax', nn.Softmax(dim=2)],
                ['sigmoid', nn.Sigmoid()]
            ])[gating_activation],
            nn.Linear(gating_units, num_heads),
            nn.ModuleDict([
                ['softmax', nn.Softmax(dim=2)],
                ['sigmoid', nn.Sigmoid()]
            ])[gating_activation]
        )

    def forward(self, x):
        logit = 0.
        gating_outputs = self.gating(x)
        index = 0
        for head in self.multi_head:
            weight = gating_outputs[:, :, index:index+1]
            logit += head(x, weight)
            index = index + 1

        return logit
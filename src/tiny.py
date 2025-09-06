import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class TinyLM(nn.Module):
    def __init__(self, d_model=8, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.A = nn.Linear(d_model, d_model, bias=False)
        self.B = nn.Linear(d_model, d_model, bias=False)

    def forward_from_embeds(self, embeds):
        # embeds: [L, D]
        L, D = embeds.shape
        h = torch.zeros(1, D, device=embeds.device)
        outs = []
        for t in range(L):
            h = torch.tanh(self.A(h) + self.B(embeds[t:t+1]))
            outs.append(h)
        return torch.cat(outs, dim=0)  # [L, D]

    def forward_tokens(self, tokens):
        embeds = self.embed(tokens)          # non-leaf, requires_grad=True
        hs = self.forward_from_embeds(embeds)
        return hs, embeds

def run_trial(mode="clone"):
    model = TinyLM(d_model=8, vocab_size=100).train()
    L, latent_pos = 6, 4
    tokens1 = torch.randint(0, 100, (L,))
    tokens2 = torch.randint(0, 100, (L,))

    # Pass 1: build prev_hidden with grad enabled
    hs1, _ = model.forward_tokens(tokens1)
    prev_hidden = hs1[latent_pos - 1]
    prev_hidden.retain_grad()

    if mode == "clone":
        ce_vec = prev_hidden.clone()  # keep graph
    elif mode == "detach":
        ce_vec = prev_hidden.detach()  # cut graph
    elif mode == "ste":
        # Straight-Through Estimator
        ce_vec = prev_hidden.detach() + (prev_hidden - prev_hidden.detach())
    else:
        raise ValueError(mode)

    # ---- Crucial: keep base as NON-leaf ----
    embeds2 = model.embed(tokens2)    # non-leaf
    embeds2 = embeds2.clone()         # still non-leaf
    embeds2[latent_pos] = ce_vec      # in-place on non-leaf is OK

    hs2 = model.forward_from_embeds(embeds2)
    y = torch.randn_like(hs2[-1])
    loss_ce = F.mse_loss(hs2[-1], y)

    model.zero_grad(set_to_none=True)
    loss_ce.backward()

    return {
        "mode": mode,
        "||grad(prev_hidden)||": None if prev_hidden.grad is None else prev_hidden.grad.norm().item(),
        "||grad(A)||": None if model.A.weight.grad is None else model.A.weight.grad.norm().item(),
        "||grad(B)||": None if model.B.weight.grad is None else model.B.weight.grad.norm().item(),
    }

for m in ["clone", "detach", "ste"]:
    print(run_trial(m))

# source: https://github.com/thuml/Time-Series-Library/tree/main
import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class GrangerCausalMask():
    def __init__(self, graph_sampled, device="cpu"):
        assert graph_sampled.dim() == 3, "graph_sampled should be a 3D tensor (B, N, N)"
        graph_sampled = (~graph_sampled.bool())
        graph_sampled.unsqueeze_(1)  # Add a dimension for heads
        
        with torch.no_grad():
            self._mask = graph_sampled.clone().detach()

    @property
    def mask(self):
        return self._mask
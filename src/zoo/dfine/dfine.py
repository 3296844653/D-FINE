"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        use_p2_roi = getattr(self.decoder, "use_p2_roi_cls", False)
        use_p2_query_init = getattr(self.decoder, "use_p2_query_init", False)
        use_p2_decoder = use_p2_roi or use_p2_query_init
        use_p2_detail = getattr(self.encoder, "use_p2_detail_fusion", False)
        if use_p2_decoder and use_p2_detail:
            raise ValueError("P2 decoder probes and P2 detail fusion are separate ablations")

        if use_p2_decoder:
            # Keep the detector's original P3-P5 path intact. The extra P2
            # feature is used only by the selected decoder-side probe.
            if len(x) != 4:
                raise ValueError("P2 decoder probes require backbone return_idx=[0,1,2,3]")
            p2_feat, x = x[0], x[1:]
        elif use_p2_detail:
            if len(x) != 4:
                raise ValueError("P2 detail fusion requires backbone return_idx=[0,1,2,3]")
            # HybridEncoder consumes [P2, P3, P4, P5] and still returns the
            # standard three levels expected by the decoder.
            p2_feat = None
        else:
            p2_feat = None
            # Toggling the experiment config off must also recover the normal
            # three-level detector even if return_idx still includes P2.
            if len(x) == 4 and list(getattr(self.backbone, "return_idx", [])) == [0, 1, 2, 3]:
                x = x[1:]
        x = self.encoder(x)
        if p2_feat is None:
            x = self.decoder(x, targets)
        else:
            x = self.decoder(x, targets, p2_feat=p2_feat)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn

from ...core import register
from .night_probes import NIGHT_PROBES, BackboneBandPass, EncoderCrossGate
from .finegrain_trials import FINEGRAIN_TRIALS, DetailOnlyCrossGate, WindowInteractionP3
from .spatial_evidence_relation_fusion import SERF_PLACEMENTS, SpatialEvidenceRelationFusion
from .box_local_extensions import BOX_LOCAL_EXTENSIONS, build_box_local_extension

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
        night_probe: str = "none",
        allow_aqs_l2_local_combo: bool = False,
        allow_aqs_l2_query_init_combo: bool = False,
        finegrain_trial: str = "none",
        finegrain_dim: int = 64,
        finegrain_residual_scale: float = 0.1,
        finegrain_grid_size: int = 5,
        finegrain_parts: int = 4,
        finegrain_window: int = 5,
        serf_placement: str = "none",
        serf_dim: int = 64,
        serf_window: int = 8,
        serf_halo: int = 2,
        serf_slots: int = 4,
        serf_heads: int = 4,
        serf_scale: float = 0.1,
        serf_chunk: int = 128,
        serf_checkpoint: bool = True,
        box_local_dim: int = 64,
        box_local_grid_size: int = 3,
        box_local_offset_scale: float = 0.15,
        box_local_residual_scale: float = 0.1,
        box_local_record_settings: bool = False,
        aqs_l2_local_fusion: str = "serial",
        box_local_extension: str = "none",
        box_extension_dim: int = 32,
        box_extension_scale: float = 0.1,
        box_extension_chunk: int = 32,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        if aqs_l2_local_fusion not in ("serial", "parallel"):
            raise ValueError("aqs_l2_local_fusion must be serial or parallel")
        if aqs_l2_local_fusion == "parallel" and not allow_aqs_l2_local_combo:
            raise ValueError("Parallel fusion requires the explicit AQS/local-L2 combination")
        self.aqs_l2_local_fusion = aqs_l2_local_fusion
        if box_local_extension not in ('none', *BOX_LOCAL_EXTENSIONS):
            raise ValueError('Choose one supported BoxLocal extension')
        if box_local_extension == 'none' and (box_extension_dim, box_extension_scale, box_extension_chunk) != (32, .1, 32):
            raise ValueError('BoxLocal extension settings require an enabled extension')
        self.box_local_extension = box_local_extension
        box_options = dict(dim=box_local_dim, grid_size=box_local_grid_size,
            offset_scale=box_local_offset_scale, residual_scale=box_local_residual_scale,
            record_settings=box_local_record_settings)
        if (box_options != dict(dim=64, grid_size=3, offset_scale=.15, residual_scale=.1, record_settings=False)
                and night_probe not in ("query_local_init", "decoder_l2_local")):
            raise ValueError("BoxLocal parameters are only supported for query_local_init / decoder_l2_local")
        if serf_placement not in ("none", *SERF_PLACEMENTS):
            raise ValueError("Unknown SERF placement")
        self.serf_placement = serf_placement
        if serf_placement != "none":
            decoder_flags = ("use_local_evidence_cls", "use_query_validity", "use_query_objectness",
                "use_p2_roi_cls", "use_p2_query_init", "use_decoder_local_query_cls", "use_task_decoupled_heads",
                "use_rw_spatial_relation", "use_rw_local_detail", "use_rw_isolated_specialist",
                "use_rw_query_calibration", "use_discriminative_evidence_l2")
            if (night_probe != "none" or finegrain_trial != "none"
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo
                    or any(getattr(backbone, f, False) for f in ("use_afta", "use_backbone_p3_local_probe"))
                    or any(getattr(encoder, f, False) for f in ("use_caf", "use_sfif", "use_mffe", "use_rfa_p3",
                        "use_p3_rfaconv_residual", "use_p2_detail_fusion", "use_encoder_highres_residual",
                        "use_encoder_p3_joint_residual", "use_sadf", "use_sdrf", "use_subpixel_context_fusion"))
                    or any(getattr(decoder, f, False) for f in decoder_flags)
                    or any(getattr(decoder, f, "none") != "none" for f in
                        ("literature_probe", "instance_evidence_module", "behavior_refinement_module"))):
                raise ValueError("SERF placements are two independent additions to plain AQS-L2 R1")
            if (not getattr(decoder, "use_aqs_refine", False)
                    or decoder.decoder.aqs_layer_indices != (1,)
                    or decoder.decoder.aqs_refiner_type != "legacy"
                    or len(encoder.in_channels) != 3 or list(encoder.feat_strides) != [8, 16, 32]
                    or list(getattr(backbone, "return_idx", [])) != [1, 2, 3]
                    or encoder.hidden_dim != decoder.hidden_dim):
                raise ValueError("SERF requires unchanged P3/P4/P5 routing and single L2 AQS")
            channels = encoder.in_channels[:2] if serf_placement == "SERF_Backbone" else [encoder.hidden_dim] * 2
            # Added branch must not change baseline RNG/pretrained initialization.
            with torch.random.fork_rng(devices=[]):
                self.serf = SpatialEvidenceRelationFusion(*channels, serf_placement,
                    dim=serf_dim, window=serf_window, halo=serf_halo, slots=serf_slots,
                    heads=serf_heads, scale=serf_scale, chunk=serf_chunk, use_checkpoint=serf_checkpoint)
        if getattr(decoder, "behavior_refinement_module", "none") != "none":
            if (night_probe != "none" or finegrain_trial != "none"
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo
                    or getattr(decoder, "literature_probe", "none") != "none"
                    or any(getattr(backbone, flag, False) for flag in ("use_afta", "use_backbone_p3_local_probe"))
                    or any(getattr(encoder, flag, False) for flag in ("use_caf", "use_sfif", "use_mffe",
                        "use_rfa_p3", "use_p3_rfaconv_residual", "use_p2_detail_fusion",
                        "use_encoder_highres_residual", "use_encoder_p3_joint_residual",
                        "use_sadf", "use_sdrf", "use_subpixel_context_fusion"))):
                raise ValueError('Behavior refinements cannot stack feature probes or BoxLocal; standalone GQC uses the plain baseline')
            if (encoder.hidden_dim != decoder.hidden_dim or len(encoder.in_channels) != 3
                    or list(encoder.feat_strides) != [8,16,32]
                    or list(getattr(backbone, "return_idx", [])) != [1,2,3]):
                raise ValueError("Behavior refinement requires unchanged P3/P4/P5 feature routing")
        if getattr(encoder, "use_subpixel_context_fusion", False):
            other_decoder_flags = ("use_local_evidence_cls", "use_query_validity", "use_query_objectness",
                "use_p2_roi_cls", "use_p2_query_init", "use_decoder_local_query_cls",
                "use_task_decoupled_heads", "use_rw_spatial_relation", "use_rw_local_detail",
                "use_rw_isolated_specialist", "use_rw_query_calibration", "use_discriminative_evidence_l2")
            if (night_probe != "none" or finegrain_trial != "none"
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo
                    or getattr(decoder, "instance_evidence_module", "none") != "none"
                    or getattr(decoder, "literature_probe", "none") != "none"
                    or any(getattr(decoder, flag, False) for flag in other_decoder_flags)
                    or any(getattr(backbone, flag, False) for flag in ("use_afta", "use_backbone_p3_local_probe"))):
                raise ValueError("SCF trial adds one Encoder module to unchanged AQS-L2 R1")
            if (not getattr(decoder, "use_aqs_refine", False)
                    or decoder.decoder.aqs_layer_indices != (1,)
                    or decoder.decoder.aqs_refiner_type != "legacy"
                    or list(getattr(backbone, "return_idx", [])) != [0, 1, 2, 3]):
                raise ValueError("SCF trial requires P2 routing and legacy AQS only at L2")
        if (getattr(decoder, "instance_evidence_module", "none") != "none"
                or getattr(decoder, "use_discriminative_evidence_l2", False)):
            if (night_probe != "none" or finegrain_trial != "none"
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo
                    or getattr(decoder, "literature_probe", "none") != "none"
                    or any(getattr(backbone, name, False) for name in
                           ("use_afta", "use_backbone_p3_local_probe"))
                    or any(getattr(encoder, name, False) for name in
                           ("use_caf", "use_sfif", "use_mffe", "use_rfa_p3",
                            "use_p3_rfaconv_residual", "use_p2_detail_fusion",
                            "use_encoder_highres_residual", "use_encoder_p3_joint_residual",
                            "use_sadf", "use_sdrf"))):
                raise ValueError("Instance evidence trials add one module to AQS, not other feature probes")
            if (encoder.hidden_dim != decoder.hidden_dim or len(encoder.in_channels) != 3
                    or list(encoder.feat_strides) != [8, 16, 32]):
                raise ValueError("Instance evidence requires standard P3/P4/P5 encoder routing")
        if finegrain_trial not in ("none", *FINEGRAIN_TRIALS):
            raise ValueError("Unknown finegrain_trial")
        self.finegrain_trial = finegrain_trial
        if finegrain_trial != "none":
            if (night_probe != "none" or getattr(decoder, "literature_probe", "none") != "none"
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo):
                raise ValueError("Finegrain trials must be independent structural experiments")
            for owner, flags in (
                (backbone, ("use_afta", "use_backbone_p3_local_probe")),
                (encoder, ("use_caf", "use_sfif", "use_mffe", "use_rfa_p3",
                    "use_p3_rfaconv_residual", "use_p2_detail_fusion", "use_encoder_highres_residual",
                    "use_encoder_p3_joint_residual", "use_sadf", "use_sdrf")),
            ):
                if any(getattr(owner, flag, False) for flag in flags):
                    raise ValueError("Finegrain trials cannot stack existing feature modules")
            if (len(encoder.in_channels) != 3 or encoder.hidden_dim != decoder.hidden_dim
                    or list(encoder.feat_strides) != [8, 16, 32]):
                raise ValueError("Finegrain trials require standard P3/P4/P5 features")
            decoder.configure_finegrain_trial(finegrain_trial, dim=finegrain_dim,
                scale=finegrain_residual_scale, grid=finegrain_grid_size, parts=finegrain_parts)
            with torch.random.fork_rng(devices=[]):
                if finegrain_trial == "DetailOnlyCrossGate":
                    encoder.finegrain_module = DetailOnlyCrossGate(encoder.hidden_dim,
                        encoder.in_channels[0], finegrain_dim, finegrain_residual_scale)
                elif finegrain_trial == "WindowInteractionP3":
                    encoder.finegrain_module = WindowInteractionP3(encoder.hidden_dim,
                        finegrain_dim, finegrain_residual_scale, finegrain_window)
        if getattr(getattr(decoder, "decoder", None), "aqs_refiner_type", "legacy") == "EvidenceContrastAQS":
            if (night_probe != "none"
                    or any(getattr(backbone, flag, False) for flag in ("use_afta", "use_backbone_p3_local_probe"))
                    or any(getattr(encoder, flag, False) for flag in ("use_caf", "use_sfif", "use_mffe",
                        "use_rfa_p3", "use_p3_rfaconv_residual", "use_p2_detail_fusion",
                        "use_encoder_highres_residual", "use_encoder_p3_joint_residual", "use_sadf", "use_sdrf"))):
                raise ValueError("EvidenceContrastAQS must first be tested as one independent mechanism")
        if getattr(encoder, "use_sadf", False) or getattr(encoder, "use_sdrf", False):
            decoder_flags = ("use_aqs_refine", "use_local_evidence_cls", "use_query_validity",
                "use_query_objectness", "use_p2_roi_cls", "use_p2_query_init",
                "use_decoder_local_query_cls", "use_task_decoupled_heads", "use_rw_spatial_relation",
                "use_rw_local_detail", "use_rw_isolated_specialist", "use_rw_query_calibration")
            if (night_probe != "none"
                    or any(getattr(backbone, flag, False) for flag in ("use_afta", "use_backbone_p3_local_probe"))
                    or any(getattr(decoder, flag, False) for flag in decoder_flags)):
                raise ValueError("SADF/SDRF must first be tested independently from backbone/decoder modules")
        if type(allow_aqs_l2_local_combo) is not bool:
            raise ValueError("allow_aqs_l2_local_combo must be a boolean")
        if type(allow_aqs_l2_query_init_combo) is not bool:
            raise ValueError("allow_aqs_l2_query_init_combo must be a boolean")
        if allow_aqs_l2_local_combo and allow_aqs_l2_query_init_combo:
            raise ValueError("Select only one explicit AQS combination")
        if allow_aqs_l2_local_combo and night_probe != "decoder_l2_local":
            raise ValueError("The explicit AQS combination is only supported for decoder_l2_local")
        if allow_aqs_l2_query_init_combo and night_probe != "query_local_init":
            raise ValueError("The explicit query-init AQS combination is only supported for query_local_init")
        self.allow_aqs_l2_local_combo = allow_aqs_l2_local_combo
        self.allow_aqs_l2_query_init_combo = allow_aqs_l2_query_init_combo
        if night_probe not in ("none", *NIGHT_PROBES):
            raise ValueError("Unknown night_probe: choose one independent screening module")
        self.night_probe = night_probe
        if night_probe != "none":
            # Keep single-variable guards, except the explicitly requested
            # AQS-L2 + local-L2 or query-init experiment validated below.
            for owner, names in (
                (backbone, ("use_afta", "use_backbone_p3_local_probe")),
                (encoder, ("use_caf", "use_sfif", "use_mffe", "use_rfa_p3",
                           "use_p3_rfaconv_residual", "use_p2_detail_fusion",
                           "use_encoder_highres_residual", "use_encoder_p3_joint_residual", "use_sadf", "use_sdrf")),
            ):
                if any(getattr(owner, name, False) for name in names):
                    raise ValueError("Night probes cannot be combined with existing feature modules")
            if len(encoder.in_channels) != 3 or encoder.hidden_dim != decoder.hidden_dim:
                raise ValueError("Night probes require the standard three-level encoder interface")
            if night_probe == "backbone_bandpass":
                with torch.random.fork_rng(devices=[]):
                    self.night_backbone = BackboneBandPass(encoder.in_channels[0])
            if night_probe == "encoder_cross_gate":
                with torch.random.fork_rng(devices=[]):
                    self.night_encoder = EncoderCrossGate(encoder.hidden_dim)
            decoder.configure_night_probe(
                night_probe, allow_aqs_l2_local_combo=allow_aqs_l2_local_combo,
                allow_aqs_l2_query_init_combo=allow_aqs_l2_query_init_combo,
                box_local_options=box_options,
                aqs_l2_local_fusion=aqs_l2_local_fusion,
            )

        if box_local_extension != 'none':
            if (night_probe != 'decoder_l2_local' or decoder.use_aqs_refine
                    or allow_aqs_l2_local_combo or allow_aqs_l2_query_init_combo
                    or aqs_l2_local_fusion != 'serial' or finegrain_trial != 'none' or serf_placement != 'none'
                    or box_options != dict(dim=64, grid_size=3, offset_scale=.15, residual_scale=.1, record_settings=False)
                    or decoder.num_classes != 3 or decoder.num_layers != 3 or decoder.eval_idx != 2
                    or list(encoder.feat_strides) != [8, 16, 32]
                    or list(getattr(backbone, 'return_idx', [])) != [1, 2, 3]):
                raise ValueError('Extensions require the unchanged SCB-S default BoxLocal-L2 baseline, without AQS')
            with torch.random.fork_rng(devices=[]):
                extension = build_box_local_extension(box_local_extension, hidden_dim=decoder.hidden_dim,
                    fine_channels=encoder.in_channels[0], coarse_channels=encoder.in_channels[1],
                    dim=box_extension_dim, scale=box_extension_scale, chunk=box_extension_chunk)
            if box_local_extension == 'DirectionalDetailBackbone':
                # Newly initialized output adapter, NOT pretrained backbone
                # weights: avoid matching the optimizer's '*backbone*' low-LR
                # regex while keeping the original optimizer config unchanged.
                self.box_detail_extension = extension
            elif box_local_extension == 'SemanticChannelRelation':
                encoder.box_semantic_extension = extension
            elif box_local_extension == 'CrossScaleNeighborhoodFusion':
                self.box_encoder_extension = extension
            elif box_local_extension == 'PartRelationQueryInit':
                decoder.box_query_initializer = extension
            elif box_local_extension == 'ReliableQueryCompetition':
                decoder.decoder.box_score_extension = extension

    def forward(self, x, targets=None):
        x = self.backbone(x)
        if hasattr(self, 'box_detail_extension'):
            x = [self.box_detail_extension(x[0], x[1]), *x[1:]]
        if self.serf_placement == "SERF_Backbone":
            # Output-side stage adapter, not an in-place modification before
            # stage3/stage4: P4/P5 backbone computation remains original.
            x = [self.serf(x[0], x[1]), *x[1:]]
        use_p2_roi = getattr(self.decoder, "use_p2_roi_cls", False)
        use_p2_query_init = getattr(self.decoder, "use_p2_query_init", False)
        use_rw_local_detail = getattr(self.decoder, "use_rw_local_detail", False)
        use_p2_night = getattr(self.decoder, "night_probe", "none") == "rw_adaptive_roi"
        use_p2_evidence = (getattr(self.decoder, "instance_evidence_module", "none") != "none"
                           or getattr(self.decoder, "use_discriminative_evidence_l2", False))
        use_p2_decoder = use_p2_roi or use_p2_query_init or use_rw_local_detail or use_p2_night or use_p2_evidence
        use_p2_detail = getattr(self.encoder, "use_p2_detail_fusion", False)
        use_p2_encoder = use_p2_detail or getattr(self.encoder, "use_subpixel_context_fusion", False)
        if use_p2_decoder and use_p2_encoder:
            raise ValueError("P2 decoder probes and P2 detail fusion are separate ablations")

        if use_p2_decoder:
            # Keep the detector's original P3-P5 path intact. The extra P2
            # feature is used only by the selected decoder-side probe.
            if len(x) != 4:
                raise ValueError("P2 decoder probes require backbone return_idx=[0,1,2,3]")
            p2_feat, x = x[0], x[1:]
        elif use_p2_encoder:
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
        if self.night_probe == "backbone_bandpass":
            x = [self.night_backbone(x[0]), *x[1:]]
        source_p3 = x[0] if self.finegrain_trial == "DetailOnlyCrossGate" else None
        x = self.encoder(x)
        if hasattr(self, 'box_encoder_extension'):
            x = [self.box_encoder_extension(x[0], x[1]), *x[1:]]
        if self.serf_placement == "SERF_Encoder":
            # After complete FPN/PAN: P3 changes, P4/P5 encoder outputs do not.
            x = [self.serf(x[0], x[1]), *x[1:]]
        if self.finegrain_trial == "DetailOnlyCrossGate":
            x = [self.encoder.finegrain_module(source_p3, x[0], x[1]), *x[1:]]
        elif self.finegrain_trial == "WindowInteractionP3":
            x = [self.encoder.finegrain_module(x[0]), *x[1:]]
        if self.night_probe == "encoder_cross_gate":
            x = [self.night_encoder(x[0], x[1]), *x[1:]]
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

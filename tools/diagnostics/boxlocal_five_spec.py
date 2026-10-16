"""Stdlib-only definitions: five independent extensions, NEVER an L2 stack."""
VARIANTS = {
    'DDB': ('DirectionalDetailBackbone', 'backbone P3 output, before encoder'),
    'SCR': ('SemanticChannelRelation', 'encoder P5 after AIFI, before FPN/PAN'),
    'CNF': ('CrossScaleNeighborhoodFusion', 'encoder P3 output, after FPN/PAN'),
    'PRQI': ('PartRelationQueryInit', 'query initialization before L1 and DN concatenation'),
    'RQC': ('ReliableQueryCompetition', 'final ordinary-query scores after L3'),
}
BOX_BASELINE = 'configs/dfine/dfine_s_scbs_hrw_110_night_n4.yml'
BASELINE_AP = .5566581543651368
BASELINE_WRITE_AP = .4910974650379758


def config_path(tag):
    if tag not in VARIANTS:
        raise ValueError('Unknown BoxLocal extension tag')
    return f'configs/dfine/dfine_s_scbs_hrw_110_BoxLocalL2_{tag}.yml'


def output_name(tag, seed, run):
    config_path(tag)
    return f'dfine_s_scbs_hrw_epochs110_BoxLocalL2_{tag}_seed{seed}_{run}'

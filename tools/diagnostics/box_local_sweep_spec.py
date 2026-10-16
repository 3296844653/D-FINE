"""Stdlib-only single-factor BoxLocal-L2 experiment definitions."""

DEFAULTS = dict(box_local_dim=64, box_local_grid_size=3,
                box_local_offset_scale=.15, box_local_residual_scale=.1)
PROFILES = {
    'r005': ('box_local_residual_scale', .05),
    'r020': ('box_local_residual_scale', .20),
    'o010': ('box_local_offset_scale', .10),
    'o020': ('box_local_offset_scale', .20),
    'd032': ('box_local_dim', 32),
    'd096': ('box_local_dim', 96),
    'g2': ('box_local_grid_size', 2),
    'g5': ('box_local_grid_size', 5),
    'o000': ('box_local_offset_scale', .0),
}
FIRST_GROUPS = ('r005', 'r020')


def settings(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown BoxLocal sweep group')
    key, value = PROFILES[profile]
    return {**DEFAULTS, key: value}


def module_name(profile):
    settings(profile)
    return f'BoxLocalQueryAdapter_l2_{profile}'

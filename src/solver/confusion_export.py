"""Export an existing Validator confusion matrix without changing evaluation.

Rows are GT classes, columns are predicted classes. The last row contains
unmatched predictions, NOT verified background. The last column contains
unmatched GT. Matching stays class-blind, greedy highest-IoU, one-to-one.
"""
import json
from pathlib import Path

import numpy as np


def export_confusion_matrix(validator, coco, output_dir, metadata=None, remap=False):
    """Include every dataset category, even if it has no observed GT/predictions."""
    categories = sorted(coco.dataset['categories'], key=lambda c: c['id'])
    if remap:
        from ..data.dataset import mscoco_category2label
        ids = [mscoco_category2label[c['id']] for c in categories]
    else:
        ids = [c['id'] for c in categories]
    names = [c['name'] for c in categories]
    index = {category: i for i, category in enumerate(ids)}
    n = len(ids)
    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)
    old = validator.conf_matrix
    old_background = len(validator.class_to_idx)
    mapping = {old_idx: index[category] for category, old_idx in validator.class_to_idx.items()}
    mapping[old_background] = n
    for r, new_r in mapping.items():
        for c, new_c in mapping.items():
            matrix[new_r, new_c] = old[r, c]
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)
    rows = names + ['unmatched_predictions']
    columns = names + ['unmatched_GT']
    report = {
        'row_labels': rows, 'column_labels': columns, 'category_ids': ids,
        'matrix_counts': matrix.tolist(), 'matrix_row_normalized': normalized.tolist(),
        'normalization': 'each row divided by its row total; empty rows are zero',
        'matching': 'class-blind greedy highest-IoU one-to-one (existing Validator)',
        'confidence_threshold': validator.conf_thresh, 'iou_threshold': validator.iou_thresh,
        'correct_class_matches': int(np.trace(matrix[:n, :n])),
        'misclassifications': int(matrix[:n, :n].sum() - np.trace(matrix[:n, :n])),
        'unmatched_predictions': int(matrix[n, :n].sum()),
        'unmatched_GT': int(matrix[:n, n].sum()),
        'GT_count': int(matrix[:n, :].sum()),
        'predictions_after_threshold': int(matrix[:, :n].sum()),
        'per_class': {}, 'metadata': metadata or {},
        'note': 'Diagnostic counts, not COCO AP. Unmatched predictions may be duplicates, '
                'localization errors, unannotated or out-of-scope targets, not proven background.',
    }
    for i, name in enumerate(names):
        tp = int(matrix[i, i]); fp = int(matrix[:, i].sum()) - tp
        fn = int(matrix[i, :].sum()) - tp
        report['per_class'][name] = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
        }
    output = Path(output_dir) / 'confusion_matrix'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'confusion_matrix.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    lines = ['Rows = GT; columns = prediction. Last row = unmatched predictions; '
             'last column = unmatched GT.',
             f'conf >= {validator.conf_thresh}; IoU >= {validator.iou_thresh}',
             'GT / Prediction\t' + '\t'.join(columns)]
    lines += [name + '\t' + '\t'.join(map(str, matrix[i])) for i, name in enumerate(rows)]
    lines += [f'{k}: {report[k]}' for k in ('correct_class_matches', 'misclassifications',
                                          'unmatched_predictions', 'unmatched_GT')]
    (output / 'confusion_matrix.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Headless PNGs need no extra dependencies beyond the existing Validator.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for values, filename, fraction in (
        (matrix, 'confusion_matrix_counts.png', False),
        (normalized, 'confusion_matrix_row_normalized.png', True),
    ):
        fig = Figure(figsize=(max(8, n * 1.7), max(6, n * 1.4)))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if fraction else None)
        ax.set_xticks(range(n + 1), columns, rotation=30, ha='right')
        ax.set_yticks(range(n + 1), rows)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('GT class')
        ax.set_title(f'conf >= {validator.conf_thresh}, IoU >= {validator.iou_thresh}'
                     + (' | Row normalized' if fraction else ' | Counts'))
        cutoff = float(values.max()) / 2
        for r in range(n + 1):
            for c in range(n + 1):
                label = f'{values[r, c]:.1%}' if fraction else str(values[r, c])
                ax.text(c, r, label, ha='center', va='center',
                        color='white' if values[r, c] > cutoff else 'black')
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output / filename, dpi=180)
    print('\n'.join(lines), flush=True)
    print('Confusion matrix saved:', output, flush=True)
    return report

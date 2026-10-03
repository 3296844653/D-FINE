"""Format-only conversion: preserve EVERY YOLO box, including invalid geometry.

No clipping, minimum-size filtering, deduplication, relabeling or split changes.
Source images and labels are never written. Unparseable labels abort conversion.
Output files must not already exist. Category IDs remain exactly as in YOLO.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def convert(root, classes_yaml=None):
    # Optional explicit source YAML supplies names without changing label IDs.
    class_names = None
    if classes_yaml is not None:
        import yaml
        source = yaml.safe_load(Path(classes_yaml).read_text(encoding='utf-8'))
        class_names = source['names']
        if isinstance(class_names, dict):
            mapping = {int(k): v for k, v in class_names.items()}
            if sorted(mapping) != list(range(int(source['nc']))):
                raise ValueError('YAML class IDs must be contiguous from zero')
            class_names = [mapping[i] for i in range(int(source['nc']))]
        if (not isinstance(class_names, list) or len(class_names) != int(source['nc'])
                or not all(isinstance(n, str) and n for n in class_names)
                or len(set(class_names)) != len(class_names)):
            raise ValueError('Invalid YAML class names')
    outputs = root / 'annotations'
    names = ['instances_train.json', 'instances_val.json', 'conversion_report.json']
    if any((outputs / name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite existing output files')
    datasets, summaries, hashes, class_ids = {}, {}, {}, set()
    for split in ('train', 'val'):
        image_dir, label_dir = root / 'images' / split, root / 'labels' / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and
                        p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        labels = {p.stem: p for p in label_dir.glob('*.txt')}
        stems = [p.stem for p in images]
        if not images or len(set(stems)) != len(stems) or set(stems) != set(labels):
            raise ValueError(f'{split}: image/label correspondence is not one-to-one')
        data = {'images': [], 'annotations': [], 'categories': []}
        counts, anomalies = Counter(), []
        for image_id, image_path in enumerate(images, 1):
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            data['images'].append({'id': image_id, 'file_name': image_path.name,
                                   'width': width, 'height': height})
            label_path = labels[image_path.stem]
            content = label_path.read_bytes()
            hashes[str(label_path.relative_to(root))] = hashlib.sha256(content).hexdigest()
            for line_number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f'{label_path}:{line_number}: expected five YOLO values')
                category, xc, yc, w, h = map(float, parts)
                if not all(math.isfinite(x) for x in (category, xc, yc, w, h)):
                    raise ValueError(f'{label_path}:{line_number}: nonfinite data')
                if category != int(category) or category < 0:
                    raise ValueError(f'{label_path}:{line_number}: invalid category ID')
                category = int(category)
                if class_names is not None and category >= len(class_names):
                    raise ValueError(f'{label_path}:{line_number}: class outside YAML definitions')
                class_ids.add(category)
                # Direct center-normalized -> pixel top-left xywh. NO repair.
                bbox = [(xc - w / 2) * width, (yc - h / 2) * height,
                        w * width, h * height]
                annotation_id = len(data['annotations']) + 1
                data['annotations'].append({
                    'id': annotation_id, 'image_id': image_id, 'category_id': category,
                    'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0,
                })
                counts[str(category)] += 1
                issues = []
                if w <= 0 or h <= 0:
                    issues.append('nonpositive_size')
                if xc-w/2 < 0 or yc-h/2 < 0 or xc+w/2 > 1 or yc+h/2 > 1:
                    issues.append('outside_image_boundary')
                if issues:
                    anomalies.append({'image': image_path.name, 'label': label_path.name,
                                      'line': line_number, 'annotation_id': annotation_id,
                                      'source_yolo': line, 'issues': issues,
                                      'action': 'preserved_without_changes'})
        datasets[split] = data
        summaries[split] = {'images': len(images), 'source_boxes': sum(counts.values()),
                            'converted_boxes': len(data['annotations']),
                            'per_category': dict(counts), 'preserved_anomalies': anomalies}
        assert summaries[split]['source_boxes'] == summaries[split]['converted_boxes']
    categories = ([{'id': c, 'name': name} for c, name in enumerate(class_names)]
                  if class_names is not None else
                  [{'id': c, 'name': f'class_{c}'} for c in sorted(class_ids)])
    for data in datasets.values():
        data['categories'] = categories
    # Confirm source labels unchanged before writing any final outputs.
    for relative, digest in hashes.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source label changed: {relative}')
    report = {'dataset': str(root), 'mode': 'format_only_no_changes',
              'classes_yaml': str(Path(classes_yaml).resolve()) if classes_yaml else None,
              'categories': categories, 'splits': summaries,
              'source_label_sha256': hashes,
              'note': 'All boxes retained, including out-of-bounds/nonpositive-size boxes. '
                      'Only YOLO coordinate representation is converted to COCO xywh pixels.'}
    outputs.mkdir(exist_ok=True)
    for split, data in datasets.items():
        path = outputs / f'instances_{split}.json'
        with path.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        with path.open(encoding='utf-8') as f:
            assert json.load(f) == data
        print(split, {k: v for k, v in summaries[split].items() if k != 'preserved_anomalies'},
              'anomalies retained:', len(summaries[split]['preserved_anomalies']))
    with (outputs / 'conversion_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print('COCO annotations:', outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--classes-yaml', help='Use exact class names from original YAML')
    args = parser.parse_args()
    convert(Path(args.dataset).resolve(), args.classes_yaml)

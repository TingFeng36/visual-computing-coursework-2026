"""Transfer COCO classifier rows by semantic category, not by dense row number."""
from pathlib import Path


def initialize_category_rows(detector, weights, source_ids, dataset_names):
    import torch
    from coco_tools import category_names
    from common import sha256
    names = category_names()
    if len(source_ids) != len(dataset_names) or len(set(source_ids)) != len(source_ids):
        raise ValueError('source_category_ids must be unique and match the dataset class count')
    if any(i not in names for i in source_ids) or [names[i] for i in source_ids] != list(dataset_names):
        raise ValueError('Source COCO category names differ from custom class order')
    state = torch.load(Path(weights), map_location='cpu', weights_only=False)['model']
    net = detector.model.model
    heads = [('class_embed', net.class_embed)]
    encoder_heads = getattr(net.transformer, 'enc_out_class_embed', ())
    heads.extend((f'transformer.enc_out_class_embed.{index}', head)
                 for index, head in enumerate(encoder_heads))
    keys, selected_parameters, previous_shapes = [], {}, {}
    rows = torch.tensor([0, *source_ids], dtype=torch.long)
    # Upstream changes Parameter.data shape without updating Linear.out_features.
    # Validate all source tensors before touching any classifier. This initializer
    # runs before the optimizer or any forward/backward graph is constructed.
    for prefix, head in heads:
        for suffix in ('weight', 'bias'):
            key = f'{prefix}.{suffix}'
            destination = getattr(head, suffix)
            if destination is None or key not in state or state[key].shape[0] <= max(source_ids):
                raise ValueError(f'COCO source classifier missing/incompatible: {key}')
            selected = state[key].index_select(0, rows)
            expected = (len(rows), head.in_features) if suffix == 'weight' else (len(rows),)
            if tuple(selected.shape) != expected or tuple(destination.shape[1:]) != expected[1:]:
                raise ValueError(f'Classifier feature dimensions differ: {key}; source={tuple(selected.shape)}, target={tuple(destination.shape)}')
            keys.append(key)
            selected_parameters[key] = selected
            previous_shapes[key] = list(destination.shape)
    with torch.no_grad():
        for prefix, head in heads:
            for suffix in ('weight', 'bias'):
                parameter = getattr(head, suffix)
                parameter.data = selected_parameters[f'{prefix}.{suffix}'].to(parameter).clone()
            head.out_features = len(rows)
            if head.weight.shape != (len(rows), head.in_features) or head.bias.shape != (len(rows),):
                raise RuntimeError(f'Classifier resize failed: {prefix}')
    return {'method': 'COCO classifier semantic row selection', 'source_category_ids': list(source_ids),
            'source_weights_sha256': sha256(weights), 'parameter_keys': keys, 'reserved_row': 0,
            'previous_parameter_shapes': previous_shapes, 'target_rows': len(rows),
            'repair': 'resize actual decoder/encoder classifier tensors and synchronize Linear metadata'}

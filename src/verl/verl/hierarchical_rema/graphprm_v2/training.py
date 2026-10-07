"""Source-only small-GNN training; frozen encoder features are never optimized."""
import json
import random
import signal
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import FEATURE_SCHEMA
from . import graphprm_core as core
from .data import atomic_json, digest
from .features import atomic_torch, layout_from_spec
from .graphprm_model import GraphPRMHybridModel, set_graphprm_layout, instantiate_model_from_checkpoint
from .evaluation import (SCOPES, targets, logits_by_scope, evaluate, write_evaluation)

STOP = False


def stop_after_epoch(signum, frame):
    global STOP
    STOP = True
    print('[training] stop requested; checkpointing at the next epoch boundary', flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(examples, device):
    counts = {(scope, label): [0, 0] for scope, labels in SCOPES.items() for label in labels}
    for ex in examples:
        gold = targets(ex.source_record)
        for scope, labels in SCOPES.items():
            groups = gold['workers'].values() if scope == 'worker' else [gold[scope]]
            for group in groups:
                for label in labels:
                    value = group.get(label)
                    if value is not None:
                        counts[scope, label][value] += 1
    weights = {}
    for key, count in counts.items():
        tensor = torch.tensor(count, dtype=torch.float32, device=device).clamp_min(1).rsqrt()
        weights[key] = tensor / tensor.mean()
    return weights, {'.'.join(k): v for k, v in counts.items()}


def supervised_loss(outputs, gold, config, weights):
    scopes, total_weight = [], 0.
    for scope, nodes in logits_by_scope(outputs).items():
        node_losses = []
        for node, logits in nodes:
            labels = gold['workers'].get(node, {}) if scope == 'worker' else gold[scope]
            losses = []
            for i, name in enumerate(SCOPES[scope]):
                value = labels.get(name)
                if value is not None:
                    # Per-observation weighting: singleton CE(weight=..., mean) cancels its own weight.
                    ce = F.cross_entropy(logits[i, [0, 2]].unsqueeze(0),
                          torch.tensor([value], device=logits.device), reduction='sum')
                    losses.append(ce * weights[scope, name][value])
            if losses:
                node_losses.append(torch.stack(losses).mean())
        if node_losses:
            weight = config['process_weight_' + scope]
            scopes.append(weight * torch.stack(node_losses).mean())
            total_weight += weight
    if not scopes or not total_weight:
        raise ValueError('No supervised labels in a training example')
    process = torch.stack(scopes).sum() / total_weight
    final = outputs['final_anchor_logit'] * 0
    if gold['graph']['global_outcome'] is not None:
        final = F.binary_cross_entropy_with_logits(outputs['final_anchor_logit'],
                    outputs['final_anchor_logit'].new_tensor(float(gold['graph']['global_outcome'])))
    return config['loss_weight_process'] * process + config['loss_weight_final'] * final


def ranking_partners(examples):
    by_task = {}
    for i, ex in enumerate(examples):
        task = ' '.join(ex.source_record['task']['prompt'].lower().split())
        y = targets(ex.source_record)['graph']['global_outcome']
        if y is not None:
            by_task.setdefault(task, {0: [], 1: []})[y].append(i)
    partners = {}
    for classes in by_task.values():
        for y in (0, 1):
            for i in classes[y]:
                partners[i] = classes[1-y]
    return partners


def checkpoint(model, config, spec):
    return {'feature_schema': FEATURE_SCHEMA, 'label_schema_name': 'binary_simplified_v2',
        'node_types': list(core.NODE_TYPES), 'edge_types': list(core.EDGE_TYPES),
        'metadata_names': list(core.METADATA_NAMES), 'label_schema': {s: list(v) for s, v in SCOPES.items()},
        'feature_spec': spec, 'config': config,
        'encoder_spec': {'backend': 'graphprm_hybrid', 'feature_layout': layout_from_spec(spec).to_json(),
            'bge_encoder': {'backend': 'sentence-transformers', 'model_name': spec['bge_model']},
            'prm_features': {'model': spec['prm_model'], 'score_used_as_node_feature': False}},
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def search(train, val, test, spec, args):
    global STOP
    for name in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_after_epoch)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    space = json.loads(Path(args.search_space).read_text())
    run_identity = digest({'spec': spec, 'space': space, 'seed': args.seed, 'val_ratio': args.val_ratio,
        'train': [digest(e.source_record) for e in train], 'val': [digest(e.source_record) for e in val],
        'test': [digest(e.source_record) for e in test], 'objective': args.objective,
        'patience': args.patience, 'epochs': args.epochs})
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity'] != run_identity:
        raise ValueError('Run settings/data changed; choose a NEW output directory')
    atomic_json(manifest_path, {'identity': run_identity, 'feature_spec': spec,
        'counts': {'train': len(train), 'validation': len(val), 'test': len(test)},
        'selection': 'validation only; maximize risk-balanced score',
        'pareto_objectives': ['mean_binary_macro_f1', 'risky_recall', 'good_recall',
                              'worst_source_macro_f1', 'original_examples_macro_f1', 'all_head_macro_f1'],
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    study = optuna.create_study(study_name='graphprm_v2', storage='sqlite:///' + str(output.resolve() / 'study.sqlite3'),
        load_if_exists=True, directions=['maximize'] * (6 if args.objective == 'pareto' else 1),
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    # A killed process may leave a RUNNING trial. We resume its saved epoch in a queued trial.
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    active_path = output / 'active_trial.json'
    if active_path.exists():
        active = json.loads(active_path.read_text())
        completed_paths = {t.user_attrs.get('directory') for t in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))}
        if active['directory'] not in completed_paths:
            study.enqueue_trial(active['parameters'], user_attrs={'resume_directory': active['directory']}, skip_if_exists=False)
    weights, counts = class_weights(train, device)
    atomic_json(output / 'training_class_counts.json', counts)
    partners = ranking_partners(train)
    print(f'[training] within-task ranking eligible={sum(bool(v) for v in partners.values())}/{len(train)}', flush=True)
    set_graphprm_layout(layout_from_spec(spec))

    def objective(trial):
        config = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}
        if config.get('loss_weight_stage', 0) != 0:
            raise ValueError('No primary-stage supervision in the simplified schema')
        directory = Path(trial.user_attrs.get('resume_directory') or output / f'trial_{trial.number:03d}')
        directory.mkdir(parents=True, exist_ok=True)
        trial.set_user_attr('directory', str(directory.resolve()))
        atomic_json(active_path, {'parameters': config, 'directory': str(directory.resolve())})
        trial_seed = args.seed + int(digest(config)[:6], 16)
        seed_everything(trial_seed)
        model = GraphPRMHybridModel(text_dim=layout_from_spec(spec).total_dim, metadata_dim=len(core.METADATA_NAMES),
            **{k: config[k] for k in ('hidden_dim', 'message_passing_layers', 'dropout',
                                      'graph_architecture', 'gat_heads', 'node_head_context')}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        best, stale, start = -float('inf'), 0, 0
        last_path = directory / 'last_epoch.pt'
        if last_path.exists():
            saved = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            best, stale, start = saved['best'], saved['stale'], saved['epoch'] + 1
            random.setstate(saved['python_rng'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available() and saved['cuda_rng']:
                torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                break
            model.train()
            indices = list(range(len(train)))
            random.shuffle(indices)
            running_loss = 0.
            for offset in tqdm(range(0, len(indices), config['batch_size']), desc=f'trial {trial.number} epoch {epoch+1}'):
                batch = indices[offset:offset + config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                for i in batch:
                    ex = train[i]
                    outputs = model(ex, device)
                    gold = targets(ex.source_record)
                    loss = supervised_loss(outputs, gold, config, weights)
                    if config['loss_weight_ranking'] and partners.get(i):
                        j = random.choice(partners[i])
                        other = model(train[j], device)['ranking_score']
                        sign = 1 if gold['graph']['global_outcome'] == 1 else -1
                        ranking = F.softplus(-sign * (outputs['ranking_score'] - other))
                        loss = loss + config['loss_weight_ranking'] * ranking
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss; refusing to save a best model')
                    (loss / len(batch)).backward()
                    running_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
            summary, _, _ = evaluate(model, val, device)
            score = summary['selection_score']
            if score > best + 1e-6:
                best, stale = score, 0
                cp = checkpoint(model, config, spec)
                cp['validation_metrics'] = summary
                cp['best_epoch'] = epoch + 1
                atomic_torch(directory / 'best_model.pkl', cp)
            else:
                stale += 1
            atomic_torch(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'best': best, 'stale': stale, 'python_rng': random.getstate(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
            with (directory / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps({'epoch': epoch+1, 'train_loss': running_loss/len(train),
                    **{k: v for k, v in summary.items() if k != 'metrics'}}) + '\n')
            print(f'[training] trial={trial.number} epoch={epoch+1} val_selection={score:.4f} patience={stale}/{args.patience}', flush=True)
            if STOP:
                raise KeyboardInterrupt('Epoch checkpoint saved; rerun the same command')
        cp = torch.load(directory / 'best_model.pkl', map_location='cpu', weights_only=False)
        summary = cp['validation_metrics']
        trial.set_user_attr('selection_score', summary['selection_score'])
        trial.set_user_attr('checkpoint', str((directory / 'best_model.pkl').resolve()))
        objectives = ['mean_binary_macro_f1', 'risky_recall', 'good_recall', 'worst_source_macro_f1',
                      'original_examples_macro_f1', 'all_head_macro_f1']
        return [summary[n] for n in objectives] if args.objective == 'pareto' else summary['selection_score']

    completed = lambda: study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    while len(completed()) < args.search_trials:
        study.optimize(objective, n_trials=1)
        if active_path.exists():
            active_path.unlink()
    # Export Pareto front, then choose one model by the predeclared validation-only policy.
    atomic_json(output / 'pareto_trials.json', [{'trial': t.number, 'objectives': t.values,
                 'parameters': t.params, 'attributes': t.user_attrs} for t in study.best_trials])
    best_trial = max(completed(), key=lambda t: t.user_attrs['selection_score'])
    cp = torch.load(best_trial.user_attrs['checkpoint'], map_location='cpu', weights_only=False)
    cp['run_identity'] = run_identity
    cp['selection_trial'] = best_trial.number
    atomic_torch(output / 'best_model.pkl', cp)
    model = instantiate_model_from_checkpoint(cp, device)
    summary, records, observations = evaluate(model, test, device)
    write_evaluation(output / 'test', summary, records, observations)
    import hashlib
    sha = hashlib.sha256((output / 'best_model.pkl').read_bytes()).hexdigest()
    atomic_json(output / 'COMPLETE.json', {'best_trial': best_trial.number, 'sha256': sha,
        'feature_schema': FEATURE_SCHEMA, 'validation_selection': best_trial.user_attrs['selection_score']})
    print(f'[training] best_model={output / "best_model.pkl"}', flush=True)

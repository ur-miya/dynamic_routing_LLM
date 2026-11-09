#!/usr/bin/env python3
"""Resume the exact judge-batch-1.0 without modifying its frozen files or judgments."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import uuid

VERSION = 'judge-recovery-1.0'
ORIGINAL_SHA = '0b8aa6f85cb74686147f5653810ded32774e0d05f2ae1725bd83caa7dd6bb89a'
EXTRA_ATTEMPTS = 1
CIRCUIT_BREAKER = 5


def original(path):
    import hashlib
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Original batch file differs; do not change hashes or frozen files')
    spec = importlib.util.spec_from_file_location('verified_original_batch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def open_context(config, source):
    b = original(source)
    cfg, pilot, pc = b.load(config)
    root, plan = b.frozen(cfg, pilot, pc)
    if not (root / 'results.sqlite3').is_file():
        raise ValueError('Existing results.sqlite3 required; this tool does not prepare a new batch')
    return b, cfg, pilot, pc, root, plan


def schema(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS recovery_diagnostics(
      attempt_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, code TEXT NOT NULL,
      safe_json TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS recovery_terminal(
      task_id TEXT PRIMARY KEY, reason TEXT NOT NULL, attempts_used INTEGER NOT NULL,
      last_code TEXT NOT NULL, created_at TEXT NOT NULL);
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS recovery_attempts_task_idx ON attempts(task_id)')
    db.commit()


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def tables(db):
    return {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def manifest_check(b, root, plan):
    path = root / 'recovery_v1' / 'manifest.json'
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding='utf-8'))
    checks = {'version': VERSION, 'recovery_source_sha256': b.sha(__file__),
              'original_plan_sha256': b.sha(root / 'plan.json'),
              'original_implementation_sha256': ORIGINAL_SHA,
              'inputs_sha256': plan['inputs_sha256'], 'extra_attempts_per_task': EXTRA_ATTEMPTS,
              'consecutive_error_circuit': CIRCUIT_BREAKER,
              'terminal_policy': 'technical_error_null_label_excluded'}
    if any(m.get(k) != v for k, v in checks.items()):
        raise ValueError('Recovery manifest/code mismatch; do not edit frozen files or hashes')
    backup = Path(m['backup_path'])
    if not backup.is_file():
        raise ValueError('Recovery backup missing')
    if b.sha(backup) != m.get('backup_sha256'):
        raise ValueError('Recovery backup missing or modified')
    return m


def initialize(b, root, plan, pc):
    m = manifest_check(b, root, plan)
    if m:
        return m
    folder = root / 'recovery_v1'
    folder.mkdir(exist_ok=True, mode=0o700)
    db = b.readonly(root / 'results.sqlite3')
    try:
        if 'recovery_terminal' in tables(db) and db.execute('SELECT count(*) FROM recovery_terminal').fetchone()[0]:
            raise ValueError('Recovery state exists without manifest; do not reset it')
        backup_path = folder / ('results_before_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.sqlite3')
        with sqlite3.connect(backup_path) as dest:
            db.backup(dest)
        os.chmod(backup_path, 0o600)
        attempts = db.execute('SELECT count(*) FROM attempts').fetchone()[0]
        completed = db.execute('SELECT count(*) FROM results').fetchone()[0]
    finally:
        db.close()
    m = {'version': VERSION, 'created_at': timestamp(), 'recovery_source_sha256': b.sha(__file__),
         'original_plan_sha256': b.sha(root / 'plan.json'),
         'original_implementation_sha256': ORIGINAL_SHA, 'inputs_sha256': plan['inputs_sha256'],
         'extra_attempts_per_task': EXTRA_ATTEMPTS, 'consecutive_error_circuit': CIRCUIT_BREAKER,
         'terminal_policy': 'technical_error_null_label_excluded',
         'backup_path': str(backup_path), 'backup_sha256': b.sha(backup_path),
         'completed_before_recovery': completed, 'http_attempts_before_recovery': attempts,
         'attempts_per_task_total': int(pc['request']['attempts_per_task']) + EXTRA_ATTEMPTS,
         'global_http_cap_unchanged': True,
         'note': 'Payload, rubric, parser, deployment binding and quality policy are unchanged.'}
    b.write_json(folder / 'manifest.json', m)
    return m


def task_maps(fdb):
    return dict(fdb.execute('SELECT task_id,split FROM tasks'))


def state(b, root, plan, db, fdb):
    known = task_maps(fdb)
    success = {x[0] for x in db.execute('SELECT task_id FROM results')}
    terminal = {x[0] for x in db.execute('SELECT task_id FROM recovery_terminal')} if 'recovery_terminal' in tables(db) else set()
    if success & terminal or (success | terminal) - set(known):
        raise ValueError('Unknown or simultaneously successful/terminal task')
    if db.execute('SELECT count(*) FROM results r JOIN cache c USING(cache_key)').fetchone()[0] != len(success):
        raise ValueError('Completed results have missing cache records')
    counts = {}
    for split in b.SPLITS:
        ids = {t for t, s in known.items() if s == split}
        counts[split] = {'planned_tasks': len(ids), 'completed_tasks': len(ids & success),
                         'terminal_tasks': len(ids & terminal), 'pending_tasks': len(ids - success - terminal)}
    codes = dict(db.execute('SELECT code,count(*) FROM recovery_diagnostics GROUP BY code')) if 'recovery_diagnostics' in tables(db) else {}
    recent = [{'attempt_id': aid, 'task_id': tid, **json.loads(safe)}
              for aid, tid, safe in db.execute('SELECT attempt_id,task_id,safe_json FROM recovery_diagnostics ORDER BY attempt_id DESC LIMIT 5')] if 'recovery_diagnostics' in tables(db) else []
    return {'version': VERSION, 'planned_tasks': plan['planned_tasks'], 'completed_tasks': len(success),
            'terminal_tasks': len(terminal), 'pending_tasks': len(known) - len(success) - len(terminal),
            'all_tasks_succeeded': len(success) == len(known),
            'all_tasks_accounted_for': len(success | terminal) == len(known),
            'http_attempts_total': db.execute('SELECT count(*) FROM attempts').fetchone()[0],
            'diagnostic_counts': codes, 'recent_diagnostics': recent, 'splits': counts}


def inspect_response(body):
    info = {'finish_reason': 'unavailable', 'content_chars': None, 'reasoning_chars': None}
    if not isinstance(body, dict):
        return info
    choices = body.get('choices')
    info['choice_count'] = len(choices) if isinstance(choices, list) else None
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        choice = choices[0]
        reason = choice.get('finish_reason')
        info['finish_reason'] = reason if reason in ('stop', 'length', 'content_filter', 'tool_calls', None) else 'other'
        msg = choice.get('message')
        if isinstance(msg, dict):
            for key, out in [('content', 'content_chars'), ('reasoning_content', 'reasoning_chars')]:
                if isinstance(msg.get(key), str):
                    info[out] = len(msg[key])
    return info


def parser_code(exc):
    if isinstance(exc, json.JSONDecodeError):
        return 'invalid_content_json'
    lookup = {
        'Response must be object': 'response_not_object',
        'Expected exactly one choice with finish_reason=stop': 'choice_or_finish_reason',
        'Missing final content': 'missing_final_content',
        'Duplicate JSON result key': 'duplicate_result_key',
        'Nonfinite JSON': 'nonfinite_json',
        'Unexpected result keys': 'unexpected_result_keys',
        'Scores must be strict integers 1..5': 'invalid_score_type_or_range',
        'Invalid rationale length': 'invalid_rationale_length',
        'Inconsistent reference presence': 'reference_presence_conflict',
    }
    msg = str(exc)
    if msg in lookup:
        return lookup[msg]
    for key in ('preferred', 'acceptable_a', 'acceptable_b', 'confidence', 'reference_assessment'):
        if msg == 'Invalid ' + key:
            return 'invalid_enum_' + key
    return 'invalid_response_structure'


def add_diagnostic(b, db, aid, tid, code, details):
    safe = {'code': code, **details}
    db.execute('INSERT INTO recovery_diagnostics VALUES(?,?,?,?,?)',
               (aid, tid, code, b.canonical(safe), timestamp()))


def terminal(db, tid, used, code):
    db.execute('INSERT INTO recovery_terminal VALUES(?,?,?,?,?)',
               (tid, 'judge_task_failed_after_bounded_retries', used, code, timestamp()))
    db.commit()


def latest_code(db, tid):
    row = db.execute('SELECT code FROM recovery_diagnostics WHERE task_id=? ORDER BY attempt_id DESC LIMIT 1', (tid,)).fetchone()
    return row[0] if row else 'legacy_attempt_details_unavailable'


def bound_deployment(b, cfg, pc, args, db):
    path = Path(cfg['output']) / 'deployment.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if not args.deployment_tag:
        args.deployment_tag = saved['deployment_tag']
    before = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    if not before or before[0] != saved.get('binding_sha256'):
        raise ValueError('Existing deployment binding required')
    # Original binding check includes URL, endpoint, requested model and operator tag.
    return b.deployment(cfg, pc, args, db)


def run(b, cfg, pilot, pc, root, plan, args, client_factory=None):
    if not args.allow_network:
        raise ValueError('run requires --allow-network')
    if args.max_new_tasks < 1 or args.max_http_attempts < 1:
        raise ValueError('Positive explicit task and HTTP budgets required')
    with b.locked(root):
        initialize(b, root, plan, pc)
        db = b.result_db(root)
        fdb = b.readonly(root / 'inputs.sqlite3')
        schema(db)
        attempted = handled = errors = hits = streak = 0
        fatal = False
        last_error = None
        stop_reason = None
        try:
            url, model, token, binding = bound_deployment(b, cfg, pc, args, db)
            base_per = int(pc['request']['attempts_per_task'])
            per_task = base_per + EXTRA_ATTEMPTS
            cap = min(int(cfg['max_total_http_attempts']), plan['planned_tasks'] * base_per)
            prior = db.execute('SELECT count(*) FROM attempts').fetchone()[0]
            marks = ','.join('?' for _ in args.splits)
            tasks = fdb.execute(f'SELECT task_id,task_json FROM tasks WHERE split IN ({marks}) ORDER BY rowid', args.splits)
            if client_factory is None:
                import httpx
                client_factory = lambda: httpx.Client(timeout=float(pc['request']['timeout_seconds']), follow_redirects=False)
            with client_factory() as client:
                for tid, tj in tasks:
                    if db.execute('SELECT 1 FROM results WHERE task_id=?', (tid,)).fetchone():
                        continue
                    if db.execute('SELECT 1 FROM recovery_terminal WHERE task_id=?', (tid,)).fetchone():
                        continue
                    if fatal or handled >= args.max_new_tasks:
                        stop_reason = stop_reason or 'run_task_budget'
                        break
                    task = json.loads(tj)
                    payload = pilot.payload(task, pc, model)
                    ck = b.digest([binding, pilot.RUBRIC_VERSION, pilot.PREPROCESS_VERSION, payload])
                    cached = db.execute('SELECT record FROM cache WHERE cache_key=?', (ck,)).fetchone()
                    if cached:
                        pilot.validate_result(b.canonical(json.loads(cached[0])['judgment']), False)
                        db.execute('INSERT INTO results VALUES(?,?)', (tid, ck))
                        db.commit()
                        hits += 1
                        handled += 1
                        continue
                    used = db.execute('SELECT count(*) FROM attempts WHERE task_id=?', (tid,)).fetchone()[0]
                    if used > per_task:
                        raise ValueError('Task attempts exceed frozen recovery allowance')
                    if used == per_task:
                        terminal(db, tid, used, latest_code(db, tid))
                        handled += 1
                        continue
                    if attempted >= args.max_http_attempts or prior + attempted >= cap:
                        stop_reason = 'global_http_cap' if prior + attempted >= cap else 'run_http_budget'
                        break
                    handled += 1
                    for index in range(used, per_task):
                        if attempted >= args.max_http_attempts or prior + attempted >= cap:
                            stop_reason = 'global_http_cap' if prior + attempted >= cap else 'run_http_budget'
                            break
                        aid = db.execute('INSERT INTO attempts(task_id,state) VALUES(?,?)', (tid, 'started')).lastrowid
                        db.commit()
                        attempted += 1
                        http_status = None
                        details = {}
                        parsed = None
                        code = None
                        stage = 'transport'
                        start = time.monotonic()
                        try:
                            response = client.post(url, headers={'Authorization': 'Bearer ' + token}, json=payload)
                            http_status = response.status_code
                            stage = 'http_status'
                            if not 200 <= http_status < 300:
                                code = 'http_' + str(int(http_status))
                                fatal = http_status not in (408, 429) and http_status < 500
                            else:
                                stage = 'response_json'
                                body = response.json()
                                details = inspect_response(body)
                                stage = 'pilot_parser'
                                parsed = pilot.parse_response(body, False)
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as exc:
                            import httpx
                            if stage == 'transport' and isinstance(exc, httpx.HTTPError):
                                code = 'transport_timeout' if isinstance(exc, httpx.TimeoutException) else 'transport_error'
                            elif stage == 'response_json' and isinstance(exc, ValueError):
                                code = 'response_body_not_json'
                            elif stage == 'pilot_parser' and isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)):
                                code = parser_code(exc)
                            else:
                                code = 'unexpected_client_or_runtime_error'
                                fatal = True
                        if parsed is not None:
                            # Storage exceptions are not treated as model/API errors.
                            parsed['latency_ms'] = round((time.monotonic() - start) * 1000, 2)
                            db.execute('INSERT INTO cache VALUES(?,?)', (ck, b.canonical(parsed)))
                            db.execute('INSERT INTO results VALUES(?,?)', (tid, ck))
                            db.execute('UPDATE attempts SET state=?,http_status=? WHERE id=?', ('ok', http_status, aid))
                            db.commit()
                            streak = 0
                            break
                        code = code or 'unclassified_response_error'
                        db.execute('UPDATE attempts SET state=?,http_status=? WHERE id=?', ('failed', http_status, aid))
                        add_diagnostic(b, db, aid, tid, code, {'stage': stage, 'http_status': http_status, **details})
                        db.commit()
                        errors += 1
                        streak += 1
                        last_error = code
                        if fatal or streak >= CIRCUIT_BREAKER:
                            fatal = True
                            stop_reason = 'non_retryable_http_or_runtime_error' if streak < CIRCUIT_BREAKER else 'consecutive_error_circuit'
                            break
                        if index + 1 >= per_task:
                            terminal(db, tid, index + 1, code)
                            break
                        if attempted < args.max_http_attempts and prior + attempted < cap:
                            time.sleep(min(8, 2 ** min(index, 3)))
                    if handled % int(cfg['runtime']['progress_every']) == 0 or fatal:
                        done = db.execute('SELECT count(*) FROM results').fetchone()[0]
                        failed = db.execute('SELECT count(*) FROM recovery_terminal').fetchone()[0]
                        print(b.canonical({'completed_tasks': done, 'terminal_tasks': failed,
                                           'http_attempts_this_run': attempted, 'stopped_on_error': fatal}), flush=True)
                    if fatal:
                        break
            report = state(b, root, plan, db, fdb)
            report.update(selected_splits=args.splits, new_tasks_this_run=handled,
                          cache_hits_this_run=hits, http_attempts_this_run=attempted,
                          http_errors_or_invalid_responses_this_run=errors, stopped_on_error=fatal,
                          last_error=last_error, stop_reason=stop_reason, network_called=attempted > 0,
                          persistent_http_limit=cap, attempts_per_task_total=per_task)
            folder = root / 'recovery_v1'
            b.write_json(folder / 'last_run.json', report)
            b.write_json(folder / ('run_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json'), report)
            # Original last_run.json and original historical reports are intentionally untouched.
            return report
        finally:
            db.close()
            fdb.close()


def iter_labels(b, fdb, db, pilot):
    for row in b.iter_labels(fdb, db, pilot):
        if row['label_status'] == 'incomplete':
            tids = [r[0] for r in fdb.execute('SELECT task_id FROM tasks WHERE pair_id=?', (row['pair_id'],))]
            bad = any(db.execute('SELECT 1 FROM recovery_terminal WHERE task_id=?', (t,)).fetchone() for t in tids)
            if bad:
                row.update(label_status='technical_error', label_reason='judge_task_failed_after_bounded_retries')
        yield row


def verify_store(b, pilot, pc, db, fdb, per_task):
    bind = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    model = db.execute("SELECT value FROM metadata WHERE key='requested_model'").fetchone()
    if not bind or not model:
        raise ValueError('Missing deployment identity')
    for tid, tj in fdb.execute('SELECT task_id,task_json FROM tasks'):
        found = db.execute('SELECT r.cache_key,c.record FROM results r JOIN cache c USING(cache_key) WHERE r.task_id=?', (tid,)).fetchone()
        term = db.execute('SELECT attempts_used FROM recovery_terminal WHERE task_id=?', (tid,)).fetchone()
        if found:
            if term:
                raise ValueError('Successful and terminal task conflict')
            ck = b.digest([bind[0], pilot.RUBRIC_VERSION, pilot.PREPROCESS_VERSION, pilot.payload(json.loads(tj), pc, model[0])])
            if found[0] != ck:
                raise ValueError('Cache payload/deployment mismatch')
            pilot.validate_result(b.canonical(json.loads(found[1])['judgment']), False)
        elif term:
            used = db.execute('SELECT count(*) FROM attempts WHERE task_id=?', (tid,)).fetchone()[0]
            if used != term[0] or used != per_task:
                raise ValueError('Invalid terminal attempt accounting')
        else:
            raise ValueError('Pending task: finish scoring before finalize')


def validate_rows(rows):
    for r in rows:
        if r['label_status'] == 'incomplete':
            raise ValueError('Incomplete row in closed scoring run')
        if r['quality_valid']:
            ds = [(r[f'T_overall_o{o}'] - r[f'S_overall_o{o}']) / 4 for o in (0, 1)]
            qt = (r['T_overall_o0'] + r['T_overall_o1'] - 2) / 8
            qs = (r['S_overall_o0'] + r['S_overall_o1'] - 2) / 8
            expected = 'easy' if max(ds) <= 0 else 'hard' if min(ds) >= .5 else 'medium'
            if (r['q_T'], r['q_S'], r['delta'], r['delta_lo'], r['delta_hi'], r['label']) != (qt, qs, qt-qs, min(ds), max(ds), expected):
                raise ValueError('Independent labeling formula mismatch')
            both = r['acceptable_T'] == r['acceptable_S'] == 'no'
            if r['train_eligible'] != (not both) or r['both_unacceptable'] != both:
                raise ValueError('Training eligibility mismatch')
        elif r['label'] is not None or r['quality_label'] is not None or r['train_eligible'] or r['kd_candidate']:
            raise ValueError('Excluded row has an invented label')


def finalize(b, cfg, pilot, pc, root, plan, validate_only=False):
    with b.locked(root):
        m = manifest_check(b, root, plan)
        if not m:
            raise ValueError('Initialize recovery with run first')
        fdb = b.readonly(root / 'inputs.sqlite3')
        db = b.result_db(root)
        c = b.connect_inputs(cfg)
        try:
            s = state(b, root, plan, db, fdb)
            if not s['all_tasks_accounted_for']:
                raise ValueError('Scoring has pending tasks; no training-ready export made')
            verify_store(b, pilot, pc, db, fdb, m['attempts_per_task_total'])
            folder = root / 'recovery_v1'
            dest = Path(cfg['paths']['labeled_dir'])
            summary_path = folder / 'label_summary.json'
            previous = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
            if previous:
                if previous.get('inputs_sha256') != plan['inputs_sha256'] or previous.get('recovery_manifest_sha256') != b.sha(folder / 'manifest.json'):
                    raise ValueError('Recovery manifest/output provenance mismatch')
                for entry in previous['outputs'].values():
                    if not Path(entry['path']).is_file() or b.sha(entry['path']) != entry['sha256']:
                        raise ValueError('Previously exported recovery labels changed')
            elif any((dest / (split + '_labeled.parquet')).exists() for split in b.SPLITS):
                raise ValueError('Existing non-recovery label outputs; not overwriting them')
            if validate_only and not previous:
                raise ValueError('Run finalize before validate')
            dest.mkdir(parents=True, exist_ok=True)
            stats = {split: {'rows': 0, 'label_status': Counter(), 'reliable_labels': Counter(),
                             'train_eligible_labels': Counter(), 'exclusion_reasons': Counter(),
                             'both_unacceptable': 0, 'kd_candidates': 0} for split in b.SPLITS}
            expected = {split: [] for split in b.SPLITS}
            for row in iter_labels(b, fdb, db, pilot):
                expected[row['split']].append(row)
                st = stats[row['split']]
                st['rows'] += 1
                st['label_status'][row['label_status']] += 1
                if row['label'] is not None:
                    st['reliable_labels'][row['label']] += 1
                if row['train_eligible']:
                    st['train_eligible_labels'][row['label']] += 1
                else:
                    st['exclusion_reasons']['both_unacceptable' if row['both_unacceptable'] else row['label_reason']] += 1
                st['both_unacceptable'] += row['both_unacceptable']
                st['kd_candidates'] += row['kd_candidate']
            for split, rows in expected.items():
                if len(rows) != plan['splits'][split]['rows']:
                    raise ValueError('Pair coverage mismatch')
                validate_rows(rows)
            if not validate_only:
                outputs = {}
                with tempfile.TemporaryDirectory(prefix='.recovery_labels_', dir=dest) as td:
                    td = Path(td)
                    types = b.output_schema(pilot)
                    columns = '{' + ','.join(b.qs(k) + ':' + b.qs(v) for k, v in types.items()) + '}'
                    for split in b.SPLITS:
                        js = td / (split + '.jsonl')
                        with js.open('w', encoding='utf-8') as f:
                            for row in expected[split]:
                                f.write(b.canonical(row) + '\n')
                        pq = td / (split + '.parquet')
                        c.execute('COPY (SELECT * FROM read_json(' + b.qs(js) + ",format='newline_delimited',columns=" + columns + ')) TO ' + b.qs(pq) + ' (FORMAT PARQUET,COMPRESSION ZSTD)')
                        final = dest / (split + '_labeled.parquet')
                        outputs[split] = {'path': str(final), 'sha256': b.sha(pq)}
                    for split in b.SPLITS:
                        os.replace(td / (split + '.parquet'), outputs[split]['path'])
                report = {'version': VERSION, 'policy': b.POLICY, 'complete': True, 'partial': False,
                          'scoring_closed': True, 'all_tasks_succeeded': s['all_tasks_succeeded'],
                          'completed_tasks': s['completed_tasks'], 'terminal_tasks': s['terminal_tasks'],
                          'inputs_sha256': plan['inputs_sha256'], 'recovery_manifest_sha256': b.sha(folder / 'manifest.json'),
                          'stats': stats, 'outputs': outputs, 'human_validation': False,
                          'interpretation': 'Judge-derived weak supervision. Technical failures are explicit exclusions, not scores.'}
                b.write_json(summary_path, report)
            else:
                report = previous
            verified = 0
            for split in b.SPLITS:
                entry = report['outputs'][split]
                if b.sha(entry['path']) != entry['sha256']:
                    raise ValueError('Labeled output hash mismatch')
                cur = c.execute('SELECT * FROM read_parquet(' + b.qs(entry['path']) + ') ORDER BY pair_id')
                cols = [x[0] for x in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                if rows != expected[split]:
                    raise ValueError('Labels differ from frozen judgments and technical exclusions')
                validate_rows(rows)
                verified += len(rows)
            valid = {'version': VERSION, 'valid': True, 'rows_verified': verified,
                     'all_tasks_accounted_for': True, 'all_tasks_succeeded': s['all_tasks_succeeded'],
                     'terminal_tasks': s['terminal_tasks'], 'complete': True, 'source_text_exported': False,
                     'inputs_sha256': plan['inputs_sha256'], 'outputs': report['outputs']}
            b.write_json(folder / 'validation.json', valid)
            return valid
        finally:
            c.close()
            fdb.close()
            db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('status', 'run', 'finalize', 'validate'))
    ap.add_argument('--config', default='configs/judge_batch_v1.yaml')
    ap.add_argument('--batch-source', default='scripts/judge_batch_v1.py')
    ap.add_argument('--allow-network', action='store_true')
    ap.add_argument('--deployment-tag', default='', help='Default: reuse existing deployment.json operator tag')
    ap.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val', 'test'])
    ap.add_argument('--max-new-tasks', type=int, default=8)
    ap.add_argument('--max-http-attempts', type=int, default=8)
    args = ap.parse_args()
    try:
        b, cfg, pilot, pc, root, plan = open_context(args.config, args.batch_source)
        if args.mode == 'run':
            result = run(b, cfg, pilot, pc, root, plan, args)
        elif args.mode in ('finalize', 'validate'):
            result = finalize(b, cfg, pilot, pc, root, plan, args.mode == 'validate')
        else:
            with b.locked(root):
                manifest_check(b, root, plan)
                db, fdb = b.readonly(root / 'results.sqlite3'), b.readonly(root / 'inputs.sqlite3')
                try:
                    result = state(b, root, plan, db, fdb)
                finally:
                    db.close()
                    fdb.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.mode == 'run' and result['stopped_on_error']:
            return 2
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Attempts already started remain accounted for. Reuse the same command.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Never print raw exception strings: third-party exceptions can contain request data.
        print(type(exc).__name__ + ': stopped; no secrets printed. Frozen originals must remain unchanged.', file=sys.stderr)
        if isinstance(exc, ValueError):
            safe = str(exc)
            if safe.startswith(('Original batch file differs', 'Frozen plan/', 'Pilot source differs',
                                'Recovery manifest/', 'Existing results.sqlite3 required', 'Existing deployment binding',
                                'Recovery backup missing', 'Missing/invalid judge environment', 'Deployment binding changed',
                                'Scoring has pending', 'Existing non-recovery', 'Previously exported', 'Run finalize before',
                                'Recovery state exists', 'Missing deployment identity', 'Cache payload/', 'Pending task:',
                                'Positive explicit', 'run requires', 'Initialize recovery', 'Result payload/',
                                'Invalid terminal', 'Task attempts exceed')):
                print(safe, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

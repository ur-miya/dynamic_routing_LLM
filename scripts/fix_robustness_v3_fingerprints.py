#!/usr/bin/env python3
from __future__ import annotations
import ast
import hashlib
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'scripts/robustness_corrected_v3.py'
OLD="""    expected=rm['inputs'];initial=fingerprint({key:item['path'] for key,item in expected.items()})
    if initial!=expected:raise ValueError('Corrected replay input changed')
"""
NEW="""    expected=rm['inputs']
    verify_recorded_inputs(expected)
"""
HELPER='def verify_recorded_inputs(records):\n    actual=fingerprint({key:item[\'path\'] for key,item in records.items()})\n    problems=[]\n    for key,expected in records.items():\n        observed=actual[key]\n        if observed[\'sha256\']!=expected[\'sha256\']:\n            problems.append(f"{key}: SHA256 mismatch; path={expected[\'path\']}")\n        if \'bytes\' in expected:\n            size=Path(expected[\'path\']).stat().st_size\n            if size!=expected[\'bytes\']:\n                problems.append(f"{key}: size mismatch; expected={expected[\'bytes\']}, actual={size}")\n    if problems:\n        raise ValueError(\'Corrected replay input changed:\\n\'+\'\\n\'.join(problems))\n\n\n'

def self_test(text):
    tree=ast.parse(text)
    names={'sha','fingerprint','verify_recorded_inputs'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes}!=names:raise RuntimeError('Expected provenance functions not found')
    namespace={'Path':Path,'hashlib':hashlib}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<provenance-selftest>','exec'),namespace)
    verify=namespace['verify_recorded_inputs']
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'evidence.txt';path.write_bytes(b'original')
        digest=hashlib.sha256(b'original').hexdigest()
        relative=os.path.relpath(path,Path.cwd())
        verify({'absolute':{'path':str(path.resolve()),'sha256':digest,'bytes':8},
                'relative':{'path':relative,'sha256':digest,'bytes':8}})
        verify({'minimal':{'path':str(path),'sha256':digest}})
        path.write_bytes(b'changed!')
        try:verify({'changed':{'path':str(path),'sha256':digest,'bytes':8}})
        except ValueError:pass
        else:raise AssertionError('Changed content was not rejected')
        path.write_bytes(b'original')
        try:verify({'wrong_size':{'path':str(path),'sha256':digest,'bytes':9}})
        except ValueError:pass
        else:raise AssertionError('Wrong size was not rejected')

def main():
    original=TARGET.read_text(encoding='utf-8')
    if 'def verify_recorded_inputs(records):' in original and NEW in original and OLD not in original:
        self_test(original)
        print('ALREADY PATCHED; SELF-TEST: PASSED')
        return
    if original.count(OLD)!=1 or original.count('def main():')!=1:
        raise RuntimeError('Unexpected script version; no file was changed. Send robustness_corrected_v3.py.')
    fixed=original.replace(OLD,NEW).replace('def main():',HELPER+'def main():',1)
    ast.parse(fixed);self_test(fixed)
    digest=hashlib.sha256(original.encode('utf-8')).hexdigest()[:16]
    backup=TARGET.with_name(TARGET.name+'.bak.'+digest)
    if backup.exists() and backup.read_text(encoding='utf-8')!=original:
        raise RuntimeError('Backup conflict; no file was changed')
    if not backup.exists():backup.write_text(original,encoding='utf-8')
    temporary=TARGET.with_name(TARGET.name+'.patch.tmp')
    temporary.write_text(fixed,encoding='utf-8')
    os.replace(temporary,TARGET)
    print('PATCH: PASSED')
    print('SELF-TEST: PASSED (metadata/path handling; SHA256 and size rejection)')
    print('BACKUP:',backup)
    print('PATCHED:',TARGET)
    print('MANIFESTS_CHANGED: False')

if __name__=='__main__':main()
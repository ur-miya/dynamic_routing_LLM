"""Pure single-request routing logic for the corrected candidate runtime."""
from __future__ import annotations
import math

def finite(value,name):
    value=float(value)
    if not math.isfinite(value):raise ValueError(f'Nonfinite {name}')
    return value

def route_one(record,*,direct_score=None,pre_score=None,gate_score=None,flags=None,prethreshold=None):
    method=record['method'];entry=record['threshold_policy']
    def optional(score):
        if entry['optional_disabled']:return False
        return finite(score,'routing score')>=finite(entry['threshold'],'threshold')
    if method.startswith('direct_'):
        teacher=optional(direct_score)
        return {'teacher':teacher,'student_needed':not teacher,'stage':'direct'}
    if not method.startswith(('cascade_','hybrid_')):raise ValueError('Unsupported frozen method')
    if method.startswith('hybrid_'):
        threshold=record.get('prethreshold',prethreshold)
        if finite(pre_score,'pre score')>=finite(threshold,'prethreshold'):
            return {'teacher':True,'student_needed':False,'stage':'pre_teacher'}
    if flags is None:raise ValueError('Student flags required on student branch')
    required=['finish_error','empty_answer','finish_length','confidence_missing']
    if any(key not in flags or flags[key] not in (0,1,False,True) for key in required):raise ValueError('Invalid student flags')
    allowed=set(record['fallback_rules'])
    checks=[('finish_error','student_error'),('empty_answer','empty_answer'),('finish_length','length_limit'),('confidence_missing','confidence_unavailable')]
    if any(bool(flags[key]) and reason in allowed for key,reason in checks):
        return {'teacher':True,'student_needed':True,'stage':'post_mandatory'}
    teacher=optional(gate_score)
    return {'teacher':teacher,'student_needed':True,'stage':'post_gate' if teacher else 'student_accepted'}

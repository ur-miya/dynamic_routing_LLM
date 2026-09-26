"""Low-memory helpers for Rasch 1PL calibration and Arrow embeddings."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

@dataclass
class RaschResult:
    abilities: np.ndarray
    difficulties: np.ndarray
    subject_ids: list[str]
    item_ids: list[str]
    item_counts: np.ndarray
    iterations: int
    converged: bool
    max_change: float
    log_loss: float


def read_response_matrix(path: Path, cfg: dict[str, Any], batch_size: int = 65536):
    cols=[cfg['subject_column'],cfg['item_column'],cfg['response_column']]
    smap: dict[str,int]={}; imap: dict[str,int]={}; subjects=[]; items=[]; ss=[]; ii=[]; yy=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,columns=cols):
        raw_s=batch.column(0).to_pylist(); raw_i=batch.column(1).to_pylist(); y=np.asarray(batch.column(2).to_numpy(zero_copy_only=False))
        if np.any(~np.isin(y,[0,1])): raise ValueError(f"{cols[2]} must contain only 0/1")
        s=np.empty(len(y),np.int32); i=np.empty(len(y),np.int32)
        for pos,value in enumerate(raw_s):
            key=str(value)
            if key not in smap: smap[key]=len(subjects); subjects.append(key)
            s[pos]=smap[key]
        for pos,value in enumerate(raw_i):
            key=str(value)
            if key not in imap: imap[key]=len(items); items.append(key)
            i[pos]=imap[key]
        ss.append(s); ii.append(i); yy.append(y.astype(np.float64,copy=False))
    if not yy: raise ValueError(f"Empty response matrix: {path}")
    return np.concatenate(ss),np.concatenate(ii),np.concatenate(yy),subjects,items


def fit_rasch(path: Path, cfg: dict[str, Any]) -> RaschResult:
    sidx,iidx,y,subjects,items=read_response_matrix(path,cfg)
    if len(subjects)<int(cfg['min_subjects']): raise ValueError(f"IRT requires >= {cfg['min_subjects']} subjects; got {len(subjects)}")
    counts=np.bincount(iidx,minlength=len(items)); required=int(cfg['min_item_responses'])
    if np.any(counts<required): raise ValueError(f"{int(np.sum(counts<required))} items have fewer than {required} responses")
    theta=np.zeros(len(subjects)); difficulty=np.zeros(len(items)); reg=float(cfg['l2']); clip=float(cfg['parameter_clip']); tol=float(cfg['tolerance']); change=float('inf'); converged=False
    for iteration in range(1,int(cfg['max_iterations'])+1):
        old_t=theta.copy(); old_d=difficulty.copy()
        z=np.clip(theta[sidx]-difficulty[iidx],-30,30); p=1/(1+np.exp(-z)); w=p*(1-p)
        theta+=(np.bincount(sidx,weights=y-p,minlength=len(theta))-reg*theta)/np.maximum(np.bincount(sidx,weights=w,minlength=len(theta))+reg,1e-8)
        z=np.clip(theta[sidx]-difficulty[iidx],-30,30); p=1/(1+np.exp(-z)); w=p*(1-p)
        difficulty+=(np.bincount(iidx,weights=p-y,minlength=len(difficulty))-reg*difficulty)/np.maximum(np.bincount(iidx,weights=w,minlength=len(difficulty))+reg,1e-8)
        shift=float(theta.mean()); theta-=shift; difficulty-=shift
        theta=np.clip(theta,-clip,clip); difficulty=np.clip(difficulty,-clip,clip)
        change=max(float(np.max(np.abs(theta-old_t))),float(np.max(np.abs(difficulty-old_d))))
        if change<tol: converged=True; break
    z=np.clip(theta[sidx]-difficulty[iidx],-30,30); p=np.clip(1/(1+np.exp(-z)),1e-12,1-1e-12); loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    return RaschResult(theta,difficulty,subjects,items,counts,iteration,converged,change,loss)


def embedding_matrix(column: pa.Array) -> np.ndarray:
    if isinstance(column,pa.ChunkedArray): column=column.combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        width=column.type.list_size
        return np.asarray(column.values.to_numpy(zero_copy_only=False),dtype=np.float32).reshape(len(column),width)
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        offsets=np.asarray(column.offsets.to_numpy(zero_copy_only=False))
        widths=np.diff(offsets)
        if len(widths)==0: return np.empty((0,0),dtype=np.float32)
        if np.any(widths!=widths[0]): raise ValueError("Embedding vectors have inconsistent dimensions")
        values=np.asarray(column.values.to_numpy(zero_copy_only=False),dtype=np.float32)
        return values.reshape(len(column),int(widths[0]))
    raise TypeError(f"Expected list embedding, got {column.type}")

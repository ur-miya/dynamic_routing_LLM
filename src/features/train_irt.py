#!/usr/bin/env python3
"""Calibrate Rasch on TRAIN responses and fit a streaming embedding predictor."""
from __future__ import annotations
import argparse,json,logging,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
import duckdb,joblib,numpy as np,pyarrow as pa,pyarrow.parquet as pq,yaml
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
try: from .irt_utils import embedding_matrix,fit_rasch
except ImportError: from irt_utils import embedding_matrix,fit_rasch
LOG=logging.getLogger('train_irt')

def atomic_dump(value,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent); os.close(fd)
    try: joblib.dump(value,name); os.replace(name,path)
    finally: Path(name).unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/irt.yaml'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--log-level',default='INFO'); a=p.parse_args(); logging.basicConfig(level=a.log_level.upper(),format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    cfg=yaml.safe_load(open(a.config,encoding='utf-8'))['irt']; response=Path(cfg['response_matrix']); embeddings=Path(cfg['embeddings_dir'])/'embeddings_train.parquet'; artifact=Path(cfg['artifact_dir']); model_path=artifact/'predictor.joblib'
    if not response.exists(): raise FileNotFoundError(f"Deferred IRT input is not ready: {response}")
    if not embeddings.exists(): raise FileNotFoundError(f"Train embeddings are required: {embeddings}")
    if model_path.exists() and not a.overwrite: raise FileExistsError(f"Use --overwrite: {model_path}")
    result=fit_rasch(response,cfg); artifact.mkdir(parents=True,exist_ok=True)
    items=pa.table({'prompt_id':pa.array(result.item_ids,pa.string()),'irt_calibrated_difficulty':pa.array(result.difficulties.astype(np.float32)),'irt_response_count':pa.array(result.item_counts.astype(np.int32))}); item_path=artifact/'train_item_difficulties.parquet'; pq.write_table(items,item_path,compression=cfg['compression'])
    temp=Path(cfg['temp_dir']); temp.mkdir(parents=True,exist_ok=True); joined=temp/'joined.parquet'; con=duckdb.connect(); con.execute("SET memory_limit='2GB'"); con.execute(f"SET temp_directory='{temp}'"); con.execute(f"COPY (SELECT e.prompt_embedding,i.irt_calibrated_difficulty FROM read_parquet('{embeddings}') e INNER JOIN read_parquet('{item_path}') i USING(prompt_id)) TO '{joined}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    parquet=pq.ParquetFile(joined); pcfg=cfg['predictor']; size=int(pcfg['batch_size']); scaler=StandardScaler(); rows=0; dim=None
    for batch in parquet.iter_batches(batch_size=size,columns=['prompt_embedding']): x=embedding_matrix(batch.column(0)); scaler.partial_fit(x); rows+=len(x); dim=x.shape[1]
    if rows<2: raise ValueError('Too few calibrated prompts for predictor training')
    model=SGDRegressor(loss='squared_error',penalty='l2',alpha=float(pcfg['alpha']),learning_rate=pcfg['learning_rate'],eta0=float(pcfg['eta0']),random_state=int(pcfg['random_state']),average=True)
    for epoch in range(int(pcfg['epochs'])):
        for batch in parquet.iter_batches(batch_size=size,columns=['prompt_embedding','irt_calibrated_difficulty']): model.partial_fit(scaler.transform(embedding_matrix(batch.column(0))),np.asarray(batch.column(1).to_numpy(zero_copy_only=False)))
        LOG.info('epoch=%d/%d',epoch+1,int(pcfg['epochs']))
    se=ae=count=0.0
    for batch in parquet.iter_batches(batch_size=size,columns=['prompt_embedding','irt_calibrated_difficulty']):
        x=scaler.transform(embedding_matrix(batch.column(0))); y=np.asarray(batch.column(1).to_numpy(zero_copy_only=False)); error=model.predict(x)-y; se+=float(error@error); ae+=float(np.abs(error).sum()); count+=len(y)
    contract=cfg['embedding_contract']; atomic_dump({'scaler':scaler,'model':model,'embedding_dimension':dim,'embedding_model_name':contract['model_name'],'embedding_normalized':bool(contract['normalized']),'trained_split':'train'},model_path)
    metrics={'created_at':datetime.now(timezone.utc).isoformat(),'subjects':len(result.subject_ids),'items':len(result.item_ids),'responses':int(result.item_counts.sum()),'rasch_iterations':result.iterations,'rasch_converged':result.converged,'rasch_max_change':result.max_change,'rasch_log_loss':result.log_loss,'predictor_train_rows':int(count),'predictor_train_mae':ae/count,'predictor_train_rmse':(se/count)**.5}; (artifact/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); joined.unlink(missing_ok=True); LOG.info('saved=%s',model_path); return 0
if __name__=='__main__': raise SystemExit(main())

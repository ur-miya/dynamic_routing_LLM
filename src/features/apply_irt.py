#!/usr/bin/env python3
"""Apply the TRAIN-only embedding-to-IRT predictor in streaming mode."""
from __future__ import annotations
import argparse,logging,os,tempfile
from pathlib import Path
import joblib,numpy as np,pyarrow as pa,pyarrow.parquet as pq,yaml
try: from .irt_utils import embedding_matrix
except ImportError: from irt_utils import embedding_matrix

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/irt.yaml'); p.add_argument('--splits',nargs='+'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--log-level',default='INFO'); a=p.parse_args(); logging.basicConfig(level=a.log_level.upper(),format='%(asctime)s | %(levelname)s | %(message)s')
    cfg=yaml.safe_load(open(a.config,encoding='utf-8'))['irt']; bundle=joblib.load(Path(cfg['artifact_dir'])/'predictor.joblib'); contract=cfg['embedding_contract']
    if bundle['embedding_model_name']!=contract['model_name'] or bundle['embedding_normalized']!=bool(contract['normalized']): raise RuntimeError('Embedding contract differs from IRT training')
    for split in a.splits or cfg['splits']:
        source=Path(cfg['embeddings_dir'])/f'embeddings_{split}.parquet'; output=Path(cfg['output_dir'])/f'irt_{split}.parquet'
        if output.exists() and not a.overwrite: raise FileExistsError(f'Use --overwrite: {output}')
        output.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=f'.{output.name}.',suffix='.tmp',dir=output.parent); os.close(fd); writer=None; rows=0
        try:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=int(cfg['predictor']['batch_size']),columns=['pair_id','prompt_id','message_tree_id','prompt_embedding']):
                x=embedding_matrix(batch.column(3));
                if x.shape[1]!=bundle['embedding_dimension']: raise RuntimeError('Embedding dimension mismatch')
                pred=bundle['model'].predict(bundle['scaler'].transform(x)).astype(np.float32); table=pa.table({'pair_id':batch.column(0),'prompt_id':batch.column(1),'message_tree_id':batch.column(2),'irt_predicted_difficulty':pa.array(pred),'irt_available':pa.array(np.ones(len(pred),dtype=bool))})
                if writer is None: writer=pq.ParquetWriter(name,table.schema,compression=cfg['compression'])
                writer.write_table(table); rows+=len(pred)
            if writer is None: raise ValueError(f'No embeddings in {source}')
            writer.close(); writer=None; os.replace(name,output); logging.info('saved=%s rows=%d',output,rows)
        finally:
            if writer is not None: writer.close()
            Path(name).unlink(missing_ok=True)
    return 0
if __name__=='__main__': raise SystemExit(main())

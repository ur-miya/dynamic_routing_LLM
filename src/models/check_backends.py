import argparse, asyncio, json, logging
from pathlib import Path
import yaml
from .teacher_client import TeacherClient

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/models.yaml"); ap.add_argument("--env-file",default=".env"); ap.add_argument("--output",default="artifacts/models/teacher_capabilities.json")
    args=ap.parse_args(); logging.basicConfig(level=logging.INFO)
    cfg=yaml.safe_load(open(args.config)); c=TeacherClient(cfg["teacher"],args.env_file)
    result=asyncio.run(c.preflight()); p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()

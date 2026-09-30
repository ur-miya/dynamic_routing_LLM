import argparse, subprocess, sys

def call(module, common, split): subprocess.run([sys.executable,"-m",module,*common,"--split",split],check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default="configs/models.yaml"); ap.add_argument("--generation",default="configs/generation.yaml"); ap.add_argument("--env-file",default=".env"); ap.add_argument("--splits",nargs="+",default=["train","val","test"]); ap.add_argument("--limit",type=int); ap.add_argument("--with-teacher-scores",action="store_true"); ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args(); common=["--models",a.models,"--generation",a.generation,"--env-file",a.env_file]+(["--limit",str(a.limit)] if a.limit else [])
    for split in a.splits:
        call("src.generation.generate_teacher",common,split); call("src.generation.generate_student",common,split)
        if a.with_teacher_scores: call("src.generation.score_teacher_sequences",common,split)
        m=[sys.executable,"-m","src.generation.merge_generations","--generation",a.generation,"--split",split]
        if a.with_teacher_scores:m.append("--include-scores")
        if a.overwrite:m.append("--overwrite")
        subprocess.run(m,check=True)
if __name__=="__main__":main()

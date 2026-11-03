from pathlib import Path
import hashlib,urllib.request,whisper
r=Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
p=r/"assets/asr_validation/tiny.pt";u=whisper._MODELS["tiny"];expected=u.split("/")[-2]
for attempt in range(3):
 start=p.stat().st_size if p.exists() else 0
 req=urllib.request.Request(u,headers={"Range":"bytes="+str(start)+"-"})
 with urllib.request.urlopen(req,timeout=30) as response:
  print("ASR_RESUME",response.status,start,response.headers.get("Content-Range"),flush=True)
  mode="ab" if response.status==206 else "wb"
  with p.open(mode) as f:
   while True:
    b=response.read(1024*1024)
    if not b:break
    f.write(b)
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 print("ASR_MODEL_HASH",p.stat().st_size,digest,flush=True)
 if digest==expected:break
else:raise RuntimeError("ASR checkpoint still fails integrity verification")

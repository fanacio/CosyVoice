import pathlib,json,urllib.request,tarfile,io
r=pathlib.Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
pin=json.loads((r/"references/matcha_pin.json").read_text())["sha"]
u="https://api.github.com/repos/shivammehta25/Matcha-TTS/tarball/"+pin
print("FETCH_MATCHA",u,flush=True)
req=urllib.request.Request(u,headers={"User-Agent":"CosyVoice-Ascend-deployment"})
with urllib.request.urlopen(req,timeout=40) as f:data=f.read(80*1024*1024)
(r/"references/Matcha-TTS-official.tar.gz").write_bytes(data)
stage=r/"src/.matcha_extract";stage.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as t:t.extractall(stage,filter="data")
dirs=[p for p in stage.iterdir() if p.is_dir()]
assert len(dirs)==1
dest=r/"src/CosyVoice/third_party/Matcha-TTS"
if dest.exists():assert not any(dest.iterdir());dest.rmdir()
dirs[0].rename(dest)
(dest/".source_commit").write_text(pin+"\n")
print("MATCHA_READY",dest,flush=True)

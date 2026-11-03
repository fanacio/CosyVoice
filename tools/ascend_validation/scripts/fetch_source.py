import pathlib,urllib.request,tarfile,io,time
root=pathlib.Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
dst=root/"src/CosyVoice"
sha="074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
urls=["https://api.github.com/repos/FunAudioLLM/CosyVoice/tarball/"+sha,"https://codeload.github.com/FunAudioLLM/CosyVoice/tar.gz/"+sha]
for u in urls:
 try:
  print("FETCH",u,flush=True)
  req=urllib.request.Request(u,headers={"User-Agent":"CosyVoice-Ascend-deployment"})
  with urllib.request.urlopen(req,timeout=30) as r:data=r.read(100*1024*1024)
  archive=root/"references/CosyVoice-official.tar.gz";archive.write_bytes(data)
  stage=root/"src/.source_extract";stage.mkdir(exist_ok=True)
  with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as t:
   t.extractall(stage,filter="data")
  folders=[p for p in stage.iterdir() if p.is_dir()]
  assert len(folders)==1
  if dst.exists():assert not any(dst.iterdir());dst.rmdir()
  folders[0].rename(dst)
  (dst/".source_commit").write_text(sha+"\n")
  print("SOURCE_READY",dst,flush=True)
  print("SUBMODULES",(dst/".gitmodules").read_text(),flush=True)
  break
 except Exception as e:print("SOURCE_FETCH_ERROR",repr(e),flush=True)
else:raise SystemExit(1)

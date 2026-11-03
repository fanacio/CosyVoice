import pathlib,json,time,socket
import torch,torchaudio,soundfile as sf
import whisper
r=pathlib.Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
socket.setdefaulttimeout(30)
torch.set_num_threads(8)
try:
 model=whisper.load_model("tiny",device="cpu",download_root=str(r/"assets/asr_validation"))
 files=[r/"outputs/cosyvoice3_ascend.wav",r/"outputs/stage2/fixed_tokens_npu.wav",r/"outputs/stage2/cosyvoice3_full_npu.wav"]
 result=[]
 for path in files:
  deadline=time.monotonic()+180
  while not path.exists() and time.monotonic()<deadline:time.sleep(1)
  if not path.exists():raise FileNotFoundError(path)
  audio,sr=sf.read(path,dtype="float32")
  wave=torch.from_numpy(audio)
  if sr!=16000:wave=torchaudio.functional.resample(wave,sr,16000)
  text=model.transcribe(wave.numpy(),language="zh",fp16=False,temperature=0.0,condition_on_previous_text=False,without_timestamps=True)["text"]
  result.append({"audio":str(path),"asr_text":text})
  print("ASR_RESULT",result[-1],flush=True)
 out={"model":"openai-whisper-tiny","device":"cpu_validation_only","results":result,"note":"Coarse content check; not a MOS or full quality evaluation"}
 (r/"reports/stage2_asr_check.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
except Exception as e:
 (r/"reports/stage2_asr_check.json").write_text(json.dumps({"status":"not_completed","error":repr(e)},indent=2))
 raise

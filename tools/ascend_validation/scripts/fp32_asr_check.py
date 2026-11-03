from pathlib import Path
import json,hashlib
import torch,torchaudio,soundfile as sf,whisper
r=Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
torch.set_num_threads(8)
model=whisper.load_model("tiny",device="cpu",download_root=str(r/"assets/asr_validation"))
cases=[]
for name in ["cosyvoice3_full_npu","cosyvoice3_full_npu_long"]:
 p=r/"outputs/fp32"/(name+".wav")
 wave,sr=sf.read(p,dtype="float32")
 x=torch.from_numpy(wave)
 if sr!=16000:x=torchaudio.functional.resample(x,sr,16000)
 text=model.transcribe(x.numpy(),language="zh",fp16=False,temperature=0.0,condition_on_previous_text=False,without_timestamps=True)["text"]
 cases.append({"audio":str(p),"text":text})
 print("FP32_ASR_CHECK",cases[-1],flush=True)
(r/"reports/fp32_asr_check.json").write_text(json.dumps({"model":"whisper-tiny","device":"cpu_validation_only","cases":cases,"scope":"coarse transcription, not MOS"},ensure_ascii=False,indent=2))

from pathlib import Path
import torch,json
from stage2_frontend import converted_model
r=Path(__file__).resolve().parents[1]
m=Path("/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512")
torch.set_num_threads(8)
result={}
for name in ["campplus.onnx","speech_tokenizer_v3.onnx"]:
 model=converted_model(m/name,r,torch.device("cpu"))
 result[name]={"parameters":sum(x.numel() for x in model.parameters()),"buffers":sum(x.numel() for x in model.buffers())}
 print("FRONTEND_CACHE_READY",name,result[name],flush=True)
 del model
(r/"reports/stage2_frontend_conversion.json").write_text(json.dumps(result,indent=2))

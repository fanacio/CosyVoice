import pathlib,json,collections
import onnx
from onnx2torch.node_converters.registry import get_converter
r=pathlib.Path("/data/tiankuan/fanyc/models_test_opt/CosyVoice")
model=pathlib.Path("/public-flash/models/q4_models/iic/Fun-CosyVoice3-0.5B-2512")
report={}
for name in ["campplus.onnx","speech_tokenizer_v3.onnx"]:
 print("ANALYZE",name,flush=True)
 graph=onnx.load_model(model/name,load_external_data=False)
 versions={x.domain:x.version for x in graph.opset_import}
 types=collections.Counter(n.op_type for n in graph.graph.node)
 missing={}
 for n in graph.graph.node:
  try:get_converter(n.op_type,versions.get(n.domain,versions.get("",13)),n.domain)
  except NotImplementedError as e:missing[n.op_type]=str(e)
 def info(v):return {"name":v.name,"dtype":v.type.tensor_type.elem_type,"shape":[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]}
 report[name]={"opsets":versions,"nodes":len(graph.graph.node),"types":dict(types),"unsupported":missing,"inputs":[info(x) for x in graph.graph.input],"outputs":[info(x) for x in graph.graph.output]}
 print(json.dumps({name:report[name]}),flush=True)
 del graph
(r/"reports/stage2_onnx_graphs.json").write_text(json.dumps(report,indent=2))

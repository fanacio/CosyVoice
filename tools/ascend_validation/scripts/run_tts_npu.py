"""Full neural NPU pipeline, retaining the official vLLM LLM integration."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llm-dtype", choices=["float32", "bfloat16"], default="float32")
    arguments, _ = parser.parse_known_args()
    root = arguments.root.resolve()
    source = root / "src/CosyVoice"
    sys.path[:0] = [str(source), str(source / "third_party/Matcha-TTS")]
    output = arguments.output
    if output is None:
        output = root / ("outputs/fp32/cosyvoice3_full_npu.wav" if arguments.llm_dtype == "float32" else "outputs/stage2/cosyvoice3_full_npu.wav")
        sys.argv.extend(["--output", str(output)])
    from stage2_backend import install_full_npu_pipeline
    state = install_full_npu_pipeline(root)
    from run_tts import main as official_pipeline
    official_pipeline()
    from vllm.platforms import current_platform
    frontend = state.pop("frontend_object")
    state["frontend"] = {"backend": "official_onnx_to_torch_npu",
                         "execution": frontend.neural_execution,
                         "neural_seconds": frontend.component_timings}
    state["vllm_platform_class"] = f"{type(current_platform).__module__}.{type(current_platform).__name__}"
    state["host_work"] = ["text tokenization", "file I/O", "shape/control metadata", "vLLM prompt IPC staging"]
    report_path = output.with_suffix(".json")
    report = json.loads(report_path.read_text())
    report["stage2"] = state
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("FULL_NPU_TTS_SUCCESS", json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

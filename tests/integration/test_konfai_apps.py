import json
import math
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Expected JSON file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON in file: {path}\n{exc}") from exc


def extract_case_metrics(data: dict, *, context: str) -> dict[str, float]:
    """
    Returns {metric_key: single_value} from:
      data["case"][metric_key] == {"P000": value}  (or potentially multiple cases)
    We enforce that each metric contains exactly 1 case value for this test.
    """
    if "case" not in data or not isinstance(data["case"], dict):
        raise AssertionError(f"{context}: JSON does not contain a 'case' dict.")

    out: dict[str, float] = {}
    for metric_key, case_map in data["case"].items():
        if not isinstance(case_map, dict) or len(case_map) == 0:
            raise AssertionError(f"{context}: metric '{metric_key}' has no case values.")
        if len(case_map) != 1:
            raise AssertionError(
                f"{context}: metric '{metric_key}' has {len(case_map)} case values, " "but this test expects exactly 1."
            )

        value = next(iter(case_map.values()))
        if not isinstance(value, (int, float)):
            raise AssertionError(f"{context}: metric '{metric_key}' value is not numeric: {value!r}")
        out[metric_key] = float(value)

    return out


def assert_metrics_close(
    baseline: dict[str, float],
    output: dict[str, float],
    *,
    name: str,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-8,
) -> None:
    base_keys = set(baseline.keys())
    out_keys = set(output.keys())

    missing = sorted(base_keys - out_keys)
    extra = sorted(out_keys - base_keys)

    if missing or extra:
        msg = [f"{name}: metric keys mismatch."]
        if missing:
            msg.append(f"Missing keys ({len(missing)}): {missing}")
        if extra:
            msg.append(f"Extra keys ({len(extra)}): {extra}")
        raise AssertionError("\n".join(msg))

    diffs: list[str] = []
    for k in sorted(base_keys):
        b = baseline[k]
        o = output[k]
        if not math.isclose(o, b, rel_tol=rel_tol, abs_tol=abs_tol):
            diffs.append(f"- {k}: baseline={b:.12g}, output={o:.12g}, diff={o-b:.12g}")

    if diffs:
        raise AssertionError(f"{name}: values differ (rel_tol={rel_tol}, abs_tol={abs_tol}).\n" + "\n".join(diffs))


def test_konfai_apps_infer(tmp_path: Path):
    # --- Load baselines
    eval_base_path = Path("tests/assets/Baselines/Evaluation.json")
    unc_base_path = Path("tests/assets/Baselines/Uncertainties.json")

    eval_baseline_data = read_json(eval_base_path)
    unc_baseline_data = read_json(unc_base_path)

    eval_baseline = extract_case_metrics(eval_baseline_data, context="Baseline Evaluation")
    unc_baseline = extract_case_metrics(unc_baseline_data, context="Baseline Uncertainties")

    # --- Run pipeline
    cmd = [
        "konfai-apps",
        "pipeline",
        "VBoussot/ImpactSynth:CBCT",
        "-i",
        "tests/assets/Dataset/P001/CBCT.mha",
        "-o",
        str(tmp_path),
        "--gt",
        "tests/assets/Dataset/P001/CT.mha",
        "--mask",
        "tests/assets/Dataset/P001/MASK.mha",
        "--ensemble",
        "2",
        "--tta",
        "0",
        "--mc",
        "0",
    ]

    p = run(cmd)
    assert p.returncode == 0, (
        "The 'konfai-apps pipeline' command failed.\n\n"
        f"CMD: {' '.join(cmd)}\n\n"
        f"STDOUT:\n{p.stdout}\n\n"
        f"STDERR:\n{p.stderr}"
    )

    # --- Load outputs
    eval_out_path = tmp_path / "Evaluations" / "ImpactSynth" / "Metric_TRAIN.json"
    unc_out_path = tmp_path / "Uncertainties" / "ImpactSynth" / "Metric_TRAIN.json"

    eval_out_data = read_json(eval_out_path)
    unc_out_data = read_json(unc_out_path)

    eval_out = extract_case_metrics(eval_out_data, context="Output Evaluation")
    unc_out = extract_case_metrics(unc_out_data, context="Output Uncertainties")

    # --- Compare
    assert_metrics_close(eval_baseline, eval_out, name="Evaluation", rel_tol=1e-6, abs_tol=1e-8)
    assert_metrics_close(unc_baseline, unc_out, name="Uncertainties", rel_tol=1e-6, abs_tol=1e-8)

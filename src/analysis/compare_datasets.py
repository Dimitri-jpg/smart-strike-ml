"""Create a metadata-only dataset alignment report.
Run after both audits:
python src/analysis/compare_datasets.py
No preprocessing, resampling, trimming, window creation or training is done.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/"reports/dataset_analysis"
SMART=DIR/"smartstrike_report.json"
SILENT=DIR/"silent_impact_report.json"
OUT=DIR/"alignment_report.json"
SMART_CHANNELS=["accel_x","accel_y","accel_z","linear_x","linear_y","linear_z",
                "gyro_x","gyro_y","gyro_z","rot_x","rot_y","rot_z","rot_w"]
SILENT_CHANNELS=["linear_x","linear_y","linear_z","gyro_x","gyro_y","gyro_z"]

def load(p):
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run both audit scripts first.")
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    s,l=load(SMART),load(SILENT)
    common=[c for c in SMART_CHANNELS if c in SILENT_CHANNELS]
    sd=s["overall_summary"]
    silent_by_file={x["file"]:x for x in l["npz_datasets"]}
    base=silent_by_file.get("data/processed/silent_impact/train_windows.npz",{})
    report={
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "status":"analysis_only_no_preprocessing",
      "sensor_channels":{"smartstrike":SMART_CHANNELS,"silent_impact":SILENT_CHANNELS,
        "common":common,"smartstrike_only":[x for x in SMART_CHANNELS if x not in common]},
      "labels":{"smartstrike":sd.get("class_distribution"),
        "silent_impact_train":base.get("label_distribution"),
        "note":"Do not assume the three SmartStrike shot types map to Silent Impact labels. Define binary SHOT/NO_SHOT separately before considering multi-class transfer."},
      "temporal":{"smartstrike":sd.get("class_duration_and_sampling_summary"),
        "silent_impact_window_samples":{k:v.get("window_samples") for k,v in silent_by_file.items() if v.get("window_samples") is not None},
        "note":"Silent Impact uses 128-sample windows. Check actual sampling intervals, units, coordinate conventions and window-generation policy before choosing SmartStrike window length."},
      "split_strategy":{
        "silent_impact":"Preserve the existing participant-wise split for the existing benchmark; verify details in preprocessing metadata.",
        "smartstrike":"Do not treat recording date as participant identity. Keep all windows from one source CSV in one split; use participant grouping if participant IDs become available.",
        "evaluation":"Compare experiments on the same held-out SmartStrike test groups; prevent windows from one source recording crossing train/test."},
      "decisions_not_yet_made":["sampling-rate conversion/resampling","SmartStrike window length and stride",
        "whether to trim recording edges","normalization strategy fitted on training data only",
        "exact label mapping","whether combined training is scientifically valid"],
      "source_reports":["reports/dataset_analysis/smartstrike_report.json","reports/dataset_analysis/silent_impact_report.json"]}
    OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("ALIGNMENT REPORT")
    print("Common channels:",", ".join(common))
    print("SmartStrike classes:",sd.get("class_distribution"))
    print("Still to decide:",", ".join(report["decisions_not_yet_made"]))
    print("Saved:",OUT)

if __name__=="__main__": main()

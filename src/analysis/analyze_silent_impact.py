"""Read-only Silent Impact NPZ audit. Run from project root:
python src/analysis/analyze_silent_impact.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/silent_impact"
MODELS = ROOT / "models/silent_impact_baseline"
REPORTS = ROOT / "reports/silent_impact_baseline"
OUT = ROOT / "reports/dataset_analysis/silent_impact_report.json"
CHANNELS = ["linear_x","linear_y","linear_z","gyro_x","gyro_y","gyro_z"]
NPZ_FILES = ["train_windows.npz","test_windows.npz",
 "threshold_experiments/train_threshold_10.npz","threshold_experiments/test_threshold_10.npz",
 "threshold_experiments/train_threshold_25.npz","threshold_experiments/test_threshold_25.npz",
 "threshold_experiments/train_threshold_50.npz","threshold_experiments/test_threshold_50.npz"]

def py(v):
    return v.item() if isinstance(v,np.generic) else v

def inspect(path):
    item={"file":str(path.relative_to(ROOT)),"exists":path.exists()}
    if not path.exists(): return item
    try:
        with np.load(path,allow_pickle=False) as z:
            item["keys"]=list(z.files)
            item["arrays"]={k:{"shape":list(z[k].shape),"dtype":str(z[k].dtype)} for k in z.files}
            if "X" in z and "y" in z:
                X,y=z["X"],z["y"]
                labels,counts=np.unique(y,return_counts=True)
                item.update({"X_shape":list(X.shape),"X_dtype":str(X.dtype),
                    "y_shape":list(y.shape),"label_distribution":{str(py(a)):int(b) for a,b in zip(labels,counts)}})
                if X.ndim==3:
                    item["window_samples"]=int(X.shape[1])
                    item["channels_per_window"]=int(X.shape[2])
                for key in ["participant_id","arm","sequence_index","start_frame","shot_ratio"]:
                    if key in z:
                        vals,cts=np.unique(z[key],return_counts=True)
                        item[key]={"unique_count":int(len(vals))}
                        if len(vals)<=50:
                            item[key]["distribution"]={str(py(a)):int(b) for a,b in zip(vals,cts)}
        item["error"]=None
    except Exception as e: item["error"]=repr(e)
    return item

def inventory(folder, suffixes):
    if not folder.exists(): return []
    return [{"path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in suffixes]

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    datasets=[inspect(DATA/name) for name in NPZ_FILES]
    threshold={}
    base=DATA/"train_windows.npz"
    if base.exists():
        with np.load(base,allow_pickle=False) as b:
            for n in [10,25,50]:
                p=DATA/f"threshold_experiments/train_threshold_{n}.npz"
                if not p.exists(): threshold[str(n)]={"exists":False}; continue
                with np.load(p,allow_pickle=False) as z:
                    labels,counts=np.unique(z["y"],return_counts=True)
                    threshold[str(n)]={"exists":True,
                        "X_same_shape":z["X"].shape==b["X"].shape,
                        "X_exactly_equal_to_base":z["X"].shape==b["X"].shape and bool(np.array_equal(z["X"],b["X"])),
                        "y_same_as_base":z["y"].shape==b["y"].shape and bool(np.array_equal(z["y"],b["y"])),
                        "label_distribution":{str(py(a)):int(c) for a,c in zip(labels,counts)}}
    metadata=DATA/"preprocessing_metadata.txt"
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"read_only":True,
        "common_channels_with_smartstrike":CHANNELS,"npz_datasets":datasets,
        "preprocessing_metadata":metadata.read_text(encoding="utf-8",errors="replace") if metadata.exists() else None,
        "threshold_comparison_to_base_train":threshold,
        "model_files":inventory(MODELS,{".joblib",".pkl"}),
        "report_files":inventory(REPORTS,{".json",".txt",".md"})}
    OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("SILENT IMPACT AUDIT")
    for d in datasets:
        print(d["file"],"MISSING" if not d["exists"] else
              f'X={d.get("X_shape")} y={d.get("y_shape")} labels={d.get("label_distribution")}')
        if d.get("error"): print("  ERROR:",d["error"])
    print("Threshold comparison:",threshold)
    print("Models found:",len(report["model_files"]),"Reports found:",len(report["report_files"]))
    print("Saved:",OUT)

if __name__=="__main__": main()

"""Read-only SmartStrike CSV audit. Run from project root:
python src/analysis/analyze_smartstrike.py
"""
import csv, json, re, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/smartstrike/SensorReadings"
FOLDERS = {"DatasetV4": DATA / "DatasetV4", "NoneDatasetV4": DATA / "NoneDatasetV4"}
OUT = ROOT / "reports/dataset_analysis/smartstrike_report.json"
COLUMNS = ["timestamp","shot_id","shot_type","accel_x","accel_y","accel_z",
           "linear_x","linear_y","linear_z","gyro_x","gyro_y","gyro_z",
           "rot_x","rot_y","rot_z","rot_w"]
COMMON = ["linear_x","linear_y","linear_z","gyro_x","gyro_y","gyro_z"]
NAME_RE = re.compile(r"^session_(.+)_(\d{8})_(\d{6})_(\d+)_score(\d+)(?:_score\d+)*\.csv$", re.I)
BUCKETS = {"<2s": lambda x:x<2, "2-3s":lambda x:2<=x<3,
           "3-4s":lambda x:3<=x<4, "4-5s":lambda x:4<=x<5, "5+s":lambda x:x>=5}

def med(v): return statistics.median(v) if v else None
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def inspect(path, folder):
    m = NAME_RE.match(path.name)
    info = m.groupdict() if m else None
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows, ts, types, ids = 0, [], Counter(), set()
            empty, malformed = 0, 0
            channel_values = defaultdict(list)
            for row in reader:
                rows += 1
                malformed += int(None in row)
                empty += sum(row.get(c) in (None, "") for c in COLUMNS)
                if row.get("shot_type"): types[row["shot_type"]] += 1
                if row.get("shot_id") not in (None, ""): ids.add(row["shot_id"])
                t = num(row.get("timestamp"))
                if t is not None: ts.append(t)
                for c in COLUMNS[3:]:
                    x = num(row.get(c))
                    if x is not None: channel_values[c].append(x)
        diffs = [b-a for a,b in zip(ts,ts[1:])]
        duration = (ts[-1]-ts[0])/1000 if len(ts)>1 else None
        return {
            "file":path.name,"folder":folder,"filename_parse_ok":bool(info),"filename":info,
            "rows":rows,"empty_cells":empty,"malformed_rows":malformed,
            "missing_columns":[c for c in COLUMNS if c not in headers],
            "extra_columns":[c for c in headers if c not in COLUMNS],
            "shot_types_in_file":list(types),"shot_ids_in_file":list(ids),
            "one_shot_id_per_file":len(ids)==1,"one_shot_type_per_file":len(types)==1,
            "duration_seconds":duration,"median_interval_ms":med(diffs),
            "timestamp_interval_counts":{
                "over_5ms":sum(x>5 for x in diffs),"over_10ms":sum(x>10 for x in diffs),
                "over_20ms":sum(x>20 for x in diffs),"equal":sum(x==0 for x in diffs),
                "negative":sum(x<0 for x in diffs)},
            "channel_stats":{c:{"count":len(v),"min":min(v),"max":max(v),
                "mean":statistics.fmean(v)} for c,v in channel_values.items()},
            "read_error":None}
    except Exception as e:
        return {"file":path.name,"folder":folder,"filename_parse_ok":bool(info),
                "filename":info,"read_error":repr(e)}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    folder_reports, all_files = {}, []
    for name, folder in FOLDERS.items():
        paths = sorted(folder.glob("*.csv")) if folder.exists() else []
        records = [inspect(p,name) for p in paths]
        folder_reports[name] = {"path":str(folder),"csv_files_found":len(paths),"files":records,
                                "error":None if folder.exists() else "Folder not found"}
        all_files.extend(records)

    valid = [x for x in all_files if not x.get("read_error")]
    classes, date_class, all_ids = defaultdict(list), Counter(), []
    for x in valid:
        label = x["shot_types_in_file"][0] if len(x["shot_types_in_file"])==1 else "MIXED_OR_UNKNOWN"
        classes[label].append(x)
        all_ids.extend(x["shot_ids_in_file"])
        if x["filename"]: date_class[(x["filename"]["date"],label)] += 1

    summaries = {}
    for label, group in sorted(classes.items()):
        durations = [x["duration_seconds"] for x in group if x["duration_seconds"] is not None]
        counts = [x["rows"] for x in group]
        summaries[label] = {
            "files":len(group),"rows_min":min(counts),"rows_median":med(counts),"rows_max":max(counts),
            "duration_min_seconds":min(durations) if durations else None,
            "duration_median_seconds":med(durations),"duration_max_seconds":max(durations) if durations else None,
            "duration_buckets":{k:sum(fn(d) for d in durations) for k,fn in BUCKETS.items()},
            "median_of_file_median_intervals_ms":med([x["median_interval_ms"] for x in group if x["median_interval_ms"] is not None]),
            "average_gaps_over_5ms_per_file":statistics.fmean(x["timestamp_interval_counts"]["over_5ms"] for x in group),
            "average_gaps_over_10ms_per_file":statistics.fmean(x["timestamp_interval_counts"]["over_10ms"] for x in group),
            "average_gaps_over_20ms_per_file":statistics.fmean(x["timestamp_interval_counts"]["over_20ms"] for x in group)}
    id_counts = Counter(all_ids)
    report = {
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"read_only":True,
        "expected_columns":COLUMNS,"folders":folder_reports,
        "overall_summary":{
            "files_read":len(valid),"files_with_read_errors":len(all_files)-len(valid),
            "filename_parse_failures":[x["file"] for x in valid if not x["filename_parse_ok"]],
            "duplicate_shot_ids_across_files":{k:v for k,v in id_counts.items() if v>1},
            "class_distribution":{k:len(v) for k,v in sorted(classes.items())},
            "class_duration_and_sampling_summary":summaries,
            "date_by_class_distribution":[{"date":d,"class":c,"files":n} for (d,c),n in sorted(date_class.items())],
            "common_channels_with_silent_impact":COMMON}}
    OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("SMARTSTRIKE AUDIT")
    for name, data in folder_reports.items(): print(name, data["csv_files_found"], "CSV files")
    print("Files read:",len(valid)," Read errors:",len(all_files)-len(valid))
    print("Filename parse failures:",report["overall_summary"]["filename_parse_failures"] or "None")
    print("Duplicate IDs:",report["overall_summary"]["duplicate_shot_ids_across_files"] or "None")
    for label,s in summaries.items():
        print(f'{label:16} files={s["files"]:3} rows median={s["rows_median"]} '
              f'duration median={s["duration_median_seconds"]:.3f}s '
              f'avg gaps >5ms={s["average_gaps_over_5ms_per_file"]:.2f}')
    print("Saved:",OUT)

if __name__=="__main__": main()

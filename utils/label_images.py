import os, csv
from pathlib import Path
import regex as re


folder = Path(r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\test_set')
out_csv = Path(r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\utils\labels.csv')

def filename_to_label(fname):
    fn = fname.lower()
    if re.search(r"normal", fname, re.IGNORECASE):
        return 0
    else:
        return 1

rows = []
for p in folder.iterdir():
    if p.suffix.lower() in (".png",".jpg",".jpeg"):
        lbl = filename_to_label(p.name)
        if lbl >= 0:
            rows.append((p.name, lbl))

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    # Write header columns
    w.writerow(["name", "initial_label"])
    for r in rows:
        w.writerow(r)

print("Wrote", len(rows), "labels to", out_csv)

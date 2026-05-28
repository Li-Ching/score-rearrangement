import os, json, csv, warnings
from music21 import musicxml
warnings.filterwarnings("ignore", category=musicxml.xmlToM21.MusicXMLWarning)

from score_to_tokens import ViolinXML_to_tokens

csv_path   = "./PDMX.csv"
mxl_root   = "./mxl"
output_dir = "./violin_tokens"
os.makedirs(output_dir, exist_ok=True)

# ── 從 CSV 找出所有純小提琴的 MXL 路徑 ───────────────────────────────────────
violin_paths = []
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tracks = row['tracks']
        if all(t == '40' for t in tracks.split('-')):
            rel = row['mxl'].lstrip('./')
            violin_paths.append(os.path.join(os.path.dirname(csv_path), rel))

print(f"Found {len(violin_paths)} violin scores to tokenize.")

success, failed = 0, 0

for in_path in violin_paths:
    if not os.path.exists(in_path):
        continue 

    rel      = os.path.relpath(in_path, mxl_root)
    out_path = (os.path.join(output_dir, rel)
                .replace('.mxl', '.json')
                .replace('.xml', '.json'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        continue  

    try:
        tokens = ViolinXML_to_tokens(in_path, note_name=True)

        if not tokens:
            failed += 1
            continue

        with open(out_path, 'w') as f:
            json.dump(tokens, f)

        success += 1
        if success % 500 == 0:
            print(f"Processed {success} files ({failed} failed)")

    except Exception as e:
        failed += 1
        print(f"Failed {in_path}: [{type(e).__name__}] {e}")

print(f"Done. {success} succeeded, {failed} failed.")
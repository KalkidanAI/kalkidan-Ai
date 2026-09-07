import json
import os
import glob

notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))

print(f"Verifying {len(notebook_files)} notebooks in {notebooks_dir}:")
all_valid = True

for nb_path in notebook_files:
    fname = os.path.basename(nb_path)
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = data.get("cells", [])
        nbformat = data.get("nbformat")
        print(f"  [OK] {fname:32s} | Format: v{nbformat} | Cells: {len(cells)}")
    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        all_valid = False

if all_valid:
    print("\nAll notebooks passed JSON validity checks successfully!")
else:
    print("\nSome notebooks failed validity checks.")
    exit(1)

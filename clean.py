import json, os

for file in os.listdir():
    if file.endswith(".ipynb"):
        with open(file) as f:
            nb = json.load(f)
        if "widgets" in nb.get("metadata", {}):
            del nb["metadata"]["widgets"]
            print("Cleaned", file)
        with open(file, "w") as f:
            json.dump(nb, f, indent=2)

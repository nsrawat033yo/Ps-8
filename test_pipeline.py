from pathlib import Path
from lunar_icenav.pipeline import run_pipeline

if __name__ == "__main__":
    print("Running pipeline...")
    root = Path(".")
    run_pipeline(root)
    print("Done!")

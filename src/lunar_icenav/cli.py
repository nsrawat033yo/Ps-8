from __future__ import annotations

import argparse
from pathlib import Path

from lunar_icenav.config import ensure_output_dirs, load_config
from lunar_icenav.io.products import discover_products
from lunar_icenav.pipeline import create_notebook, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="LunaQuest / Lunar IceNav prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect workspace products and write inventory CSV")
    inspect_p.add_argument("--root", default=".", help="Project root")
    inspect_p.add_argument("--config", default="configs/pipeline.json")

    run_p = sub.add_parser("run", help="Run the end-to-end prototype pipeline")
    run_p.add_argument("--root", default=".", help="Project root")
    run_p.add_argument("--config", default="configs/pipeline.json")

    notebook_p = sub.add_parser("notebook", help="Regenerate the research notebook from current outputs")
    notebook_p.add_argument("--root", default=".", help="Project root")
    notebook_p.add_argument("--config", default="configs/pipeline.json")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)

    if args.command == "inspect":
        df = discover_products(root)
        df.to_csv(paths["tables"] / "product_inventory.csv", index=False)
        df.to_csv(paths["metadata"] / "product_inventory.csv", index=False)
        print(f"Wrote {len(df)} inventory rows to {paths['tables'] / 'product_inventory.csv'}")
    elif args.command == "run":
        manifest = run_pipeline(root, Path(args.config))
        print("Prototype complete.")
        print(f"Selected SAR: {manifest['selected_sar'].get('product_id')}")
        print("Figures: outputs/figures")
        print("Tables: outputs/tables")
        print("Summary: reports/LUNAQUEST_PROTOTYPE_SUMMARY.md")
    elif args.command == "notebook":
        notebook_path = root / "notebooks" / "LunaQuest_BAH2026_Workflow.ipynb"
        create_notebook(notebook_path)
        print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    main()

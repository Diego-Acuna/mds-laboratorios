"""Smoke test for local repository: run scoring pipeline against local data/artifacts.

Run this script from the repository root (it uses relative paths).
"""
from pathlib import Path
import traceback


def main():
    repo = Path.cwd()
    data_dir = repo / "airflow" / "data"
    artifacts_dir = repo / "airflow" / "artifacts_entrega1" / "artifacts_optuna"

    print("Data dir:", data_dir)
    print("Artifacts dir:", artifacts_dir)

    try:
        # Import here so we can catch missing deps
        import sys
        # Ensure local airflow package is importable
        sys.path.insert(0, str(repo / "airflow"))
        from sodai_core import inference

        print("Imported sodai_core.inference")

        # Try a quick scoring pipeline call
        print("Running run_scoring_pipeline(...) (top_k=5) ...")
        ranking = inference.run_scoring_pipeline(
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            top_k=5,
        )

        print("Ranking head:")
        try:
            print(ranking.head())
        except Exception:
            print(repr(ranking))

        print("Smoke test completed successfully.")
    except Exception as e:
        print("Smoke test failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()

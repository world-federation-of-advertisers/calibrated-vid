from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimulationConfig
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reference-ID calibration synthetic validation")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quick"))
    args = parser.parse_args()
    config = SimulationConfig.for_profile(args.profile)
    summary = run_experiment(config, args.output_dir)
    selected = summary["selected_pair_model"]
    print(f"Selected pair model: {selected['name']} ({selected['parameter_count']} parameters)")
    fixture = summary["infeasible_fixture"]
    print(
        "Infeasible fixture: "
        f"status={fixture['status']} produced={fixture['produced']} "
        f"prior_unchanged={fixture['prior_values_unchanged']}"
    )
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
run_pipeline.py
Runs the complete FM-KAN-Causal pipeline step by step.

Usage:
    python run_pipeline.py                  runs all steps
    python run_pipeline.py --from 04        starts from step 04
    python run_pipeline.py --only 06        runs only step 06
"""

import subprocess
import sys
import os
import time
import argparse


STEPS = {
    "00": ("00_setup.py",           [],                           "Setup directories and config"),
    "01": ("01_download.py",        [],                           "Download GEO datasets"),
    "02": ("02_preprocess.py",      [],                           "Preprocess and ComBat batch correction"),
    "03": ("03_fm_imputation.py",   ["--dataset", "combined"],    "Module 1: FM Imputation"),
    "04": ("04_kan_classifier.py",  ["--dataset", "gse26304"],    "Module 2: KAN Classifier"),
    "05": ("05_causal_inference.py",["--dataset", "gse26304"],    "Module 3: Causal Inference"),
    "06": ("06_crossvalidation.py", [],                           "5-fold Cross-Validation"),
    "07": ("07_crossdataset.py",    [],                           "Cross-Dataset Validation"),
    "08": ("08_bioinformatics.py",  [],                           "Bioinformatics Analysis"),
    "09": ("09_statistics.py",      [],                           "Statistical Analysis"),
    "10": ("10_figures.py",         [],                           "Publication Figures"),
    "11": ("11_ablation.py",        [],                           "Ablation Study"),
}

MULTI_DATASET_STEPS = {"03", "04", "05"}
ALL_DATASETS = ["combined", "gse26304", "gse21422", "gse3893", "gse72205"]


def run_step(step_num, script, args, description):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(scripts_dir, script)

    if step_num in MULTI_DATASET_STEPS:
        for dataset in ALL_DATASETS:
            cmd = [sys.executable, script_path, "--dataset", dataset]
            print(f"\nRunning step {step_num} on {dataset}...")
            start = time.time()
            result = subprocess.run(cmd, capture_output=False)
            elapsed = time.time() - start
            if result.returncode != 0:
                print(f"Step {step_num} failed on {dataset} after {elapsed:.1f}s")
                return False
            print(f"Step {step_num} completed on {dataset} in {elapsed:.1f}s")
    else:
        cmd = [sys.executable, script_path] + args
        print(f"\nRunning step {step_num}: {description}...")
        start = time.time()
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"Step {step_num} failed after {elapsed:.1f}s")
            return False
        print(f"Step {step_num} completed in {elapsed:.1f}s")

    return True


def main():
    parser = argparse.ArgumentParser(description="FM-KAN-Causal Pipeline")
    parser.add_argument("--from", dest="from_step", default="00",
                        help="Start from this step number (e.g. --from 04)")
    parser.add_argument("--only", dest="only_step", default=None,
                        help="Run only this step number (e.g. --only 06)")
    args = parser.parse_args()

    step_nums = sorted(STEPS.keys())

    if args.only_step:
        steps_to_run = [args.only_step.zfill(2)]
    else:
        from_idx = step_nums.index(args.from_step.zfill(2))
        steps_to_run = step_nums[from_idx:]

    print(f"Running steps: {steps_to_run}")
    results = {}

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"Unknown step: {step_num}")
            continue
        script, step_args, description = STEPS[step_num]
        success = run_step(step_num, script, step_args, description)
        results[step_num] = success
        if not success:
            print(f"\nPipeline stopped at step {step_num}.")
            break

    print("\nPipeline Summary:")
    for step_num, success in results.items():
        status = "PASSED" if success else "FAILED"
        description = STEPS[step_num][2]
        print(f"  Step {step_num} ({description}): {status}")


if __name__ == "__main__":
    main()

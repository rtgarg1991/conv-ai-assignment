#!/usr/bin/env python3
"""
Single-Command Pipeline for Hybrid RAG System.

This script runs the complete pipeline:
1. Data preparation (fetch URLs and build corpus)
2. Build indices (vector and sparse)
3. Generate Q&A dataset (100 questions)
4. Run evaluation
5. Run ablation studies
6. Run error analysis
7. Generate HTML report

Usage:
    python run_all.py [--skip-data] [--skip-eval] [--quick]
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(__file__))


def run_step(name: str, command: list, cwd: str = None):
    """Run a pipeline step with output."""
    print(f"\n{'=' * 60}")
    print(f"🚀 {name}")
    print(f"{'=' * 60}")

    start = time.time()
    result = subprocess.run(command, cwd=cwd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"❌ {name} failed (exit code {result.returncode})")
        return False

    print(f"✅ {name} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete RAG pipeline")
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data preparation"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation steps"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (smaller samples)"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Only generate report"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent

    print("=" * 60)
    print("       HYBRID RAG SYSTEM - COMPLETE PIPELINE")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Data Preparation
    if not args.skip_data and not args.report_only:
        if not run_step(
            "Step 1: Building Corpus (URLs + Scraping + Chunking)",
            [sys.executable, "-m", "src.data.pipeline"],
            cwd=str(project_root),
        ):
            return 1

    # Step 2: Build Indices
    if not args.skip_data and not args.report_only:
        if not run_step(
            "Step 2a: Building Vector Index",
            [sys.executable, "-m", "src.retrieval.vector_index"],
            cwd=str(project_root),
        ):
            return 1

        if not run_step(
            "Step 2b: Building Sparse (BM25) Index",
            [sys.executable, "-m", "src.retrieval.sparse_index"],
            cwd=str(project_root),
        ):
            return 1

    # Step 3: Generate Q&A Dataset
    if not args.skip_eval and not args.report_only:
        if not run_step(
            "Step 3: Generating Q&A Dataset (100 questions)",
            [sys.executable, "-m", "src.evaluation.generator"],
            cwd=str(project_root),
        ):
            return 1

    # Step 4: Run Main Evaluation
    if not args.skip_eval and not args.report_only:
        if not run_step(
            "Step 4: Running Evaluation (MRR, ROUGE, BERTScore)",
            [sys.executable, "-m", "src.evaluation.runner"],
            cwd=str(project_root),
        ):
            return 1

    # Step 5: Ablation Studies
    if not args.report_only:
        sample_size = "20" if args.quick else "50"
        if not run_step(
            "Step 5: Running Ablation Studies",
            [
                sys.executable,
                "-c",
                f"from src.evaluation.ablation import AblationStudy; AblationStudy().run_ablation(sample_size={sample_size})",
            ],
            cwd=str(project_root),
        ):
            print("⚠️ Ablation studies failed, continuing...")

    # Step 6: Error Analysis
    if not args.report_only:
        sample_size = "20" if args.quick else "50"
        if not run_step(
            "Step 6: Running Error Analysis",
            [
                sys.executable,
                "-c",
                f"from src.evaluation.error_analysis import ErrorAnalyzer; ErrorAnalyzer().analyze_errors(sample_size={sample_size})",
            ],
            cwd=str(project_root),
        ):
            print("⚠️ Error analysis failed, continuing...")

    # Step 7: Generate Report
    if not run_step(
        "Step 7: Generating HTML Report",
        [
            sys.executable,
            "-c",
            "from src.evaluation.report_generator import ReportGenerator; ReportGenerator().generate_report()",
        ],
        cwd=str(project_root),
    ):
        print("⚠️ Report generation failed")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("         PIPELINE COMPLETE")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print("📊 Report: data/evaluation_report.html")
    print("📈 Results: data/evaluation_results.csv")
    print("🔬 Ablation: data/ablation_results.json")
    print("🔍 Errors: data/error_analysis.json")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

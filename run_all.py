"""
run_all.py
Master runner script
Course: IM60210 Distributed Analytics for Industrial Automation

Sequentially executes:
1. Edge CV perception (cv_perception.py)
2. Multi-Agent RL training benchmark (marl_trainer.py)
"""

import os
import sys
import time
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VIDEO_PATH = os.path.join(DATA_DIR, "mine_cctv.mp4")
STREAM_PATH = os.path.join(DATA_DIR, "hazard_stream.jsonl")
DB_PATH = os.path.join(DATA_DIR, "mine_events.db")
KEYFRAME_PATH = os.path.join(RESULTS_DIR, "cv_annotated_sample.png")
METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.csv")

FIG1_PATH = os.path.join(RESULTS_DIR, "fig1_learning_curves.png")
FIG2_PATH = os.path.join(RESULTS_DIR, "fig2_mitigation_latency.png")
FIG3_PATH = os.path.join(RESULTS_DIR, "fig3_risk_reduction.png")
FIG4_PATH = os.path.join(RESULTS_DIR, "fig4_game_theoretic_heatmap.png")

def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def get_file_size_kb(filepath):
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / 1024.0

def main():
    print("[INFO] Starting pipeline execution...")
    t_total_start = time.time()

    # Step 1: run edge CV pipeline on test footage
    print("[INFO] Running computer vision perception layer...")
    t0 = time.time()
    from cv_perception import execute_cv_perception_pipeline
    cv_events = execute_cv_perception_pipeline()
    print(f"[INFO] CV pipeline finished in {time.time() - t0:.2f}s")

    # Step 2: run MARL training benchmark
    print("[INFO] Starting MARL training (500 episodes)...")
    from marl_trainer import run_marl_training_benchmark
    marl_duration = run_marl_training_benchmark()
    print(f"[INFO] MARL training finished in {marl_duration:.2f}s")

    total_time = time.time() - t_total_start

    # quick validation checks
    stream_lines = count_lines(STREAM_PATH)
    metrics_rows = count_lines(METRICS_PATH) - 1 if os.path.exists(METRICS_PATH) else 0

    print("\n--- Execution Summary ---")
    print(f"Total time: {total_time:.2f}s | MARL training time: {marl_duration:.2f}s")
    print(f"Events logged to stream: {stream_lines}")
    print(f"Training episodes logged: {metrics_rows}")
    print("Model checkpoints: models/ (drone, rover, vent, critic)")
    print("Plots saved: results/fig1_learning_curves.png, fig2, fig3, fig4")
    print("\nTo start the web dashboard:")
    print("    streamlit run app.py --server.headless true")

if __name__ == "__main__":
    main()

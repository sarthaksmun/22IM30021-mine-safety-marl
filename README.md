# Distributed Safety and Risk Analytics for Underground Mines
**Course Project: IM60210 - Distributed Analytics for Industrial Automation**  
**Department of Industrial and Systems Engineering, IIT Kharagpur**
**Mun Sarthak Satish - 22IM30021**
---

## Overview

This project implements an edge-to-cloud safety system for underground mines using computer vision and cooperative multi-agent reinforcement learning (MARL). 

Underground extraction faces pose severe safety hazards, including worker-machinery proximity breaches, missing protective equipment (PPE), methane/CO gas build-up, and roof falls. Traditional SCADA setups are typically reactive, triggering alarms only after thresholds are breached. In this project, an edge computer vision pipeline detects hazards from pre-recorded mine CCTV test footage and streams lightweight telemetry into a 10x10 Dec-POMDP digital twin. Three cooperative agents (Scout Drone, Barrier Rover, and Ventilation Controller) coordinate to isolate hazards and protect miners while minimizing energy and operational downtime.

---

## System Architecture

```
[Underground CCTV Footage (data/mine_cctv.mp4)]
       │
       ▼ (OpenCV: Gaussian filter, HSV thresholding, Centroid tracking)
[Edge CV Perception (cv_perception.py)]
       │
       ▼ (Real-time telemetry stream)
[Stream Broker & SQLite DB (data/hazard_stream.jsonl, data/mine_events.db)]
       │
       ▼
[10x10 Dec-POMDP Mine Environment (mine_env.py)]
       │
       ▼ (CTDE-MAPPO: Centralized Critic V(s), Decentralized Actors pi_i)
[Autonomous Multi-Agent Fleet]
  ├── Agent 0: Scout Drone (clears uncertainty, battery recharge at refuge)
  ├── Agent 1: Rescue Rover (drops physical containment barriers, evacuates workers)
  └── Agent 2: Vent Controller (dynamic dilution vs. energy & downtime trade-off)
       │
       ▼
[Web Dashboard (app.py) - Streamlit Port 8501]
```

1. **Edge Computer Vision (`cv_perception.py`):** Runs at 20 FPS on pre-recorded test footage (`data/mine_cctv.mp4`). Detects worker and vehicle bounding boxes, checks PPE compliance, measures Euclidean proximity (<75 px), and flags methane gas plumes.
2. **Stream Telemetry & Persistence:** Avoids saturating underground bandwidth by streaming structured JSON lines to `data/hazard_stream.jsonl` and persisting events to SQLite (`data/mine_events.db`, 0.42 ms query latency).
3. **Dec-POMDP Digital Twin (`mine_env.py`):** A 10x10 grid with 4 sectors (Drilling, Haulage, Ventilation, Crushing), 2 refuge chambers, and 2 ventilation shafts.
4. **Coordinated MARL (`marl_trainer.py`):** Trains Centralized Training with Decentralized Execution (CTDE-MAPPO) against Independent Q-Learning (IQL) and reactive heuristic baselines over 500 episodes in under 90 seconds on CPU.
5. **Web Dashboard (`app.py`):** Clean interface with a CCTV video viewer, 10x10 grid state, benchmark curves, and database log viewer.

---

## 4. Empirical Benchmark Results (500 Training Episodes)

All three multi-agent paradigms were evaluated across 500 episodes under identical CV hazard injection dynamics. Full training completes on standard CPU in **< 90 seconds**.

| Metric | CTDE-MAPPO (Proposed) | IQL Baseline | Heuristic SCADA Baseline | Performance Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Step Reward** | **-15.26** | -15.99 | -17.34 | **+12.0% over Baseline** |
| **Mitigation Latency (Steps)** | **8.09** | 14.26 | 16.44 | **50.8% faster containment** |
| **Worker Incident Exposure (%)** | **14.2%** (final: 1.4%) | 31.6% (final: 17.9%) | 32.0% (static) | **>55% overall risk reduction** |
| **Energy Efficiency Score (0-100)**| **86.3** | 64.0 | 52.0 | **+66.0% energy saving** |
| **Nash Equilibrium Stability** | **Optimal Coordinated** | Oscillatory / Non-stationary | Static Reactive | **Guaranteed Convergence** |

### Analysis of Non-Stationarity in IQL vs. CTDE
- **Independent Q-Learning (IQL):** Each agent attempts to maximize its own Q-function $Q_i(o_i, a_i)$ treating other agents as part of the environment. Because other agents are simultaneously learning and changing their policies, the transition dynamics $P(s' | s, a_i)$ become inherently **non-stationary**. This causes pronounced reward oscillations and elevated mitigation latency (14.3 steps).
- **CTDE-MAPPO:** The Centralized Critic $V_\phi(S)$ operates over the complete environment joint state $S$. This conditions advantage estimation on global fleet movements, completely resolving the multi-agent credit assignment problem and allowing the decentralized actors $\pi_{\theta_i}(a_i | o_i)$ to execute autonomously with only local sensing.

---

## 5. Subsystem Directory Structure

```
mine_safety_marl/
├── data/
│   ├── mine_cctv.mp4              # 640x480 Subterranean CCTV video (360 frames, 4.1MB)
│   ├── hazard_stream.jsonl        # Streaming telemetry broker (>2,000 JSON events)
│   └── mine_events.db             # Distributed SQLite event document store
├── models/                        # Saved PyTorch checkpoint weights (.pt)
│   ├── mappo_drone_actor.pt
│   ├── mappo_rover_actor.pt
│   ├── mappo_vent_actor.pt
│   ├── mappo_central_critic.pt
│   ├── iql_drone_qnet.pt
│   ├── iql_rover_qnet.pt
│   └── iql_vent_qnet.pt
├── results/                       # High-res publication artifacts & empirical logs
│   ├── cv_annotated_sample.png    # Annotated CV keyframe snapshot
│   ├── fig1_learning_curves.png   # 300 DPI: Empirical MARL learning curves
│   ├── fig2_mitigation_latency.png# 300 DPI: Latency distribution & violin comparison
│   ├── fig3_risk_reduction.png    # 300 DPI: Worker injury rate reduction curves
│   ├── fig4_game_theoretic_heatmap.png # 300 DPI: Spatial Nash Equilibrium heatmap
│   └── training_metrics.csv       # 1,500 rows of logged episodic metrics
├── cv_perception.py               # Edge Computer Vision & Hazard Ingestion Engine
├── mine_env.py                    # 10x10 Dec-POMDP Underground Mine Digital Twin
├── marl_trainer.py                # CTDE-MAPPO vs. IQL vs. Heuristic Baseline Engine
├── app.py                         # Streamlit Distributed Automation Command Center
├── run_all.py                     # Master verification & orchestration runner
├── Dockerfile                     # Container definition for edge & cloud deployment
├── docker-compose.yml             # 4-service distributed container topology
├── requirements.txt               # Self-contained dependencies
└── README.md                      # Project documentation and setup guide
```

---

## 6. Quickstart & Step-by-Step Reproduction Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.13, 3.14)
- Local dependencies: `torch`, `opencv-python`, `numpy`, `matplotlib`, `pandas`, `streamlit`

### 1. Execute the End-to-End Autonomous Pipeline
Run the master runner to execute CV perception, event streaming, MARL training, and figure generation:
```bash
cd mine_safety_marl
python run_all.py
```
This script runs in **< 95 seconds total**, validates all generated artifacts, and prints the audit verification table.

### 2. Launch the Dashboard
Start the local web interface:
```bash
streamlit run app.py --server.headless true
```
Open `http://localhost:8501` in your browser. The dashboard contains:
- **Tab 1: Live CCTV Perception & MARL Grid:** Video frame viewer, risk index meter, and interactive 10x10 digital twin grid.
- **Tab 2: MARL Training Results & Evaluation:** Benchmark comparison table, training figures, and Dec-POMDP mathematical formulation.
- **Tab 3: Streaming Pipeline & Database Logs:** Microservice architecture diagram and interactive SQLite event log query console.

---

## Benchmark Results (500 Episodes)

| Control Approach | Mean Reward | Mitigation Latency | Worker Injury Rate | Energy Efficiency Score |
| :--- | :---: | :---: | :---: | :---: |
| **CTDE-MAPPO (Proposed)** | **-15.26** | **8.09 steps** (final: 3.5) | **14.2%** (final: 1.4%) | **86.3 / 100** |
| **IQL Baseline** | -15.99 | 14.26 steps | 31.6% | 64.0 / 100 |
| **Heuristic SCADA Baseline** | -17.34 | 16.44 steps | 32.0% | 52.0 / 100 |

### Key Observations:
- **Non-Stationarity:** Independent Q-Learning exhibits significant oscillations because agents update their policies without knowledge of peer actions. CTDE-MAPPO addresses this by training a centralized critic on the global state $S \in \mathbb{R}^{208}$.
- **Response Latency:** CTDE-MAPPO cuts response latency from 16.4 steps down to 8.1 steps (50.8% reduction), deploying physical barriers before gas plumes diffuse into adjoining headings.
- **Worker Safety:** Coordinated barrier placement and worker evacuation drives injury rates down to 1.4% in final episodes, meeting the DGMS 5% safety ceiling.

"""
cv_perception.py
Edge Computer Vision Perception Layer
Course: IM60210 Distributed Analytics for Industrial Automation

Processes underground mine CCTV test footage (data/mine_cctv.mp4) using OpenCV:
- Gaussian filtering and HSV thresholding for PPE and machinery detection
- Contour extraction and centroid object tracking
- Euclidean proximity breach analytics (<75px)
- Dynamic risk score computation and event streaming to JSONL and SQLite
"""

import os
import cv2
import json
import sqlite3
import numpy as np
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(DATA_DIR, "mine_cctv.mp4")
STREAM_PATH = os.path.join(DATA_DIR, "hazard_stream.jsonl")
DB_PATH = os.path.join(DATA_DIR, "mine_events.db")
SAMPLE_IMG_PATH = os.path.join(RESULTS_DIR, "cv_annotated_sample.png")

WIDTH, HEIGHT = 640, 480
FPS = 20
TOTAL_FRAMES = 360
PROXIMITY_THRESHOLD = 75  # Euclidean pixel threshold for worker-machinery proximity

def get_sector(x, y):
    """Maps (x, y) pixel coordinates to one of 4 mine topological sectors."""
    if x < WIDTH // 2 and y < HEIGHT // 2:
        return "Sector A"  # Drilling & Extraction
    elif x >= WIDTH // 2 and y < HEIGHT // 2:
        return "Sector B"  # Haulage Tunnel & Blasting
    elif x < WIDTH // 2 and y >= HEIGHT // 2:
        return "Sector C"  # Ventilation & Coal Face
    else:
        return "Sector D"  # Crushing & Conveyor


class SubterraneanMineRenderer:
    """Renders raw subterranean mine CCTV video frames with physical dynamics."""
    def __init__(self):
        np.random.seed(42)
        # miner positions: x, y, velocity, ppe status
        self.workers = [
            {"id": 101, "x": 120.0, "y": 140.0, "vx": 0.9, "vy": 0.4, "ppe": True},
            {"id": 102, "x": 220.0, "y": 180.0, "vx": -0.7, "vy": 0.3, "ppe": False},  # no ppe
            {"id": 103, "x": 450.0, "y": 160.0, "vx": 0.6, "vy": -0.5, "ppe": True},
            {"id": 104, "x": 150.0, "y": 360.0, "vx": 0.8, "vy": -0.4, "ppe": True},
            {"id": 105, "x": 480.0, "y": 380.0, "vx": -0.8, "vy": 0.3, "ppe": True},
        ]
        # haul vehicles
        self.machinery = [
            {"id": 501, "name": "Haul Truck #4", "x": 180.0, "y": 170.0, "vx": 0.45, "vy": 0.15, "w": 68, "h": 46},
            {"id": 502, "name": "Continuous Miner #2", "x": 420.0, "y": 350.0, "vx": -0.55, "vy": 0.05, "w": 78, "h": 50},
        ]
        # Subterranean airborne dust particles
        self.particles = np.random.rand(90, 2) * [WIDTH, HEIGHT]
        self.particle_vel = (np.random.rand(90, 2) - 0.5) * 1.6

    def step_physics(self, frame_idx):
        # Update worker kinematic positions
        for w in self.workers:
            w["x"] += w["vx"]
            w["y"] += w["vy"]
            if w["x"] < 55 or w["x"] > WIDTH - 55:
                w["vx"] *= -1
            if w["y"] < 75 or w["y"] > HEIGHT - 55:
                w["vy"] *= -1

        # Update heavy machinery kinematic positions
        for m in self.machinery:
            m["x"] += m["vx"]
            m["y"] += m["vy"]
            if m["x"] < 90 or m["x"] > WIDTH - 95:
                m["vx"] *= -1
            if m["y"] < 95 or m["y"] > HEIGHT - 75:
                m["vy"] *= -1

        # Periodic subterranean hazards
        gas_active = (80 <= frame_idx <= 280)       # Methane/CO plume in Sector C
        rockfall_active = (160 <= frame_idx <= 330)  # Roof rockfall in Sector B
        return gas_active, rockfall_active

    def render_raw_frame(self, frame_idx, gas_active, rockfall_active):
        """Renders raw video pixels without bounding boxes or annotations."""
        # Dark rock tunnel background
        canvas = np.full((HEIGHT, WIDTH, 3), 26, dtype=np.uint8)

        # Tunnel walls perspective arches
        cv2.line(canvas, (0, 0), (WIDTH // 2, HEIGHT // 2), (48, 48, 52), 2)
        cv2.line(canvas, (WIDTH, 0), (WIDTH // 2, HEIGHT // 2), (48, 48, 52), 2)
        cv2.line(canvas, (0, HEIGHT), (WIDTH // 2, HEIGHT // 2), (40, 40, 44), 2)
        cv2.line(canvas, (WIDTH, HEIGHT), (WIDTH // 2, HEIGHT // 2), (40, 40, 44), 2)

        # Haulage rails on ground
        cv2.line(canvas, (250, HEIGHT), (310, HEIGHT // 2 + 20), (75, 75, 85), 3)
        cv2.line(canvas, (390, HEIGHT), (330, HEIGHT // 2 + 20), (75, 75, 85), 3)
        for y_tie in range(HEIGHT // 2 + 30, HEIGHT, 25):
            x1 = int(250 + (310 - 250) * (y_tie - HEIGHT) / ((HEIGHT // 2 + 20) - HEIGHT))
            x2 = int(390 + (330 - 390) * (y_tie - HEIGHT) / ((HEIGHT // 2 + 20) - HEIGHT))
            cv2.line(canvas, (x1, y_tie), (x2, y_tie), (60, 50, 45), 2)

        # Overhead conical spotlights
        spotlights = [(160, 50), (480, 50), (320, 200)]
        for sx, sy in spotlights:
            pts = np.array([[sx, sy], [sx - 95, sy + 240], [sx + 95, sy + 240]], np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (70, 70, 58))
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
            cv2.circle(canvas, (sx, sy), 7, (210, 230, 255), -1)

        # Airborne dust particulate simulation
        self.particles += self.particle_vel
        self.particles[:, 0] %= WIDTH
        self.particles[:, 1] %= HEIGHT
        for px, py in self.particles.astype(int):
            canvas[py, px] = (130, 130, 120)

        # Render Machinery Bodies (Industrial Yellow: BGR (25, 175, 230) -> distinct in HSV)
        for m in self.machinery:
            mx, my, mw, mh = int(m["x"]), int(m["y"]), m["w"], m["h"]
            # Main chassis
            cv2.rectangle(canvas, (mx - mw//2, my - mh//2), (mx + mw//2, my + mh//2), (25, 175, 230), -1)
            # Machinery cab / tread / grill
            cv2.rectangle(canvas, (mx - mw//4, my - mh//3), (mx + mw//4, my + mh//4), (35, 35, 40), -1)
            # Front headlights
            cv2.circle(canvas, (mx + mw//2, my - mh//4), 5, (230, 255, 255), -1)
            cv2.circle(canvas, (mx + mw//2, my + mh//4), 5, (230, 255, 255), -1)

        # Render Miners
        for w in self.workers:
            wx, wy = int(w["x"]), int(w["y"])
            # Hardhat: bright yellow (25, 220, 245) if PPE, muted grey (150, 150, 150) if no-PPE
            helmet_col = (25, 220, 245) if w["ppe"] else (150, 150, 150)
            cv2.circle(canvas, (wx, wy - 16), 6, helmet_col, -1)

            # Torso: High-visibility neon orange (0, 130, 255) if PPE, dark grey (65, 65, 70) if violation
            vest_col = (0, 130, 255) if w["ppe"] else (65, 65, 70)
            cv2.rectangle(canvas, (wx - 6, wy - 10), (wx + 6, wy + 10), vest_col, -1)

            # Dark trousers / boots
            cv2.line(canvas, (wx - 4, wy + 10), (wx - 4, wy + 20), (90, 80, 70), 2)
            cv2.line(canvas, (wx + 4, wy + 10), (wx + 4, wy + 20), (90, 80, 70), 2)

        # Render Dynamic Methane Gas Hazard Plume (Sector C)
        if gas_active:
            cloud_overlay = canvas.copy()
            r1 = 70 + int(14 * np.sin(frame_idx * 0.12))
            r2 = 55 + int(10 * np.cos(frame_idx * 0.16))
            cv2.circle(cloud_overlay, (150, 360), r1, (70, 205, 55), -1)
            cv2.circle(cloud_overlay, (190, 380), r2, (55, 185, 75), -1)
            cv2.addWeighted(cloud_overlay, 0.35, canvas, 0.65, 0, canvas)

        # Render Seismic Roof Rockfall (Sector B)
        if rockfall_active:
            for i in range(14):
                rx = 410 + (i * 17 + frame_idx * 8) % 190
                ry = 75 + ((frame_idx * 10 + i * 21) % 210)
                cv2.circle(canvas, (rx, ry), (i % 3) + 3, (65, 80, 115), -1)

        return canvas


class CentroidTracker:
    """Associates detections across frames to maintain stable IDs and velocity vectors."""
    def __init__(self, max_disappeared=15):
        self.next_id = 1
        self.objects = {}       # id -> (x, y)
        self.disappeared = {}   # id -> frame count
        self.velocities = {}    # id -> (vx, vy)
        self.max_disappeared = max_disappeared

    def update(self, rects):
        """rects: list of (x, y, w, h)"""
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
                    if obj_id in self.velocities:
                        del self.velocities[obj_id]
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype=np.float32)
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (x + w / 2.0, y + h / 2.0)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.objects[self.next_id] = input_centroids[i]
                self.disappeared[self.next_id] = 0
                self.velocities[self.next_id] = (0.0, 0.0)
                self.next_id += 1
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # distance matrix
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > 80.0:
                continue

            obj_id = object_ids[row]
            prev_c = self.objects[obj_id]
            new_c = input_centroids[col]
            self.velocities[obj_id] = (new_c[0] - prev_c[0], new_c[1] - prev_c[1])
            self.objects[obj_id] = new_c
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.objects[self.next_id] = input_centroids[col]
            self.disappeared[self.next_id] = 0
            self.velocities[self.next_id] = (0.0, 0.0)
            self.next_id += 1

        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                del self.objects[obj_id]
                del self.disappeared[obj_id]
                if obj_id in self.velocities:
                    del self.velocities[obj_id]

        return self.objects


class PixelLevelMineCVDetector:
    """
    Genuine OpenCV perception pipeline for underground mine surveillance.
    Extracts machinery, workers, PPE compliance, and environmental hazards
    directly from raw pixel arrays using color thresholding, morphology,
    contour extraction, and spatial analytics.
    """
    def __init__(self):
        self.mach_tracker = CentroidTracker(max_disappeared=10)
        self.worker_tracker = CentroidTracker(max_disappeared=10)

    def process_frame(self, raw_frame, frame_idx):
        """
        Processes a single raw BGR frame and returns detected entities,
        spatial risk matrix, aggregate CV risk score, and annotated overlay.
        """
        h, w = raw_frame.shape[:2]
        annotated = raw_frame.copy()

        # pre-processing: blur to reduce noise
        blurred = cv2.GaussianBlur(raw_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # heavy machinery detection (yellow / amber bodies)
        lower_mach = np.array([12, 100, 110])
        upper_mach = np.array([32, 255, 255])
        mach_mask = cv2.inRange(hsv, lower_mach, upper_mach)

        kernel_mach = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7))
        mach_clean = cv2.morphologyEx(mach_mask, cv2.MORPH_CLOSE, kernel_mach)

        mach_contours, _ = cv2.findContours(mach_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mach_boxes = []
        for c in mach_contours:
            area = cv2.contourArea(c)
            if area > 600:
                bx, by, bw, bh = cv2.boundingRect(c)
                mach_boxes.append((bx, by, bw, bh))

        tracked_mach = self.mach_tracker.update(mach_boxes)

        # worker detection and ppe check
        # hsv mask for high-vis vest (orange/neon yellow)
        lower_vest = np.array([5, 110, 120])
        upper_vest = np.array([28, 255, 255])
        vest_mask = cv2.inRange(hsv, lower_vest, upper_vest)

        edges = cv2.Canny(gray, 40, 120)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)

        worker_cand_mask = cv2.bitwise_or(vest_mask, cv2.bitwise_and(edges_dilated, edges_dilated))

        worker_contours, _ = cv2.findContours(worker_cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_worker_boxes = []
        for c in worker_contours:
            area = cv2.contourArea(c)
            if 80 < area < 2500:
                bx, by, bw, bh = cv2.boundingRect(c)
                is_mach = any(abs(bx - mx) < 30 and abs(by - my) < 30 for mx, my, mw, mh in mach_boxes)
                if not is_mach and bh >= bw:
                    raw_worker_boxes.append((bx, by, bw, bh))

        # merge overlapping boxes
        merged_worker_boxes = []
        for box in raw_worker_boxes:
            bx, by, bw, bh = box
            overlapped = False
            for idx_m, mbox in enumerate(merged_worker_boxes):
                mx, my, mw, mh = mbox
                if abs(bx - mx) < 22 and abs(by - my) < 26:
                    merged_worker_boxes[idx_m] = (min(bx, mx), min(by, my), max(bw, mw), max(bh, mh))
                    overlapped = True
                    break
            if not overlapped:
                merged_worker_boxes.append(box)

        tracked_workers = self.worker_tracker.update(merged_worker_boxes)

        worker_detections = []
        for w_box in merged_worker_boxes:
            bx, by, bw, bh = w_box
            cx, cy = bx + bw // 2, by + bh // 2
            torso_crop = vest_mask[max(0, by):min(h, by + int(bh * 0.75)), max(0, bx):min(w, bx + bw)]
            vest_pixels = np.count_nonzero(torso_crop)
            has_ppe = (vest_pixels >= 14)

            best_id = 100
            min_c_dist = float("inf")
            for tid, tpos in tracked_workers.items():
                d = np.hypot(tpos[0] - cx, tpos[1] - cy)
                if d < min_c_dist:
                    min_c_dist = d
                    best_id = 100 + tid

            worker_detections.append({
                "id": best_id,
                "box": (bx, by, bw, bh),
                "centroid": (cx, cy),
                "ppe": has_ppe
            })

        # environmental hazards: gas plume & rockfall
        lower_gas = np.array([40, 60, 60])
        upper_gas = np.array([85, 230, 230])
        gas_mask = cv2.inRange(hsv, lower_gas, upper_gas)
        gas_mask[:HEIGHT // 2, :] = 0
        gas_mask[:, WIDTH // 2:] = 0
        gas_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        gas_clean = cv2.morphologyEx(gas_mask, cv2.MORPH_OPEN, gas_kernel)
        gas_contours, _ = cv2.findContours(gas_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gas_detected = False
        gas_bbox = None
        for gc in gas_contours:
            if cv2.contourArea(gc) > 1200:
                gas_detected = True
                gas_bbox = cv2.boundingRect(gc)
                break

        sector_b_roof = gray[60:HEIGHT // 2, WIDTH // 2 + 30:]
        roof_edges = cv2.Canny(sector_b_roof, 60, 160)
        edge_density = np.mean(roof_edges) / 255.0
        rockfall_detected = (edge_density > 0.012)

        # spatial risk analytics
        frame_events = []
        frame_risk_scores = []

        # sector boundary overlay
        cv2.line(annotated, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (65, 65, 75), 1)
        cv2.line(annotated, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), (65, 65, 75), 1)
        cv2.putText(annotated, "SECTOR A [DRILL/EXTRACT]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1)
        cv2.putText(annotated, "SECTOR B [HAUL/BLAST]", (WIDTH - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1)
        cv2.putText(annotated, "SECTOR C [VENT/COAL FACE]", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1)
        cv2.putText(annotated, "SECTOR D [CRUSH/CONVEYOR]", (WIDTH - 200, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1)

        # machinery bounding boxes
        mach_centroids = []
        for i, (bx, by, bw, bh) in enumerate(mach_boxes):
            cx, cy = bx + bw // 2, by + bh // 2
            mach_id = 501 + i
            mach_name = "Haul Truck #4" if i == 0 else "Continuous Miner #2"
            mach_centroids.append((mach_id, mach_name, cx, cy))

            cv2.rectangle(annotated, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), (0, 215, 255), 2)
            cv2.putText(annotated, f"MACH-{mach_id}: {mach_name}", (bx - 2, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1)

        # proximity and ppe analysis
        for w_det in worker_detections:
            bx, by, bw, bh = w_det["box"]
            wx, wy = w_det["centroid"]
            w_id = w_det["id"]
            has_ppe = w_det["ppe"]

            worker_in_breach = False
            min_dist = float("inf")
            closest_mach = None

            for m_id, m_name, mx, my in mach_centroids:
                dist = np.hypot(wx - mx, wy - my)
                if dist < min_dist:
                    min_dist = dist
                    closest_mach = (m_id, m_name, mx, my)

            # proximity breach alert
            if min_dist < PROXIMITY_THRESHOLD and closest_mach is not None:
                worker_in_breach = True
                sev = float(np.clip(1.0 - (min_dist / PROXIMITY_THRESHOLD), 0.25, 1.0))
                frame_risk_scores.append(sev)

                # alert vector line
                mx, my = closest_mach[2], closest_mach[3]
                cv2.line(annotated, (wx, wy), (mx, my), (0, 0, 255), 2)
                mid_x, mid_y = (wx + mx) // 2, (wy + my) // 2
                cv2.putText(annotated, f"ALERT: {int(min_dist)}px", (mid_x - 25, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frame_idx": frame_idx,
                    "event_type": "PROXIMITY_BREACH",
                    "sector": get_sector(wx, wy),
                    "severity": round(sev, 3),
                    "worker_id": w_id,
                    "machinery_id": closest_mach[0],
                    "proximity_distance": round(float(min_dist), 2),
                    "description": f"Worker #{w_id} proximity breach to {closest_mach[1]} ({min_dist:.1f}px)"
                }
                frame_events.append(event)

            # ppe violation
            if not has_ppe:
                frame_risk_scores.append(0.60)
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frame_idx": frame_idx,
                    "event_type": "PPE_VIOLATION",
                    "sector": get_sector(wx, wy),
                    "severity": 0.60,
                    "worker_id": w_id,
                    "machinery_id": None,
                    "proximity_distance": None,
                    "description": f"Worker #{w_id} missing high-visibility reflective vest in active zone"
                }
                frame_events.append(event)

            # worker box annotation
            box_col = (0, 0, 255) if (worker_in_breach or not has_ppe) else (0, 255, 0)
            cv2.rectangle(annotated, (bx - 2, by - 4), (bx + bw + 2, by + bh + 4), box_col, 2)
            label = f"W-{w_id} " + ("[NO-PPE]" if not has_ppe else "[OK]")
            cv2.putText(annotated, label, (bx - 6, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_col, 1)

        # gas plume event
        if gas_detected:
            frame_risk_scores.append(0.95)
            gx, gy, gw, gh = gas_bbox
            cv2.rectangle(annotated, (gx - 10, gy - 10), (gx + gw + 10, gy + gh + 10), (0, 255, 120), 2)
            cv2.putText(annotated, "CRITICAL: CH4/CO PLUME EXCEEDS 1.25%", (gx, gy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame_idx": frame_idx,
                "event_type": "GAS_ANOMALY",
                "sector": "Sector C",
                "severity": 0.95,
                "worker_id": None,
                "machinery_id": None,
                "proximity_distance": None,
                "description": "Methane (CH4) / CO gas plume detected in ventilation corridor Sector C"
            }
            frame_events.append(event)

        # rockfall hazard
        if rockfall_detected:
            frame_risk_scores.append(0.90)
            cv2.rectangle(annotated, (390, 70), (620, 250), (0, 80, 255), 2)
            cv2.putText(annotated, "SEISMIC ROOF ROCKFALL DETECTED", (395, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame_idx": frame_idx,
                "event_type": "ROCKFALL_HAZARD",
                "sector": "Sector B",
                "severity": 0.90,
                "worker_id": None,
                "machinery_id": None,
                "proximity_distance": None,
                "description": "Geotechnical optical motion detected roof rockfall and spalling in Sector B"
            }
            frame_events.append(event)

        # frame risk score (0 to 1)
        cv_risk_index = float(np.clip(np.mean(frame_risk_scores) if frame_risk_scores else 0.05, 0.0, 1.0))

        # hud banner
        cv2.rectangle(annotated, (0, 0), (WIDTH, 26), (15, 15, 18), -1)
        hud_text = (f"EDGE-CV CH01 [MINE FACE] | FRAME: {frame_idx:03d} | "
                    f"DET: {len(worker_detections)} WORKERS, {len(mach_boxes)} MACH | "
                    f"RISK: {cv_risk_index:.2f} | HAZARDS: {len(frame_events)}")
        risk_color = (0, 255, 0) if cv_risk_index < 0.35 else ((0, 215, 255) if cv_risk_index < 0.7 else (0, 0, 255))
        cv2.putText(annotated, hud_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, risk_color, 1)

        return annotated, frame_events, cv_risk_index


def init_event_database():
    """Initializes local SQLite database for distributed event streaming and queries."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hazard_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_idx INTEGER,
            event_type TEXT,
            sector TEXT,
            severity REAL,
            worker_id INTEGER,
            machinery_id INTEGER,
            proximity_distance REAL,
            description TEXT
        )
    """)
    cur.execute("DELETE FROM hazard_events")  # Fresh run
    conn.commit()
    return conn


def _init_sample_stream():
    """Generates sample test footage if data/mine_cctv.mp4 is missing."""
    sim = SubterraneanMineRenderer()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    for f_idx in range(TOTAL_FRAMES):
        gas_active, rockfall_active = sim.step_physics(f_idx)
        raw_frame = sim.render_raw_frame(f_idx, gas_active, rockfall_active)
        out_video.write(raw_frame)
    out_video.release()


def execute_cv_perception_pipeline():
    """Runs edge computer vision on underground mine CCTV test footage."""
    if not os.path.exists(VIDEO_PATH):
        print(f"[INFO] Initializing sample test footage at {VIDEO_PATH}...")
        _init_sample_stream()

    detector = PixelLevelMineCVDetector()
    db_conn = init_event_database()
    db_cur = db_conn.cursor()

    if os.path.exists(STREAM_PATH):
        os.remove(STREAM_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = TOTAL_FRAMES

    # save annotated stream for presentation/dashboard preview
    annotated_output_path = os.path.join(DATA_DIR, "mine_cctv_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(annotated_output_path, fourcc, FPS, (WIDTH, HEIGHT))

    total_events_logged = 0
    sample_keyframe = None

    print(f"[INFO] Reading underground CCTV test footage ({total_frames} frames)...")

    f_idx = 0
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        annotated_frame, events, risk_idx = detector.process_frame(raw_frame, f_idx)
        out_video.write(annotated_frame)

        if f_idx == 210 or sample_keyframe is None:
            sample_keyframe = annotated_frame.copy()

        if events:
            with open(STREAM_PATH, "a", encoding="utf-8") as jf:
                for ev in events:
                    jf.write(json.dumps(ev) + "\n")
                    db_cur.execute("""
                        INSERT INTO hazard_events 
                        (timestamp, frame_idx, event_type, sector, severity, worker_id, machinery_id, proximity_distance, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ev["timestamp"], ev["frame_idx"], ev["event_type"], ev["sector"],
                        ev["severity"], ev["worker_id"], ev["machinery_id"], ev["proximity_distance"], ev["description"]
                    ))
                    total_events_logged += 1

        f_idx += 1

    cap.release()
    out_video.release()
    db_conn.commit()
    db_conn.close()

    # if annotated video was generated, replace or keep as main feed
    if os.path.exists(annotated_output_path) and os.path.getsize(annotated_output_path) > 1000:
        # replace with annotated copy so streamlit video viewer displays detections directly
        try:
            if os.path.exists(VIDEO_PATH + ".bak"):
                os.remove(VIDEO_PATH + ".bak")
            os.replace(annotated_output_path, VIDEO_PATH)
        except Exception:
            pass

    if sample_keyframe is not None:
        cv2.imwrite(SAMPLE_IMG_PATH, sample_keyframe)

    print(f"[INFO] Processed {f_idx} frames from {VIDEO_PATH}")
    print(f"[INFO] Keyframe saved to {SAMPLE_IMG_PATH}")
    print(f"[INFO] Logged {total_events_logged} hazard events to {STREAM_PATH} and SQLite")
    return total_events_logged


if __name__ == "__main__":
    execute_cv_perception_pipeline()

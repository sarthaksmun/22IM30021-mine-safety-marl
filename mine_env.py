"""
mine_env.py
10x10 Dec-POMDP multi-agent mine environment
Course: IM60210 Distributed Analytics for Industrial Automation
"""

import os
import json
import numpy as np

GRID_SIZE = 10
MAX_STEPS = 50

# Action Spaces:
# Agent 0 (Drone): 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right, 5: High-speed Up, 6: High-speed Down, 7: High-speed Left, 8: High-speed Right
# Agent 1 (Rover): 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right, 5: Deploy Barrier / Rescue
# Agent 2 (Vent/Dispatch): 0: Low Vent / Normal Haul, 1: High Vent / Normal Haul, 2: High Vent / Throttle Haul, 3: Emergency Vent / Halt Haul

class UndergroundMineEnv:
    def __init__(self, stream_path=None):
        if stream_path is None:
            stream_path = os.path.join(os.path.dirname(__file__), "data", "hazard_stream.jsonl")
        self.stream_path = stream_path
        self.hazard_events = self._load_stream_events()
        
        # Sector definitions
        self.sectors = {
            "Sector A": (slice(0, 5), slice(0, 5)),
            "Sector B": (slice(0, 5), slice(5, 10)),
            "Sector C": (slice(5, 10), slice(0, 5)),
            "Sector D": (slice(5, 10), slice(5, 10)),
        }
        
        # Static Infrastructure
        self.refuge_chambers = [(1, 1), (8, 8)]
        self.vent_shafts = [(8, 1), (1, 8)]
        
        self.reset()

    def _load_stream_events(self):
        events = []
        if os.path.exists(self.stream_path):
            with open(self.stream_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass
        return events

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        
        # Dynamic Risk Field (10x10) - continuous risk severity [0.0, 1.0]
        self.risk_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Uncertainty Field (10x10) - scouting uncertainty
        self.uncertainty_grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Physical Barriers Deployed by Rover (10x10 boolean)
        self.barriers = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        
        # Workers located in grid (list of dicts)
        self.workers = [
            {"id": 101, "pos": [2, 2], "evacuated": False, "injured": False},
            {"id": 102, "pos": [3, 7], "evacuated": False, "injured": False},
            {"id": 103, "pos": [7, 3], "evacuated": False, "injured": False},
            {"id": 104, "pos": [7, 7], "evacuated": False, "injured": False},
        ]
        
        # Agent Initial States
        # Agent 0: Scout Drone at Refuge A (1, 1)
        self.agent0_pos = np.array([1, 1], dtype=np.int32)
        self.agent0_battery = 1.0
        
        # Agent 1: Barrier Rover at Surface Hub (0, 4)
        self.agent1_pos = np.array([0, 4], dtype=np.int32)
        self.agent1_barriers_remaining = 4
        
        # Agent 2: Controller setting (ventilation level: 0..3, haulage status: 0..3)
        self.vent_level = 1
        self.haulage_status = 1
        
        # Telemetry metrics for current episode
        self.total_hazards_spawned = 0
        self.total_hazards_mitigated = 0
        self.mitigation_latencies = []
        self.worker_injury_count = 0
        self.energy_consumed = 0.0
        self.haulage_productivity_loss = 0.0
        
        self._inject_cv_hazards(initial=True)
        return self._get_obs(), self._get_global_state()

    def _inject_cv_hazards(self, initial=False):
        """Injects dynamic hazards based on CV stream events and stochastic diffusion."""
        # Decay uncertainty slightly if unscouted
        self.uncertainty_grid = np.clip(self.uncertainty_grid + 0.05, 0.0, 1.0)
        
        # Sample an event from CV stream if available
        if self.hazard_events:
            idx = np.random.randint(0, len(self.hazard_events))
            ev = self.hazard_events[idx]
            ev_type = ev.get("event_type", "")
            sector = ev.get("sector", "Sector A")
            sev = float(ev.get("severity", 0.7))
            
            s_rows, s_cols = self.sectors.get(sector, (slice(0, 5), slice(0, 5)))
            # Place hazard in that sector
            hr = np.random.randint(s_rows.start, s_rows.stop)
            hc = np.random.randint(s_cols.start, s_cols.stop)
            
            if not self.barriers[hr, hc]:
                self.risk_grid[hr, hc] = np.clip(self.risk_grid[hr, hc] + sev, 0.0, 1.0)
                self.total_hazards_spawned += 1
        else:
            # Fallback stochastic hazard spawn
            if np.random.rand() < 0.4:
                hr, hc = np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)
                if not self.barriers[hr, hc]:
                    self.risk_grid[hr, hc] = np.clip(self.risk_grid[hr, hc] + 0.6, 0.0, 1.0)
                    self.total_hazards_spawned += 1

    def _move_agent(self, pos, action, speed=1):
        moves = {
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1),   # Right
        }
        dr, dc = moves.get(action, (0, 0))
        new_pos = pos + np.array([dr * speed, dc * speed])
        return np.clip(new_pos, 0, GRID_SIZE - 1)

    def step(self, actions):
        """
        Executes joint action (a0, a1, a2) in the Dec-POMDP Markov Game.
        actions: tuple or list of (a0, a1, a2)
        """
        self.step_count += 1
        a0, a1, a2 = actions
        
        reward_0 = 0.0
        reward_1 = 0.0
        reward_2 = 0.0
        
        # Agent 0: drone scout
        if a0 in [1, 2, 3, 4]:
            self.agent0_pos = self._move_agent(self.agent0_pos, a0, speed=1)
        elif a0 in [5, 6, 7, 8]:
            self.agent0_pos = self._move_agent(self.agent0_pos, a0 - 4, speed=2)
        
        self.agent0_battery = max(0.0, self.agent0_battery - 0.015)
        # recharge near refuge stations
        if any(np.array_equal(self.agent0_pos, rc) for rc in self.refuge_chambers):
            self.agent0_battery = min(1.0, self.agent0_battery + 0.25)
            
        # clear uncertainty in 3x3 window around drone
        r, c = self.agent0_pos
        r_min, r_max = max(0, r - 1), min(GRID_SIZE, r + 2)
        c_min, c_max = max(0, c - 1), min(GRID_SIZE, c + 2)
        uncertainty_cleared = np.sum(self.uncertainty_grid[r_min:r_max, c_min:c_max])
        self.uncertainty_grid[r_min:r_max, c_min:c_max] *= 0.1
        
        scouted_risk = np.sum(self.risk_grid[r_min:r_max, c_min:c_max])
        reward_0 += 1.5 * scouted_risk + 0.2 * uncertainty_cleared
        
        # Agent 1: barrier rover
        if a1 in [1, 2, 3, 4]:
            self.agent1_pos = self._move_agent(self.agent1_pos, a1, speed=1)
        elif a1 == 5:
            # deploy barrier at current cell
            rr, rc = self.agent1_pos
            if self.risk_grid[rr, rc] > 0.1 and self.agent1_barriers_remaining > 0:
                mitigated_val = self.risk_grid[rr, rc]
                self.risk_grid[rr, rc] = 0.0
                self.barriers[rr, rc] = True
                self.agent1_barriers_remaining -= 1
                self.total_hazards_mitigated += 1
                self.mitigation_latencies.append(self.step_count)
                reward_1 += 12.0 * mitigated_val
            
            # check for worker rescue nearby
            for w in self.workers:
                if not w["evacuated"]:
                    if np.hypot(w["pos"][0] - rr, w["pos"][1] - rc) <= 1.5:
                        w["evacuated"] = True
                        reward_1 += 15.0
        
        # distance shaping toward nearest hazard
        mask = self.risk_grid > 0.3
        if mask.any():
            r_idx, c_idx = np.nonzero(mask)
            min_dist = np.min(np.hypot(self.agent1_pos[0] - r_idx, self.agent1_pos[1] - c_idx))
            reward_1 -= 0.15 * min_dist
        
        # Agent 2: ventilation and haulage controller
        # a2: 0=low/normal, 1=med/normal, 2=high/throttled, 3=emergency/halt
        vent_power = {0: 1.0, 1: 2.2, 2: 4.0, 3: 6.5}[a2]
        haulage_halt = {0: 0.0, 1: 0.0, 2: 0.4, 3: 1.0}[a2]
        
        self.energy_consumed += vent_power
        self.haulage_productivity_loss += haulage_halt
        
        # ventilation dilutes Sector C and D risks
        vent_dilution = {0: 0.02, 1: 0.08, 2: 0.18, 3: 0.35}[a2]
        self.risk_grid[5:10, :] = np.clip(self.risk_grid[5:10, :] - vent_dilution, 0.0, 1.0)
        
        total_remaining_risk = np.sum(self.risk_grid)
        reward_2 += 4.0 * max(0.0, 5.0 - total_remaining_risk)
        reward_2 -= (0.35 * vent_power + 1.2 * haulage_halt)
        
        # worker injury penalties
        team_penalty = 0.0
        for w in self.workers:
            if not w["evacuated"] and not w["injured"]:
                wr, wc = w["pos"]
                cell_risk = self.risk_grid[wr, wc]
                if cell_risk > 0.65:
                    w["injured"] = True
                    self.worker_injury_count += 1
                    team_penalty -= 25.0
                elif cell_risk > 0.2:
                    team_penalty -= 2.0 * cell_risk
        
        shared_reward = -0.5 * total_remaining_risk + team_penalty
        total_team_reward = (reward_0 + reward_1 + reward_2) + (3.0 * shared_reward)
        
        # Step hazard injection
        self._inject_cv_hazards()
        
        done = self.step_count >= MAX_STEPS
        
        info = {
            "total_risk": float(total_remaining_risk),
            "worker_injuries": self.worker_injury_count,
            "hazards_mitigated": self.total_hazards_mitigated,
            "mean_latency": float(np.mean(self.mitigation_latencies)) if self.mitigation_latencies else 0.0,
            "energy_consumed": float(self.energy_consumed),
            "productivity_loss": float(self.haulage_productivity_loss),
            "reward_breakdown": (reward_0, reward_1, reward_2, shared_reward)
        }
        
        return self._get_obs(), total_team_reward, done, info

    def _get_obs(self):
        """
        Decentralized Partially Observable Observations:
        Each agent receives their localized feature vector (concatenated local patch + internal state).
        """
        # Decentralized local risk patches using 2D padding (vectorized)
        padded_risk = np.pad(self.risk_grid, 2, mode="constant", constant_values=0.0)
        
        # Agent 0 Obs: 5x5 localized risk around drone, pos (x,y), battery
        r0, c0 = self.agent0_pos
        local_0 = padded_risk[r0:r0+5, c0:c0+5].flatten()
        obs_0 = np.concatenate([local_0, [r0 / 9.0, c0 / 9.0, self.agent0_battery]])
        
        # Agent 1 Obs: 5x5 localized risk around rover, pos (x,y), remaining barriers
        r1, c1 = self.agent1_pos
        local_1 = padded_risk[r1:r1+5, c1:c1+5].flatten()
        obs_1 = np.concatenate([local_1, [r1 / 9.0, c1 / 9.0, self.agent1_barriers_remaining / 4.0]])
        
        # Agent 2 Obs: Sector aggregate risks (4), vent level, haulage status, total workers at risk
        sec_risks = [
            np.mean(self.risk_grid[0:5, 0:5]),
            np.mean(self.risk_grid[0:5, 5:10]),
            np.mean(self.risk_grid[5:10, 0:5]),
            np.mean(self.risk_grid[5:10, 5:10]),
        ]
        un_evac = sum(1 for w in self.workers if not w["evacuated"]) / 4.0
        obs_2 = np.array(sec_risks + [self.vent_level / 3.0, self.haulage_status / 3.0, un_evac], dtype=np.float32)
        
        return (obs_0.astype(np.float32), obs_1.astype(np.float32), obs_2.astype(np.float32))

    def _get_global_state(self):
        """Centralized Critic Global State (Full Environment Knowledge during CTDE training)."""
        state_vec = np.concatenate([
            self.risk_grid.flatten(),
            self.uncertainty_grid.flatten(),
            self.agent0_pos / 9.0,
            [self.agent0_battery],
            self.agent1_pos / 9.0,
            [self.agent1_barriers_remaining / 4.0],
            [self.vent_level / 3.0, self.haulage_status / 3.0]
        ]).astype(np.float32)
        return state_vec

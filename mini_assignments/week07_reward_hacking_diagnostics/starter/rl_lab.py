"""Week 7 hands-on lab: tabular Q-learning and a reward-hacking diagnosis.

Pure-Python + NumPy, CPU-only, runs in seconds. Two pieces:

1. A generic tabular `q_learning` that works with any Gymnasium-style discrete
   environment (reset() -> (obs, info); step(a) -> (obs, reward, terminated,
   truncated, info)). Used on Gymnasium FrozenLake in Part A.

2. `RewardHackGrid`: a tiny corridor world with a DELIBERATELY mis-specified
   reward. A "polish" tile pays out every time the agent re-enters it, so the
   reward-optimal policy farms the proxy forever instead of reaching the goal.
   The gap between *reward earned* and *task success* is the reward-hacking
   signal students learn to read.

Attribution: the Q-learning workflow mirrors the HuggingFace Deep RL Course
(Unit 2) and Farama Gymnasium tutorials. The reward-hacking framing follows
DeepMind's specification-gaming catalogue and OpenAI's "Faulty Reward
Functions in the Wild" (CoastRunners). All code here is original to ESP3201.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# A mis-specified-reward environment
# --------------------------------------------------------------------------- #
class RewardHackGrid:
    """A 1-D corridor:  [S] [ ] [P] [ ] [G]

    - The agent starts at S (cell 0).
    - P (the "polish"/proxy tile) pays `proxy_reward` every time the agent
      *enters* it. The agent can leave and re-enter to farm it.
    - G (the true goal) ends the episode and pays `goal_reward`.
    - Each step costs `step_cost`.

    reward_mode:
      "proxy"  -> entering P pays proxy_reward (the mis-specified reward).
      "fixed"  -> entering P pays nothing; only reaching G is rewarded.

    Gymnasium-style API so the same q_learning() works here and on FrozenLake.
    Observation is the integer cell index. Actions: 0 = left, 1 = right.
    """

    def __init__(self, length: int = 5, proxy_cell: int = 2, proxy_reward: float = 1.0,
                 goal_reward: float = 5.0, step_cost: float = 0.0,
                 reward_mode: str = "proxy", max_steps: int = 50,
                 custom_reward=None):
        self.length = length
        self.proxy_cell = proxy_cell
        self.goal_cell = length - 1
        self.proxy_reward = proxy_reward
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        # custom_reward(prev_pos, new_pos, entered, env) -> float, used when
        # reward_mode == "custom". Lets students design and TEST their own fix.
        self.custom_reward = custom_reward
        self.n_states = length
        self.n_actions = 2
        self.pos = 0
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict]:
        self.pos = 0
        self.steps = 0
        return self.pos, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        self.steps += 1
        prev = self.pos
        if action == 1:
            self.pos = min(self.length - 1, self.pos + 1)
        else:
            self.pos = max(0, self.pos - 1)

        entered = self.pos != prev

        if self.reward_mode == "custom" and self.custom_reward is not None:
            # Student-authored reward fully defines the step reward.
            reward = float(self.custom_reward(prev, self.pos, entered, self))
        else:
            reward = self.step_cost
            if self.pos == self.proxy_cell and entered and self.reward_mode == "proxy":
                reward += self.proxy_reward  # <-- the reward hack lives here

        terminated = False
        if self.pos == self.goal_cell:
            if self.reward_mode != "custom":
                reward += self.goal_reward
            terminated = True

        truncated = self.steps >= self.max_steps
        info = {"reached_goal": self.pos == self.goal_cell, "cell": self.pos}
        return self.pos, reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# Generic tabular Q-learning
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeLog:
    ret: float
    success: bool
    steps: int


@dataclass
class TrainResult:
    Q: np.ndarray
    logs: List[EpisodeLog] = field(default_factory=list)

    def returns(self) -> List[float]:
        return [e.ret for e in self.logs]

    def successes(self) -> List[int]:
        return [int(e.success) for e in self.logs]

    def moving_success_rate(self, window: int = 50) -> float:
        tail = self.successes()[-window:]
        return sum(tail) / max(1, len(tail))


def q_learning(env, n_states: int, n_actions: int, episodes: int = 2000,
               alpha: float = 0.1, gamma: float = 0.95,
               epsilon: float = 1.0, epsilon_min: float = 0.05,
               epsilon_decay: float = 0.999, max_steps: int = 100,
               is_success: Optional[Callable] = None,
               seed: int = 0) -> TrainResult:
    """Standard epsilon-greedy tabular Q-learning.

    is_success(terminated, reward, info) -> bool defines TASK success, which is
    intentionally separate from reward. Defaults to "terminated with positive
    reward" (works for FrozenLake) but RewardHackGrid passes a goal check.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions), dtype=float)
    if is_success is None:
        def is_success(terminated, reward, info):
            return bool(terminated and reward > 0)

    logs: List[EpisodeLog] = []
    eps = epsilon
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = 0.0
        success = False
        for _ in range(max_steps):
            if rng.random() < eps:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q[obs]))
            nxt, reward, terminated, truncated, info = env.step(action)
            best_next = 0.0 if terminated else np.max(Q[nxt])
            Q[obs, action] += alpha * (reward + gamma * best_next - Q[obs, action])
            obs = nxt
            total += reward
            if is_success(terminated, reward, info):
                success = True
            if terminated or truncated:
                break
        logs.append(EpisodeLog(total, success, _ + 1))
        eps = max(epsilon_min, eps * epsilon_decay)
    return TrainResult(Q, logs)


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)


def evaluate(env, Q: np.ndarray, max_steps: int = 100,
             is_success: Optional[Callable] = None) -> Dict:
    """Run the greedy policy once; report return, success, and trajectory."""
    if is_success is None:
        def is_success(terminated, reward, info):
            return bool(terminated and reward > 0)
    obs, _ = env.reset()
    total = 0.0
    success = False
    traj = [obs]
    for _ in range(max_steps):
        action = int(np.argmax(Q[obs]))
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append(obs)
        total += reward
        if is_success(terminated, reward, info):
            success = True
        if terminated or truncated:
            break
    return {"return": total, "success": success, "trajectory": traj}


# --------------------------------------------------------------------------- #
# Convenience builders
# --------------------------------------------------------------------------- #
def make_frozenlake(slippery: bool = False):
    """Lazy import so the reward-hacking part has no Gymnasium dependency."""
    import gymnasium as gym
    return gym.make("FrozenLake-v1", is_slippery=slippery)


def reached_goal_success(terminated, reward, info):
    return bool(info.get("reached_goal", False))


if __name__ == "__main__":
    # Part B preview: train under the mis-specified reward, then the fixed one.
    print("=== RewardHackGrid under the MIS-SPECIFIED ('proxy') reward ===")
    env = RewardHackGrid(reward_mode="proxy")
    res = q_learning(env, env.n_states, env.n_actions, episodes=1500,
                     max_steps=env.max_steps, is_success=reached_goal_success)
    ev = evaluate(RewardHackGrid(reward_mode="proxy"), res.Q,
                  is_success=reached_goal_success)
    print(f"greedy return={ev['return']:.1f}  reached_goal={ev['success']}  "
          f"trajectory={ev['trajectory'][:12]}...")

    print("\n=== RewardHackGrid under the FIXED reward ===")
    env2 = RewardHackGrid(reward_mode="fixed", step_cost=-0.05)
    res2 = q_learning(env2, env2.n_states, env2.n_actions, episodes=1500,
                      max_steps=env2.max_steps, is_success=reached_goal_success)
    ev2 = evaluate(RewardHackGrid(reward_mode="fixed", step_cost=-0.05), res2.Q,
                   is_success=reached_goal_success)
    print(f"greedy return={ev2['return']:.1f}  reached_goal={ev2['success']}  "
          f"trajectory={ev2['trajectory']}")

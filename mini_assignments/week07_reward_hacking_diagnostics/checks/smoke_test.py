"""Week 7 lab smoke test. CPU-only.

Run from the lab root:  python checks/smoke_test.py
Verifies tabular Q-learning learns and that the reward-hacking signal appears.
The FrozenLake check is skipped if Gymnasium is not installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from rl_lab import (  # noqa: E402
    RewardHackGrid, q_learning, evaluate, reached_goal_success,
)


def test_proxy_reward_is_hacked():
    """Under the mis-specified reward, the policy farms the proxy and never
    reaches the goal: high return, task failure."""
    env = RewardHackGrid(reward_mode="proxy")
    res = q_learning(env, env.n_states, env.n_actions, episodes=1500,
                     max_steps=env.max_steps, is_success=reached_goal_success)
    ev = evaluate(RewardHackGrid(reward_mode="proxy"), res.Q,
                  is_success=reached_goal_success)
    assert ev["return"] > env.goal_reward, "proxy farming should out-earn the goal"
    assert ev["success"] is False, "hacked policy should NOT reach the goal"


def test_fixed_reward_reaches_goal():
    """With the reward fixed, the optimal policy goes straight to the goal."""
    env = RewardHackGrid(reward_mode="fixed", step_cost=-0.05)
    res = q_learning(env, env.n_states, env.n_actions, episodes=1500,
                     max_steps=env.max_steps, is_success=reached_goal_success)
    ev = evaluate(RewardHackGrid(reward_mode="fixed", step_cost=-0.05), res.Q,
                  is_success=reached_goal_success)
    assert ev["success"] is True, "fixed reward should reach the goal"
    assert ev["trajectory"][-1] == env.goal_cell


def test_reward_success_divergence_is_visible():
    """The core teaching signal: reward earned and task success disagree."""
    proxy = RewardHackGrid(reward_mode="proxy")
    rp = q_learning(proxy, proxy.n_states, proxy.n_actions, episodes=1200,
                    max_steps=proxy.max_steps, is_success=reached_goal_success)
    final_success_rate = rp.moving_success_rate(100)
    final_return = sum(rp.returns()[-100:]) / 100.0
    assert final_return > 5.0 and final_success_rate < 0.5


def test_custom_reward_hook():
    """A student-authored reward with no proxy payout should stop the hacking;
    a custom reward that still pays the proxy should still be hacked."""
    def good_fix(prev, pos, entered, env):
        return -0.05 + (5.0 if pos == env.goal_cell else 0.0)

    def bad_fix(prev, pos, entered, env):
        r = 1.0 if (pos == env.proxy_cell and entered) else 0.0
        return r + (5.0 if pos == env.goal_cell else 0.0)

    for fn, expect_goal in ((good_fix, True), (bad_fix, False)):
        env = RewardHackGrid(reward_mode="custom", custom_reward=fn)
        res = q_learning(env, env.n_states, env.n_actions, episodes=1500,
                         max_steps=env.max_steps, is_success=reached_goal_success)
        ev = evaluate(RewardHackGrid(reward_mode="custom", custom_reward=fn), res.Q,
                      is_success=reached_goal_success)
        assert ev["success"] is expect_goal


def test_frozenlake_learns_if_available():
    try:
        from rl_lab import make_frozenlake
        env = make_frozenlake(slippery=False)
    except Exception as exc:  # gymnasium missing or env unavailable
        print(f"SKIP test_frozenlake_learns_if_available ({exc})")
        return
    ns, na = env.observation_space.n, env.action_space.n
    res = q_learning(env, ns, na, episodes=3000, max_steps=100)
    assert res.moving_success_rate(100) > 0.6, "FrozenLake should learn to solve"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} smoke tests passed.")

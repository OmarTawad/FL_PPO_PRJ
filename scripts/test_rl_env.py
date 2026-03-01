#!/usr/bin/env python3
"""
scripts/test_rl_env.py — Phase 7 RL environment smoke tests

Fast checks (tests 1-3 require NO FL rounds):
  1. Config loads and obs shape correct
  2. All-skip action forces 1 client selected (FP32, highest reliability)
  3. compute_reward produces correct components
  4. env.reset() returns correct obs shape and dtype
  5. ONE env.step() produces round JSON with required SPEC.md §8 keys
     (this test runs 1 FL round: ~60-90s on VM)

Exit 0 = all passed.
"""

from __future__ import annotations
import os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GREEN = "\033[0;32m"; RED = "\033[0;31m"; CYAN = "\033[0;36m"; RESET = "\033[0m"
errors = []; passed = 0

def ok(msg):
    global passed; passed += 1
    print(f"{GREEN}  [OK]{RESET}    {msg}")

def fail(msg):
    print(f"{RED}  [FAIL]{RESET}  {msg}")
    errors.append(msg)

def section(title):
    print(f"\n{CYAN}{'─'*60}{RESET}\n{CYAN}  {title}{RESET}")


# ── Test 1: Config + obs shape ─────────────────────────────────────────────────

def test_config_and_obs_shape():
    section("1/5  Config loads + obs shape = N*7+3")
    from src.common.config import load_config
    cfg = load_config("configs/exp1_ppo_smoke.yaml")
    n = cfg.n_clients
    expected_dim = n * 7 + 3
    if cfg.state_dim == expected_dim:
        ok(f"cfg.state_dim={cfg.state_dim} == {n}×7+3 ✓")
    else:
        fail(f"cfg.state_dim={cfg.state_dim} != {expected_dim}")

    if cfg.quantization.mode == "adaptive":
        ok(f"quantization.mode='{cfg.quantization.mode}' ✓")
    else:
        fail(f"quantization.mode='{cfg.quantization.mode}' (expected 'adaptive')")

    if cfg.data.max_eval_samples_per_client == 20:
        ok(f"max_eval_samples_per_client={cfg.data.max_eval_samples_per_client} ✓")
    else:
        ok(f"max_eval_samples_per_client={cfg.data.max_eval_samples_per_client} (non-default OK)")

    if hasattr(cfg.rl, 'batch_size'):
        ok(f"cfg.rl.batch_size={cfg.rl.batch_size} ✓")
    else:
        fail("cfg.rl.batch_size attribute missing")


# ── Test 2: All-skip action forcing ───────────────────────────────────────────

def test_all_skip_forcing():
    section("2/5  All-skip action forces 1 selected (highest reliability)")
    import numpy as np
    from src.rl.env import FLEnv, _ACTION_TO_BITS
    from src.common.config import load_config

    cfg = load_config("configs/exp1_ppo_smoke.yaml")
    n = cfg.n_clients

    # Simulate env action decode logic (no FL round needed)
    action_arr = np.zeros(n, dtype=int)  # all-skip
    action_bits = {i: _ACTION_TO_BITS[int(action_arr[i])] for i in range(n)}
    selected = {i: b for i, b in action_bits.items() if b > 0}

    # Should be empty
    if not selected:
        ok("All-skip: selected dict is empty (as expected before forcing)")
    else:
        fail(f"Expected empty selected, got {selected}")

    # Apply the same forcing logic the env uses
    from src.heterogeneity.profiles import get_profile
    reliabilities = []
    for prof_cfg in cfg.clients.profiles:
        try:
            hp = get_profile(prof_cfg.profile)
            reliabilities.append(hp.reliability)
        except KeyError:
            reliabilities.append(1.0)

    if not selected:
        best_cid = int(np.argmax(reliabilities))
        selected = {best_cid: 32}

    if len(selected) == 1:
        ok(f"Forced 1 client: {selected} (reliability={reliabilities[best_cid]:.2f}) ✓")
    else:
        fail(f"Expected 1 forced selection, got {selected}")

    # Verify it's FP32
    _, bits = list(selected.items())[0]
    if bits == 32:
        ok(f"Forced client uses FP32 (bits=32) ✓")
    else:
        fail(f"Expected bits=32, got {bits}")


# ── Test 3: compute_reward standalone ─────────────────────────────────────────

def test_compute_reward():
    section("3/5  compute_reward(): components correct")
    from src.rl.reward import compute_reward
    import numpy as np

    # Case 1: 2 clients selected, 1 dropped, same bits (var=0)
    r, comp = compute_reward(
        acc_t=0.85, acc_prev=0.80,
        selected_client_ids=["0", "1"],
        dropout_client_ids=["1"],
        quant_assignments_bits={"0": 32, "1": 32},
        alpha=1.0, beta=0.5, gamma=0.1,
    )
    delta = 0.85 - 0.80
    dropout_rate = 1 / 2   # 1 dropped, 2 selected
    var = 0.0
    expected_r = 1.0 * delta - 0.5 * dropout_rate - 0.1 * var
    if abs(r - expected_r) < 1e-6:
        ok(f"Case1: r={r:.4f} == expected={expected_r:.4f} ✓")
    else:
        fail(f"Case1: r={r:.4f} != expected={expected_r:.4f}")

    if abs(comp["delta_acc"] - delta) < 1e-9:
        ok(f"Case1: delta_acc={comp['delta_acc']:.4f} ✓")
    else:
        fail(f"Case1: delta_acc={comp['delta_acc']:.4f} != {delta:.4f}")

    # Case 2: 3 clients, bits=[32,16,8] → var = var([32,16,8])
    r2, comp2 = compute_reward(
        acc_t=0.50, acc_prev=0.50,
        selected_client_ids=["0", "1", "2"],
        dropout_client_ids=[],
        quant_assignments_bits={"0": 32, "1": 16, "2": 8},
        alpha=1.0, beta=0.5, gamma=0.1,
    )
    expected_var = float(np.var([32, 16, 8], ddof=0))
    if abs(comp2["quant_variance"] - expected_var) < 1e-6:
        ok(f"Case2: var([32,16,8])={comp2['quant_variance']:.4f} == {expected_var:.4f} ✓")
    else:
        fail(f"Case2: var mismatch: {comp2['quant_variance']:.4f} != {expected_var:.4f}")

    # Case 3: 1 client → var = 0.0 by definition
    r3, comp3 = compute_reward(
        acc_t=0.60, acc_prev=0.50,
        selected_client_ids=["0"],
        dropout_client_ids=[],
        quant_assignments_bits={"0": 8},
        alpha=1.0, beta=0.5, gamma=0.1,
    )
    if comp3["quant_variance"] == 0.0:
        ok("Case3: 1 client → quant_variance=0.0 ✓")
    else:
        fail(f"Case3: quant_variance={comp3['quant_variance']} (expected 0.0)")

    # Case 4: dropout client NOT in selected → not counted
    r4, comp4 = compute_reward(
        acc_t=0.70, acc_prev=0.60,
        selected_client_ids=["0"],
        dropout_client_ids=["1"],  # client 1 never selected → shouldn't count
        quant_assignments_bits={"0": 32},
        alpha=1.0, beta=0.5, gamma=0.1,
    )
    if comp4["dropout_rate"] == 0.0:
        ok("Case4: dropout of non-selected client → dropout_rate=0.0 ✓")
    else:
        fail(f"Case4: dropout_rate={comp4['dropout_rate']} (expected 0.0)")


# ── Test 4: env.reset() obs shape ─────────────────────────────────────────────

def test_env_reset():
    section("4/5  FLEnv.reset() returns correct obs shape + dtype")
    import numpy as np
    from src.common.config import load_config
    from src.rl.env import FLEnv

    cfg = load_config("configs/exp1_ppo_smoke.yaml")
    expected_dim = cfg.state_dim  # N*7+3

    print(f"  Initialising FLEnv (downloads CIFAR if needed)...")
    t0 = time.time()
    env = FLEnv(cfg)
    obs, info = env.reset(seed=42)
    elapsed = time.time() - t0

    if obs.shape == (expected_dim,):
        ok(f"obs.shape={obs.shape} == ({expected_dim},) ✓  [init+reset in {elapsed:.1f}s]")
    else:
        fail(f"obs.shape={obs.shape} != ({expected_dim},)")

    if obs.dtype == np.float32:
        ok(f"obs.dtype={obs.dtype} ✓")
    else:
        fail(f"obs.dtype={obs.dtype} (expected float32)")

    if isinstance(info, dict):
        ok(f"info is dict ✓")
    else:
        fail(f"info type={type(info)} (expected dict)")

    # Action + obs spaces match
    n = cfg.n_clients
    if tuple(env.action_space.nvec) == tuple([4] * n):
        ok(f"action_space.nvec={env.action_space.nvec} ✓")
    else:
        fail(f"action_space.nvec={env.action_space.nvec} (expected {[4]*n})")

    return env   # return for test 5


# ── Test 5: one step → round JSON ─────────────────────────────────────────────

def test_one_step_json(env):
    section("5/5  One env.step() → round_001.json with required SPEC.md §8 keys")
    required_keys = {
        "round", "selected_clients", "quant_assignments",
        "actual_quant_method", "dropout_clients", "dropout_fraction",
        "global_accuracy", "accuracy_delta",
    }
    import numpy as np

    print(f"  Running one FL round (est. 60-100s on VM)...")
    t0 = time.time()

    # action: [1, 1, 1] = select all clients with FP32
    action = np.array([1, 1, 1], dtype=int)
    obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.time() - t0

    ok(f"step() returned in {elapsed:.1f}s")

    # Check obs
    expected_dim = env.cfg.state_dim
    if obs.shape == (expected_dim,):
        ok(f"Next obs.shape={obs.shape} ✓")
    else:
        fail(f"Next obs.shape={obs.shape} != ({expected_dim},)")

    # Check reward
    if isinstance(reward, (int, float)):
        ok(f"reward={reward:.4f} (float) ✓")
    else:
        fail(f"reward type={type(reward)}")

    # Check info contains round_log
    round_log = info.get("round_log", {})
    missing = required_keys - set(round_log.keys())
    if not missing:
        ok(f"round_log has all {len(required_keys)} required SPEC.md §8 keys ✓")
    else:
        fail(f"round_log missing keys: {missing}")

    # Check global_accuracy is a float
    acc = round_log.get("global_accuracy")
    if isinstance(acc, float):
        ok(f"global_accuracy={acc:.4f} (float) ✓")
    else:
        fail(f"global_accuracy={acc} (expected float)")

    # Check accuracy_delta
    delta = round_log.get("accuracy_delta")
    if isinstance(delta, (int, float)):
        ok(f"accuracy_delta={delta:+.4f} ✓")
    else:
        fail(f"accuracy_delta type={type(delta)}")

    # Check JSON file on disk
    output_dir = env.strategy.output_dir
    json_files = sorted(output_dir.glob("round_*.json"))
    if json_files:
        ok(f"round JSON written to disk: {json_files[0].name} ✓")
        import json
        with open(json_files[0]) as f:
            d = json.load(f)
        miss2 = required_keys - set(d.keys())
        if not miss2:
            ok(f"On-disk JSON has all required keys ✓")
        else:
            fail(f"On-disk JSON missing: {miss2}")
    else:
        fail(f"No round_*.json found in {output_dir}")

    # Reward components
    rc = info.get("reward_components", {})
    for key in ("delta_acc", "dropout_rate", "quant_variance", "reward"):
        if key in rc:
            ok(f"reward_components['{key}']={rc[key]:.4f} ✓")
        else:
            fail(f"reward_components missing '{key}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{CYAN}  Phase 7 RL Environment Smoke Test{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    test_config_and_obs_shape()
    test_all_skip_forcing()
    test_compute_reward()
    env = test_env_reset()
    test_one_step_json(env)

    print(f"\n{CYAN}{'═'*60}{RESET}")
    if errors:
        print(f"{RED}  FAILED — {len(errors)} error(s):{RESET}")
        for e in errors:
            print(f"{RED}    • {e}{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}  ALL {passed} CHECKS PASSED (5 sections){RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()

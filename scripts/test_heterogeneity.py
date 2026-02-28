#!/usr/bin/env python3
"""
scripts/test_heterogeneity.py — Phase 5 smoke test for the heterogeneity layer

Tests WITHOUT any network calls or downloads.
Verifies:
    1.  All 4 profiles exist and have correct attribute types
    2.  Profile resource limits are monotonically ordered (strong > weak)
    3.  Dropout probabilities: p(8) > p(16) > p(32) for all profiles
    4.  stochastic_dropout_probability() returns expected values
    5.  get_profile() raises KeyError on unknown name
    6.  ClientProfile.state_features() returns 7 keys matching SPEC.md §2.1
    7.  DropoutTracker: record() increments consecutive_drops correctly
    8.  DropoutTracker: successful round resets consecutive_drops to 0
    9.  DropoutTracker: record_not_selected() preserves existing dropout state
   10.  simulate_dropout(): extreme_weak + INT8 → dropout fraction ≈ 0.60 (±0.1)
   11.  simulate_dropout(): strong + FP32 → dropout fraction ≈ 0.01 (±0.05)
   12.  simulate_round_dropout(): tracker records outcomes for all client IDs
   13.  simulate_round_dropout(): mismatched lengths raise ValueError
   14.  DropoutTracker.reset() zeroes all counters
   15.  ClientProfile is immutable (frozen dataclass)

Exit codes: 0 = all passed, 1 = failure
"""

from __future__ import annotations

import sys
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GREEN = "\033[0;32m"
RED   = "\033[0;31m"
CYAN  = "\033[0;36m"
RESET = "\033[0m"

errors: list = []
passed = 0

def ok(msg):
    global passed; passed += 1
    print(f"{GREEN}  [OK]{RESET}    {msg}")

def fail(msg):
    print(f"{RED}  [FAIL]{RESET}  {msg}")
    errors.append(msg)

def section(title):
    print(f"\n{CYAN}{'─'*60}{RESET}\n{CYAN}  {title}{RESET}")


# Imports
from src.heterogeneity.profiles import (
    get_profile, list_all_profiles, stochastic_dropout_probability,
    ClientProfile, PROFILE_NAMES, VALID_BITS,
)
from src.heterogeneity.dropout import (
    DropoutTracker, simulate_dropout, simulate_round_dropout,
)

ORDERED_PROFILES = ["strong", "medium", "weak", "extreme_weak"]


# ── Test 1: All profiles exist ────────────────────────────────────────────────

def test_profiles_exist():
    section("1/15  All 4 Profiles Exist")
    all_p = list_all_profiles()
    for name in ORDERED_PROFILES:
        if name in all_p:
            p = all_p[name]
            ok(f"'{name}': cpu_cores={p.cpu_cores}, mem={p.mem_limit_mb}MB, reliability={p.reliability}")
        else:
            fail(f"Profile '{name}' missing from registry")


# ── Test 2: Resource limits ordered ──────────────────────────────────────────

def test_resource_ordering():
    section("2/15  Resource Limits Monotonically Ordered (strong > weak)")
    profiles = [get_profile(n) for n in ORDERED_PROFILES]

    for i in range(len(profiles) - 1):
        a, b = profiles[i], profiles[i+1]
        # mem_limit_mb should be non-increasing
        if a.mem_limit_mb >= b.mem_limit_mb:
            ok(f"{a.name}.mem={a.mem_limit_mb} >= {b.name}.mem={b.mem_limit_mb}")
        else:
            fail(f"mem ordering violated: {a.name}={a.mem_limit_mb} < {b.name}={b.mem_limit_mb}")

        # reliability should be non-increasing
        if a.reliability >= b.reliability:
            ok(f"{a.name}.rel={a.reliability} >= {b.name}.rel={b.reliability}")
        else:
            fail(f"reliability ordering violated: {a.name}={a.reliability} < {b.name}={b.reliability}")


# ── Test 3: Dropout probs ordered per profile ─────────────────────────────────

def test_dropout_ordering():
    section("3/15  Dropout Probabilities: p(8) > p(16) > p(32) Per Profile")
    for name in ORDERED_PROFILES:
        p = get_profile(name)
        p32, p16, p8 = p.dropout_p[32], p.dropout_p[16], p.dropout_p[8]
        if p8 > p16 > p32:
            ok(f"{name}: p(8)={p8} > p(16)={p16} > p(32)={p32} ✓")
        else:
            fail(f"{name}: ordering violated p(8)={p8}, p(16)={p16}, p(32)={p32}")


# ── Test 4: stochastic_dropout_probability() values ─────────────────────────

def test_dropout_probability_values():
    section("4/15  stochastic_dropout_probability() Expected Values")
    cases = [
        ("strong",       32, 0.01),
        ("strong",       8,  0.03),
        ("medium",       16, 0.08),
        ("weak",         32, 0.10),
        ("weak",         8,  0.35),
        ("extreme_weak", 8,  0.60),
    ]
    for name, bits, expected in cases:
        actual = stochastic_dropout_probability(get_profile(name), bits)
        if abs(actual - expected) < 1e-9:
            ok(f"{name} bits={bits}: p={actual} ✓")
        else:
            fail(f"{name} bits={bits}: expected {expected}, got {actual}")


# ── Test 5: get_profile() raises on unknown name ──────────────────────────────

def test_unknown_profile():
    section("5/15  get_profile() Raises KeyError on Unknown Name")
    try:
        get_profile("nonexistent")
        fail("get_profile('nonexistent') should raise KeyError")
    except KeyError as e:
        ok(f"Raises KeyError: {str(e)[:60]}")

    try:
        stochastic_dropout_probability(get_profile("strong"), 4)
        fail("quant_bits=4 should raise ValueError")
    except ValueError as e:
        ok(f"quant_bits=4 raises ValueError: {str(e)[:60]}")


# ── Test 6: state_features() returns 7 keys ───────────────────────────────────

def test_state_features():
    section("6/15  state_features() Returns 7 SPEC.md §2.1 Keys")
    expected_keys = {
        "cpu_cores_i", "mem_limit_mb_i", "mem_used_frac_i",
        "reliability_i", "last_dropout_i", "consecutive_drops_i", "last_quant_i",
    }
    for name in ORDERED_PROFILES:
        feat = get_profile(name).state_features()
        actual_keys = set(feat.keys())
        if actual_keys == expected_keys:
            ok(f"{name}: 7 expected keys present ✓")
        else:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            fail(f"{name}: missing={missing} extra={extra}")


# ── Test 7: DropoutTracker consecutive_drops increments ──────────────────────

def test_tracker_consecutive():
    section("7/15  DropoutTracker: consecutive_drops Increments on Drop")
    tracker = DropoutTracker(n_clients=2)

    tracker.record(0, dropped=True)
    tracker.record(0, dropped=True)
    tracker.record(0, dropped=True)
    state = tracker.get_state(0)
    if state["consecutive_drops"] == 3:
        ok(f"3 drops → consecutive_drops=3 ✓")
    else:
        fail(f"Expected 3, got {state['consecutive_drops']}")

    if state["last_dropout"] == 1:
        ok("last_dropout=1 ✓")
    else:
        fail(f"last_dropout={state['last_dropout']} (expected 1)")


# ── Test 8: Successful round resets consecutive_drops ────────────────────────

def test_tracker_reset_on_success():
    section("8/15  DropoutTracker: Success Resets consecutive_drops to 0")
    tracker = DropoutTracker(n_clients=1)
    tracker.record(0, dropped=True)
    tracker.record(0, dropped=True)
    tracker.record(0, dropped=False)  # success
    state = tracker.get_state(0)
    if state["consecutive_drops"] == 0:
        ok("consecutive_drops reset to 0 on success ✓")
    else:
        fail(f"Expected 0, got {state['consecutive_drops']}")
    if state["last_dropout"] == 0:
        ok("last_dropout=0 after success ✓")
    else:
        fail(f"last_dropout={state['last_dropout']} (expected 0)")


# ── Test 9: record_not_selected preserves state ───────────────────────────────

def test_tracker_not_selected():
    section("9/15  record_not_selected() Preserves Existing Dropout State")
    tracker = DropoutTracker(n_clients=2)
    tracker.record(0, dropped=True)
    tracker.record(0, dropped=True)
    consecutive_before = tracker.get_state(0)["consecutive_drops"]

    tracker.record_not_selected(0)  # should NOT reset

    state = tracker.get_state(0)
    if state["consecutive_drops"] == consecutive_before:
        ok(f"consecutive_drops unchanged ({consecutive_before}) after record_not_selected ✓")
    else:
        fail(f"consecutive_drops changed: {consecutive_before} → {state['consecutive_drops']}")


# ── Test 10: simulate_dropout() extreme_weak + INT8 distribution ─────────────

def test_dropout_distribution_extreme():
    section("10/15  simulate_dropout(): extreme_weak + INT8 ≈ 60% Dropout")
    profile = get_profile("extreme_weak")
    rng = random.Random(42)
    n_trials = 2000
    n_dropped = sum(simulate_dropout(0, profile, 8, rng=rng) for _ in range(n_trials))
    rate = n_dropped / n_trials
    expected = 0.60
    tolerance = 0.07  # ±7% at n=2000 is well within 3σ
    if abs(rate - expected) <= tolerance:
        ok(f"extreme_weak+INT8: dropout rate={rate:.3f} (expected≈{expected}, tol±{tolerance}) ✓")
    else:
        fail(f"extreme_weak+INT8: rate={rate:.3f} far from {expected} (tol {tolerance})")


# ── Test 11: simulate_dropout() strong + FP32 distribution ───────────────────

def test_dropout_distribution_strong():
    section("11/15  simulate_dropout(): strong + FP32 ≈ 1% Dropout")
    profile = get_profile("strong")
    rng = random.Random(99)
    n_trials = 5000
    n_dropped = sum(simulate_dropout(0, profile, 32, rng=rng) for _ in range(n_trials))
    rate = n_dropped / n_trials
    expected = 0.01
    tolerance = 0.015  # ±1.5% at n=5000
    if abs(rate - expected) <= tolerance:
        ok(f"strong+FP32: dropout rate={rate:.4f} (expected≈{expected}, tol±{tolerance}) ✓")
    else:
        fail(f"strong+FP32: rate={rate:.4f} far from {expected} (tol {tolerance})")


# ── Test 12: simulate_round_dropout() records into tracker ───────────────────

def test_round_dropout():
    section("12/15  simulate_round_dropout(): Records All Outcomes in Tracker")
    tracker = DropoutTracker(n_clients=3)
    profiles = [get_profile("strong"), get_profile("weak"), get_profile("extreme_weak")]
    quant_assignments = {0: 32, 1: 16, 2: 8}
    rng = random.Random(7)

    n_rounds = 100
    for _ in range(n_rounds):
        simulate_round_dropout([0, 1, 2], profiles, quant_assignments, tracker=tracker, rng=rng)

    # Each client should have n_rounds recorded
    for cid in range(3):
        s = tracker.get_state(cid)
        total = tracker.total_rounds[cid]
        if total == n_rounds:
            ok(f"client {cid}: {total} rounds recorded ✓")
        else:
            fail(f"client {cid}: expected {n_rounds} rounds, got {total}")

    # Dropout rates should follow profile ordering (strong < weak < extreme_weak approx)
    rate0 = tracker.total_drops[0] / n_rounds
    rate2 = tracker.total_drops[2] / n_rounds
    if rate2 >= rate0:
        ok(f"extreme_weak dropout rate ({rate2:.2f}) >= strong ({rate0:.2f}) ✓")
    else:
        fail(f"Ordering violated: extreme_weak={rate2:.2f}, strong={rate0:.2f}")


# ── Test 13: simulate_round_dropout() mismatched lengths ─────────────────────

def test_round_dropout_mismatch():
    section("13/15  simulate_round_dropout(): Mismatched Lengths Raise ValueError")
    try:
        simulate_round_dropout(
            [0, 1], [get_profile("strong")], {0: 32, 1: 32}
        )
        fail("Should have raised ValueError for mismatched lengths")
    except ValueError as e:
        ok(f"Raises ValueError: {str(e)[:60]}")


# ── Test 14: DropoutTracker.reset() ──────────────────────────────────────────

def test_tracker_reset():
    section("14/15  DropoutTracker.reset() Zeroes All Counters")
    tracker = DropoutTracker(n_clients=2)
    tracker.record(0, dropped=True)
    tracker.record(0, dropped=True)
    tracker.record(1, dropped=True)
    tracker.reset()

    for cid in range(2):
        s = tracker.get_state(cid)
        if s["consecutive_drops"] == 0 and s["last_dropout"] == 0 and s["dropout_rate"] == 0.0:
            ok(f"client {cid}: all counters zeroed ✓")
        else:
            fail(f"client {cid}: reset failed: {s}")


# ── Test 15: ClientProfile is immutable ──────────────────────────────────────

def test_profile_immutable():
    section("15/15  ClientProfile Is Immutable (frozen dataclass)")
    profile = get_profile("strong")
    try:
        profile.reliability = 0.5  # type: ignore
        fail("Should raise FrozenInstanceError for mutation attempt")
    except Exception as e:
        if "frozen" in type(e).__name__.lower() or "can't set" in str(e).lower() or "FrozenInstanceError" in type(e).__name__:
            ok(f"Mutation raises {type(e).__name__} ✓")
        else:
            ok(f"Mutation raises {type(e).__name__}: {str(e)[:50]} (immutability enforced)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{CYAN}  Phase 5 Heterogeneity Layer Smoke Test{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    test_profiles_exist()
    test_resource_ordering()
    test_dropout_ordering()
    test_dropout_probability_values()
    test_unknown_profile()
    test_state_features()
    test_tracker_consecutive()
    test_tracker_reset_on_success()
    test_tracker_not_selected()
    test_dropout_distribution_extreme()
    test_dropout_distribution_strong()
    test_round_dropout()
    test_round_dropout_mismatch()
    test_tracker_reset()
    test_profile_immutable()

    print(f"\n{CYAN}{'═'*60}{RESET}")
    if errors:
        print(f"{RED}  FAILED — {len(errors)} error(s):{RESET}")
        for e in errors:
            print(f"{RED}    • {e}{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}  ALL {passed} CHECKS PASSED (15 sections){RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()

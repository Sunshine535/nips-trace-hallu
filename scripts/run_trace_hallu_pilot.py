#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
DEFAULT_INPUT = "methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv"


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def has_final(text):
    return "final answer" in (text or "").lower()


def risk_score(row):
    raw64 = row.get("fixed_64_raw", "")
    p64 = row.get("fixed_64_pred", "")
    p128 = row.get("fixed_128_pred", "")
    score = 0.0
    score += 1.2 if not has_final(raw64) else 0.0
    score += 0.8 if p64 != p128 else 0.0
    score += 0.4 * min(len(NUM_RE.findall(raw64)), 8) / 8.0
    score += 0.3 if to_float(row.get("fixed_64_tokens", 0)) >= 62 else 0.0
    return score


def prf(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def eval_policy(rows, threshold, lambda_cost):
    n = max(1, len(rows))
    total_correct = 0
    total_tokens = 0.0
    out_rows = []
    for r in rows:
        s = risk_score(r)
        act = 256 if s >= threshold else 64
        c = to_int(r[f"fixed_{act}_correct"])
        t = to_float(r[f"fixed_{act}_tokens"])
        total_correct += c
        total_tokens += t
        out_rows.append(
            {
                "idx": r.get("idx", ""),
                "score": s,
                "action_budget": act,
                "correct": c,
                "tokens": t,
            }
        )
    acc = total_correct / n
    avg_tokens = total_tokens / n
    utility = acc - lambda_cost * (avg_tokens / 256.0)
    return {"accuracy": acc, "avg_tokens": avg_tokens, "utility": utility, "rows": out_rows}


def fixed_metrics(rows, budget, lambda_cost):
    n = max(1, len(rows))
    acc = sum(to_int(r[f"fixed_{budget}_correct"]) for r in rows) / n
    avg_tokens = sum(to_float(r[f"fixed_{budget}_tokens"]) for r in rows) / n
    utility = acc - lambda_cost * (avg_tokens / 256.0)
    return {"accuracy": acc, "avg_tokens": avg_tokens, "utility": utility}


def main():
    ap = argparse.ArgumentParser(description="TRACE-Hallu pilot from AdaThink per-sample traces")
    ap.add_argument("--input_csv", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", type=str, default="methods/02_trace_hallu/results")
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    for r in rows:
        r["_risk"] = risk_score(r)
        r["_label"] = 1 - to_int(r.get("fixed_256_correct", 0))

    train = [r for r in rows if (to_int(r.get("idx", 0)) % 5) != 0]
    test = [r for r in rows if (to_int(r.get("idx", 0)) % 5) == 0]
    if not test:
        test = rows[-max(1, len(rows) // 5) :]
        train = rows[: len(rows) - len(test)]

    best_t = 0.0
    best_f1 = -1.0
    train_y = [r["_label"] for r in train]
    for i in range(0, 81):
        t = i * 0.05
        pred = [1 if r["_risk"] >= t else 0 for r in train]
        f1 = prf(train_y, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    test_y = [r["_label"] for r in test]
    test_pred = [1 if r["_risk"] >= best_t else 0 for r in test]
    det = prf(test_y, test_pred)

    policy = eval_policy(test, best_t, args.lambda_cost)
    fixed64 = fixed_metrics(test, 64, args.lambda_cost)
    fixed128 = fixed_metrics(test, 128, args.lambda_cost)
    fixed256 = fixed_metrics(test, 256, args.lambda_cost)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"trace_hallu_pilot_{ts}.json")
    out_csv = os.path.join(args.output_dir, f"trace_hallu_policy_{ts}.csv")

    result = {
        "meta": {
            "timestamp_utc": ts,
            "input_csv": args.input_csv,
            "train_size": len(train),
            "test_size": len(test),
            "lambda_cost": args.lambda_cost,
            "best_threshold": best_t,
        },
        "detector_test": det,
        "policy_test": {
            "accuracy": policy["accuracy"],
            "avg_tokens": policy["avg_tokens"],
            "utility": policy["utility"],
        },
        "baselines_test": {
            "fixed64": fixed64,
            "fixed128": fixed128,
            "fixed256": fixed256,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "score", "action_budget", "correct", "tokens"])
        writer.writeheader()
        writer.writerows(policy["rows"])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

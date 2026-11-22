"""Mixed-order check: feed a known list of parts in a known order, compare with what the sorter decided.

  python sorter.py check                 20 parts, order chosen by the program (same every time)
  python sorter.py check 30 --seed 4     30 parts, a different order
  python sorter.py check --order "bolt-0.25, nut-0.375, washer-0.25"     your own order
  python sorter.py check --order myorder.txt                             ... or one label per line

Extra detections (the belt seam, a hand) and parts the camera never saw are handled: the two lists
are lined up the way a spell-checker lines up two words, so one slip does not shift everything after it.
"""
import os
import random

RESORT = "resort"


def make_order(labels, n=20, seed=1):
    """n parts, as even a mix as possible, shuffled, never the same part three times running."""
    labels = sorted(labels)
    rng = random.Random(seed)
    for _ in range(200):
        order = [labels[i % len(labels)] for i in range(n)]
        rng.shuffle(order)
        if all(not (order[i] == order[i + 1] == order[i + 2]) for i in range(n - 2)):
            break
    return order


def read_order(text, known):
    if os.path.exists(text):
        with open(text) as fh:
            text = fh.read()
    order = [w.strip() for w in text.replace("\n", ",").split(",") if w.strip() and not w.strip().startswith("#")]
    bad = sorted(set(order) - set(known))
    if bad:
        raise SystemExit(f"Unknown label(s) in the order: {', '.join(bad)}. Known: {', '.join(sorted(known))}")
    return order


def align(expected, observed, blind=()):
    """Line up expected labels with observed decisions. Returns rows of (expected | None, observed | None)."""
    n, m = len(expected), len(observed)

    def pair(e, o):
        return 0.0 if e == o else 0.5 if o == RESORT else 1.0

    def extra(j):                        # a detection with no part behind it
        if j in blind:                   # never measured (belt seam, hand): almost free to leave out
            return 0.05
        return 0.45 if observed[j] == RESORT else 1.2

    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < n and j < m:
                cost[i + 1][j + 1] = min(cost[i + 1][j + 1], c + pair(expected[i], observed[j]))
            if i < n:
                cost[i + 1][j] = min(cost[i + 1][j], c + 1.0)            # part never seen
            if j < m:
                cost[i][j + 1] = min(cost[i][j + 1], c + extra(j))
    rows, i, j = [], n, m
    while i or j:
        c = cost[i][j]
        if i and j and abs(cost[i - 1][j - 1] + pair(expected[i - 1], observed[j - 1]) - c) < 1e-9:
            rows.append((i - 1, j - 1)); i -= 1; j -= 1
        elif j and abs(cost[i][j - 1] + extra(j - 1) - c) < 1e-9:
            rows.append((None, j - 1)); j -= 1
        else:
            rows.append((i - 1, None)); i -= 1
    return rows[::-1]


def report(expected, log):
    """log: the pipeline's per-part records, in the order the parts crossed the decision line."""
    log = sorted(log, key=lambda r: r["t_decide"])
    blind = {j for j, r in enumerate(log) if not r["frames"]}     # never measured: belt seam, a hand
    observed = [r["final"] for r in log]
    rows = align(expected, observed, blind)
    left_out = sum(1 for i, j in rows if i is None and j in blind)
    rows = [(i, j) for i, j in rows if not (i is None and j in blind)]
    count = dict(correct=0, resort=0, wrong=0, unseen=0, extra=0)
    out = [f"{'#':>3}  {'you fed':<14} {'sorter said':<14} result"]
    for i, j in rows:
        r = log[j] if j is not None else None
        e = expected[i] if i is not None else None
        if r is None:
            kind, text = "unseen", "NOT SEEN by the camera"
        elif e is None:
            kind = "extra"
            text = "extra detection, not one of your parts" + (f" ({r['final_reason']})" if r["final_reason"] else "")
        elif r["final"] == e:
            kind, text = "correct", "correct"
        elif r["final"] == RESORT:
            kind = "resort"
            text = f"re-sort: {r['final_reason']}"
            if r.get("geo_label") or r.get("app_label"):
                text += f"   [size model: {r.get('geo_label')} ({r.get('geo_level')}), " \
                        f"look model: {r.get('app_label')} ({r.get('app_level')})]"
        else:
            kind, text = "wrong", "WRONG BIN"
        count[kind] += 1
        out.append(f"{(i + 1) if i is not None else '':>3}  {e or '-':<14} {(r['final'] if r else '-'):<14} {text}")
    n = len(expected)
    out.append("")
    out.append(f"{n} parts fed:  correct bin {count['correct']}   re-sort {count['resort']}   "
               f"WRONG BIN {count['wrong']}   not seen {count['unseen']}   extra detections {count['extra']}")
    if left_out:
        out.append(f"Left out: {left_out} detection(s) that were never measured and match none of your parts "
                   f"(belt seam, a hand).")
    if count["extra"] or count["unseen"]:
        out.append("(If parts were fed in a different order than listed, the rows will not line up: check the crops "
                   "saved in the run folder.)")
    return "\n".join(out), count

"""
Compare two replays (verify_replay.py .jsonl) of one recording, event by event on exact file samples. Unlike compare_labels.py
there is no clock to fit and all CLAP scores are compared. Events pair within one hop. Pairs whose clips start on the same file
sample saw the same audio, so their scores can be compared numerically (run twice = determinism, two GPUs = numeric parity).
Usage: python compare_replay.py <A.jsonl> <B.jsonl>
"""
import json, statistics, sys

A = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
B = [json.loads(x) for x in open(sys.argv[2], encoding="utf-8")]
HOP = 1024
candidates = sorted((abs(a["sample"] - b["sample"]), i, j) for i, a in enumerate(A) for j, b in enumerate(B) if abs(a["sample"] - b["sample"]) <= HOP)
ua, ub, pairs = set(), set(), []
for _, i, j in candidates:
    if i not in ua and j not in ub:
        ua.add(i); ub.add(j); pairs.append((i, j))
pairs.sort()
exact = [(i, j) for i, j in pairs if A[i]["clip"] == B[j]["clip"]]
same = sum(A[i]["clap"] == B[j]["clap"] for i, j in pairs)
print(f"A {len(A)} events, B {len(B)}; paired {len(pairs)} (same clip {len(exact)}); A only {len(A) - len(ua)}, B only {len(B) - len(ub)}")
print(f"same CLAP label {same}/{len(pairs)}")
if exact:
    d = sorted(abs(x - y) for i, j in exact for x, y in zip(A[i]["scores"], B[j]["scores"]))
    identical = sum(A[i]["scores"] == B[j]["scores"] for i, j in exact)
    print(f"same-clip events with all scores bit-identical: {identical}/{len(exact)}")
    print(f"|score difference| over {len(d)} scores: median {statistics.median(d):.3e}, p99 {d[int(len(d) * .99)]:.3e}, max {d[-1]:.3e}")
for i, j in pairs:
    if A[i]["clap"] != B[j]["clap"]:
        print(f"  {A[i]['sample'] / 48000:8.3f} s  A {A[i]['clap']:<17} {A[i]['p']:.3f}   B {B[j]['clap']:<17} {B[j]['p']:.3f}  (clip diff {B[j]['clip'] - A[i]['clip']:+d})")
for k, e in enumerate(A):
    if k not in ua: print(f"  {e['sample'] / 48000:8.3f} s  A only: {e['clap']} {e['p']:.3f}")
for k, e in enumerate(B):
    if k not in ub: print(f"  {e['sample'] / 48000:8.3f} s  B only: {e['clap']} {e['p']:.3f}")

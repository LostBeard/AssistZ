"""What did ?memlog=1 see? Usage: python mem_read.py <run> <tag> [memsample csv]"""
import csv, re, sys

run, tag = sys.argv[1], sys.argv[2]
rows, sites = [], {}
for line in open(f"{run}/replay{tag}_mem.log", encoding="utf-8"):
    m = re.match(r"\[AssistZ\] mem n=(\d+) storage=(\d+) staging=(\d+) buffers=(\d+) pool=(\d+)", line)
    if m:
        rows.append(tuple(int(x) for x in m.groups()))
        continue
    m = re.match(r"\[AssistZ\] memsite n=(\d+) (.*)", line)
    if m:
        sites.setdefault(int(m.group(1)), []).append(m.group(2).strip())
print(f"{len(rows)} events")
print("  event  storage MB  staging MB  buffers  pool allocations")
for r in rows:
    if r[0] in (1, 2, 3, 5, 10, 25, 50, 100, 150, 200, 250, 300) or r[0] == rows[-1][0]:
        print(f"  {r[0]:5d} {r[1] / 1e6:11.1f} {r[2] / 1e6:11.1f} {r[3]:8d} {r[4]:8d}")
for n, s in sorted(sites.items()):
    print(f"Creation sites at event {n}:")
    for x in s:
        print("   ", x[:210])
if len(sys.argv) > 3:
    t = list(csv.DictReader(open(sys.argv[3])))
    print("Chrome dedicated MB by replay second:", " ".join(f"{x['t']}s:{x['dedicated_mb']}" for x in t if int(x["t"]) % 16 < 2))

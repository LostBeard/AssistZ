"""
Read-only. 1) Which ammo classes use the flyby and crack sound sets. 2) List the game's projectile sound samples and copy them into the scratchpad for measurement (kept
local, never committed). Decoding happens in a later step, depending on the format found here.
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_dayz_config import load, pbo_entries, lzss

ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_samples")
os.makedirs(OUT, exist_ok=True)

names = sorted(os.path.basename(p) for p in glob.glob(ADDONS + r"\*.pbo"))
print("pbo names with ammo or ammunition:", [n for n in names if re.search(r"(?i)ammo|ammunition", n)])
for pbo in glob.glob(ADDONS + r"\*.pbo"):
    base = os.path.basename(pbo)
    if not re.search(r"(?i)ammo|ammunition|^weapons_data|^weapons_core", base) or os.path.getsize(pbo) > 60_000_000:
        continue
    try:
        configs = load(pbo)
    except Exception as e:
        print(" ", base, "parse failed:", e)
        continue
    for cfg, rap in configs:
        raw = rap.d
        if b"soundSetBulletFly" not in raw and b"soundSetSonicCrack" not in raw and b"bulletFlyBy" not in raw and b"SuperSonicCrack" not in raw:
            continue
        print(f"  {base} {cfg}: mentions flyby or crack")

        def visit(offset, path):
            node = rap.body(offset)
            for k, v in node["values"].items():
                if re.search(r"(?i)flyby|bulletfly|soniccrack|supersoniccrack|soundset.*fly|soundset.*crack", k + " " + str(v)):
                    print(f"     {path} : {node['_parent']}   {k} = {v}")
            for name, off in node["classes"].items():
                if len(path) < 200:
                    visit(off, path + "/" + name)
        visit(16, "")

f, entries = pbo_entries(ADDONS + r"\sounds_weapons.pbo")
hits = {n: e for n, e in entries.items() if re.search(r"(?i)projectiles[\\/](bullet_flyby|sonic_crack)", n)}
print(f"\nsounds_weapons.pbo: {len(entries)} files, {len(hits)} projectile samples")
for n, (offset, size, mime, orig) in sorted(hits.items()):
    f.seek(offset)
    data = f.read(size)
    if mime == 0x43707273 and orig:
        data = lzss(data, orig)
    name = os.path.basename(n.replace("\\", "/"))
    with open(os.path.join(OUT, name), "wb") as o:
        o.write(data)
    print(f"  {name:<36} {len(data):8d} bytes, starts {data[:4]!r}")

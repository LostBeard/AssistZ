"""
Packs Lab/mod/AssistZLab into <out>/@AssistZLab/addons/AssistZLab.pbo (uncompressed PBO: version header with a
'prefix' property, one entry header per file, the end sentinel, the file data, then a zero byte and the SHA-1 of
everything before it). Unsigned: the lab server runs verifySignatures = 0.
Usage: python build_mod.py <out_mods_dir>
"""
import hashlib, os, struct, sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AssistZLab")
out_root = os.path.join(sys.argv[1], "@AssistZLab")
addons = os.path.join(out_root, "addons")
os.makedirs(addons, exist_ok=True)

def cstr(s):
    return s.encode("ascii") + b"\0"

def meta(packing, original, size):
    return struct.pack("<5I", packing, original, 0, 0, size)

files = []
for root, _, names in os.walk(SRC):
    for n in sorted(names):
        full = os.path.join(root, n)
        files.append((os.path.relpath(full, SRC).replace("/", "\\"), open(full, "rb").read()))
files.sort(key=lambda f: f[0].lower())

body = bytearray()
body += cstr("") + meta(0x56657273, 0, 0)           # "sreV" version header
body += cstr("prefix") + cstr("AssistZLab") + cstr("product") + cstr("dayz") + cstr("")
for name, data in files:
    body += cstr(name) + meta(0, len(data), len(data))
body += cstr("") + meta(0, 0, 0)                     # end of headers
for _, data in files:
    body += data
body += b"\0" + hashlib.sha1(body).digest()

pbo = os.path.join(addons, "AssistZLab.pbo")
open(pbo, "wb").write(body)
open(os.path.join(out_root, "meta.cpp"), "w").write('protocol = 1;\npublishedid = 0;\nname = "AssistZLab";\ntimestamp = 0;\n')
open(os.path.join(out_root, "mod.cpp"), "w").write('name = "AssistZLab";\nauthor = "SpawnDev";\n')
print(f"packed {len(files)} files -> {pbo} ({len(body)} bytes)")
for name, data in files:
    print("  ", name, len(data))

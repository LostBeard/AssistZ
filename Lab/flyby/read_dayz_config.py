"""
Read-only: parse a binarized DayZ config.bin out of a PBO and print the classes whose names match. Nothing in the game folder is written.
Usage: python -I read_dayz_config.py <pbo> <regex for class names> [max depth to print]
"""
import re, struct, sys


def pbo_entries(path):
    f = open(path, "rb")
    entries = []
    while True:
        name = b""
        while (c := f.read(1)) != b"\0":
            name += c
        mime, orig, res, stamp, size = struct.unpack("<5I", f.read(20))
        if name == b"" and mime == 0x56657273:  # 'Vers' header with properties
            while True:
                key = b""
                while (c := f.read(1)) != b"\0":
                    key += c
                if not key:
                    break
                while f.read(1) != b"\0":
                    pass
            continue
        if name == b"":
            break
        entries.append((name.decode("latin1"), mime, orig, size))
    offset = f.tell()
    out = {}
    for name, mime, orig, size in entries:
        out[name] = (offset, size, mime, orig)
        offset += size
    return f, out


def lzss(data, out_size):
    out = bytearray()
    i = 0
    while len(out) < out_size and i < len(data):
        flags = data[i]
        i += 1
        for bit in range(8):
            if len(out) >= out_size or i >= len(data):
                break
            if flags & (1 << bit):
                out.append(data[i])
                i += 1
            else:
                a, b = data[i], data[i + 1]
                i += 2
                rel = len(out) - (a + ((b & 0xF0) << 4))
                count = (b & 0x0F) + 3
                for k in range(count):
                    p = rel + k
                    out.append(out[p] if p >= 0 else 0x20)
    return bytes(out)


class Rap:
    def __init__(self, data):
        assert data[:4] == b"\0raP", "not a rapified config"
        self.d = data

    def asciiz(self, p):
        e = self.d.index(b"\0", p)
        return self.d[p:e].decode("latin1"), e + 1

    def cint(self, p):
        v = shift = 0
        while True:
            b = self.d[p]
            p += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return v, p

    def array(self, p):
        n, p = self.cint(p)
        out = []
        for _ in range(n):
            t = self.d[p]
            p += 1
            if t in (0, 4):
                s, p = self.asciiz(p)
                out.append(s)
            elif t == 1:
                out.append(round(struct.unpack_from("<f", self.d, p)[0], 5))
                p += 4
            elif t == 2:
                out.append(struct.unpack_from("<i", self.d, p)[0])
                p += 4
            elif t == 3:
                sub, p = self.array(p)
                out.append(sub)
            else:
                raise ValueError(t)
        return out, p

    def body(self, offset):
        parent, p = self.asciiz(offset)
        n, p = self.cint(p)
        node = dict(_parent=parent, values={}, classes={})
        for _ in range(n):
            t = self.d[p]
            p += 1
            if t == 0:
                name, p = self.asciiz(p)
                off = struct.unpack_from("<I", self.d, p)[0]
                p += 4
                node["classes"][name] = off
            elif t == 1:
                sub = self.d[p]
                p += 1
                name, p = self.asciiz(p)
                if sub in (0, 4):
                    v, p = self.asciiz(p)
                elif sub == 1:
                    v = round(struct.unpack_from("<f", self.d, p)[0], 5)
                    p += 4
                else:
                    v = struct.unpack_from("<i", self.d, p)[0]
                    p += 4
                node["values"][name] = v
            elif t == 2:
                name, p = self.asciiz(p)
                v, p = self.array(p)
                node["values"][name] = v
            elif t in (3, 4):
                _, p = self.asciiz(p)
            elif t == 5:
                p += 4
                name, p = self.asciiz(p)
                v, p = self.array(p)
                node["values"][name + "[+]"] = v
            else:
                raise ValueError(f"entry type {t} at {p}")
        return node


def walk(rap, offset, path, pattern, depth, out):
    node = rap.body(offset)
    for name, off in node["classes"].items():
        full = path + "/" + name
        child = rap.body(off)
        if pattern.search(name):
            out.append((full, child))
        if depth > 0:
            walk(rap, off, full, pattern, depth - 1, out)


def load(pbo):
    f, entries = pbo_entries(pbo)
    configs = [n for n in entries if n.lower().endswith("config.bin")]
    res = []
    for n in configs:
        offset, size, mime, orig = entries[n]
        f.seek(offset)
        data = f.read(size)
        if mime == 0x43707273 and orig:
            data = lzss(data, orig)
        res.append((n, Rap(data)))
    return res


if __name__ == "__main__":
    pbo, regex = sys.argv[1], re.compile(sys.argv[2], re.I)
    for name, rap in load(pbo):
        out = []
        walk(rap, 16, "", regex, int(sys.argv[3]) if len(sys.argv) > 3 else 3, out)
        print(f"{name}: {len(out)} classes match")
        for full, node in out[:60]:
            print(f"  {full} : {node['_parent']}")
            for k, v in node["values"].items():
                print(f"      {k} = {v}")

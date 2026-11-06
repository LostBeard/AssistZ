"""
Data for the open-set sound probe: the game's sound bank mapped to classes, split by whole folder into train, validation and test; and the lab's real recordings as fit_sound_probe.py labels them, all for training. Writes open_lab.npz once. Usage: open_data.py <game samples dir> (builds the lab part)
"""
import hashlib, os, re, sys
import numpy as np
BANK = "L:/AssistZLabWork/bank"
FLY = "D:/users/tj/Projects/AssistZ/AssistZ/Lab/flyby"
for p in (FLY, "D:/users/tj/Projects/AssistZ/AssistZ/Lab", os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
from bank_lib import group, load

# What each of the game's folders counts as. None: left out of training (sounds that belong to two kinds).
MAP = {
    "gunshot": "gunfire", "gunshot silenced": "gunfire", "gun tail": "gunfire",
    "gun handling": "gun_handling", "other weapon": None, "gun unclear": None, "explosion": "explosion",
    "footsteps": "footsteps", "crawling": "footsteps", "body fall": "player_action", "clothes and kit": "player_action",
    "ladder": "player_action", "swimming": "water", "other movement": "player_action", "voice": "voice", "player actions": "player_action",
    "infected voice": "infected_voice", "infected movement": "footsteps", "animal voice": "animal", "animal movement": "footsteps",
    "thunder": "thunder", "rain and wind": "weather", "artillery": "gas_strike", "heli crash": "heli_crash", "doors and buildings": "door",
    "birds": "birds", "ambience": "ambience", "water": "water", "spooky area": "ambience", "events": "ambience", "world objects": "world_object",
    "vehicles": "vehicle", "music and tones": "music", "melee hit": "player_action", "bullet hit": "world_object", "shell case": "player_action", "bullet flyby": "gunfire",
}
# Never held out: too few clips or too few folders to spare.
KEEP_IN_TRAIN = {"artillery", "heli crash", "thunder", "events", "swimming"}
CLASSES = ["other", "gunfire", "gas_strike", "heli_crash", "thunder", "infected_attack", "own_move", "own_shot", "gun_handling", "explosion", "footsteps", "voice", "player_action", "infected_voice", "animal", "weather", "door", "birds", "ambience", "water", "world_object", "vehicle", "music"]
# Sounds that may honestly be called gunfire, for scoring false gunfire.
GUNLIKE = {"gunshot", "gunshot silenced", "gun tail", "explosion", "other weapon", "gun unclear", "bullet flyby"}


def kind_of(path):
    """A whole kind of sound: the file's folder, and its name without take numbers. Holding these out means new kinds, not new takes; a folder like infected/VoiceFX holds many kinds."""
    parts = path.lower().split("/")
    stem = re.sub(r"(_?\d+[a-z]?)*\.ogg$", "", parts[-1])
    stem = re.sub(r"_?(close|dist|distant|mid|near|far|int|ext|interior|exterior|1st|3rd|tail|tails)\b", "", stem)
    return "/".join(parts[:-1]) + "/" + stem


def split(g, sub, path):
    """Whole kinds of sound: 60% train, 20% validation, 20% test, stable across runs. Guns are held out as whole guns."""
    if g in KEEP_IN_TRAIN:
        return "train"
    h = int(hashlib.sha1(kind_of(path).encode()).hexdigest()[:8], 16) % 10
    # Guns are held out as whole guns, across every group that gun's sounds appear in.
    if g in ("gunshot", "gunshot silenced", "gun tail", "gun handling", "gun unclear"):
        h = int(hashlib.sha1(f"gun|{sub}".encode()).hexdigest()[:8], 16) % 10
    return "val" if h in (6, 7) else "test" if h in (8, 9) else "train"


def bank():
    rows, emb = load()
    groups = [group(r) for r in rows]
    y = np.array([CLASSES.index(MAP[g]) if MAP.get(g) else -1 for g, _ in groups])
    sp = np.array([split(g, s, r["path"]) for (g, s), r in zip(groups, rows)])
    names = np.array([g for g, _ in groups])
    subs = np.array([s for _, s in groups])
    return rows, emb, y, sp, names, subs


if __name__ == "__main__":
    samples = sys.argv[1]
    os.chdir(FLY)
    import multiprocessing as mp
    from world_classes import GATE, WORLD, STORM, ATTACK, SURVEY, GUNSTORM, FAR, NEWGUNS, OWN, U, gun_rows, world_rows, storm_rows, attack_rows, survey_rows, gunstorm_rows, init, strike, triggers
    from own_cv import own_rows
    with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
        located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
    strikes = {(n, t, d): b for n, t, d, b in located}
    d = [r for n in GATE for r in gun_rows(n, "_emb")] + [r for n in WORLD for r in world_rows(n, strikes)] + [r for n in STORM for r in storm_rows(n)] + [r for n in ATTACK for r in attack_rows(n)] + [r for n in SURVEY for r in survey_rows(n)] + [r for n in GUNSTORM for r in gunstorm_rows(n)] + [r for n in FAR for r in gunstorm_rows(n, "_far")] + [r for n in NEWGUNS for r in gunstorm_rows(n, "_w")] + [r for n in OWN for r in own_rows(U + n)]
    X = np.stack([r["x"] for r in d]).astype(np.float32)
    y = np.array([r["y"] for r in d])
    out = os.path.join(BANK, "open_lab.npz")
    np.savez(out, X=X, y=y)
    print(len(y), {c: int((y == c).sum()) for c in sorted(set(y.tolist()))})

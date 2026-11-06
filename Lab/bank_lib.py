"""The sound bank's clips, embeddings and groups, for scoring and fitting probes. Groups come from the game's own folders and file names. Stays local."""
import csv, glob, os, re
import numpy as np
BANK = "L:/AssistZLabWork/bank"
LEVELS = (-6, -24)
# A gun's own mechanism, not a shot. Their echoes (dry_tail_interior) are handling too.
HANDLING = re.compile(r"dry|jam|reload|rld|closure|mech|charge|chamber|mag_|clip|hand_|pull|cycl|uncock|load|empty|bolt|cock|open|weapon_down|bullet|shell|arrow|string|grenate|pressure")


def gun_part(path, name, gun):
    """(group, gun) of a file in a gun's folder, from its name. A tail is the echo of a real shot, so it counts as one; the first rule here sent 1,300 of them to handling."""
    folders = [x.lower() for x in path.split("/")[:-1]]
    if "handling" in folders or HANDLING.search(name):
        return ("gun handling", gun)
    if "shot" in name:
        return ("gunshot silenced" if "silenced" in name else "gunshot", gun)
    if "tail" in name or "slapback" in name:
        return ("gun tail", gun)
    if re.search(r"close_?\d*\.ogg$", name):
        return ("gun handling", gun)
    return ("gun unclear", gun)


def load():
    rows = list(csv.DictReader(open(os.path.join(BANK, "manifest.csv"), encoding="utf-8")))
    emb = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(BANK, "emb_*.npy")))], axis=0)
    assert len(rows) == len(emb), (len(rows), emb.shape)
    return rows, emb


def group(r):
    """(group, subgroup) of a clip: what a player would call the sound."""
    bank, path = r["bank"][7:-4], r["path"]
    p = path.split("/")
    name = p[-1].lower()
    head = p[0].lower()
    two = "/".join(p[:2]).lower()
    if bank == "weapons":
        if head in ("firearms", "shotguns"):
            return gun_part(path, name, p[1])
        if head == "launchers":
            return gun_part(path, name, "launcher")
        if head in ("crossbow", "nonlethal"):
            part = gun_part(path, name, head)
            # Quiet shots: not gunfire to teach, not a false alarm to count.
            return ("other weapon", head) if part[0] != "gun handling" else part
        if head == "generic":
            return ("gun handling", "generic")
        if head == "tails":
            return ("gun tail", p[1] if len(p) > 2 else "tails")
        if head in ("grenades", "landmine", "claymoremine", "plastic", "gasstationexplosion", "gascanisterexplosion"):
            # Pins, hiss and toys in the same folders are not blasts.
            if re.search(r"explosion|detonation|flashbang|tail", name):
                return ("explosion", head)
            return ("player actions", head)
        if head in ("meleehits", "hits", "shells", "projectiles"):
            return ({"meleehits": "melee hit", "hits": "bullet hit", "shells": "shell case", "projectiles": "bullet flyby"}[head], p[1] if len(p) > 2 else head)
        return ("other weapon", head)
    if bank == "characters":
        if head == "movement":
            sub = p[1].lower()
            if sub == "footstep":
                return ("footsteps", p[2].lower() if len(p) > 3 else "footstep")
            if sub in ("handstep", "prone"):
                return ("crawling", p[2].lower() if len(p) > 3 else sub)
            if sub == "bodyfall":
                return ("body fall", p[2].lower() if len(p) > 3 else sub)
            if sub == "attachment":
                return ("clothes and kit", p[2].lower() if len(p) > 3 else sub)
            if sub == "ladder":
                return ("ladder", sub)
            if sub in ("swim", "swimming", "water"):
                return ("swimming", sub)
            return ("other movement", sub)
        if head == "voice":
            return ("voice", p[1].lower().split("_")[0])
        if head in ("actions", "crafting", "gestures", "misc", "throw", "attacks", "impacts", "injuries", "unconscious"):
            return ("player actions", two)
        return ("player actions", head)
    if bank in ("ai", "bliss") and (head == "infected" or two.startswith("ai/infected")):
        return ("infected voice" if "voice" in two.lower() else "infected movement", two)
    if bank == "ai" or two.startswith("ai/animals"):
        return ("animal movement" if "movement" in two else "animal voice", "/".join(p[:3]).lower())
    if bank in ("environment", "bliss"):
        if head == "environment":
            head = p[1].lower()
            two = "/".join(p[1:3]).lower()
        if head == "weather":
            return ("thunder", head) if "thunder" in path.lower() else ("rain and wind", two)
        if head == "artillery":
            return ("artillery", head)
        if head == "helicrash":
            return ("heli crash", head)
        if head == "buildings":
            return ("doors and buildings", two)
        if head == "birds":
            return ("birds", two)
        if head in ("ambients", "insect", "insects", "water", "spookyarea", "events", "objects"):
            return ({"ambients": "ambience", "insect": "ambience", "insects": "ambience", "water": "water", "spookyarea": "spooky area", "events": "events", "objects": "world objects"}[head], two)
        return ("ambience", head)
    if bank == "vehicles":
        return ("vehicles", head)
    if bank in ("music", "hpp"):
        return ("music and tones", head)
    return ("unknown", two)

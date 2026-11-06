"""The sound bank's clips, embeddings and groups, for scoring and fitting probes. Groups come from the game's own folders and file names. Stays local."""
import csv, glob, os, re
import numpy as np
BANK = "L:/AssistZLabWork/bank"
LEVELS = (-6, -24)


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
        if head == "firearms":
            gun = p[1]
            if re.search(r"shot", name) and "tail" not in name:
                silenced = "silenced" in name
                return ("gunshot silenced" if silenced else "gunshot", gun)
            if "tail" in name:
                return ("gun tail", gun)
            return ("gun handling", gun)
        if head == "shotguns":
            return ("gunshot", p[1]) if "shot" in name else ("gun handling", p[1])
        if head == "tails":
            return ("gun tail", p[1] if len(p) > 2 else "tails")
        if head in ("launchers", "crossbow", "nonlethal"):
            return ("other weapon", head)
        if head in ("grenades", "landmine", "claymoremine", "plastic", "gasstationexplosion", "gascanisterexplosion", "generic"):
            return ("explosion", head)
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

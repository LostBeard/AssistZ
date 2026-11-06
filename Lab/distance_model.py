"""The shipped distance probe in Python, the same arithmetic as AssistZ/Audio/DistanceProbe.cs, for checking the app against it."""
import json
import numpy as np


def load(path):
    probe = json.load(open(path, encoding="ascii"))

    def value(part, q):
        return ((np.asarray(q, dtype=np.float64) - np.array(part["Mean"])) / np.array(part["Std"])) @ np.array(part["Weights"]) + part["Intercept"]

    def estimate(q):
        if value(probe["FarGate"], q) > 0:
            return max(probe["CutMeters"], 2 ** value(probe["Far"], q))
        return 2 ** value(probe["Near"], q)

    return estimate

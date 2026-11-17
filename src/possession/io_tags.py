"""Smart Tagging XML -> frame intervals (RED tier when applied to BAR-ATH: return only to code, print nothing)."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

from .paths import ROOT

sys.path.insert(0, ROOT)
import games  # noqa: E402


def load(data_dir=None, fps=25):
    path = games.require(["smart_tagging"], games.GAME, data_dir)[0]
    out = []
    for ins in ET.parse(path).iter("instance"):
        s, e = float(ins.findtext("start")), float(ins.findtext("end"))
        out.append({"code": ins.findtext("code"), "start_frame": int(round(s * fps)), "end_frame": int(round(e * fps)),
                    "tags": {lab.findtext("group"): lab.findtext("text") for lab in ins.findall("label")}})
    return out

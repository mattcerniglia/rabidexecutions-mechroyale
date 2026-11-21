"""Silent camper: never makes a footstep/gun sound unless it has to, stays ahead of the zone, faces whoever hits it."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechfight as mf

PREFER = ["armor", "plating", "nanites", "servos", "capacitor", "reactor", "sensors"]

class Bot:
    name = "stealth-camper"
    def __init__(self, hello):
        c = hello["config"]
        self.silent = 0.95 * c["sound"]["step_speed"] / c["mech"]["walk_speed"]   # move-vector length that stays under step_speed
        self.reach = c["loot"]["pickup_radius"] - 0.5
        self.threat = None            # (bearing, t)
        self.mode = "init"

    def act(self, obs):
        me, t = obs["self"], obs["t"]; x, y, z = me["pos"]; zone = obs["zone"]
        act = {"tick": obs["tick"]}
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None: self.threat = (e["bearing"], t)
        if obs["offer"]:
            ids = [ch["id"] for ch in obs["offer"]["choices"]]
            act["pick"] = next((ids.index(p) for p in PREFER if p in ids), 0)
        enemies = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"], me["pos"]))
        # 1) where to go
        gx, gy = zone["next_center"]; d = math.hypot(gx - x, gy - y); mode = "camp"
        mag = 0.0
        if not me["in_zone"]: mode, mag = "EMERGENCY (outside zone)", 1.0          # noise is cheaper than zone dps
        elif d > 0.55 * max(zone["next_radius"], 1): mode, mag = "silent-walk", self.silent
        # 2) loot a lightning/laser pickup if close and nobody is near
        want = [p for p in obs["pickups"] if p["kind"] in ("lightning", "laser") and math.dist(p["pos"][:2], (x, y)) < 25]
        if me["weapons"][1] is None and want and not (enemies and math.dist(enemies[0]["pos"], me["pos"]) < 60):
            p = min(want, key=lambda q: math.dist(q["pos"][:2], (x, y))); gx, gy = p["pos"][:2]; mode, mag = "loot", max(mag, self.silent)
            if math.dist(p["pos"][:2], (x, y)) < self.reach and abs(p["pos"][2] - z) < 3: act["take"] = p["id"]; mag = 0.0
        yaw = math.atan2(gy - y, gx - x)
        act["move"] = [mag * math.cos(yaw), mag * math.sin(yaw)]; act["boost"] = False
        # 3) facing: toward whoever hit us (denies the x1.5 rear-armor multiplier), else look where we walk
        aim = yaw
        if self.threat and t - self.threat[1] < 3.0: aim, mode = self.threat[0], mode + " +face-attacker"
        # 4) self-defense only: shoot back when we were hit in the last 2 s and the shooter is visible
        if enemies and self.threat and t - self.threat[1] < 2.0:
            e = enemies[0]; dist = math.dist(e["pos"], me["pos"]); lead = dist / me["weapons"][0]["stats"]["speed"]
            tx, ty, tz = (e["pos"][k] + e["vel"][k] * lead for k in range(3))
            aim = math.atan2(ty - y, tx - x)
            act["aim"] = [aim, math.atan2(tz + 4.5 - (z + 7), math.hypot(tx - x, ty - y))]
            lt = me["weapons"][1] and me["weapons"][1]["kind"] == "lightning"
            act["fire"] = [True, bool(lt and dist < 40)]; mode += " +RETURN FIRE"
        else: act["aim"] = [aim, 0.0]
        self.mode = mode; mf.draw.label(mode); mf.draw.point([zone["next_center"][0], zone["next_center"][1], z], "#40c0ff", 6)
        return act

if __name__ == "__main__": mf.run(Bot)

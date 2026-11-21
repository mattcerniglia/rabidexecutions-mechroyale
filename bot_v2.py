"""v2: pathfinds around buildings, stays silent until engaged, loots lightning/laser, fights in lightning range, faces attackers."""
import math, os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechfight as mf

try:
    from nav import Nav
except Exception:                       # numpy missing etc.: fall back to straight lines
    Nav = None

PREFER = ["armor", "plating", "lightning_efficiency", "nanites", "lightning_range", "servos", "capacitor", "gun_damage", "gun_mag", "reactor", "gyros", "sensors"]

def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

class Bot:
    name = "rabid-v2"

    def __init__(self, hello):
        c = hello["config"]; self.step_speed = c["sound"]["step_speed"]; self.reach = c["loot"]["pickup_radius"] - 0.5
        self.nav = None
        if Nav:
            try: self.nav = Nav(hello["map"]["dir"])
            except Exception as e: mf.log("nav init failed:", repr(e))
        self.field = None; self.field_key = None; self.want_key = None; self.busy = False
        self.threat = None; self.hist = []; self.unstick_until = -1; self.unstick_sign = 1; self.black = {}; self.mode = "init"
        sys.setswitchinterval(0.002)

    # ---- navigation -------------------------------------------------------------------------------------------
    def _compute(self, key):
        try:
            f = self.nav.field(key[0], key[1], key[2])
            self.field, self.field_key = f, key
        except Exception as e: mf.log("field failed:", repr(e))
        self.busy = False

    def goal(self, zone):
        gx, gy = zone["next_center"]; r = zone["next_radius"]; return (round(gx), round(gy), 6 if r <= 0 else max(8, round(0.45 * r)))

    def route(self, x, y, zone):
        """-> (target_xy or None, remaining_m, arrived)"""
        key = self.goal(zone)
        if self.nav and key != self.field_key and not self.busy:
            self.busy = True; threading.Thread(target=self._compute, args=(key,), daemon=True).start()
        if self.nav and self.field is not None and self.field_key == key:
            r = self.nav.follow(self.field, x, y)
            if r: return r[0], r[1], r[1] <= 0
        gx, gy = zone["next_center"]; d = math.hypot(gx - x, gy - y)
        return (gx, gy), d, d <= 0.45 * max(zone["next_radius"], 1)

    # ---- per tick ---------------------------------------------------------------------------------------------
    def act(self, obs):
        me, t, tick, zone = obs["self"], obs["t"], obs["tick"], obs["zone"]; x, y, z = me["pos"]; st = me["stats"]
        act = {"tick": tick, "boost": False}
        walk = st["walk_speed"]; silent = 0.95 * self.step_speed / walk
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None: self.threat = (e["bearing"], t, e.get("by"))
        if obs["offer"]:
            ids = [c["id"] for c in obs["offer"]["choices"]]; act["pick"] = next((ids.index(p) for p in PREFER if p in ids), 0)
        slots = {(w or {}).get("kind"): i for i, w in enumerate(me["weapons"]) if w}
        lt_i, ls_i = slots.get("lightning"), slots.get("laser")
        enemies = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"], me["pos"]))
        recent = self.threat is not None and t - self.threat[1] < 3.0

        # 1) zone route
        tgt, remaining, arrived = self.route(x, y, zone); mag = 0.0; mode = "camp"
        tleft = max(1.0, zone["shrink_end"] - t) if t < zone["shrink_end"] else 1.0
        if not me["in_zone"]:
            mode, mag = "OUT OF ZONE", 1.0
            if zone["dps"] >= 60 and me["energy"] > 25: act["boost"] = True
        elif not arrived:
            edge = max(0.0, remaining - max(0.0, zone["next_radius"] - self.goal(zone)[2])); need = edge / tleft / walk; mode = "walk"; mag = silent if need < silent * 0.9 else min(1.0, need * 1.3)
            if mag > silent: mode = "hurry"
        goal_xy = tgt

        # 2) opportunistic loot (only with time to spare and nobody close)
        near_enemy = enemies and math.dist(enemies[0]["pos"], me["pos"]) < 70
        if me["in_zone"] and mag <= silent and not near_enemy:
            want = []
            if me["weapons"][1] is None:
                want += [(0 if p["kind"] == "lightning" else 1, p) for p in obs["pickups"] if p["kind"] in ("lightning", "laser") and self.black.get(p["id"], -1) < t and math.dist(p["pos"][:2], (x, y)) < 70]
            want += [(2, c) for c in obs["cubes"] if me["cubes"] < 10 and self.black.get(c["id"], -1) < t and math.dist(c["pos"][:2], (x, y)) < 14]
            if want:
                _, p = min(want, key=lambda w: (w[0], math.dist(w[1]["pos"][:2], (x, y)))); d = math.dist(p["pos"][:2], (x, y))
                goal_xy, mag, mode = p["pos"][:2], max(mag, silent), "loot"
                if "kind" in p and d < self.reach and abs(p["pos"][2] - z) < 3: act["take"] = p["id"]; mag = 0.0
                self.cur_loot = p["id"]
        # 3) combat
        aim, fire = None, [False, False]; move_override = None
        e = enemies[0] if enemies else None
        if e:
            d = math.dist(e["pos"], me["pos"]); lt_range = me["weapons"][lt_i]["stats"]["range"] if lt_i is not None else 0
            engage = (d <= 42 and (lt_i is not None or me["hp"] >= e.get("hp", 1e9) * 0.9)) or (recent and d <= 150)
            if engage:
                lead = d / me["weapons"][0]["stats"]["speed"] if me["weapons"][0] else 0
                tx, ty, tz = (e["pos"][k] + e["vel"][k] * lead for k in range(3)); yaw = math.atan2(ty - y, tx - x)
                aim = [yaw, math.atan2(tz + 4.5 - (z + st["muzzle_height"]), math.hypot(tx - x, ty - y))]
                if me["weapons"][0] and d <= 200: fire[0] = True
                if lt_i is not None and d <= lt_range * 0.95 and me["energy"] >= 4: fire[lt_i] = True
                if ls_i is not None:
                    w = me["weapons"][ls_i]; fire[ls_i] = not (w["cooldown"] > 0 or (w.get("charge", 0) >= 0.7 and abs(angdiff(me["aim"][0], yaw)) < 0.06))
                    if w["cooldown"] > 0: fire[ls_i] = False
                side = 1 if (tick // 12) % 2 == 0 else -1; rad = max(-1.0, min(1.0, (d - 30) / 30)) * 0.6 if lt_i is not None else 0.0
                move_override = [0.8 * math.cos(yaw + side * math.pi / 2) + rad * math.cos(yaw), 0.8 * math.sin(yaw + side * math.pi / 2) + rad * math.sin(yaw)]
                mode = "FIGHT d=%.0f" % d
        elif recent:
            aim = [self.threat[0], 0.0]; mag = max(mag, 1.0) if mag > 0 else 0.0; mode += " +face-bearing"

        # 4) movement vector
        if move_override and me["in_zone"]: mv = move_override
        elif goal_xy is not None and mag > 0:
            yaw = math.atan2(goal_xy[1] - y, goal_xy[0] - x); mv = [mag * math.cos(yaw), mag * math.sin(yaw)]
        else: mv = [0.0, 0.0]
        # 5) stuck detection: wanted to move, barely moved for 3 s
        self.hist.append((t, x, y, math.hypot(*mv) > 0.2)); self.hist = [h for h in self.hist if t - h[0] <= 3.0]
        if tick >= self.unstick_until and len(self.hist) >= 25 and all(h[3] for h in self.hist) and math.hypot(x - self.hist[0][1], y - self.hist[0][2]) < 1.5:
            self.unstick_until = tick + 16; self.unstick_sign = -self.unstick_sign
            if mode == "loot": self.black[getattr(self, "cur_loot", -1)] = t + 60
        if tick < self.unstick_until:
            yaw = math.atan2(mv[1], mv[0]) if math.hypot(*mv) > 0.01 else 0.0; yaw += self.unstick_sign * math.pi / 2
            mv = [math.cos(yaw), math.sin(yaw)]; act["jump"] = tick % 4 == 0; act["boost"] = not me["grounded"]; mode += " UNSTICK"
        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        if aim is None:
            h = math.atan2(mv[1], mv[0]) if math.hypot(*mv) > 0.05 else 1.2 * math.sin(t * 0.6)
            aim = [h if math.hypot(*mv) > 0.05 else h, 0.0]
        act["aim"] = aim; act["fire"] = fire
        self.mode = mode; mf.draw.label(mode)
        if goal_xy: mf.draw.point([goal_xy[0], goal_xy[1], z], "#40c0ff", 6)
        return act

if __name__ == "__main__": mf.run(Bot)

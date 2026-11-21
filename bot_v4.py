"""rabid v4 - built for the practice format: the house squad hunts you one bot at a time (more as you kill them),
so score ~= house bots that die before you. Win duels, bank cubes, heal between fights, never die to the zone.

  Memory    enemy tracks (with velocity history) + sound / damage cues that decay with time
  Aim       projectile intercept with strafe-damped lead; fire only when the torso's ACTUAL aim is on target;
            line of fire checked against the real map (buildings/terrain; trees do not stop shots)
  Legs      zone routing on goal fields, seeded random strafing, cover-scored retreat, farming NPCs/crates/cubes
  Selector  priority rules choose legs and torso independently every decision
Deterministic apart from wall-clock nav timing: all randomness comes from Random(hello.seed).
"""
import gc, math, os, random, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechfight as mf

try:
    from nav import Nav
except Exception:                       # numpy missing etc.: straight-line fallback
    Nav = None

TAU, PI = 2 * math.pi, math.pi


def wrap(a): return (a + PI) % TAU - PI
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


UPGRADES = ["armor", "plating", "lightning_range", "nanites", "gun_rate", "gun_damage", "lightning_efficiency",
            "laser_damage", "laser_charge", "gun_mag", "lightning_chain", "servos", "gun_velocity", "reactor",
            "capacitor", "jets", "gyros", "sensors", "laser_pierce", "mortar_reload", "mortar_splash", "mortar_velocity"]
CUE_W = {"step": 1.0, "boost": 1.0, "jet": 0.9, "reload": 0.8, "laser_charge": 1.0, "gun": 0.6, "laser": 0.7,
         "lightning": 0.8, "mortar": 0.5}
DIST_EST = {"near": 35.0, "mid": 110.0, "far": 230.0}
FIELD_POPS = 3000          # Dijkstra pops per decision (~6 ms here); a full field takes ~5 s of match time


def intercept(sx, sy, sz, px, py, pz, vx, vy, vz, speed, skip=3.0):
    """Aim point for a projectile of `speed` leaving `skip` m out from (sx,sy,sz) at a target moving linearly."""
    t = 0.0; qx, qy, qz = px, py, pz
    for _ in range(3):
        qx, qy, qz = px + vx * t, py + vy * t, pz + vz * t
        t = max(0.0, math.sqrt((qx - sx) ** 2 + (qy - sy) ** 2 + (qz - sz) ** 2) - skip) / speed
    return qx, qy, qz, t


class Memory:
    """Belief about enemy mechs. The server keeps reporting a seen mech for memory_s; beyond that we decay."""

    def __init__(self):
        self.tracks = {}
        self.cues = []

    def update(self, obs, x, y):
        t = obs["t"]
        for m in obs["mechs"]:
            if m["kind"] != "mech": continue
            tr = self.tracks.get(m["id"])
            if tr is None: tr = self.tracks[m["id"]] = {"vh": []}
            tr["pos"] = m["pos"]; tr["vel"] = m["vel"]; tr["t"] = t
            vh = tr["vh"]; vh.append((t, m["vel"][0], m["vel"][1]))
            while vh and t - vh[0][0] > 1.0: vh.pop(0)
        for s in obs["sounds"]:
            w = CUE_W.get(s["kind"])
            if w is None or s.get("bearing") is None: continue
            d = DIST_EST.get(s.get("dist"), 120.0)
            self.cues.append((t - s.get("age", 0.0), x + d * math.cos(s["bearing"]), y + d * math.sin(s["bearing"]), w))
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None and e.get("weapon") not in ("npc", "zone"):
                tr = self.tracks.get(e.get("by"))
                if tr is not None and t - tr["t"] < 0.5: continue
                self.cues.append((t, x + 90 * math.cos(e["bearing"]), y + 90 * math.sin(e["bearing"]), 2.5))
            elif e["type"] == "kill":
                self.tracks.pop(e.get("target"), None)
        if len(self.cues) > 160: self.cues = self.cues[-160:]
        self.cues = [c for c in self.cues if t - c[0] < 15.0]
        for k in [k for k, tr in self.tracks.items() if t - tr["t"] > 45.0]: del self.tracks[k]

    def avg_vel(self, mid, vel):
        tr = self.tracks.get(mid)
        if not tr or not tr["vh"]: return vel[0], vel[1]
        vh = tr["vh"]; n = len(vh)
        return sum(v[1] for v in vh) / n, sum(v[2] for v in vh) / n

    def threat_point(self, t, x, y):
        """Most likely enemy position (x, y, weight) from stale tracks and recent cues, or None."""
        best, bw = None, 0.15
        for tr in self.tracks.values():
            age = t - tr["t"]
            if age < 0.15 or age > 30.0: continue
            d = math.hypot(tr["pos"][0] - x, tr["pos"][1] - y)
            w = 3.0 * math.exp(-age / 10.0) * (1.0 if d < 220 else 0.4)
            if w > bw: best, bw = (tr["pos"][0], tr["pos"][1]), w
        for (tc, cx, cy, cw) in self.cues:
            w = cw * math.exp(-(t - tc) / 4.0)
            if w > bw: best, bw = (cx, cy), w
        return (best[0], best[1], bw) if best else None


class Bot:
    name = "rabid-v4"

    def __init__(self, hello):
        c = hello["config"]; M = c["mech"]
        self.rng = random.Random(hello.get("seed") or 0)
        self.map_dir = hello["map"]["dir"]
        self.nav = None; self.nav_loading = False; self.nav_tries = 0
        if Nav:
            try: self.nav = Nav(self.map_dir)
            except Exception as e: mf.log("nav init failed:", repr(e))
        self.base_r, self.base_h = M["radius"], M["height"]
        self.size_per_cube = c["cubes"].get("size_per_cube", 0.05); self.cube_cap = c["cubes"].get("stat_cap", 10)
        self.reach = c["loot"]["pickup_radius"] - 0.7
        cr = c.get("crates", {}); self.crate_r, self.crate_h = cr.get("radius", 1.4), cr.get("height", 2.4)
        npc = c.get("npc", {}); self.npc_r, self.npc_h = npc.get("radius", 1.8), npc.get("height", 3.5)
        self.mortar_g = c.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.step_speed = c["sound"]["step_speed"]
        self.mem = Memory()
        self.field = None; self.field_key = None; self.pending = None; self.center_key = None; self.center_pt = None
        self.side = 1; self.flip_t = 0.0
        self.hist = []; self.unstick_until = -1; self.unstick_sign = 1; self.black = {}
        self.goal = None; self.goal_since = 0.0
        self.target_id = None; self.scan = 0.0
        self.last_dmg_t = self.last_mech_dmg_t = self.last_seen_t = -99.0
        self.npc_angry = {}
        self.retreating = False; self.cover = None; self.cover_t = -99.0
        self.my_id = hello.get("slot")
        sys.setswitchinterval(0.002)
        gc.collect(); gc.freeze()                            # map grids never need scanning by the cyclic GC

    # ---- map helpers --------------------------------------------------------------------------------------------
    def _load_nav(self):
        try: self.nav = Nav(self.map_dir)
        except Exception as e: mf.log("nav retry failed:", repr(e))
        self.nav_loading = False

    def los(self, x0, y0, z0, x1, y1, z1):
        return self.nav.los(x0, y0, z0, x1, y1, z1) if self.nav else True

    def steer(self, x, y, yaw):
        return self.nav.steer(x, y, yaw) if self.nav else yaw

    def advance_field(self, zone):
        """Incremental goal field toward the next circle: a fixed number of Dijkstra pops per decision, no thread."""
        if not self.nav: return
        gx, gy = zone["next_center"]; r = zone["next_radius"]
        key = (round(gx), round(gy), 6 if r <= 0 else max(8, round(0.55 * r)))
        if key != self.field_key and (self.pending is None or self.pending[0] != key):
            try: self.pending = (key, self.nav.field_start(*key))  # setup gets a decision of its own
            except Exception as e: mf.log("field start failed:", repr(e)); self.pending = None
            return
        if self.pending is not None:
            try: done = self.nav.field_step(self.pending[1], FIELD_POPS)
            except Exception as e: mf.log("field step failed:", repr(e)); self.pending = None; return
            if done: self.field, self.field_key = self.pending[1][0], self.pending[0]; self.pending = None

    def route(self, x, y, zone):
        """Next waypoint toward the comfortable part of the next circle."""
        gx, gy = zone["next_center"]; r = zone["next_radius"]
        key = (round(gx), round(gy), 6 if r <= 0 else max(8, round(0.55 * r)))
        if self.nav and self.field is not None:              # a stale field still points into the (nested) new circle
            res = self.nav.follow(self.field, x, y)
            if res and res[1] > 0 and math.hypot(res[0][0] - x, res[0][1] - y) > 0.5: return res[0]
            if res and self.field_key == key:                # inside the goal set: stand on the open cell nearest the centre
                if self.center_key != key: self.center_key, self.center_pt = key, self.nav.nearest_free(gx, gy)
                return self.center_pt
        return (gx, gy)

    def toward(self, x, y, px, py, mag=1.0):
        d = math.hypot(px - x, py - y)
        if d < 0.5: return (0.0, 0.0)
        yaw = math.atan2(py - y, px - x)
        if self.nav and d > 2.0: yaw = self.nav.steer(x, y, yaw, (min(3.5, 0.5 * d), min(7.0, d)))  # never probe past the target
        return (mag * math.cos(yaw), mag * math.sin(yaw))

    # ---- zone ---------------------------------------------------------------------------------------------------
    def zone_status(self, x, y, zone, t, walk):
        cx, cy = zone["center"]; r = zone["radius"]; ncx, ncy = zone["next_center"]; nr = zone["next_radius"]
        s0, s1 = zone["shrink_start"], zone["shrink_end"]
        f = clamp((t - s0) / (s1 - s0), 0.0, 1.0) if s1 > s0 else (1.0 if t >= s1 else 0.0)
        lc = (cx + (ncx - cx) * f, cy + (ncy - cy) * f); lr = r + (nr - r) * f
        margin = clamp(0.25 * nr, 3.0, 35.0) if nr > 0 else 0.0
        dn = math.hypot(x - ncx, y - ncy)
        need = max(0.0, dn - (nr - margin)); tl = max(0.0, s1 - t)
        slack = tl - need / (0.75 * walk) - 6.0
        return {"lc": lc, "lr": lr, "nc": (ncx, ncy), "nr": nr, "margin": margin, "need": need, "tl": tl, "slack": slack,
                "urgent": need > 0 and slack < 0, "mag": clamp(need / max(tl, 1.0) / walk * 1.5, 0.5, 1.0), "dn": dn}

    def zone_ok(self, px, py, zs, walk):
        if math.hypot(px - zs["lc"][0], py - zs["lc"][1]) > zs["lr"] - 6: return False
        over = math.hypot(px - zs["nc"][0], py - zs["nc"][1]) - (zs["nr"] - zs["margin"])
        return over <= max(0.0, zs["slack"]) * walk * 0.3

    # ---- weapons ------------------------------------------------------------------------------------------------
    def engage(self, me, W, slot, e, x, y, mz, mech):
        """Aim + trigger for target e (dict: pos, vel, r, h, d, los, id)."""
        px, py, pz = e["pos"]; vx, vy, vz = e["vel"]; cz = pz + 0.5 * e["h"]; d = max(e["d"], 1.0)
        fire = [False, False]; ax, ay = me["aim"]
        gi, li, ki = slot.get("gun"), slot.get("laser"), slot.get("lightning")
        yaw_d = math.atan2(py - y, px - x); pit_d = math.atan2(cz - mz, math.hypot(px - x, py - y))
        aim = [yaw_d, pit_d]
        tol_y = math.atan2(0.7 * e["r"], d) + 0.004; tol_p = math.atan2(0.35 * e["h"], d) + 0.012
        if gi is not None:
            g = W[gi]; gs = g["stats"]
            if mech:                                         # damp the lead toward the 1 s mean velocity on long flights
                avx, avy = self.mem.avg_vel(e["id"], e["vel"]); w = clamp((d / gs["speed"]) / 0.6, 0.0, 1.0) * 0.5
                lvx, lvy = (1 - w) * vx + w * avx, (1 - w) * vy + w * avy
            else: lvx, lvy = vx, vy
            qx, qy, qz, tf = intercept(x, y, mz, px, py, cz, lvx, lvy, 0.4 * vz, gs["speed"])
            yaw_l = math.atan2(qy - y, qx - x); pit_l = math.atan2(qz - mz, math.hypot(qx - x, qy - y))
            aim = [yaw_l, pit_l]
            on = abs(wrap(ax - yaw_l)) < tol_y and abs(ay - pit_l) < tol_p
            moving = math.hypot(vx, vy) > 3.0
            in_rng = d <= gs["range"] * 0.97 and (d <= 150 or not moving or not mech)
            fire[gi] = on and in_rng and e["los"] and g.get("ammo", 1) > 0
        if li is not None:
            lw = W[li]; ls = lw["stats"]; ch = lw.get("charge") or 0.0
            if lw.get("cooldown", 0) <= 0 and e["los"] and d <= ls["range"] * 0.95:
                if ch >= ls["charge"] * 0.97:                 # charged: hitscan, aim straight, release when on target
                    aim = [yaw_d, pit_d]
                    on = abs(wrap(ax - yaw_d)) < tol_y and abs(ay - pit_d) < tol_p
                    fire[li] = not on
                else: fire[li] = True
            elif ch > 0: fire[li] = True                     # keep holding rather than beam a wall
        if ki is not None:
            kw = W[ki]; ks = kw["stats"]
            if (e["los"] and e["d"] <= ks["range"] * 0.93 and abs(wrap(ax - yaw_d)) < math.radians(ks["cone_deg"]) * 0.4
                    and me["energy"] >= ks["energy"] + (2 if mech else 40)):
                fire[ki] = True
        return aim, fire

    def pick(self, offer):
        ids = [ch["id"] for ch in offer["choices"]]
        return next((ids.index(p) for p in UPGRADES if p in ids), 0)

    # ---- legs ---------------------------------------------------------------------------------------------------
    def strafe(self, x, y, px, py, d, lo, hi, t, hold=False):
        if hold: self.flip_t = max(self.flip_t, t + 0.3)     # a charged laser is on us: keep the lateral run going
        if t >= self.flip_t:
            if self.rng.random() < 0.8: self.side = -self.side
            self.flip_t = t + self.rng.uniform(0.35, 1.0)
        to = math.atan2(py - y, px - x)
        rad = min(1.0, (d - hi) / 15.0) if d > hi else (-min(1.0, (lo - d) / 10.0) if d < lo else 0.0)
        vx = -math.sin(to) * self.side * 0.9 + rad * math.cos(to); vy = math.cos(to) * self.side * 0.9 + rad * math.sin(to)
        yaw = self.steer(x, y, math.atan2(vy, vx)); return (math.cos(yaw), math.sin(yaw))

    def find_cover(self, x, y, z, threats, zs):
        cx = sum(p[0] for p in threats) / len(threats); cy = sum(p[1] for p in threats) / len(threats)
        away = math.atan2(y - cy, x - cx); best, bs, bc = None, -1e9, 0.0
        for k in range(16):
            a = away + k * TAU / 16; ca, sa = math.cos(a), math.sin(a)
            for dd in (12.0, 24.0):
                px, py = x + dd * ca, y + dd * sa
                if self.nav and not (self.nav.free_at(px, py) and self.nav.free_at(x + 0.5 * dd * ca, y + 0.5 * dd * sa)): continue
                gz = self.nav.ground(px, py) if self.nav else z
                cov = sum(1 for (tx, ty, tz) in threats if not self.los(tx, ty, tz + 8.0, px, py, gz + 4.5)) / len(threats)
                sc = 2.5 * cov + math.cos(a - away) - 0.02 * dd
                if math.hypot(px - zs["lc"][0], py - zs["lc"][1]) > zs["lr"] - 4: sc -= 4.0
                if sc > bs: bs, best, bc = sc, (px, py), cov
        return (best or (x + 15 * math.cos(away), y + 15 * math.sin(away))), bc

    def mortar_dodge(self, obs, x, y, z):
        for p in obs["projectiles"]:
            if p["kind"] != "mortar": continue
            px, py, pz = p["pos"]; vx, vy, vz = p["vel"]; g = self.mortar_g
            a, b, c = -0.5 * g, vz, pz - z                   # pz + vz t - g t^2/2 = z
            disc = b * b - 4 * a * c
            if disc < 0: continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl < 0 or tl > 3.0: continue
            lx, ly = px + vx * tl, py + vy * tl
            if math.hypot(lx - x, ly - y) < 10.0:
                yaw = self.steer(x, y, math.atan2(y - ly, x - lx)); return (math.cos(yaw), math.sin(yaw))
        return None

    def farm_target(self, obs, me, x, y, z, zs, slot, walk, t):
        W = me["weapons"]; cands = []
        empty = any(w is None for w in W)
        for p in obs["pickups"]:
            if self.black.get(p["id"], -1) > t or abs(p["pos"][2] - z) > 3.5: continue
            k = p["kind"]
            if k == "lightning" and "lightning" not in slot and (empty or "laser" in slot): pr = 0.4
            elif k == "laser" and empty and "lightning" not in slot and "laser" not in slot: pr = 0.6
            else: continue
            d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
            if d < 160: cands.append((d * pr, "pickup", p))
        if me["cubes"] < self.cube_cap:
            for c in obs["cubes"]:
                if self.black.get(c["id"], -1) > t or abs(c["pos"][2] - z) > 3.5: continue
                d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if d < 110: cands.append((d * 0.6, "cube", c))
        hpf = me["hp"] / max(1.0, me["max_hp"])
        for dr in obs.get("drops") or []:
            if dr.get("holder") not in (None, self.my_id): continue
            px, py = dr["pos"][0], dr["pos"][1]; d = math.hypot(px - x, py - y)
            if d < 260 and dr.get("lands_in", 0) <= d / walk + 4:
                cands.append((d * 0.5 - (60 if hpf < 0.7 else 0) - 20, "drop", dr))
        if "gun" in slot or "lightning" in slot:
            for n in obs["mechs"]:
                if n["kind"] != "npc" or self.black.get(n["id"], -1) > t: continue
                d = math.hypot(n["pos"][0] - x, n["pos"][1] - y)
                if d < 170: cands.append((d + 25 - (60 if t - self.npc_angry.get(n["id"], -99) < 6 else 0), "npc", n))
            for c in obs["crates"]:
                if self.black.get(c["id"], -1) > t or abs(c["pos"][2] - z) > 3.0: continue
                d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if d < 130: cands.append((d + 45, "crate", c))
        cands = [c for c in cands if self.zone_ok(c[2]["pos"][0], c[2]["pos"][1], zs, walk)]
        if not cands: self.goal = None; return None
        cands.sort(key=lambda c: c[0]); best = cands[0]
        if self.goal is not None:
            cur = next((c for c in cands if c[1] == self.goal[0] and c[2]["id"] == self.goal[1]), None)
            if cur is not None and cur[0] < best[0] + 25: best = cur
        if self.goal is None or (best[1], best[2]["id"]) != self.goal: self.goal = (best[1], best[2]["id"]); self.goal_since = t
        elif t - self.goal_since > 30 and best[1] in ("pickup", "cube", "crate", "npc"):
            self.black[best[2]["id"]] = t + 60; self.goal = None; return None
        return best

    # ---- per decision -------------------------------------------------------------------------------------------
    def act(self, obs):
        t, tick, me, zone = obs["t"], obs["tick"], obs["self"], obs["zone"]
        x, y, z = me["pos"]; st = me["stats"]
        if self.nav is None and Nav and not self.nav_loading and self.nav_tries < 5 and tick % 20 == 0:
            self.nav_loading = True; self.nav_tries += 1; threading.Thread(target=self._load_nav, daemon=True).start()
        A = {"tick": tick}
        self.mem.update(obs, x, y)
        for e in obs["events"]:
            if e["type"] == "damaged":
                self.last_dmg_t = t
                if e.get("weapon") == "npc" and e.get("by") is not None: self.npc_angry[e["by"]] = t
                elif e.get("weapon") not in ("zone", None): self.last_mech_dmg_t = t
        if obs.get("offer"): A["pick"] = self.pick(obs["offer"])
        W = me["weapons"]; slot = {w["kind"]: i for i, w in enumerate(W) if w}
        walk = st.get("walk_speed", 11.7); hpf = me["hp"] / max(1.0, me["max_hp"]); energy = me["energy"]
        mz = z + st.get("muzzle_height", 7.0)
        zs = self.zone_status(x, y, zone, t, walk)
        self.advance_field(zone)
        gun = W[slot["gun"]] if "gun" in slot else None; lt = W[slot["lightning"]] if "lightning" in slot else None
        las = W[slot["laser"]] if "laser" in slot else None
        lt_ready = lt is not None and energy >= lt["stats"]["energy"] * 4

        # ---- perception ----
        enemies = []
        for m in obs["mechs"]:
            if m["kind"] != "mech": continue
            s = 1 + self.size_per_cube * m.get("cubes", 0); px, py, pz = m["pos"]
            enemies.append({"m": m, "id": m["id"], "pos": m["pos"], "vel": m["vel"], "r": self.base_r * s, "h": self.base_h * s,
                            "d": math.dist((x, y, z), (px, py, pz)), "los": self.los(x, y, mz, px, py, pz + 0.5 * self.base_h * s)})
        if any(e["los"] for e in enemies): self.last_seen_t = t
        max_rng = max(gun["stats"]["range"] * 0.9 if gun else 0, las["stats"]["range"] * 0.9 if las else 0, lt["stats"]["range"] if lt else 0)
        tgt, best = None, 1e9
        for e in enemies:
            if not e["los"] or e["d"] > max_rng: continue
            sc = e["d"] + 120.0 * e["m"].get("hp", 1400) / max(1.0, e["m"].get("max_hp", 1400)) - (60.0 if e["id"] == self.target_id else 0.0)
            if sc < best: best, tgt = sc, e
        if tgt: self.target_id = tgt["id"]
        threats = [e for e in enemies if e["d"] < 170 and (e["los"] or e["d"] < 60)]
        hidden = [e for e in enemies if not e["los"]]

        # ---- fight or flight ----
        if threats and not self.retreating:
            weakest = min(e["m"].get("hp", 1400) for e in threats)
            n_los = sum(1 for e in threats if e["los"])
            near_los = min((e["d"] for e in threats if e["los"]), default=1e9)
            heal_first = hpf < 0.55 and near_los > 80               # next fight is still coming in: heal behind cover
            if (hpf < 0.32 and weakest > me["hp"] * 0.8) or (n_los >= 2 and hpf < 0.55) or heal_first:
                pts = [(e["pos"][0], e["pos"][1], e["pos"][2]) for e in threats]
                self.cover, cov = self.find_cover(x, y, z, pts, zs); self.cover_t = t
                if cov >= 0.5: self.retreating = True           # running without cover only hands out free shots
        if self.retreating and (hpf > 0.85 or (not threats and t - self.last_dmg_t > 12)): self.retreating = False
        if self.retreating and tgt and lt_ready and tgt["d"] < lt["stats"]["range"] * 0.85 and tgt["m"].get("hp", 1400) < me["hp"] * 1.3:
            self.retreating = False                          # in arc range of a beatable target: finish it
        unseen_hit = tgt is None and t - self.last_mech_dmg_t < 1.5
        tp = self.mem.threat_point(t, x, y)

        # ---- legs ----
        mv = (0.0, 0.0); boost = False; mode = "idle"; farm = None
        dodge = self.mortar_dodge(obs, x, y, z)
        if not me["in_zone"]:
            wx, wy = self.route(x, y, zone); mv = self.toward(x, y, wx, wy); mode = "ZONE!"
            boost = energy > 25 and (zone.get("dps", 0) >= 35 or hpf < 0.5)
        elif dodge:
            mv = dodge; boost = energy > 20; mode = "dodge"
        elif self.retreating and (threats or unseen_hit):
            pts = [(e["pos"][0], e["pos"][1], e["pos"][2]) for e in threats] or ([(tp[0], tp[1], z)] if tp else [(x + 1, y, z)])
            if self.cover is None or t - self.cover_t > 1.0 or math.hypot(self.cover[0] - x, self.cover[1] - y) < 3:
                self.cover = self.find_cover(x, y, z, pts, zs)[0]; self.cover_t = t
            mv = self.toward(x, y, self.cover[0], self.cover[1]); boost = energy > (35 if lt else 20); mode = "retreat"
        elif zs["urgent"]:
            wx, wy = self.route(x, y, zone); mv = self.toward(x, y, wx, wy, zs["mag"]); mode = "zone"
            boost = zs["mag"] >= 1.0 and zs["slack"] < -5 and energy > 30
        elif tgt and self.retreating:
            mode = "regen-hold"                              # out of its reach for now: stay put, let regen start
        elif tgt:
            m = tgt["m"]; ew = m.get("weapons") or []
            if lt_ready: lo, hi = 16.0, 32.0
            elif "lightning" in ew: lo, hi = 55.0, 100.0
            elif las is not None: lo, hi = 50.0, 120.0
            else: lo, hi = 35.0, 80.0
            if hpf < 0.5 and not lt_ready: lo, hi = lo + 20, hi + 30
            aimed_at_me = bool(m.get("charging") and m.get("aim") and abs(wrap(m["aim"][0] - math.atan2(y - m["pos"][1], x - m["pos"][0]))) < 0.15)
            mv = self.strafe(x, y, m["pos"][0], m["pos"][1], tgt["d"], lo, hi, t, aimed_at_me); mode = "fight %d d=%.0f" % (m["id"], tgt["d"])
            boost = (aimed_at_me and energy > 15) or (not lt_ready and energy > 45 and tgt["d"] > 40)
        elif unseen_hit:
            pts = [(tp[0], tp[1], z)] if tp else [(x + 1, y, z)]
            if self.cover is None or t - self.cover_t > 1.0 or math.hypot(self.cover[0] - x, self.cover[1] - y) < 3:
                self.cover = self.find_cover(x, y, z, pts, zs)[0]; self.cover_t = t
            mv = self.toward(x, y, self.cover[0], self.cover[1]); boost = energy > 30; mode = "evade"
        elif hidden and t - self.last_seen_t < 4.0:
            mode = "hold"                                    # it went behind cover: hold the angle, it has to come back out
        elif hidden and hpf > 0.7 and not self.retreating:
            e = min(hidden, key=lambda e: e["d"]); mv = self.toward(x, y, e["pos"][0], e["pos"][1], 0.6); mode = "hunt %d" % e["id"]
        else:
            farm = None if (self.retreating or t - self.last_mech_dmg_t < 4.0) else self.farm_target(obs, me, x, y, z, zs, slot, walk, t)
            if farm:
                _, kind, o = farm; px, py = o["pos"][0], o["pos"][1]; d = math.hypot(px - x, py - y); mode = "farm " + kind
                if kind == "pickup":
                    if d < self.reach and abs(o["pos"][2] - z) < 3.5:
                        A["take"] = o["id"]
                        if not any(w is None for w in W) and "laser" in slot: A["drop"] = slot["laser"]
                    else: mv = self.toward(x, y, px, py)
                elif kind == "cube": mv = self.toward(x, y, px, py)
                elif kind == "drop":
                    if d > o.get("radius", 8) - 2.5: mv = self.toward(x, y, px, py)
                elif kind == "npc": mv = self.strafe(x, y, px, py, d, 40.0, 65.0, t)
                elif kind == "crate":
                    if d > 30: mv = self.toward(x, y, px, py)
            elif zs["dn"] > max(6.0, 0.55 * zs["nr"]):
                wx, wy = self.route(x, y, zone); mv = self.toward(x, y, wx, wy, 0.95 * self.step_speed / walk); mode = "walk"
        # stuck: wanted to move for 3 s but never got 1.5 m from where we were
        moving = math.hypot(*mv) > 0.2
        self.hist.append((t, x, y, moving)); self.hist = [h for h in self.hist if t - h[0] <= 3.0]
        if (tick >= self.unstick_until and len(self.hist) >= 25 and all(h[3] for h in self.hist)
                and max(math.hypot(h[1] - self.hist[0][1], h[2] - self.hist[0][2]) for h in self.hist) < 1.5):
            self.unstick_until = tick + 16; self.unstick_sign = -self.unstick_sign
            if self.goal: self.black[self.goal[1]] = t + 60; self.goal = None
        jump = False
        if tick < self.unstick_until and moving:
            yaw = math.atan2(mv[1], mv[0]) + self.unstick_sign * PI / 2; mv = (math.cos(yaw), math.sin(yaw))
            jump = tick % 4 == 0; boost = not me["grounded"]; mode += " UNSTICK"

        # ---- torso ----
        aim, fire = None, [False, False]
        far = [e for e in enemies if e["los"] and e is not tgt]
        if tgt:
            aim, fire = self.engage(me, W, slot, tgt, x, y, mz, True)
            if self.retreating and t - self.last_dmg_t > 2.5:
                fire = [bool(fire[i]) and W[i] is not None and W[i]["kind"] == "laser" and (W[i].get("charge") or 0) > 0 for i in range(2)]
        elif farm and farm[1] in ("npc", "crate"):
            o = farm[2]; px, py, pz = o["pos"]; npc = farm[1] == "npc"
            e = {"id": o["id"], "pos": o["pos"], "vel": o.get("vel", (0.0, 0.0, 0.0)), "r": self.npc_r if npc else self.crate_r,
                 "h": self.npc_h if npc else self.crate_h, "d": math.dist((x, y, z), (px, py, pz))}
            e["los"] = self.los(x, y, mz, px, py, pz + 0.5 * e["h"])
            if e["d"] < 130: aim, fire = self.engage(me, W, slot, e, x, y, mz, False)
            else: aim = [math.atan2(py - y, px - x), 0.0]
        elif hidden:
            e = min(hidden, key=lambda e: e["d"]); px, py, pz = e["pos"]
            aim = [math.atan2(py - y, px - x), math.atan2(pz + 4.5 - mz, max(1.0, math.hypot(px - x, py - y)))]
            if las is not None and (las.get("charge") or 0) > 0: fire[slot["laser"]] = True
        elif far:
            e = min(far, key=lambda e: e["d"]); aim = [math.atan2(e["pos"][1] - y, e["pos"][0] - x), 0.0]
        elif tp:
            aim = [math.atan2(tp[1] - y, tp[0] - x), 0.0]
        if aim is None:
            if math.hypot(*mv) > 0.05: aim = [math.atan2(mv[1], mv[0]) + 0.7 * math.sin(t * 1.3), 0.0]
            else: self.scan = wrap(self.scan + 0.11); aim = [self.scan, 0.0]
        if (gun and not tgt and t - self.last_seen_t > 2.0 and gun.get("reload", 0) == 0
                and gun.get("ammo", 99) < gun["stats"]["mag"] * 0.5 and not fire[slot["gun"]]):
            A["reload"] = True

        A["move"] = [round(mv[0], 3), round(mv[1], 3)]; A["boost"] = bool(boost); A["jump"] = jump
        A["aim"] = [round(aim[0], 4), round(clamp(aim[1], -0.69, 1.29), 4)]; A["fire"] = fire
        mf.draw.label(mode)
        return A


if __name__ == "__main__": mf.run(Bot)

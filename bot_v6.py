"""rabid v4 - built for the practice format: the house squad hunts you one bot at a time (more as you kill them),
so score ~= house bots that die before you. Win duels, bank cubes, heal between fights, never die to the zone.

  Memory    enemy tracks (with velocity history) + sound / damage cues that decay with time
  Aim       projectile intercept with strafe-damped lead; fire only when the torso's ACTUAL aim is on target;
            line of fire checked against the real map (buildings/terrain; trees do not stop shots)
  Legs      zone routing on goal fields, seeded random strafing, cover-scored retreat, farming NPCs/crates/cubes
  Selector  priority rules choose legs and torso independently every decision
Deterministic apart from wall-clock nav timing: all randomness comes from Random(hello.seed).
"""
import gc, json, math, os, random, sys, threading, time
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
# ---- tunables (module constants so the local sims can sweep them) ----
GUN_MOVING_RANGE = 200.0   # do not spend bullets on a moving mech farther than this
BAND_GUN = (25.0, 55.0)    # fight distance band with only the gun
BAND_LT = (16.0, 32.0)     # ... when our lightning is ready
BAND_VS_LT = (55.0, 100.0) # ... when they carry lightning and we cannot arc
BAND_LASER = (50.0, 120.0) # ... when we carry a laser
BAND_LOW_HP = (0.0, 0.0) # band shift when below half HP without lightning
FLIP_T = (0.5, 1.5)       # strafe direction change interval (s)
FLIP_P = 0.8               # chance a scheduled change actually reverses
STRAFE_W = 1.0             # lateral weight vs radial
BOOST_E = 45.0             # boost-strafe only above this energy (gun fights)
LEAD_DAMP = 0.0            # max weight of the 1 s mean velocity in the lead
RETREAT_HPF = 0.32
HIT_BONUS = 50.0           # target preference for whoever is hitting us right now
GLANCE_S = 0.6             # look toward an unseen attacker for this long (it then enters the vision wedge)
REPULSE = 0.7              # weight of 'pull away from the other hunters' in the fight movement
GRAB_R = 35.0              # detour radius for a wanted weapon pickup during a fight
# ---- ablation toggles (overridden by bot_config.json next to this file; see ablation_config.json / build_variants.py) ----
VARIANT = "baseline"
USE_LEGACY_RETREAT = False    # True: v2-style - never fall back to cover, no heal-first, no evade; just face the attacker
MAX_ENGAGEMENT_DIST = None    # hard cap (m) on firing / targeting range; None = weapon ranges
HEAL_SAFETY_THRESHOLD = 0.5   # 0..1: cover fraction required to fall back, and heal-first needs the enemy > 40+80*this m away
AGGRESSION_SCALE = 1.0        # scales the push toward targets, chasing hidden enemies and mid-fight pickup detours
ADAPTIVE_AGGRESSION = False   # in-match: after 3 outnumbered fall-backs, halve AGGRESSION_SCALE for the rest of the match
FIGHT_RESET = False           # A: engage only while <= 1 hunter has sight of us; otherwise break sight to cover and heal
RESET_HPF = 0.45              # A: also reset below this HP fraction (unless the target is nearly dead)
RESET_EXIT_HPF = 0.9          # A: leave cover once healed to this fraction (and at most one hunter has sight)
EXPOSURE_COVER = False        # B: cover = nearest walkable cell hidden from every known hunter (BFS + grid raycast)
BACK_WALL = False             # C: among nearby hidden cells prefer one with a wall behind us (no x1.5 rear hits)
NPC_DEFENSE = False           # a drone shooting us becomes the target whenever no hunter is in sight (kill it, don't stand in it)
STRICT_HIDDEN = False         # "am I in cover": centre and top rays at full resolution, and any hunter hit in the last 0.6 s = exposed
CAMP = False                  # top-practice style: stand still to fight (<70 m), move silently while hunters are within 150 m,
CAMP_CONTACT = 150.0          # no chasing / farming while in contact, full speed only when alone
SNIPER = False                # prefer the laser (hitscan) and fight at SNIPER_BAND, kiting inward hunters: sloppy house aim decays with range
SNIPER_BAND = (110.0, 170.0)
LOOT_CUBES = False            # grab kill-drop cubes / wanted weapons within LOOT_R whenever no enemy has sight within 70 m
LOOT_R = 70.0
LOOT_SAFE = 60.0              # ... and no enemy in sight closer than this
LASER_EVADE = False           # sidestep ANY visible laser charging at us; break sight (to cover) from lasers hitting us from > 110 m
LASER_FAR = 110.0             # beyond this a laser duel is a losing trade for our gun
CUBE_AVOID = False            # at the stat cap, sidestep cubes: past 10 they only make us bigger (+5% size each)
NOOK = False                  # once hunters are known, hold the most enclosed reachable cell (a chokepoint) facing its opening
NOOK_MIN = 8                  # a nook has walls in >= this many of 16 directions within 18 m
NOOK_SIGHT = 30.0             # ... and a mean sightline (16 directions, capped 60 m) at most this long
NOOK_R = 220.0                # consider nooks within this straight-line distance
NOOK_PREF = 4.0               # metres of extra walk worth 1 m shorter mean sightline
AMBUSH = False                # with >= 2 hunters known and none in sight: wait hidden (back to a wall) instead of hunting/farming
COVER_CELLS = 300             # B: cells examined per cover search (bounds the per-tick cost)
CONFIG_KEYS = ("VARIANT", "USE_LEGACY_RETREAT", "MAX_ENGAGEMENT_DIST", "HEAL_SAFETY_THRESHOLD", "AGGRESSION_SCALE", "ADAPTIVE_AGGRESSION", "TUNABLES")


def load_config(path):
    """Apply bot_config.json: the toggles above plus "TUNABLES": {NAME: value} for any tunable constant. Bad input is
    logged and ignored so a broken config can never stop the bot from answering."""
    try:
        with open(path) as fh: cfg = json.load(fh)
    except FileNotFoundError: return {}
    except Exception as e: mf.log("bot_config.json unreadable, using defaults:", repr(e)); return {}
    g = globals(); applied = {}
    for k, v in cfg.items():
        if k not in CONFIG_KEYS: mf.log("bot_config.json: unknown key ignored:", k); continue
        if k == "TUNABLES":
            for tk, tv in (v or {}).items():
                if tk.isupper() and tk in g and tk not in CONFIG_KEYS: g[tk] = tuple(tv) if isinstance(tv, list) else tv; applied[tk] = tv
                else: mf.log("bot_config.json: unknown tunable ignored:", tk)
        else: g[k] = v; applied[k] = v
    return applied


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


class DeathStateLogger:
    """Snapshot of our situation every decision; the one before death is what a LOCAL harness reads (practice_sim
    aggregates them). Real matches cannot return it: one life per match, a fresh container each match, stderr hidden."""

    def __init__(self):
        self.last = None; self.hp_hist = []

    def record(self, t, hp, visible_enemies, goal, cover_dist):
        self.hp_hist.append((t, hp)); self.hp_hist = [h for h in self.hp_hist if t - h[0] <= 3.0]
        self.last = {"time_alive": round(t, 1), "hp": round(hp), "visible_enemies_count": visible_enemies,
                     "active_goal_state": goal, "distance_to_nearest_cover": cover_dist,
                     "hp_lost_3s": round(max(0.0, self.hp_hist[0][1] - hp))}


class Bot:
    name = "rabid-v6"

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
        self.hit_by = {}; self.glance_until = -1.0; self.glance_yaw = 0.0; self.last_glance = -99.0
        self.my_id = hello.get("slot")
        self.deathlog = DeathStateLogger(); self.cover_dist = None; self.cover_dist_t = -99.0
        self.aggr = AGGRESSION_SCALE; self.outnumbered_falls = 0
        self.resetting = False; self.reset_cover = None; self.reset_cover_t = -99.0; self.resets = 0
        self.npc_hit = None; self.laser_hit = None
        self.nooks = None; self.nook = None; self.nook_t = -99.0; self.nook_field = None
        if NOOK and self.nav is not None and hasattr(self.nav, "ENC"): self.build_nooks()
        if VARIANT != "baseline": self.name = ("rabid-v6-" + str(VARIANT))[:32]
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
                avx, avy = self.mem.avg_vel(e["id"], e["vel"]); w = clamp((d / gs["speed"]) / 0.6, 0.0, 1.0) * LEAD_DAMP
                lvx, lvy = (1 - w) * vx + w * avx, (1 - w) * vy + w * avy
            else: lvx, lvy = vx, vy
            qx, qy, qz, tf = intercept(x, y, mz, px, py, cz, lvx, lvy, 0.4 * vz, gs["speed"])
            yaw_l = math.atan2(qy - y, qx - x); pit_l = math.atan2(qz - mz, math.hypot(qx - x, qy - y))
            aim = [yaw_l, pit_l]
            on = abs(wrap(ax - yaw_l)) < tol_y and abs(ay - pit_l) < tol_p
            moving = math.hypot(vx, vy) > 3.0
            in_rng = d <= gs["range"] * 0.97 and (d <= GUN_MOVING_RANGE or not moving or not mech)
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
    def strafe(self, x, y, px, py, d, lo, hi, t, hold=False, bias=(0.0, 0.0)):
        if hold: self.flip_t = max(self.flip_t, t + 0.3)     # a charged laser is on us: keep the lateral run going
        if t >= self.flip_t:
            if self.rng.random() < FLIP_P: self.side = -self.side
            self.flip_t = t + self.rng.uniform(*FLIP_T)
        to = math.atan2(py - y, px - x)
        rad = min(1.0, (d - hi) / 15.0) * self.aggr if d > hi else (-min(1.0, (lo - d) / 10.0) if d < lo else 0.0)
        vx = -math.sin(to) * self.side * STRAFE_W + rad * math.cos(to) + bias[0]; vy = math.cos(to) * self.side * STRAFE_W + rad * math.sin(to) + bias[1]
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

    def threat_points(self, t, enemies, x=0.0, y=0.0, z=0.0):
        """(x, y, eye_z) of every hunter we see or saw in the last 25 s (the squad shares vision: assume they still see
        us), plus a point 80 m along the bearing of any hit in the last 3 s whose shooter we cannot place."""
        pts = {}
        for e in enemies: pts[e["id"]] = (e["pos"][0], e["pos"][1], e["pos"][2] + 8.0)
        for k, tr in self.mem.tracks.items():
            if k not in pts and t - tr["t"] < 25.0: pts[k] = (tr["pos"][0], tr["pos"][1], tr["pos"][2] + 8.0)
        for k, (ht, hb, _) in self.hit_by.items():
            if k not in pts and hb is not None and t - ht < 3.0: pts[k] = (x + 80 * math.cos(hb), y + 80 * math.sin(hb), z + 8.0)
        return list(pts.values())

    def build_nooks(self):
        nv = self.nav
        self.nooks = [(nv.xy(u)[0], nv.xy(u)[1], nv.SIGHT[u], nv.OPEN[u], u) for u, e in enumerate(nv.ENC)
                      if e >= NOOK_MIN and nv.F[u] and nv.SIGHT[u] <= NOOK_SIGHT]

    def choose_nook(self, x, y, zs, walk):
        if self.nooks is None: self.build_nooks()
        best, bs = None, 1e18
        for nk in self.nooks:
            d = math.hypot(nk[0] - x, nk[1] - y)
            if d > NOOK_R or not self.zone_ok(nk[0], nk[1], zs, walk): continue
            sc = d + NOOK_PREF * nk[2]
            if sc < bs: bs, best = sc, nk
        return best

    def nook_waypoint(self, x, y):
        """Next waypoint toward the chosen nook on a goal field that stops as soon as our own cell is settled."""
        nv = self.nav; nk = self.nook
        if self.nook_field is not None and self.nook_field[0] == nk[4]:
            res = nv.follow(self.nook_field[1][0], x, y)
            if res and res[1] > 0 and math.hypot(res[0][0] - x, res[0][1] - y) > 0.5: return res[0]
        return (nk[0], nk[1])                                # until the field reaches us: straight line + wall steering

    def background_work(self, x, y, zone, t0):
        """Path fields only on light ticks: nook field (start, then steps until our cell is settled), else zone field."""
        if time.perf_counter() - t0 > 0.012: return
        nv = self.nav
        if NOOK and nv is not None and self.nook is not None:
            nk = self.nook
            if self.nook_field is None or self.nook_field[0] != nk[4]:
                self.nook_field = [nk[4], nv.field_start(nk[0], nk[1], 3.0)]; return
            j, i = nv.cell(x, y); me_u = j * nv.nx2 + i
            if self.nook_field[1][0][me_u] == math.inf and self.nook_field[1][1]:
                nv.field_step(self.nook_field[1], 2500, me_u if nv.F[me_u] else None); return
        self.advance_field(zone)

    def laser_dodge(self, t, x, y, z, enemies, zs, energy):
        """-> (move, boost, mode) or None. 1) any visible laser charging at us: keep a boosted lateral run;
        2) a laser that hit us in the last 1.5 s from beyond LASER_FAR: break its line of sight (cover) instead of trading."""
        if getattr(self, "_ld_t", None) == t: return self._ld
        res = None
        for e in enemies:
            m = e["m"]
            if not (e["los"] and m.get("charging") and m.get("aim")) or e["d"] < 30: continue
            to_me = math.atan2(y - m["pos"][1], x - m["pos"][0])
            if abs(wrap(m["aim"][0] - to_me)) < 0.12:
                if t >= self.flip_t: self.flip_t = t + 0.9                  # commit to one direction for the charge
                a = to_me + self.side * PI / 2; a = self.steer(x, y, a)
                res = ((math.cos(a), math.sin(a)), energy > 10, "laser-dodge"); break
        if res is None and self.laser_hit and t - self.laser_hit[0] < 1.5:
            shooter = next((e for e in enemies if e["id"] == self.laser_hit[1]), None)
            if shooter is not None and shooter["d"] > LASER_FAR:
                thr = self.threat_points(t, enemies, x, y, z)
                pc = self.pick_cover(x, y, z, thr, zs) if (EXPOSURE_COVER and thr) else None
                if pc is not None:
                    wx, wy = pc[2]; res = (self.toward(x, y, wx, wy), energy > 20, "laser-cover")
        self._ld_t, self._ld = t, res
        return res

    def loot_target(self, obs, me, W, slot, x, y, z, t):
        """Nearest wanted pickup or (below the stat cap) cube within LOOT_R on our level; kill drops are the main source."""
        if getattr(self, "_lt_t", None) == t: return self._lt
        best = None; R = 30.0 if (CAMP and getattr(self, "contact_d", 1e9) < CAMP_CONTACT) else LOOT_R
        wp = self.wanted_pickup(obs, W, slot, x, y, z, t, R)
        if wp is not None: best = (math.hypot(wp["pos"][0] - x, wp["pos"][1] - y) * 0.7, wp)
        if me["cubes"] < self.cube_cap:
            for c in obs["cubes"]:
                if self.black.get(c["id"], -1) > t or abs(c["pos"][2] - z) > 3.5: continue
                d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if d < R and (best is None or d < best[0]): best = (d, c)
        self._lt_t, self._lt = t, (best[1] if best else None)
        return self._lt

    def is_hidden(self, t, x, y, z, thr):
        if not self.nav or not thr: return False
        if not STRICT_HIDDEN: return self.nav.hidden(x, y, z, thr)
        if t - self.last_mech_dmg_t < 0.6: return False
        return self.nav.hidden(x, y, z, thr, 9.0, 1.5) and self.nav.hidden(x, y, z, thr, 4.5, 1.5)

    def pick_cover(self, x, y, z, threats, zs):
        """B/C: nearest hidden cell (BACK_WALL: a wall behind it is worth 8 m of path). -> (cx, cy, (wx, wy)) or None."""
        if not self.nav or not threats or not hasattr(self.nav, "cover_search"): return None
        c = self.nav.cover_search(x, y, threats, COVER_CELLS, 46.0, 4 if BACK_WALL else 1)
        c = [k for k in c if math.hypot(k[1] - zs["lc"][0], k[2] - zs["lc"][1]) < zs["lr"] - 4]
        if not c: return None
        best = min(c, key=lambda k: k[0] - (8.0 if (BACK_WALL and k[4]) else 0.0))
        return best[1], best[2], best[3]

    def nearest_cover(self, x, y, z, threats):
        """Walking distance (straight line) to the closest point hidden from every threat; None if none within 30 m."""
        if not self.nav or not threats: return None
        for dd in (6.0, 12.0, 18.0, 24.0, 30.0):
            for k in range(12):
                a = k * TAU / 12; px, py = x + dd * math.cos(a), y + dd * math.sin(a)
                if not self.nav.free_at(px, py): continue
                gz = self.nav.ground(px, py)
                if all(not self.los(tx, ty, tz + 8.0, px, py, gz + 4.5) for (tx, ty, tz) in threats): return dd
        return None

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

    def pickup_pref(self, k, slot, empty):
        """Priority weight (lower = more wanted) for a weapon pickup of kind k, or None if we do not want it."""
        first, second = ("laser", "lightning") if SNIPER else ("lightning", "laser")
        if k == first and first not in slot and (empty or second in slot): return 0.4
        if k == second and empty and first not in slot and second not in slot: return 0.6
        return None

    def swap_slot(self, slot, W):
        """Slot to drop when taking the preferred weapon with both slots full."""
        if any(w is None for w in W): return None
        second = "lightning" if SNIPER else "laser"
        return slot.get(second)

    def wanted_pickup(self, obs, W, slot, x, y, z, t, maxd):
        empty = any(w is None for w in W); best = None
        for p in obs["pickups"]:
            if self.black.get(p["id"], -1) > t or abs(p["pos"][2] - z) > 3.5: continue
            k = p["kind"]
            ok = self.pickup_pref(k, slot, empty) is not None
            d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
            if ok and d < maxd and (best is None or d < best[0]): best = (d, p)
        return best[1] if best else None

    def others_bias(self, x, y, t, enemies, tid):
        """Push away from every known hunter except the target: keeps them in front and arriving one at a time."""
        bx = by = 0.0; seen = set()
        pts = [(e["id"], e["pos"]) for e in enemies] + [(k, tr["pos"]) for k, tr in self.mem.tracks.items() if t - tr["t"] < 8.0]
        for k, p in pts:
            if k == tid or k in seen: continue
            seen.add(k); dx, dy = x - p[0], y - p[1]; d = math.hypot(dx, dy)
            if d < 1 or d > 260: continue
            w = REPULSE / (1.0 + d / 100.0); bx += w * dx / d; by += w * dy / d
        return bx, by

    def farm_target(self, obs, me, x, y, z, zs, slot, walk, t):
        W = me["weapons"]; cands = []
        empty = any(w is None for w in W)
        for p in obs["pickups"]:
            if self.black.get(p["id"], -1) > t or abs(p["pos"][2] - z) > 3.5: continue
            k = p["kind"]
            pr = self.pickup_pref(k, slot, empty)
            if pr is None: continue
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
                if e.get("weapon") == "npc" and e.get("by") is not None: self.npc_angry[e["by"]] = t; self.npc_hit = (t, e.get("bearing"), e["by"])
                elif e.get("weapon") not in ("zone", None):
                    if e.get("weapon") == "laser" and e.get("by") is not None: self.laser_hit = (t, e["by"])
                    self.last_mech_dmg_t = t
                    if e.get("by") is not None: self.hit_by[e["by"]] = (t, e.get("bearing"), bool(e.get("rear")))
        if obs.get("offer"): A["pick"] = self.pick(obs["offer"])
        W = me["weapons"]; slot = {w["kind"]: i for i, w in enumerate(W) if w}
        walk = st.get("walk_speed", 11.7); hpf = me["hp"] / max(1.0, me["max_hp"]); energy = me["energy"]
        mz = z + st.get("muzzle_height", 7.0)
        zs = self.zone_status(x, y, zone, t, walk)
        t_act0 = time.perf_counter()
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
        if MAX_ENGAGEMENT_DIST is not None: max_rng = min(max_rng, float(MAX_ENGAGEMENT_DIST))
        tgt, best = None, 1e9
        for e in enemies:
            if not e["los"] or e["d"] > max_rng: continue
            sc = e["d"] + 120.0 * e["m"].get("hp", 1400) / max(1.0, e["m"].get("max_hp", 1400)) - (60.0 if e["id"] == self.target_id else 0.0)
            if t - self.hit_by.get(e["id"], (-99,))[0] < 2.0: sc -= HIT_BONUS
            if sc < best: best, tgt = sc, e
        if tgt: self.target_id = tgt["id"]
        threats = [e for e in enemies if e["d"] < 170 and (e["los"] or e["d"] < 60)]
        hidden = [e for e in enemies if not e["los"]]

        # ---- fight or flight ----
        if threats and not self.retreating and not USE_LEGACY_RETREAT:
            weakest = min(e["m"].get("hp", 1400) for e in threats)
            n_los = sum(1 for e in threats if e["los"])
            near_los = min((e["d"] for e in threats if e["los"]), default=1e9)
            heal_first = hpf < 0.55 and near_los > 40 + 80 * HEAL_SAFETY_THRESHOLD   # next fight still coming in: heal behind cover
            if (hpf < RETREAT_HPF and weakest > me["hp"] * 0.8) or (n_los >= 2 and hpf < 0.55) or heal_first:
                pts = [(e["pos"][0], e["pos"][1], e["pos"][2]) for e in threats]
                if EXPOSURE_COVER:
                    pc = self.pick_cover(x, y, z, self.threat_points(t, enemies, x, y, z), zs)
                    self.cover, cov = ((pc[2], 1.0) if pc else (None, 0.0)); self.cover_t = t
                else: self.cover, cov = self.find_cover(x, y, z, pts, zs); self.cover_t = t
                if cov >= HEAL_SAFETY_THRESHOLD:                # running without cover only hands out free shots
                    self.retreating = True
                    if n_los >= 2:
                        self.outnumbered_falls += 1
                        if ADAPTIVE_AGGRESSION and self.outnumbered_falls == 3: self.aggr *= 0.5
        if self.retreating and (hpf > 0.85 or (not threats and t - self.last_dmg_t > 12)): self.retreating = False
        if self.retreating and tgt and lt_ready and tgt["d"] < lt["stats"]["range"] * 0.85 and tgt["m"].get("hp", 1400) < me["hp"] * 1.3:
            self.retreating = False                          # in arc range of a beatable target: finish it
        if FIGHT_RESET:
            self.retreating = False                          # A replaces the legacy fall-back logic
            n_los = sum(1 for e in enemies if e["los"])
            tgt_low = tgt is not None and tgt["m"].get("hp", 1400) < 0.25 * tgt["m"].get("max_hp", 1400)
            if not self.resetting and (n_los >= 2 or (hpf < RESET_HPF and not tgt_low and (tgt is not None or t - self.last_mech_dmg_t < 2))):
                self.resetting = True; self.resets += 1; self.reset_cover = None
            if self.resetting and hpf >= RESET_EXIT_HPF and n_los <= 1: self.resetting = False
            if self.resetting and any(e["los"] and e["d"] < 18 for e in enemies): self.resetting = False   # too close to hide: fight
            if self.resetting and t - self.last_mech_dmg_t < 0.2: self.reset_cover = None   # hit while "hidden": that spot is not cover
            if self.resetting and (self.reset_cover is None or t - self.reset_cover_t > 0.5):
                thr = self.threat_points(t, enemies, x, y, z)
                if EXPOSURE_COVER: pc = self.pick_cover(x, y, z, thr, zs)
                else:
                    fc, cov = self.find_cover(x, y, z, [(p[0], p[1], p[2] - 8.0) for p in thr], zs) if thr else (None, 0.0)
                    pc = (fc[0], fc[1], fc) if fc and cov >= 0.5 else None
                self.reset_cover = pc; self.reset_cover_t = t
                if pc is None and n_los >= 1: self.resetting = False     # nowhere to hide: fighting beats running in the open
        unseen_hit = tgt is None and t - self.last_mech_dmg_t < 1.5 and not USE_LEGACY_RETREAT
        tp = self.mem.threat_point(t, x, y)

        hunters_known = bool(enemies) or any(t - tr["t"] < 30.0 for tr in self.mem.tracks.values()) or t - self.last_mech_dmg_t < 10.0
        self.contact_d = min([e["d"] for e in enemies] + [math.hypot(tr["pos"][0] - x, tr["pos"][1] - y) for tr in self.mem.tracks.values() if t - tr["t"] < 10.0] + [1e9])
        at_nook = False
        if NOOK and self.nav and hasattr(self.nav, "ENC") and hunters_known:
            if self.nook is None or t - self.nook_t > 8.0 or not self.zone_ok(self.nook[0], self.nook[1], zs, walk):
                nk = self.choose_nook(x, y, zs, walk)
                if nk is not None and (self.nook is None or nk[4] != self.nook[4]): self.nook = nk
                elif nk is None: self.nook = None
                self.nook_t = t
            at_nook = self.nook is not None and math.hypot(self.nook[0] - x, self.nook[1] - y) <= 4.0
        angry_npc = None
        if NPC_DEFENSE and tgt is None:
            cands = [m for m in obs["mechs"] if m["kind"] == "npc" and t - self.npc_angry.get(m["id"], -99) < 3.0]
            if cands:
                n = min(cands, key=lambda m: math.dist(m["pos"], (x, y, z))); px, py, pz = n["pos"]
                angry_npc = {"id": n["id"], "pos": n["pos"], "vel": n.get("vel", (0.0, 0.0, 0.0)), "r": self.npc_r, "h": self.npc_h,
                             "d": math.dist((x, y, z), (px, py, pz))}
                angry_npc["los"] = self.los(x, y, mz, px, py, pz + 0.5 * self.npc_h)

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
            if self.cover is None or t - self.cover_t > (0.5 if EXPOSURE_COVER else 1.0) or math.hypot(self.cover[0] - x, self.cover[1] - y) < 3:
                if EXPOSURE_COVER:
                    pc = self.pick_cover(x, y, z, self.threat_points(t, enemies, x, y, z), zs)
                    self.cover = pc[2] if pc else self.find_cover(x, y, z, pts, zs)[0]
                else: self.cover = self.find_cover(x, y, z, pts, zs)[0]
                self.cover_t = t
            mv = self.toward(x, y, self.cover[0], self.cover[1]); boost = energy > (35 if lt else 20); mode = "retreat"
        elif LASER_EVADE and self.laser_dodge(t, x, y, z, enemies, zs, energy) is not None:
            mv, boost, mode = self.laser_dodge(t, x, y, z, enemies, zs, energy)
        elif angry_npc is not None and not any(e["los"] for e in enemies):
            mv = self.strafe(x, y, angry_npc["pos"][0], angry_npc["pos"][1], angry_npc["d"], 30.0, 55.0, t); mode = "npc-defense"
        elif FIGHT_RESET and self.resetting and self.reset_cover is not None:
            thr = self.threat_points(t, enemies, x, y, z)
            if self.is_hidden(t, x, y, z, thr): mode = "heal"                # in cover: stand still, regen
            else:
                wx, wy = self.reset_cover[2]; mv = self.toward(x, y, wx, wy); boost = energy > 20; mode = "reset"
        elif zs["urgent"]:
            wx, wy = self.route(x, y, zone); mv = self.toward(x, y, wx, wy, zs["mag"]); mode = "zone"
            boost = zs["mag"] >= 1.0 and zs["slack"] < -5 and energy > 30
        elif tgt and self.retreating:
            mode = "regen-hold"                              # out of its reach for now: stay put, let regen start
        elif tgt:
            m = tgt["m"]; ew = m.get("weapons") or []
            if SNIPER and las is not None: lo, hi = SNIPER_BAND
            elif lt_ready: lo, hi = BAND_LT
            elif "lightning" in ew: lo, hi = BAND_VS_LT
            elif las is not None: lo, hi = BAND_LASER
            else: lo, hi = BAND_GUN
            if hpf < 0.5 and not lt_ready: lo, hi = lo + BAND_LOW_HP[0], hi + BAND_LOW_HP[1]
            aimed_at_me = bool(m.get("charging") and m.get("aim") and abs(wrap(m["aim"][0] - math.atan2(y - m["pos"][1], x - m["pos"][0]))) < 0.15)
            mv = self.strafe(x, y, m["pos"][0], m["pos"][1], tgt["d"], lo, hi, t, aimed_at_me, self.others_bias(x, y, t, enemies, m["id"]))
            mode = "fight %d d=%.0f" % (m["id"], tgt["d"])
            if NOOK and self.nook is not None and hunters_known:
                if at_nook:                                  # hold the chokepoint: small sidesteps, pulled back to its centre
                    cx, cy = self.nook[0] - x, self.nook[1] - y
                    mv = (0.35 * mv[0] + 0.3 * cx, 0.35 * mv[1] + 0.3 * cy); mode += " @nook"
                elif tgt["d"] > 22:                          # fall back to the nook while firing
                    wx, wy = self.nook_waypoint(x, y); mv = self.toward(x, y, wx, wy); mode += " ->nook"
            grab = self.wanted_pickup(obs, W, slot, x, y, z, t, GRAB_R * min(1.5, self.aggr)) if hpf > 0.3 else None
            if grab:
                gd = math.hypot(grab["pos"][0] - x, grab["pos"][1] - y); mode += " +grab"
                if gd < self.reach: A["take"] = grab["id"]; A["drop"] = self.swap_slot(slot, W)
                else: mv = self.toward(x, y, grab["pos"][0], grab["pos"][1])
                if A.get("drop") is None: A.pop("drop", None)
            boost = (aimed_at_me and energy > 15) or (not lt_ready and energy > BOOST_E and tgt["d"] > 40)
            if SNIPER and las is not None and tgt["d"] < lo and energy > 30: boost = True        # kite: keep them at range
        elif LOOT_CUBES and t - self.last_mech_dmg_t > 2.0 and not any(e["los"] and e["d"] < LOOT_SAFE for e in enemies) \
                and self.loot_target(obs, me, W, slot, x, y, z, t) is not None:
            o = self.loot_target(obs, me, W, slot, x, y, z, t); px, py = o["pos"][0], o["pos"][1]; d = math.hypot(px - x, py - y)
            if "kind" in o and d < self.reach:
                A["take"] = o["id"]
                if self.swap_slot(slot, W) is not None: A["drop"] = self.swap_slot(slot, W)
            else: mv = self.toward(x, y, px, py); boost = energy > 40 and d > 15
            mode = "loot"
        elif NOOK and self.nook is not None and hunters_known and not unseen_hit:
            if at_nook: mode = "nook"
            else: wx, wy = self.nook_waypoint(x, y); mv = self.toward(x, y, wx, wy); mode = "->nook"
        elif unseen_hit:
            pts = [(tp[0], tp[1], z)] if tp else [(x + 1, y, z)]
            if self.cover is None or t - self.cover_t > 1.0 or math.hypot(self.cover[0] - x, self.cover[1] - y) < 3:
                self.cover = self.find_cover(x, y, z, pts, zs)[0]; self.cover_t = t
            mv = self.toward(x, y, self.cover[0], self.cover[1]); boost = energy > 30; mode = "evade"
        elif AMBUSH and len(self.threat_points(t, enemies, x, y, z)) >= 2 and not any(e["los"] for e in enemies):
            thr = self.threat_points(t, enemies, x, y, z)
            if self.nav and not self.is_hidden(t, x, y, z, thr):
                if self.reset_cover is None or t - self.reset_cover_t > 0.5:
                    self.reset_cover = self.pick_cover(x, y, z, thr, zs); self.reset_cover_t = t
                if self.reset_cover is not None:
                    wx, wy = self.reset_cover[2]; mv = self.toward(x, y, wx, wy); mode = "ambush-move"
                else: mode = "ambush"
            else: mode = "ambush"                            # hidden: let them come around the corner one at a time
        elif hidden and t - self.last_seen_t < 4.0:
            mode = "hold"                                    # it went behind cover: hold the angle, it has to come back out
        elif hidden and hpf > 0.7 and not self.retreating and self.aggr >= 0.3 and not CAMP:
            e = min(hidden, key=lambda e: e["d"]); mv = self.toward(x, y, e["pos"][0], e["pos"][1], min(1.0, 0.6 * self.aggr)); mode = "hunt %d" % e["id"]
        else:
            farm = None if (self.retreating or t - self.last_mech_dmg_t < 4.0 or (CAMP and self.contact_d < CAMP_CONTACT)) else self.farm_target(obs, me, x, y, z, zs, slot, walk, t)
            if farm:
                _, kind, o = farm; px, py = o["pos"][0], o["pos"][1]; d = math.hypot(px - x, py - y); mode = "farm " + kind
                if kind == "pickup":
                    if d < self.reach and abs(o["pos"][2] - z) < 3.5:
                        A["take"] = o["id"]
                        if self.swap_slot(slot, W) is not None: A["drop"] = self.swap_slot(slot, W)
                    else: mv = self.toward(x, y, px, py)
                elif kind == "cube": mv = self.toward(x, y, px, py)
                elif kind == "drop":
                    if d > o.get("radius", 8) - 2.5: mv = self.toward(x, y, px, py)
                elif kind == "npc": mv = self.strafe(x, y, px, py, d, 40.0, 65.0, t)
                elif kind == "crate":
                    if d > 30: mv = self.toward(x, y, px, py)
            elif zs["dn"] > max(6.0, 0.55 * zs["nr"]):
                wx, wy = self.route(x, y, zone); mv = self.toward(x, y, wx, wy, 0.95 * self.step_speed / walk); mode = "walk"
        if CUBE_AVOID and me["cubes"] >= self.cube_cap and math.hypot(*mv) > 0.2:
            px = py = 0.0
            for c in obs["cubes"]:
                dx, dy = x - c["pos"][0], y - c["pos"][1]; d = math.hypot(dx, dy)
                if 0.1 < d < 7.0 and abs(c["pos"][2] - z) < 4: px += dx / d * (7.0 - d) / 7.0; py += dy / d * (7.0 - d) / 7.0
            if px or py:
                vx, vy = mv[0] + 1.5 * px, mv[1] + 1.5 * py; n = math.hypot(vx, vy) or 1.0; m0 = math.hypot(*mv)
                mv = (vx / n * m0, vy / n * m0)
        if CAMP:
            if mode.startswith("fight") and tgt is not None and tgt["d"] >= 15:
                mv = (0.0, 0.0); boost = False                       # stand and shoot
            elif self.contact_d < CAMP_CONTACT and mode.split(" ")[0] in ("walk", "farm", "loot", "->nook", "ambush-move", "hunt", "idle", "hold", "nook", "ambush"):
                m0 = math.hypot(*mv); cap = 0.95 * self.step_speed / walk
                if m0 > cap: mv = (mv[0] / m0 * cap, mv[1] / m0 * cap)
                boost = False                                        # silent: no footsteps while hunters are near
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
        visible_ids = {e["id"] for e in enemies if e["los"]}
        for aid, (ht, hb, rear) in self.hit_by.items():       # hit by someone we cannot see: look, so targeting can weigh it
            if t - ht < 0.6 and aid not in visible_ids and hb is not None and t - self.last_glance > 3.0 and \
                    (tgt is None or tgt["m"].get("hp", 1400) > 0.25 * tgt["m"].get("max_hp", 1400)):
                self.glance_until, self.glance_yaw, self.last_glance = t + GLANCE_S, hb, t
        if NPC_DEFENSE and angry_npc is None and tgt is None and self.npc_hit and t - self.npc_hit[0] < 0.6 and self.npc_hit[1] is not None \
                and t - self.last_glance > 2.0:
            self.glance_until, self.glance_yaw, self.last_glance = t + GLANCE_S, self.npc_hit[1], t   # drone behind us: turn to see it
        if t < self.glance_until:
            aim = [self.glance_yaw, 0.0]
            if las is not None and (las.get("charge") or 0) > 0: fire[slot["laser"]] = True
        elif tgt:
            aim, fire = self.engage(me, W, slot, tgt, x, y, mz, True)
            if self.retreating and t - self.last_dmg_t > 2.5:
                fire = [bool(fire[i]) and W[i] is not None and W[i]["kind"] == "laser" and (W[i].get("charge") or 0) > 0 for i in range(2)]
        elif angry_npc is not None:
            aim, fire = self.engage(me, W, slot, angry_npc, x, y, mz, False)
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
        elif NOOK and at_nook and not tp:
            aim = [self.nook[3], 0.0]
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
        if hpf < 0.4 and threats and t - self.cover_dist_t > 0.5:
            self.cover_dist = self.nearest_cover(x, y, z, [(e["pos"][0], e["pos"][1], e["pos"][2]) for e in threats]); self.cover_dist_t = t
        self.deathlog.record(t, me["hp"], sum(1 for e in enemies if e["los"]), mode.split(" ")[0], self.cover_dist if hpf < 0.4 else None)
        if self.field is None and self.pending is None: self.advance_field(zone)
        else: self.background_work(x, y, zone, t_act0)
        mf.draw.label(mode)
        return A


CONFIG_APPLIED = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_config.json"))

if __name__ == "__main__": mf.run(Bot)

"""Map model built once at start (numpy at init only, plain lists afterwards for fast per-tick lookups).

- walkability grid (2 m cells, dilated by one cell) + goal distance fields for zone routing
- solid-height grid (1 m): max(terrain, building/stadium top) for line-of-fire checks (bridges ignored: shots pass under)
- ground-height grid (1 m) for z of candidate points
"""
import collections, heapq, json, math, os, time
import numpy as np

NB = [(-1, 0, 2.0), (1, 0, 2.0), (0, -1, 2.0), (0, 1, 2.0), (-1, -1, 2.83), (-1, 1, 2.83), (1, -1, 2.83), (1, 1, 2.83)]


def _inpoly(px, py, ring):
    xs, ys = ring[0::2], ring[1::2]; ins = np.zeros(px.shape, bool); j = len(xs) - 1
    for i in range(len(xs)):
        c = ((ys[i] > py) != (ys[j] > py)) & (px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]); ins ^= c; j = i
    return ins


class Nav:
    def __init__(self, mapdir, nooks=True, roofs=True):
        with open(os.path.join(mapdir, "v2/world.json")) as fh:
            w = json.load(fh)
        T = w["terrain"]
        nx, ny, self.minX, self.minY = T["nx"], T["ny"], T["minX"], T["minY"]; minX, minY = self.minX, self.minY
        H = np.fromfile(os.path.join(mapdir, "v2/terrain.i16"), dtype="<i2").reshape(ny, nx).astype(np.float32) / 100
        blk = np.zeros((ny, nx), bool)
        top = H.copy()                                                  # solid height for line of fire
        m = np.abs(np.diff(H, axis=1)) > 1.19; blk[:, :-1] |= m; blk[:, 1:] |= m
        m = np.abs(np.diff(H, axis=0)) > 1.19; blk[:-1, :] |= m; blk[1:, :] |= m
        for p in w["prisms"]:
            xs, ys = p["ring"][0::2], p["ring"][1::2]
            i0, i1 = max(0, int(min(xs) - minX) - 1), min(nx, int(max(xs) - minX) + 2)
            j0, j1 = max(0, int(min(ys) - minY) - 1), min(ny, int(max(ys) - minY) + 2)
            if i0 >= i1 or j0 >= j1: continue
            gx, gy = np.meshgrid(minX + np.arange(i0, i1), minY + np.arange(j0, j1)); mk = _inpoly(gx, gy, p["ring"])
            holes = p.get("holes") or []
            if holes and not isinstance(holes[0], list): holes = [holes]
            for h in holes: mk &= ~_inpoly(gx, gy, h)
            g = H[j0:j1, i0:i1]
            if p.get("kind") != "bridge":
                sub = top[j0:j1, i0:i1]; np.maximum(sub, np.where(mk & (p["top"] > g), np.float32(p["top"]), sub), out=sub)
            mk &= (p["base"] < g + 9) & (p["top"] > g + 1.2); blk[j0:j1, i0:i1] |= mk
        self.blk, self.nx, self.ny = blk, nx, ny                       # 1 m, undilated
        self.S = top.tolist()                                          # rows of floats: solid height
        self.Gnp = H                                                   # ground height (numpy; read a few hundred times per tick)
        self.nx2, self.ny2 = nx // 2, ny // 2
        b2 = blk[:self.ny2 * 2, :self.nx2 * 2].reshape(self.ny2, 2, self.nx2, 2).any((1, 3)); pad = np.pad(b2, 1); d = b2.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1): d |= pad[1 + dy:1 + dy + self.ny2, 1 + dx:1 + dx + self.nx2]
        self.free = ~d; self.F = self.free.ravel().tolist()
        self._inf = [math.inf] * (self.nx2 * self.ny2)                 # copied (pointer copy, ~1 ms) to start each field
        if nooks:                                                      # only builds that hold nooks pay for this
            # enclosure: for each walkable 2 m cell, which of 16 directions hit a wall (building/ledge) within 18 m.
            # Highly enclosed cells are nooks: attackers can only come through the open arc, one at a time.
            ndir = 16; blocked = np.zeros((ndir, ny, nx), bool); tall = (top - H) > 5.0            # walls that also stop shots
            for k in range(ndir):
                a = 2 * math.pi * k / ndir; ca, sa = math.cos(a), math.sin(a)
                for dist in range(2, 19, 2):
                    dx, dy = int(round(dist * ca)), int(round(dist * sa))
                    src = tall[max(0, dy):ny + min(0, dy), max(0, dx):nx + min(0, dx)]
                    blocked[k, max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)] |= src
            enc = blocked.sum(0)[1:self.ny2 * 2:2, 1:self.nx2 * 2:2]                    # sample at 2 m cell centres
            ox = sum(np.cos(2 * math.pi * k / ndir) * ~blocked[k, 1:self.ny2 * 2:2, 1:self.nx2 * 2:2] for k in range(ndir))
            oy = sum(np.sin(2 * math.pi * k / ndir) * ~blocked[k, 1:self.ny2 * 2:2, 1:self.nx2 * 2:2] for k in range(ndir))
            enc = np.where(self.free, enc, 0)
            self.ENC = enc.ravel().tolist(); self.OPEN = np.arctan2(oy, ox).ravel().tolist()
            del blocked
            # sightlines: mean over 16 directions of the distance to the first tall wall (capped at 60 m) on the 2 m grid.
            # Low = few long lines of fire reach this cell (what a hold position needs; enclosure alone can face a field).
            t2 = tall[:self.ny2 * 2, :self.nx2 * 2].reshape(self.ny2, 2, self.nx2, 2).any((1, 3))
            cap = 60.0; total = np.zeros((self.ny2, self.nx2), np.float32)
            for k in range(ndir):
                a = 2 * math.pi * k / ndir; ca, sa = math.cos(a), math.sin(a); dk = np.full((self.ny2, self.nx2), cap, np.float32)
                for st in range(1, 31):
                    dx, dy = int(round(st * ca)), int(round(st * sa))
                    if dx == 0 and dy == 0: continue
                    src = t2[max(0, dy):self.ny2 + min(0, dy), max(0, dx):self.nx2 + min(0, dx)]
                    view = dk[max(0, -dy):self.ny2 - max(0, dy), max(0, -dx):self.nx2 - max(0, dx)]
                    np.minimum(view, np.where(src, np.float32(2.0 * st), np.float32(cap)), out=view)
                total += dk
            self.SIGHT = np.where(self.free, total / ndir, cap).ravel().tolist()
        self.roofs = []
        if roofs:
            try: self.roofs = self._roof_catalog(w["prisms"])
            except Exception: self.roofs = []
        jj, ii = np.mgrid[0:self.ny2, 0:self.nx2]; self.cx = (minX + 2 * ii + 1).ravel(); self.cy = (minY + 2 * jj + 1).ravel()

    # ---- grid lookups ------------------------------------------------------------------------------------------
    def cell(self, x, y):
        return min(max(int((y - self.minY) // 2), 0), self.ny2 - 1), min(max(int((x - self.minX) // 2), 0), self.nx2 - 1)

    def xy(self, u):
        j, i = divmod(u, self.nx2); return self.minX + 2 * i + 1, self.minY + 2 * j + 1

    def free_at(self, x, y):
        i, j = int((x - self.minX) // 2), int((y - self.minY) // 2)
        return 0 <= i < self.nx2 and 0 <= j < self.ny2 and self.F[j * self.nx2 + i]

    def ground(self, x, y):
        i, j = int(x - self.minX + 0.5), int(y - self.minY + 0.5)
        if 0 <= i < self.nx and 0 <= j < self.ny: return float(self.Gnp.item(j, i))
        return 0.0

    def los(self, x0, y0, z0, x1, y1, z1, skip0=2.0, skip1=2.0, step=1.5):
        """True if the straight segment clears terrain and buildings (trees ignored: they do not stop shots)."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0; L = math.hypot(dx, dy)
        if L < skip0 + skip1 + 0.5: return True
        S, minX, minY, nx, ny = self.S, self.minX, self.minY, self.nx, self.ny
        s = skip0; end = L - skip1; inv = 1.0 / L
        while s <= end:
            f = s * inv; i = int(x0 + dx * f - minX + 0.5); j = int(y0 + dy * f - minY + 0.5)
            if 0 <= i < nx and 0 <= j < ny and S[j][i] > z0 + dz * f: return False
            s += step
        return True

    def steer(self, x, y, yaw, dists=(3.5, 7.0)):
        """Closest heading to `yaw` whose probe points are walkable; `yaw` itself if nothing is."""
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8):
            a = yaw + k * 0.3927; c, s = math.cos(a), math.sin(a)
            if all(self.free_at(x + d * c, y + d * s) for d in dists): return a
        return yaw

    def hidden(self, x, y, z, threats, top=9.0, step=3.0):
        """True if no threat (tx, ty, eye_z) can see the top of a mech standing at (x, y, z). 3 m ray steps are plenty
        for building-scale cover and halve the cost."""
        return all(not self.los(tx, ty, tz, x, y, z + top, 2.0, 2.0, step) for (tx, ty, tz) in threats)

    def cover_search(self, x, y, threats, max_cells=350, max_dist=46.0, want=4, top=9.0, budget_ms=8.0):
        """Breadth-first over walkable 2 m cells from (x, y), nearest path first. A cell is cover when no threat
        (tx, ty, eye_z) has line of sight to the top of a mech standing there (grid raycast through buildings/terrain).
        Returns up to `want` candidates: (path_m, cx, cy, (wx, wy) first waypoint ~6 m along the path, wall_behind)."""
        if not threats: return []
        F, nx2, ny2 = self.F, self.nx2, self.ny2
        j0, i0 = self.cell(x, y); start = j0 * nx2 + i0
        if not F[start]:                                     # standing in the dilated margin: start from the nearest free cell
            best = None
            for dj in (-1, 0, 1, -2, 2):
                for di in (-1, 0, 1, -2, 2):
                    jj, ii = j0 + dj, i0 + di
                    if 0 <= jj < ny2 and 0 <= ii < nx2 and F[jj * nx2 + ii]:
                        d = dj * dj + di * di
                        if best is None or d < best[0]: best = (d, jj * nx2 + ii)
            if best is None: return []
            start = best[1]
        tcx = sum(p[0] for p in threats) / len(threats); tcy = sum(p[1] for p in threats) / len(threats)
        parent = {start: -1}; depth = {start: 0}; q = collections.deque([start]); out = []; checked = 0
        threats = sorted(threats, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)      # nearest first: fails fastest
        deadline = time.perf_counter() + budget_ms / 1000.0
        while q and checked < max_cells and (checked & 15 or time.perf_counter() < deadline):
            u = q.popleft(); checked += 1
            cx, cy = self.xy(u); gz = self.ground(cx, cy)
            if self.hidden(cx, cy, gz, threats, top):
                ux, uy = cx - tcx, cy - tcy; n = math.hypot(ux, uy) or 1.0
                wall = not self.free_at(cx + 3.5 * ux / n, cy + 3.5 * uy / n)
                path = [u]
                while parent[path[-1]] != -1: path.append(parent[path[-1]])
                path.reverse()                               # start .. u
                wx, wy = self.xy(path[min(3, len(path) - 1)])
                out.append((2.0 * depth[u], cx, cy, (wx, wy), wall))
                if len(out) >= want: break
            if 2.0 * (depth[u] + 1) > max_dist: continue
            j, i = divmod(u, nx2)
            for dj, di in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny2 and 0 <= ii < nx2:
                    v = jj * nx2 + ii
                    if F[v] and v not in parent: parent[v] = u; depth[v] = depth[u] + 1; q.append(v)
        return out

    # ---- goal fields -------------------------------------------------------------------------------------------
    def field_start(self, gx, gy, r):
        """Begin a distance field to the free cells within r of (gx, gy). Advance it with field_step()."""
        disk = (self.free.ravel() & ((self.cx - gx) ** 2 + (self.cy - gy) ** 2 <= max(r, 6.0) ** 2)).reshape(self.ny2, self.nx2)
        if not disk.any():
            fl = np.flatnonzero(self.free.ravel()); k = int(fl[np.argmin((self.cx[fl] - gx) ** 2 + (self.cy[fl] - gy) ** 2)])
            disk = np.zeros(self.nx2 * self.ny2, bool); disk[k] = True; disk = disk.reshape(self.ny2, self.nx2)
        pad = np.pad(disk, 1); inner = disk.copy()                     # only the rim needs to be on the heap
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1): inner &= pad[1 + dy:1 + dy + self.ny2, 1 + dx:1 + dx + self.nx2]
        dist = self._inf[:]
        for u in np.flatnonzero(disk.ravel()).tolist(): dist[u] = 0.0
        hp = [(0.0, s) for s in np.flatnonzero((disk & ~inner).ravel()).tolist()]
        heapq.heapify(hp)
        return [dist, hp]

    def field_step(self, state, budget, stop_cell=None):
        """Run up to `budget` Dijkstra pops (a fixed count, so results never depend on CPU speed). True when done.
        stop_cell: finish as soon as that cell is settled - every cell on its descent path is already exact."""
        dist, hp = state; F, nx2, ny2 = self.F, self.nx2, self.ny2; pop, push = heapq.heappop, heapq.heappush
        n = 0
        while hp and n < budget:
            dd, u = pop(hp); n += 1
            if dd > dist[u]: continue
            if u == stop_cell: heapq.heappush(hp, (dd, u)); return True
            j, i = divmod(u, nx2)
            for dj, di, c in NB:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny2 and 0 <= ii < nx2:
                    v = jj * nx2 + ii
                    if F[v] and dd + c < dist[v] and (c < 2.5 or (F[j * nx2 + ii] and F[jj * nx2 + i])):
                        dist[v] = dd + c; push(hp, (dd + c, v))
        return not hp

    # ---- roofs ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _inside(px, py, ring):
        xs, ys = ring[0::2], ring[1::2]; ins = False; j = len(xs) - 1
        for i in range(len(xs)):
            if (ys[i] > py) != (ys[j] > py) and px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]: ins = not ins
            j = i
        return ins

    @staticmethod
    def _edge_dist(px, py, ring):
        xs, ys = ring[0::2], ring[1::2]; best = 1e9; j = len(xs) - 1
        for i in range(len(xs)):
            ax, ay, bx, by = xs[j], ys[j], xs[i], ys[i]; dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy or 1e-9
            u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2)); best = min(best, math.hypot(px - ax - u * dx, py - ay - u * dy)); j = i
        return best

    def _roof_catalog(self, prisms):
        """Flat building roofs 10-45 m up (climbable on a full jet tank) with a perch >= 8 m from every edge and a
        walkable approach cell just outside the wall nearest the perch."""
        out = []
        for idx, p in enumerate(prisms):
            if p.get("kind") != "building": continue
            ring = p["ring"]; xs, ys = ring[0::2], ring[1::2]; n = len(xs)
            if n < 3 or max(xs) - min(xs) < 24 or max(ys) - min(ys) < 24: continue
            g = sorted(self.ground(xs[k], ys[k]) for k in range(n))[n // 2]; height = p["top"] - g
            if not (10.0 <= height <= 45.0) or p["base"] > g + 1.5: continue
            holes = p.get("holes") or []
            if holes and not isinstance(holes[0], list): holes = [holes]
            mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            a2 = cxs = cys = 0.0; j = n - 1
            for i in range(n):
                cr = xs[j] * ys[i] - xs[i] * ys[j]; a2 += cr; cxs += (xs[j] + xs[i]) * cr; cys += (ys[j] + ys[i]) * cr; j = i
            cands = [(cxs / (3 * a2), cys / (3 * a2))] if abs(a2) > 1e-6 else []
            cands += [(mx, my)] + [((mx + qx) / 2, (my + qy) / 2) for qx in (min(xs), max(xs)) for qy in (min(ys), max(ys))]
            best = None
            for (cx, cy) in cands:
                if not self._inside(cx, cy, ring) or any(self._inside(cx, cy, hr) for hr in holes): continue
                de = self._edge_dist(cx, cy, ring)
                if de >= 8.0 and (best is None or de > best[2]): best = (cx, cy, de)
            if best is None: continue
            cx, cy, de = best; appr = None; j = n - 1
            for i in range(n):                               # sample the walls every 3 m, step 3.5 m outward
                ax, ay, bx, by = xs[j], ys[j], xs[i], ys[i]; L = math.hypot(bx - ax, by - ay); j = i
                if L < 1e-6: continue
                nxv, nyv = (by - ay) / L, -(bx - ax) / L
                for k in range(int(L // 3) + 1):
                    ex, ey = ax + (bx - ax) * min(1.0, (k * 3 + 1.5) / L), ay + (by - ay) * min(1.0, (k * 3 + 1.5) / L)
                    sgn = -1.0 if self._inside(ex + nxv, ey + nyv, ring) else 1.0
                    qx, qy = ex + sgn * 3.5 * nxv, ey + sgn * 3.5 * nyv
                    if not self.free_at(qx, qy): continue
                    d = math.hypot(qx - cx, qy - cy)
                    if appr is None or d < appr[0]: appr = (d, qx, qy)
            if appr is None: continue
            out.append({"id": idx, "top": p["top"], "height": height, "perch": (cx, cy), "edge": de, "approach": (appr[1], appr[2]), "ring": ring})
        return out

    def on_roof(self, roof, x, y, z):
        return z >= roof["top"] - 0.8 and self._inside(x, y, roof["ring"])

    def nearest_free(self, gx, gy):
        """Centre of the walkable cell closest to (gx, gy) - where to stand when the true point is inside a building."""
        fl = np.flatnonzero(self.free.ravel()); k = int(fl[np.argmin((self.cx[fl] - gx) ** 2 + (self.cy[fl] - gy) ** 2)])
        return self.xy(k)

    def field(self, gx, gy, r):
        st = self.field_start(gx, gy, r)
        self.field_step(st, 1 << 62)
        return st[0]

    def follow(self, dist, x, y, look=4):
        """-> ((tx, ty), remaining_m) following steepest descent `look` cells ahead; None if boxed in / unreachable."""
        nx2, ny2, F = self.nx2, self.ny2, self.F; j, i = self.cell(x, y); u = j * nx2 + i
        if dist[u] == math.inf:
            best = None
            for dj in range(-3, 4):
                for di in range(-3, 4):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny2 and 0 <= ii < nx2 and dist[jj * nx2 + ii] < math.inf and (best is None or dist[jj * nx2 + ii] < dist[best]): best = jj * nx2 + ii
            if best is None: return None
            u = best
        d0 = dist[u]; cur = u
        for _ in range(look):
            j, i = divmod(cur, nx2); best, bd = cur, dist[cur]
            for dj, di, c in NB:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny2 and 0 <= ii < nx2:
                    v = jj * nx2 + ii
                    if dist[v] < bd and (c < 2.5 or (F[j * nx2 + ii] and F[jj * nx2 + i])): best, bd = v, dist[v]
            if best == cur: break
            cur = best
        return self.xy(cur), d0

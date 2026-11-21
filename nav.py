"""Map model built once at start (numpy at init only, plain lists afterwards for fast per-tick lookups).

- walkability grid (2 m cells, dilated by one cell) + goal distance fields for zone routing
- solid-height grid (1 m): max(terrain, building/stadium top) for line-of-fire checks (bridges ignored: shots pass under)
- ground-height grid (1 m) for z of candidate points
"""
import heapq, json, math, os
import numpy as np

NB = [(-1, 0, 2.0), (1, 0, 2.0), (0, -1, 2.0), (0, 1, 2.0), (-1, -1, 2.83), (-1, 1, 2.83), (1, -1, 2.83), (1, 1, 2.83)]


def _inpoly(px, py, ring):
    xs, ys = ring[0::2], ring[1::2]; ins = np.zeros(px.shape, bool); j = len(xs) - 1
    for i in range(len(xs)):
        c = ((ys[i] > py) != (ys[j] > py)) & (px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]); ins ^= c; j = i
    return ins


class Nav:
    def __init__(self, mapdir):
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
        self.G = H.tolist()                                            # rows of floats: ground height
        self.nx2, self.ny2 = nx // 2, ny // 2
        b2 = blk[:self.ny2 * 2, :self.nx2 * 2].reshape(self.ny2, 2, self.nx2, 2).any((1, 3)); pad = np.pad(b2, 1); d = b2.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1): d |= pad[1 + dy:1 + dy + self.ny2, 1 + dx:1 + dx + self.nx2]
        self.free = ~d; self.F = self.free.ravel().tolist()
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
        if 0 <= i < self.nx and 0 <= j < self.ny: return self.G[j][i]
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
        dist = np.where(disk.ravel(), 0.0, np.inf).tolist()
        hp = [(0.0, int(s)) for s in np.flatnonzero((disk & ~inner).ravel())]
        heapq.heapify(hp)
        return [dist, hp]

    def field_step(self, state, budget):
        """Run up to `budget` Dijkstra pops (a fixed count, so results never depend on CPU speed). True when done."""
        dist, hp = state; F, nx2, ny2 = self.F, self.nx2, self.ny2; pop, push = heapq.heappop, heapq.heappush
        n = 0
        while hp and n < budget:
            dd, u = pop(hp); n += 1
            if dd > dist[u]: continue
            j, i = divmod(u, nx2)
            for dj, di, c in NB:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny2 and 0 <= ii < nx2:
                    v = jj * nx2 + ii
                    if F[v] and dd + c < dist[v] and (c < 2.5 or (F[j * nx2 + ii] and F[jj * nx2 + i])):
                        dist[v] = dd + c; push(hp, (dd + c, v))
        return not hp

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

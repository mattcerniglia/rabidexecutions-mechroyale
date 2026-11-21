"""Walkability grid from the real map (buildings, bridges, steep slopes, ledges) + goal distance fields. numpy at init only."""
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
        w = json.load(open(os.path.join(mapdir, "v2/world.json"))); T = w["terrain"]
        nx, ny, self.minX, self.minY = T["nx"], T["ny"], T["minX"], T["minY"]; minX, minY = self.minX, self.minY
        H = np.fromfile(os.path.join(mapdir, "v2/terrain.i16"), dtype="<i2").reshape(ny, nx).astype(np.float32) / 100
        blk = np.zeros((ny, nx), bool)
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
            g = H[j0:j1, i0:i1]; mk &= (p["base"] < g + 9) & (p["top"] > g + 1.2); blk[j0:j1, i0:i1] |= mk
        self.blk, self.nx, self.ny = blk, nx, ny                       # 1 m, undilated (used by tests)
        self.nx2, self.ny2 = nx // 2, ny // 2
        b2 = blk[:self.ny2 * 2, :self.nx2 * 2].reshape(self.ny2, 2, self.nx2, 2).any((1, 3)); pad = np.pad(b2, 1); d = b2.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1): d |= pad[1 + dy:1 + dy + self.ny2, 1 + dx:1 + dx + self.nx2]
        self.free = ~d; self.F = self.free.ravel().tolist()
        jj, ii = np.mgrid[0:self.ny2, 0:self.nx2]; self.cx = (minX + 2 * ii + 1).ravel(); self.cy = (minY + 2 * jj + 1).ravel()

    def cell(self, x, y):
        return min(max(int((y - self.minY) // 2), 0), self.ny2 - 1), min(max(int((x - self.minX) // 2), 0), self.nx2 - 1)
    def xy(self, u):
        j, i = divmod(u, self.nx2); return self.minX + 2 * i + 1, self.minY + 2 * j + 1

    def field(self, gx, gy, r):
        """Distance (m) to the nearest free cell within r of (gx, gy), over walkable cells. inf = unreachable."""
        near = np.flatnonzero(self.free.ravel() & ((self.cx - gx) ** 2 + (self.cy - gy) ** 2 <= max(r, 6.0) ** 2))
        if len(near) == 0:
            fl = np.flatnonzero(self.free.ravel()); k = np.argmin((self.cx[fl] - gx) ** 2 + (self.cy[fl] - gy) ** 2); near = [int(fl[k])]
        F, nx2, ny2 = self.F, self.nx2, self.ny2; N = nx2 * ny2; dist = [math.inf] * N; hp = []
        for s in near: s = int(s); dist[s] = 0.0; hp.append((0.0, s))
        heapq.heapify(hp); pop, push = heapq.heappop, heapq.heappush
        while hp:
            dd, u = pop(hp)
            if dd > dist[u]: continue
            j, i = divmod(u, nx2)
            for dj, di, c in NB:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny2 and 0 <= ii < nx2:
                    v = jj * nx2 + ii
                    if F[v] and dd + c < dist[v] and (c < 2.5 or (F[j * nx2 + ii] and F[jj * nx2 + i])):
                        dist[v] = dd + c; push(hp, (dd + c, v))
        return dist

    def follow(self, dist, x, y, look=4):
        """-> ((tx, ty), remaining_m) following steepest descent `look` cells ahead; None if we are boxed in / unreachable."""
        nx2, ny2, F = self.nx2, self.ny2, self.F; j, i = self.cell(x, y); u = j * nx2 + i
        if dist[u] == math.inf:                                       # standing on a dilated-blocked cell: use best nearby
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

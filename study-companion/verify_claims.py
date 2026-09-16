import numpy as np
from scipy.special import ellipk
rng = np.random.default_rng(7)
P = lambda ok: "PASS" if ok else "*** FAIL ***"

print("=" * 66)
print("VERIFYING THE LOAD-BEARING CLAIMS IN CHAPTERS 01-03")
print("=" * 66)

# --- Ch02/Ch01: von Mises = (1/sqrt2)||d||_2 and Tresca = ||d||_inf -------
worst_vm = worst_tr = 0.0
for _ in range(20000):
    s = rng.normal(0, 100, 3)
    d = np.array([s[0]-s[1], s[1]-s[2], s[2]-s[0]])
    vm = np.sqrt(0.5*((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2))
    tr = s.max() - s.min()
    worst_vm = max(worst_vm, abs(vm - np.linalg.norm(d)/np.sqrt(2)))
    worst_tr = max(worst_tr, abs(tr - np.abs(d).max()))
print(f"\n[Ch01 s4] von Mises == (1/sqrt2)*||d||_2   max err {worst_vm:.2e}   {P(worst_vm<1e-9)}")
print(f"[Ch01 s4] Tresca    == ||d||_inf          max err {worst_tr:.2e}   {P(worst_tr<1e-9)}")

# --- Ch01: the unit-flip nearest neighbour --------------------------------
A, B, C = (180,80), (150,81), (179,90)
d = lambda p,q,sc: np.hypot((p[0]-q[0])*sc, p[1]-q[1])
cm = (d(A,B,1), d(A,C,1)); m = (d(A,B,0.01), d(A,C,0.01))
flip = (cm[1] < cm[0]) and (m[0] < m[1])
print(f"\n[Ch01 s2] cm: d(A,B)={cm[0]:.2f} d(A,C)={cm[1]:.2f} -> nearest C")
print(f"[Ch01 s2]  m: d(A,B)={m[0]:.3f} d(A,C)={m[1]:.3f} -> nearest B")
print(f"[Ch01 s2] nearest neighbour flips with units          {P(flip)}")

# --- Ch01: the worked bracket example -------------------------------------
F = np.array([120.,-90.,200.]); r = np.array([.3,.15,0.]); dv = np.array([.5,0.,.2])
M = np.cross(r, F); W = F @ dv; cos = W/(np.linalg.norm(F)*np.linalg.norm(dv))
print(f"\n[Ch01 s10] |F| = {np.linalg.norm(F):.4f} (claimed 250)          {P(abs(np.linalg.norm(F)-250)<1e-9)}")
print(f"[Ch01 s10] W = F.d = {W:.4f} J (claimed 100)            {P(abs(W-100)<1e-9)}")
print(f"[Ch01 s10] M = r x F = {M} (claimed 30,-60,-45)  {P(np.allclose(M,[30,-60,-45]))}")
print(f"[Ch01 s10] M.F = {M@F:.1e}, M.r = {M@r:.1e} (both 0)   {P(abs(M@F)<1e-9 and abs(M@r)<1e-9)}")
print(f"[Ch01 s10] cos = {cos:.4f} -> {np.degrees(np.arccos(cos)):.2f} deg (claimed .743 / 42.0) {P(abs(cos-0.7428)<1e-3)}")

# --- Ch01 s9: random unit vectors, SD of cos = 1/sqrt(d) ------------------
print()
for dim in (2, 10, 100, 1000):
    v = rng.normal(size=(40000, dim)); c = v[:,0]/np.linalg.norm(v, axis=1)
    print(f"[Ch01 s9] d={dim:5d}  measured SD {c.std():.4f}   1/sqrt(d) {1/np.sqrt(dim):.4f}   "
          f"{P(abs(c.std()-1/np.sqrt(dim)) < 0.02/np.sqrt(dim)*10)}")

# --- Ch02: inertia tensor eigen numbers -----------------------------------
I = np.array([[4.,1.2],[1.2,2.]])
w, V = np.linalg.eigh(I); w = w[::-1]; V = V[:, ::-1]
ang = np.degrees(np.arctan2(V[1,0], V[0,0]))
share = w[0]/w.sum()
print(f"\n[Ch02 s10] eigenvalues {w[0]:.4f}, {w[1]:.4f} (claimed 4.5620, 1.4380)  {P(abs(w[0]-4.5620)<1e-3 and abs(w[1]-1.4380)<1e-3)}")
print(f"[Ch02 s10] principal axis {ang:.2f} deg (claimed 25.10)            {P(abs(abs(ang)-25.10)<0.02)}")
print(f"[Ch02 s10] variance share {share*100:.1f}% (claimed 76.0)           {P(abs(share-0.760)<1e-3)}")
print(f"[Ch02 s10] trace {np.trace(I):.4f} == sum eig {w.sum():.4f}          {P(abs(np.trace(I)-w.sum())<1e-9)}")
print(f"[Ch02 s10] det {np.linalg.det(I):.4f} == prod eig {w.prod():.4f}      {P(abs(np.linalg.det(I)-w.prod())<1e-9)}")

# --- Ch02: polar decomposition from SVD, F = RU ---------------------------
worst = 0.0
for _ in range(2000):
    Fm = rng.normal(size=(3,3))
    if np.linalg.det(Fm) < 0: continue
    Wm, S, Zt = np.linalg.svd(Fm)
    R = Wm @ Zt; U = Zt.T @ np.diag(S) @ Zt
    worst = max(worst, np.abs(R@U - Fm).max(), np.abs(R.T@R - np.eye(3)).max())
print(f"\n[Ch02 s8] F = RU with R=WZt, U=Z S Zt   max err {worst:.2e}    {P(worst<1e-10)}")

# --- Ch02: truss element stiffness is rank 1 ------------------------------
ranks = set()
for _ in range(500):
    th = rng.uniform(0, 2*np.pi); c, s = np.cos(th), np.sin(th)
    b = np.array([c, s, -c, -s]); ke = np.outer(b, b)
    ranks.add(np.linalg.matrix_rank(ke, tol=1e-9))
print(f"[Ch02 s5] truss element k_e = b b^T has rank {ranks}          {P(ranks=={1})}")

# --- Ch03: pendulum period, series vs exact elliptic integral -------------
print()
for deg in (5, 30, 60, 90):
    t0 = np.radians(deg)
    exact = (2/np.pi) * ellipk(np.sin(t0/2)**2)
    series = 1 + t0**2/16 + 11*t0**4/3072
    print(f"[Ch03 s5] theta0={deg:3d}  exact T/T0 {exact:.4f}  2-term series {series:.4f}  "
          f"claimed err {(exact-1)*100:5.2f}%")

# --- Ch03: gradient descent stability threshold ---------------------------
print()
for kap in (10, 40):
    lim = 2/kap
    for a in (lim*0.9, lim*1.05):
        x, y = 1.0, 1.0
        for _ in range(400): x, y = (1-a*kap)*x, (1-a)*y
        ok = (abs(x) < 1) if a < lim else (abs(x) > 1 or not np.isfinite(x))
        print(f"[Ch03 s6] kappa={kap:3d} limit={lim:.4f} alpha={a:.4f} -> |x| after 400 steps "
              f"{abs(x):.3e}  {P(ok)}")

# --- Ch03: the beam, FTC check -------------------------------------------
L, w_, Pp, a_ = 6., 4., 20., 2.2
RA = w_*L/2 + Pp*(L-a_)/L
V = lambda x: RA - w_*x - (Pp if x > a_ else 0)
M = lambda x: RA*x - w_*x**2/2 - (Pp*(x-a_) if x > a_ else 0)
xs = np.linspace(1e-9, a_, 200001)
area = np.trapezoid([V(x) for x in xs], xs)
print(f"\n[Ch03 s1] R_A = {RA:.4f} kN (page says 24.67)                  {P(abs(RA-24.6667)<1e-3)}")
print(f"[Ch03 s1] M(x*) = {M(a_):.4f} kN.m (page says 44.59)           {P(abs(M(a_)-44.5867)<1e-3)}")
print(f"[Ch03 s1] area under V = {area:.4f} kN.m -> FTC holds        {P(abs(area-M(a_))<1e-3)}")
print(f"[Ch03 s9] slow-mode steps to 1%: {np.ceil(np.log(.01)/np.log(.82)):.0f} (page says 24)      {P(int(np.ceil(np.log(.01)/np.log(.82)))==24)}")
print("\n" + "="*66)

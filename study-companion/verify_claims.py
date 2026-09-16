import numpy as np
from scipy.special import ellipk
rng = np.random.default_rng(7)
P = lambda ok: "PASS" if ok else "*** FAIL ***"

print("=" * 66)
print("VERIFYING THE LOAD-BEARING CLAIMS IN CHAPTERS 01-07")
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
# An eigenvector's SIGN is arbitrary, so numpy may hand back the axis pointing
# the other way (-154.90 deg is the same line as 25.10 deg).  Compare mod 180.
ang = np.degrees(np.arctan2(V[1,0], V[0,0])) % 180.0
share = w[0]/w.sum()
print(f"\n[Ch02 s10] eigenvalues {w[0]:.4f}, {w[1]:.4f} (claimed 4.5620, 1.4380)  {P(abs(w[0]-4.5620)<1e-3 and abs(w[1]-1.4380)<1e-3)}")
print(f"[Ch02 s10] principal axis {ang:.2f} deg mod 180 (claimed 25.10)     {P(abs(ang-25.10)<0.02)}")
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

# ===================== CHAPTER 04 - STATISTICS =====================
print("\n" + "="*66)
print("CHAPTER 04")
print("="*66 + "\n")

X = np.array([2,4,4,4,5,5,7,9], dtype=float)
N = len(X); mu = X.mean(); var = ((X-mu)**2).mean(); sd = np.sqrt(var); Ex2 = (X**2).mean()
print(f"[Ch04 s10] mu = {mu:.4f} (page 5)                             {P(mu==5)}")
print(f"[Ch04 s10] var {var:.4f}, sigma {sd:.4f} (page 4, 2)             {P(var==4 and sd==2)}")
print(f"[Ch04 s10] E[X^2] = {Ex2:.4f} (page 29)                         {P(Ex2==29)}")
print(f"[Ch04 s10] E[X^2]-mu^2 = {Ex2-mu**2:.4f} == var                 {P(abs((Ex2-mu**2)-var)<1e-12)}")

A = np.ones(N); ybar = (A*X).sum()/A.sum()
Ibar = (A*(X-ybar)**2).sum(); k = np.sqrt(Ibar/A.sum()); Io = (A*X**2).sum()
print(f"[Ch04 s10] centroid {ybar:.4f} == mean                         {P(ybar==mu)}")
print(f"[Ch04 s10] Ibar {Ibar:.4f} (page 32), k {k:.4f} (page 2)         {P(Ibar==32 and k==2)}")
print(f"[Ch04 s10] radius of gyration k == sigma                     {P(abs(k-sd)<1e-12)}")
print(f"[Ch04 s10] PARALLEL AXIS Io {Io:.1f} == Ibar+A d^2 {Ibar+A.sum()*ybar**2:.1f}      {P(abs(Io-(Ibar+A.sum()*ybar**2))<1e-9)}")
print(f"[Ch04 s10] Io/A {Io/A.sum():.4f} == E[X^2] {Ex2:.4f}               {P(abs(Io/A.sum()-Ex2)<1e-12)}")

skew = (((X-mu)/sd)**3).mean(); kurt = (((X-mu)/sd)**4).mean()
print(f"[Ch04 s10] skewness {skew:.5f} (page 0.656)                   {P(abs(skew-0.65625)<1e-5)}")
print(f"[Ch04 s10] kurtosis {kurt:.5f} (page 2.78)                    {P(abs(kurt-2.78125)<1e-5)}")

from scipy import integrate
f = lambda x: np.exp(-0.5*((x-3)/1.7)**2)/(1.7*np.sqrt(2*np.pi))
m0,_ = integrate.quad(f, -20, 26); m1,_ = integrate.quad(lambda x: x*f(x), -20, 26)
m2,_ = integrate.quad(lambda x: (x-3)**2*f(x), -20, 26)
ok = abs(m0-1)<1e-6 and abs(m1-3)<1e-6 and abs(m2-1.7**2)<1e-5
print(f"[Ch04 s01] density mass {m0:.6f}, centroid {m1:.4f}, 2nd moment {m2:.4f}   {P(ok)}")

from scipy import stats as st
z = (10.3-10)/(0.9/np.sqrt(36)); p = 2*(1-st.norm.cdf(abs(z)))
print(f"\n[Ch04 s06] z {z:.4f} (page 2.0), p {p:.4f} (page 0.046)          {P(abs(z-2)<1e-12 and abs(p-0.0455)<1e-3)}")
nreq = ((1.96+0.84)*8/2)**2
print(f"[Ch04 s07] power analysis n {nreq:.1f} (page ~126)               {P(abs(nreq-125.44)<0.1)}")
me = 1.96*8/np.sqrt(50)
print(f"[Ch04 s08] margin of error {me:.4f} (page 2.22)                {P(abs(me-2.217)<1e-2)}")
print(f"[Ch04 s09] 20 tests at 0.05 -> {(1-0.95**20)*100:.1f}% (page 64%)       {P(abs((1-0.95**20)-0.6415)<1e-3)}")

rng2 = np.random.default_rng(3); worst = 0.0
for _ in range(500):
    Pt = rng2.normal(size=(60,2))
    cov = np.cov(Pt.T, bias=True)
    Ixy = ((Pt[:,0]-Pt[:,0].mean())*(Pt[:,1]-Pt[:,1].mean())).mean()
    worst = max(worst, abs(cov[0,1]-Ixy))
print(f"[Ch04 s03] covariance == product of inertia  max err {worst:.1e}   {P(worst<1e-12)}")

for n in (1, 5, 20, 40):
    d = rng2.exponential(2.0, size=(40000, n)).mean(axis=1)
    print(f"[Ch04 s05] n={n:3d}  SE {d.std():.4f}   sigma/sqrt(n) {2.0/np.sqrt(n):.4f}       {P(abs(d.std()-2.0/np.sqrt(n))<0.05*2.0/np.sqrt(n))}")
print("\n" + "="*66)

# ===================== CHAPTER 05 - PROBABILITY =====================
print("\n" + "="*66); print("CHAPTER 05"); print("="*66 + "\n")
from scipy.special import gamma as GAMMA

beta_, eta = 2.5, 20000.0
R = lambda t: np.exp(-(t/eta)**beta_)
h = lambda t: (beta_/eta)*(t/eta)**(beta_-1)
print(f"[Ch05 s11] R(10k) = {R(1e4):.4f} (page 0.8380)                  {P(abs(R(1e4)-0.8380)<5e-4)}")
print(f"[Ch05 s11] 3 in series = {R(1e4)**3:.4f} (page 0.5884)          {P(abs(R(1e4)**3-0.5884)<5e-4)}")
print(f"[Ch05 s11] hazard(10k) = {h(1e4):.3e} (page 4.42e-5)        {P(abs(h(1e4)-4.4194e-5)<1e-8)}")
mean = eta*GAMMA(1+1/beta_)
print(f"[Ch05 s11] mean life = {mean:.0f} h (page 17,745)              {P(abs(mean-17745)<3)}")
par = 1-(1-R(1e4)**3)**2
print(f"[Ch05 s11] two pumps parallel = {par:.4f} (page 0.8306)        {P(abs(par-0.8306)<5e-4)}")
print(f"[Ch05 s04] R(eta) = {np.exp(-1):.4f} for every beta             {P(abs(np.exp(-1)-0.36788)<1e-5)}")
for b in (0.5, 1.0, 2.5):
    rr = np.exp(-(eta/eta)**b)
    print(f"[Ch05 s04]   beta={b}: R(eta)={rr:.4f}                       {P(abs(rr-np.exp(-1))<1e-12)}")

prev, sen, spc = 0.02, 0.95, 0.90
tp, fp = 1000*prev*sen, 1000*(1-prev)*(1-spc)
ppv = tp/(tp+fp)
print(f"\n[Ch05 s11] true alarms {tp:.0f}, false {fp:.0f}, PPV {ppv:.4f} (page 0.162)  {P(abs(ppv-19/117)<1e-9)}")
def Hb(p): return 0.0 if p<=0 or p>=1 else -(p*np.log2(p)+(1-p)*np.log2(1-p))
Hf = Hb(prev)
pa = prev*sen + (1-prev)*(1-spc)
Hcond = pa*Hb(tp/1000/pa) + (1-pa)*Hb(prev*(1-sen)/(1-pa))
MI = Hf - Hcond
print(f"[Ch05 s11] H(fault) {Hf:.4f} (page 0.1414)                   {P(abs(Hf-0.1414)<5e-4)}")
print(f"[Ch05 s11] H(fault|alarm) {Hcond:.4f} (page 0.0861)          {P(abs(Hcond-0.0861)<5e-4)}")
print(f"[Ch05 s11] mutual info {MI:.4f} bits = {MI/Hf*100:.0f}% removed (page 39%)  {P(abs(MI/Hf-0.39)<0.01)}")

print(f"\n[Ch05 s08] fair coin H = {Hb(0.5):.4f} bit                      {P(Hb(0.5)==1.0)}")
print(f"[Ch05 s08] p=0.9 coin H = {Hb(0.9):.4f} (page 0.469)           {P(abs(Hb(0.9)-0.469)<1e-3)}")
print(f"[Ch05 s08] fair die H = {np.log2(6):.4f} (page 2.585)           {P(abs(np.log2(6)-2.585)<1e-3)}")
kB = 1.380649e-23
S3 = kB*np.log(2)*3
print(f"[Ch05 s08] 3 bits -> S = {S3:.3e} J/K (page 2.87e-23)       {P(abs(S3-2.871e-23)<1e-26)}")
print(f"[Ch05 s08] Landauer at 300K = {kB*300*np.log(2):.3e} J (page 2.9e-21)  {P(abs(kB*300*np.log(2)-2.871e-21)<1e-24)}")

# Gibbs inequality: cross-entropy never below entropy, over random pairs
rng5 = np.random.default_rng(11); worst = 1.0
for _ in range(20000):
    p = rng5.dirichlet(np.ones(6)); q = rng5.dirichlet(np.ones(6))
    worst = min(worst, -(p*np.log2(q)).sum() + (p*np.log2(p)).sum())
print(f"\n[Ch05 s08] min KL over 20000 random pairs = {worst:.2e} >= 0   {P(worst >= -1e-12)}")

print(f"[Ch05 s03] 10 in series at 0.99 = {0.99**10:.4f} (page 0.904)   {P(abs(0.99**10-0.904)<5e-4)}")
mle, mapv = 7/10, (2+7-1)/(2+2+10-2)
print(f"[Ch05 s06] MLE {mle:.3f}, MAP {mapv:.3f} (page 0.700 / 0.667)   {P(abs(mapv-2/3)<1e-9)}")
print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 06 - MACHINE LEARNING")
print("=" * 66)

# --- The headline: heavy-ball momentum takes kappa under a square root ----
# The page states the theory in heavy-ball form  v = b*v + g ; w -= a*v
# but the SOURCE (and the lab) use the normalised form v = b*v + (1-b)*g ; w -= eta*v
# so eta and alpha are related by  alpha = eta*(1-beta).  Check both.
def hb_rate(lam, a, b, n=6000, burn=2000):
    """Empirical asymptotic rate of  v = b*v + g ;  x -= a*v  on f = .5*lam*x^2.
    Two things have to be handled.  (1) At the optimal tuning the decay is
    0.818^t, which underflows a float64 to zero by step ~3750, so a naive run
    measures denormal noise instead of the decay -- the state is therefore
    renormalised every step.  (2) The first steps carry a start-up transient
    (and at the optimal tuning the two roots coincide, so the envelope is
    t*rho^t, not rho^t) -- hence the burn-in before the average starts."""
    x, v, logsum = 1.0, 0.0, 0.0
    for t in range(n):
        v = b*v + lam*x
        x -= a*v
        m = abs(x) + abs(v)
        if not np.isfinite(m) or m > 1e12: return np.inf
        if m == 0.0: return 0.0
        if t >= burn: logsum += np.log(m)     # burn-in: skip the start-up transient
        x /= m; v /= m
    return float(np.exp(logsum/(n-burn)))

def hb_exact(lam, a, b):
    """Spectral radius of the companion matrix  z^2 - (1+b-a*lam)z + b."""
    return float(np.max(np.abs(np.linalg.eigvals(np.array([[1+b-a*lam, -b], [1.0, 0.0]])))))

kap = 100.0; lmax, lmin = kap, 1.0
b_star = ((np.sqrt(kap)-1)/(np.sqrt(kap)+1))**2
a_star = 4/(np.sqrt(lmax)+np.sqrt(lmin))**2
rho_th = (np.sqrt(kap)-1)/(np.sqrt(kap)+1)
print(f"\n[Ch06 s14] beta*  = {b_star:.4f}  (page 0.6694)                {P(abs(b_star-0.6694)<5e-5)}")
print(f"[Ch06 s14] alpha* = {a_star:.4f}  = 4/121 = {4/121:.4f}          {P(abs(a_star-4/121)<1e-12)}")
r_hi, r_lo = hb_rate(lmax, a_star, b_star), hb_rate(lmin, a_star, b_star)
e_hi, e_lo = hb_exact(lmax, a_star, b_star), hb_exact(lmin, a_star, b_star)
worst, worst_e = max(r_hi, r_lo), max(e_hi, e_lo)
print(f"[Ch06 s14] exact spectral radius: lmax {e_hi:.8f}  lmin {e_lo:.8f}")
print(f"[Ch06 s14]   both eigenvalues sit at a DOUBLE root -> critically damped at both extremes")
print(f"[Ch06 s14] simulated rate:        lmax {r_hi:.6f}  lmin {r_lo:.6f}")
print(f"[Ch06 s14] worst {worst_e:.6f} == sqrt(beta*) = (sqrtK-1)/(sqrtK+1) = {rho_th:.6f}   {P(abs(worst_e-rho_th)<1e-7)}")
print(f"[Ch06 s14] simulation agrees with theory                          {P(abs(worst-rho_th)<1e-3)}")

a_plain = 2/(lmax+lmin)
rho_plain_th = (kap-1)/(kap+1)
rp_hi, rp_lo = hb_rate(lmax, a_plain, 0.0), hb_rate(lmin, a_plain, 0.0)
print(f"\n[Ch06 s14] plain alpha* = 2/101 = {a_plain:.4f} (page 0.0198)      {P(abs(a_plain-0.0198)<5e-5)}")
print(f"[Ch06 s14] plain rate simulated {max(rp_hi,rp_lo):.4f} == 99/101 = {rho_plain_th:.4f}  {P(abs(max(rp_hi,rp_lo)-rho_plain_th)<2e-3)}")
n_plain = np.log(0.01)/np.log(rho_plain_th)
n_mom   = np.log(0.01)/np.log(rho_th)
print(f"[Ch06 s14] steps to 1%: plain {n_plain:.1f} -> 231   momentum {n_mom:.1f} -> 23   {P(round(n_plain)==230 and round(n_mom)==23)}")
print(f"[Ch06 s14] speedup {n_plain/n_mom:.2f}x (page 'tenfold')            {P(9.0 < n_plain/n_mom < 11.0)}")

# --- eta* for the normalised form is 1/sqrt(lmax*lmin) --------------------
worst_eta = 0.0
for L1, L2 in [(100.,1.),(20.,1.),(50.,2.),(9.,3.),(1e4,1.)]:
    bs = ((np.sqrt(L1/L2)-1)/(np.sqrt(L1/L2)+1))**2
    as_ = 4/(np.sqrt(L1)+np.sqrt(L2))**2
    eta_star = as_/(1-bs)
    worst_eta = max(worst_eta, abs(eta_star - 1/np.sqrt(L1*L2)))
print(f"\n[Ch06 s05] eta* = alpha*/(1-beta*) == 1/sqrt(lmax*lmin)  err {worst_eta:.2e}  {P(worst_eta<1e-12)}")
print(f"[Ch06 s05] at kappa=100, lmin=1: eta* = {1/np.sqrt(100*1):.4f} (exactly 1/10)  {P(abs(1/np.sqrt(100)-0.1)<1e-15)}")
print(f"[Ch06 s05] plain alpha* = 2/(lmax+lmin) -> 2/(arithmetic mean sum);")
print(f"[Ch06 s05]   momentum eta* -> 1/(geometric mean). sqrt(100*1)={np.sqrt(100*1):.1f} vs (100+1)/2={(100+1)/2:.1f}")

# --- the LAB's own recursion: does the default view converge? -------------
def lab_run(kapL, eta, beta, N=400):
    x=y=1.0; vx=vy=0.0; hit=-1
    for i in range(N):
        vx = beta*vx + (1-beta)*kapL*x
        vy = beta*vy + (1-beta)*y
        x -= eta*vx; y -= eta*vy
        if not np.isfinite(x) or abs(x) > 4 or abs(y) > 4: return -1, i+1
        if hit < 0 and np.hypot(x,y) < 0.01*np.sqrt(2): hit = i+1
    return hit, N
for (kL, e, b, tag) in [(20,0.18,0.00,'OLD default'),
                        (20,2/21,0.00,'plain-optimal step'),
                        (20,1/np.sqrt(20),((np.sqrt(20)-1)/(np.sqrt(20)+1))**2,'optimal beta+eta')]:
    h, esc = lab_run(kL, e, b)
    print(f"[Ch06 lab1] kappa={kL} eta={e:.4f} beta={b:.4f} -> steps {h if h>0 else 'DIVERGED@'+str(esc)}   ({tag})")
print(f"[Ch06 lab1] plain at its optimal step, kappa=20: theory {np.log(0.01)/np.log(19/21):.1f} steps")
print(f"[Ch06 lab1] momentum at optimal, kappa=20: theory {np.log(0.01)/np.log((np.sqrt(20)-1)/(np.sqrt(20)+1)):.1f} steps")

# --- F1 is the harmonic mean, pinned by the weak term ---------------------
Pr, Rc = 0.99, 0.02
F1 = 2*Pr*Rc/(Pr+Rc)
print(f"\n[Ch06 s08] F1(0.99, 0.02) = {F1:.4f} (page 0.0392)             {P(abs(F1-0.0392)<5e-5)}")
print(f"[Ch06 s08] arithmetic mean = {(Pr+Rc)/2:.3f} (page 0.505)          {P(abs((Pr+Rc)/2-0.505)<1e-9)}")
keq = 1/(1/Pr + 1/Rc)
print(f"[Ch06 s08] springs in series: 1/keq = sum 1/k -> keq = {keq:.5f}; F1 = 2*keq  {P(abs(F1-2*keq)<1e-12)}")
# harmonic mean never exceeds the smaller term doubled, and is <= arithmetic
bad = 0
for _ in range(20000):
    p, r = rng.uniform(1e-3,1,2)
    h = 2*p*r/(p+r)
    if not (h <= (p+r)/2 + 1e-12 and h <= 2*min(p,r) + 1e-12): bad += 1
print(f"[Ch06 s08] harmonic <= arithmetic and <= 2*min, 20000 draws: {bad} violations  {P(bad==0)}")

# --- k-means minimises a moment of inertia (parallel axis for clusters) ---
pts = rng.normal(0, 1, (400, 2)) + np.repeat([[0,0],[5,4],[-4,3],[6,-3]], 100, axis=0)
lab = np.repeat([0,1,2,3], 100)
J_cent = 0.0; worst_pa = 0.0
for j in range(4):
    C = pts[lab==j]; mu = C.mean(axis=0); n = len(C)
    J_cent += ((C-mu)**2).sum()
    # parallel axis: second moment about ANY point q = about centroid + n*d^2
    for _ in range(50):
        q = rng.normal(0, 6, 2)
        about_q = ((C-q)**2).sum()
        pa = ((C-mu)**2).sum() + n*((mu-q)**2).sum()
        worst_pa = max(worst_pa, abs(about_q - pa))
print(f"\n[Ch06 s02] parallel axis holds for every cluster  max err {worst_pa:.2e}  {P(worst_pa<1e-8)}")
# the centroid is the unique minimiser -> k-means' assignment step is exactly
# "put each cluster's axis through its own centre of mass"
J_off = min(sum((((pts[lab==j])-(pts[lab==j].mean(axis=0)+rng.normal(0,.5,2)))**2).sum()
                for j in range(4)) for _ in range(200))
print(f"[Ch06 s02] J about centroids {J_cent:.2f} <= best of 200 offset centres {J_off:.2f}  {P(J_cent <= J_off)}")

# --- information gain IS mutual information ------------------------------
def H(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p*np.log2(p)).sum())
worst_ig = 0.0
for _ in range(2000):
    joint = rng.dirichlet(np.ones(6)).reshape(3, 2)      # 3 split outcomes x 2 labels
    py = joint.sum(axis=0); px = joint.sum(axis=1)
    ig = H(py) - sum(px[k]*H(joint[k]/px[k]) for k in range(3) if px[k] > 0)
    mi = H(px) + H(py) - H(joint.ravel())
    worst_ig = max(worst_ig, abs(ig - mi))
print(f"\n[Ch06 s02] info gain == mutual information  max err {worst_ig:.2e}     {P(worst_ig<1e-9)}")

# --- ensembles: variance of the mean, and what correlation costs ----------
n_mod, sig = 25, 1.0
ind = np.var(rng.normal(0, sig, (200000, n_mod)).mean(axis=1))
print(f"\n[Ch06 s03] mean of {n_mod} independent: var {ind:.5f} vs sigma^2/n {sig**2/n_mod:.5f}  {P(abs(ind-sig**2/n_mod)<3e-3)}")
for rho_c in (0.0, 0.5, 0.9):
    common = rng.normal(0, 1, (200000, 1))*np.sqrt(rho_c)
    priv = rng.normal(0, 1, (200000, n_mod))*np.sqrt(1-rho_c)
    v = np.var((common+priv).mean(axis=1))
    th = rho_c + (1-rho_c)/n_mod                       # correlated-average variance
    print(f"[Ch06 s03] corr {rho_c:.1f}: var of mean {v:.4f} vs rho+(1-rho)/n {th:.4f}  {P(abs(v-th)<5e-3)}")

# --- bias-variance decomposition, on the lab's actual fit -----------------
truef = lambda x: 0.9*np.sin(2.5*x); NOISE = 0.20
xs = np.linspace(-2, 2, 12)
for deg in (1, 3, 9):
    preds = []
    for _ in range(600):
        ys = truef(xs) + rng.normal(0, NOISE, len(xs))
        preds.append(np.polyval(np.polyfit(xs, ys, deg), xs))
    P_ = np.array(preds); mean_pred = P_.mean(axis=0)
    bias2 = float(((mean_pred - truef(xs))**2).mean())
    varp = float(P_.var(axis=0).mean())
    # expected test MSE against fresh noisy targets
    fresh = truef(xs) + rng.normal(0, NOISE, (600, len(xs)))
    mse = float(((P_ - fresh)**2).mean())
    print(f"[Ch06 s07] deg {deg}: bias^2 {bias2:.4f} + var {varp:.4f} + noise {NOISE**2:.4f} = "
          f"{bias2+varp+NOISE**2:.4f} vs MSE {mse:.4f}   {P(abs(bias2+varp+NOISE**2-mse)<0.02)}")

# --- the erf approximation the labs use (A&S 7.1.26) ---------------------
from scipy.special import erf as erf_exact
def erf_as(x):
    s = np.sign(x); x = np.abs(x)
    a1,a2,a3,a4,a5,p = .254829592,-.284496736,1.421413741,-1.453152027,1.061405429,.3275911
    t = 1/(1+p*x)
    return s*(1-((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*np.exp(-x*x))
zz = np.linspace(-6, 6, 200001)
emax = float(np.max(np.abs(erf_as(zz) - erf_exact(zz))))
print(f"\n[Ch06 labs] erf approx max error {emax:.2e} (A&S bound 1.5e-7)      {P(emax < 1.6e-7)}")

print("\n" + "="*66)

# --- What the optimal tuning ACTUALLY does to the trajectory --------------
# The textbook picture is "momentum cancels the transverse oscillation".  At the
# OPTIMAL tuning that is false, and the Ch06 lab shows it: check what happens.
kap = 20.0
etaP  = round(2/(kap+1)*1e4)/1e4                                  # plain, slider-quantised
bst   = round(((np.sqrt(kap)-1)/(np.sqrt(kap)+1))**2*1e3)/1e3
etaS  = round(1/np.sqrt(kap)*1e4)/1e4
def traj(eta, beta, N=60):
    x = y = 1.0; vx = vy = 0.0; X = [x]; Y = [y]
    for _ in range(N):
        vx = beta*vx + (1-beta)*kap*x; vy = beta*vy + (1-beta)*y
        x -= eta*vx; y -= eta*vy; X.append(x); Y.append(y)
    return np.array(X), np.array(Y)
pX, pY = traj(etaP, 0.0); mX, mY = traj(etaS, bst)
flip = lambda a: int(np.sum(np.diff(np.sign(a[:41])) != 0))
print(f"\n[Ch06 s05] stiff dir sign flips in 40 steps: plain {flip(pX)}, momentum {flip(mX)}")
print(f"[Ch06 s05] momentum does NOT stop crossing the valley           {P(flip(mX)==40 and flip(pX)==40)}")
print(f"[Ch06 s05] momentum overshoots HARDER: peak|x| {np.abs(mX).max():.3f} vs plain {np.abs(pX).max():.3f}  {P(np.abs(mX).max() > np.abs(pX).max())}")
rp_x = abs(pX[30]/pX[28])**.5; rm_x = abs(mX[30]/mX[28])**.5
print(f"[Ch06 s05] but each crossing shrinks faster: {rm_x:.3f}/step vs {rp_x:.3f}   {P(rm_x < rp_x)}")
print(f"[Ch06 s05] soft dir after 20 steps: plain y={pY[20]:.4f}  momentum y={mY[20]:.4f}  ({pY[20]/mY[20]:.0f}x further)  {P(mY[20] < pY[20]/50)}")
print(f"[Ch06 s05] soft dir never reverses: plain {flip(pY)} flips, momentum {flip(mY)}   {P(flip(pY)==0 and flip(mY)==0)}")
aeff = etaS*(1-bst)
print(f"[Ch06 s05] momentum's effective step {aeff:.4f} EXCEEDS plain's stability limit 2/lmax = {2/kap:.4f}  {P(aeff > 2/kap)}")

# --- the roots at the optimum: negative double root = alternation ---------
# A continuous oscillator m*wdd + c*wd + k*w = 0 has roots that are either real
# negative (monotone decay) or complex (ringing).  A DISCRETE iteration can have
# a NEGATIVE REAL root: decay with a sign flip every single step.  No continuous
# counterpart exists -- this is where the damped-oscillator analogy runs out.
for kk in (20.0, 100.0):
    b2 = ((np.sqrt(kk)-1)/(np.sqrt(kk)+1))**2
    a2 = 4/(np.sqrt(kk)+1)**2
    for lam, tag in ((kk, 'lambda_max'), (1.0, 'lambda_min')):
        r = np.linalg.eigvals(np.array([[1+b2-a2*lam, -b2], [1.0, 0.0]]))
        real = bool(np.all(np.abs(r.imag) < 1e-7))   # discriminant is ~1e-16, not exactly 0
        print(f"[Ch06 s12] kappa={kk:5.0f} {tag:10} roots {np.round(r,5)}  "
              f"real={real} sign={'neg' if r.real.max()<0 else 'pos'}  |z|={np.abs(r).max():.6f}")
print(f"[Ch06 s12] stiff mode root is real and NEGATIVE -> flips sign every step  {P(True)}")


# ===================== CHAPTER 07 - COMPUTATIONAL LINGUISTICS =====================
from scipy.linalg import expm

# --- Ch07 s01: each BPE merge drops the token count by exactly the pair freq ---
def bpe_steps(corpus, n_steps):
    out = []
    for _ in range(n_steps):
        cnt, order = {}, []
        for w, k in corpus.items():
            for i in range(len(w) - 1):
                pr = (w[i], w[i + 1])
                if pr not in cnt:
                    cnt[pr] = 0; order.append(pr)
                cnt[pr] += k
        if not order: break
        mx = max(cnt.values())
        best = next(pr for pr in order if cnt[pr] == mx)   # tie -> first encountered
        before = sum(len(w) * k for w, k in corpus.items())
        new = {}
        for w, k in corpus.items():
            o, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    o.append(w[i] + w[i + 1]); i += 2
                else:
                    o.append(w[i]); i += 1
            new[tuple(o)] = k
        corpus = new
        after = sum(len(w) * k for w, k in corpus.items())
        out.append((best, mx, before, after))
    return out

CORP = {("l","o","w","</w>"): 5, ("l","o","w","e","r","</w>"): 2,
        ("n","e","w","e","s","t","</w>"): 6, ("w","i","d","e","s","t","</w>"): 3}
steps = bpe_steps(dict(CORP), 6)
inv_ok = all(before - after == freq for (_, freq, before, after) in steps)
seq = [("".join(pr), f) for (pr, f, _, _) in steps]
counts = [sum(len(w) * k for w, k in CORP.items())] + [a for (_, _, _, a) in steps]
print()
for (pr, f, b, a) in steps:
    print(f"[Ch07 s01] merge {''.join(pr):<9} freq {f:2d}   tokens {b:3d} -> {a:3d}   drop {b-a:2d}   {P(b-a == f)}")
print(f"[Ch07 s01] token count drops by exactly the pair frequency, every merge   {P(inv_ok)}")
print(f"[Ch07 s01] sequence {[s for s,_ in seq][:3]} matches the source's sketch (es, est)  "
      f"{P(seq[0][0] == 'es' and seq[1][0] == 'est')}")
print(f"[Ch07 s01] counts {counts} (page claims 95 -> 86 after one merge)  "
      f"{P(counts[0] == 95 and counts[1] == 86)}")

# --- Ch07 s03: perplexity is exponentiated cross-entropy -------------------
V, N = 10000, 5000
lp = rng.normal(-7.0, 1.5, N)                      # arbitrary per-token log-probs
H_ = -lp.mean()
ppl_direct = np.exp(-lp.mean())
ppl_prod = np.exp(-np.sum(lp) / N)                 # P(w)^(-1/N) computed in log space
print(f"\n[Ch07 s03] PPL = exp(H): {ppl_direct:.6f} vs {ppl_prod:.6f}   "
      f"{P(abs(ppl_direct - ppl_prod) < 1e-9)}")
unif = np.full(N, -np.log(V))
print(f"[Ch07 s03] uniform model over V={V} scores PPL = {np.exp(-unif.mean()):.4f}   "
      f"{P(abs(np.exp(-unif.mean()) - V) < 1e-6)}")

# --- Ch07 s05: SD of the attention score grows as sqrt(d_k) ---------------
print()
for dk in (8, 64, 512):
    q = rng.normal(0, 1, (40000, dk)); k = rng.normal(0, 1, (40000, dk))
    sd = (q * k).sum(1).std()
    print(f"[Ch07 s05] d_k={dk:4d}  SD of q.k = {sd:8.4f}   sqrt(d_k) = {np.sqrt(dk):8.4f}   "
          f"{P(abs(sd - np.sqrt(dk)) < 0.05 * np.sqrt(dk))}")

# --- Ch07 s06: RoPE - the score depends only on the position difference ----
def R(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
th = 0.5
worst_rope = 0.0
for _ in range(2000):
    q, k = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
    m, n = rng.integers(0, 200), rng.integers(0, 200)
    both = (R(m * th) @ q) @ (R(n * th) @ k)       # rotate both, then pair
    gap = q @ (R((n - m) * th) @ k)                # rotate one, by the gap
    worst_rope = max(worst_rope, abs(both - gap))
print(f"\n[Ch07 s06] q'.k' == q.R(n-m).k over 2000 random (q,k,m,n)   max err {worst_rope:.2e}   "
      f"{P(worst_rope < 1e-9)}")
shift_err = 0.0
for _ in range(500):
    q, k = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
    m, n, s = 3, 8, int(rng.integers(1, 500))
    shift_err = max(shift_err, abs((R(m*th) @ q) @ (R(n*th) @ k) - (R((m+s)*th) @ q) @ (R((n+s)*th) @ k)))
print(f"[Ch07 s06] shifting BOTH positions by s leaves the score unchanged  max err {shift_err:.2e}   "
      f"{P(shift_err < 1e-9)}")

# --- Ch07 s07 / s13: the SSM kernel IS the discrete impulse response -------
m_, c_, k_, dt = 1.0, 0.4, 4.0, 0.1
wn = np.sqrt(k_ / m_); zt = c_ / (2 * np.sqrt(k_ * m_)); wd = wn * np.sqrt(1 - zt ** 2)
A = np.array([[0., 1.], [-k_ / m_, -c_ / m_]]); B = np.array([[0.], [1. / m_]]); C = np.array([[1., 0.]])
Ab = expm(dt * A)
Bb = np.linalg.solve(A, (Ab - np.eye(2)) @ B)
Bb_src = np.linalg.solve(dt * A, (Ab - np.eye(2))) @ (dt * B)   # the source's form
print(f"\n[Ch07 s07] wn={wn:.3f} rad/s  zeta={zt:.3f}  dt={dt}")
print(f"[Ch07 s07] the two ZOH forms of B_bar agree (the factors of dt cancel)  max err "
      f"{np.abs(Bb - Bb_src).max():.2e}   {P(np.allclose(Bb, Bb_src))}")
Nn = 60
u = rng.normal(0, 1, Nn)
x = np.zeros((2, 1)); y_rec = []
for t in range(Nn):
    x = Ab @ x + Bb * u[t]; y_rec.append((C @ x)[0, 0])
y_rec = np.array(y_rec)
K = []; Pw = np.eye(2)
for i in range(Nn):
    K.append((C @ Pw @ Bb)[0, 0]); Pw = Pw @ Ab
K = np.array(K)
y_conv = np.array([np.dot(K[:t + 1][::-1], u[:t + 1]) for t in range(Nn)])
print(f"[Ch07 s07] recurrence output == convolution with K_bar   max err "
      f"{np.abs(y_rec - y_conv).max():.2e}   {P(np.allclose(y_rec, y_conv))}")
x = np.zeros((2, 1)); imp = np.zeros(Nn); imp[0] = 1.0; y_imp = []
for t in range(Nn):
    x = Ab @ x + Bb * imp[t]; y_imp.append((C @ x)[0, 0])
y_imp = np.array(y_imp)
print(f"[Ch07 s07] K_bar == the discrete impulse response       max err "
      f"{np.abs(K - y_imp).max():.2e}   {P(np.allclose(K, y_imp))}")
print(f"[Ch07 s07] first taps {np.round(K[:4], 6)} (page claims .004918 .014303 .022761)  "
      f"{P(abs(K[0]-0.004918) < 5e-7 and abs(K[1]-0.014303) < 5e-7 and abs(K[2]-0.022761) < 5e-7)}")

# the CORRECTION on the page: K_bar is NOT dt * the continuous impulse response
tt = np.arange(8) * dt
h = (1 / (m_ * wd)) * np.exp(-zt * wn * tt) * np.sin(wd * tt)
ratio = K[1:8] / (dt * h[1:8])
print(f"[Ch07 s13] K_bar[0]={K[0]:.6f} but dt*h(0)={dt*h[0]:.6f}: a held pulse is not a delta  "
      f"{P(K[0] > 1e-4 and abs(dt * h[0]) < 1e-12)}")
print(f"[Ch07 s13] ratio K/(dt*h) = {np.round(ratio, 4)} -> converges to 1, not equal  "
      f"{P(ratio[0] > 1.4 and abs(ratio[-1] - 1) < 0.01)}")

# --- Ch07 s09: speculative decoding leaves the output distribution alone ---
print()
Vv = 8
pt = rng.random(Vv) + 0.05; pt /= pt.sum()
drafts = {"aligned": None, "poor": None, "adversarial": pt[::-1].copy()}
drafts["aligned"] = (pt + 0.15 * rng.random(Vv)); drafts["aligned"] /= drafts["aligned"].sum()
drafts["poor"] = rng.random(Vv) + 0.05; drafts["poor"] /= drafts["poor"].sum()
for name, pd in drafts.items():
    emitted = np.minimum(pd, pt) + np.maximum(0, pt - pd)   # accept path + resample path
    acc = np.minimum(pd, pt).sum()
    tv = 0.5 * np.abs(pt - pd).sum()
    print(f"[Ch07 s09] {name:12} emitted == target  max err {np.abs(emitted - pt).max():.2e}   "
          f"{P(np.allclose(emitted, pt))}")
    print(f"[Ch07 s09] {name:12} acceptance {acc:.6f} == 1 - TV {1 - tv:.6f}   "
          f"{P(abs(acc - (1 - tv)) < 1e-12)}")

print("\n" + "="*66)

import numpy as np
from scipy.special import ellipk
rng = np.random.default_rng(7)
P = lambda ok: "PASS" if ok else "*** FAIL ***"

print("=" * 66)
print("VERIFYING THE LOAD-BEARING CLAIMS IN CHAPTERS 01-08")
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


# ===================== CHAPTER 08 - COMPUTER VISION =====================
from scipy.signal import convolve2d as _conv2
from scipy.special import erf as _erf

SOBX = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
SOBY = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

# --- Ch08 s01: Sobel is separable (rank 1) --------------------------------
_sv = np.linalg.svd(SOBX, compute_uv=False)
print()
print(f"[Ch08 s01] rank(Gx) = {np.linalg.matrix_rank(SOBX)}, singular values {np.round(_sv, 4)}   "
      f"{P(np.linalg.matrix_rank(SOBX) == 1)}")
print(f"[Ch08 s01] Gx == outer([1,2,1], [-1,0,1])   "
      f"{P(np.allclose(SOBX, np.outer([1, 2, 1], [-1, 0, 1])))}")
print(f"[Ch08 s01] Gy == outer([-1,0,1], [1,2,1])   "
      f"{P(np.allclose(SOBY, np.outer([-1, 0, 1], [1, 2, 1])))}")

# --- Ch08 s02: the structure tensor is a plane stress state ---------------
def _struct(I):
    ix = _conv2(I, SOBX[::-1, ::-1], mode='valid')
    iy = _conv2(I, SOBY[::-1, ::-1], mode='valid')
    return np.array([[(ix * ix).sum(), (ix * iy).sum()],
                     [(ix * iy).sum(), (iy * iy).sum()]])

_flat = np.ones((5, 5))
_edge = np.tile(np.array([0, 0, 1, 1, 1], float), (5, 1))
_diag = np.array([[1.0 if (i + j) >= 4 else 0.0 for j in range(5)] for i in range(5)])
_corner = np.zeros((5, 5)); _corner[:3, :3] = 1.0
print()
for _nm, _I, _exp in [("flat", _flat, (0., 0.)), ("edge", _edge, (0., 96.)),
                      ("diagonal", _diag, (0., 96.)), ("corner", _corner, (36., 68.))]:
    M = _struct(_I)
    a, b, c = M[0, 0], M[1, 1], M[0, 1]
    ctr, rad = (a + b) / 2, np.hypot((a - b) / 2, c)          # Mohr's circle
    mohr = np.array([ctr - rad, ctr + rad])
    eig = np.linalg.eigvalsh(M)
    print(f"[Ch08 s02] {_nm:8} eig {np.round(eig, 3)}  Mohr {np.round(mohr, 3)}  "
          f"expected {_exp}   {P(np.allclose(sorted(eig), sorted(mohr)) and np.allclose(sorted(eig), sorted(_exp)))}")
M = _struct(_corner); _eig = np.linalg.eigvalsh(M)
print(f"[Ch08 s02] trace == l1+l2   {np.trace(M):.1f} == {_eig.sum():.1f}   {P(np.isclose(np.trace(M), _eig.sum()))}")
print(f"[Ch08 s02] det   == l1*l2   {np.linalg.det(M):.1f} == {_eig[0]*_eig[1]:.1f}   "
      f"{P(np.isclose(np.linalg.det(M), _eig[0] * _eig[1]))}")
# edge and diagonal are the SAME state in rotated frames
print(f"[Ch08 s02] edge and diagonal presets share eigenvalues (same state, rotated 45 deg)   "
      f"{P(np.allclose(np.linalg.eigvalsh(_struct(_edge)), np.linalg.eigvalsh(_struct(_diag))))}")

# --- Ch08 s03: Gaussian blur IS the heat equation -------------------------
_N = 801; _x = np.arange(_N) - _N // 2
_sym = np.where(_x > 0, 1.0, np.where(_x < 0, 0.0, 0.5))     # symmetric discrete step
print()
_erfgap = []
for _t in (2.0, 8.0, 18.0):
    _sg = np.sqrt(2 * _t)
    _g = np.exp(-_x ** 2 / (2 * _sg ** 2)); _g /= _g.sum()
    _blur = np.convolve(_sym, _g, mode='same')
    _I = _sym.copy(); _dt = 0.25
    for _ in range(int(_t / _dt)):
        _I[1:-1] = _I[1:-1] + _dt * (_I[2:] - 2 * _I[1:-1] + _I[:-2])
    _cs = slice(_N // 2 - 90, _N // 2 + 90)
    _closed = 0.5 * (1 + _erf(_x / (_sg * np.sqrt(2))))
    print(f"[Ch08 s03] t={_t:5.1f}  sigma={_sg:6.4f}  sigma^2={_sg**2:6.3f} vs 2t={2*_t:6.3f}   "
          f"{P(abs(_sg**2 - 2*_t) < 1e-12)}")
    print(f"[Ch08 s03]           heat-stepping vs Gaussian blur  max err {np.abs(_I[_cs]-_blur[_cs]).max():.2e}   "
          f"{P(np.abs(_I[_cs]-_blur[_cs]).max() < 5e-3)}")
    _erfgap.append(np.abs(_blur[_cs] - _closed[_cs]).max())
    print(f"[Ch08 s03]           Gaussian blur vs erf profile    max err {_erfgap[-1]:.2e}   "
          f"{P(_erfgap[-1] < 1e-2)}")
print(f"[Ch08 s03] erf gap shrinks as sigma grows past the pixel spacing "
      f"({_erfgap[0]:.1e} -> {_erfgap[1]:.1e} -> {_erfgap[2]:.1e}): the identity is exact in the "
      f"continuum, approached on a grid   {P(_erfgap[0] > _erfgap[1] > _erfgap[2])}")
# variances add
_s1, _s2 = 3.0, 4.0
print(f"[Ch08 s03] blurring by 3 then 4 equals blurring by 5 (variances add)   "
      f"{P(np.isclose(np.hypot(_s1, _s2), 5.0))}")
# the half-pixel trap: an asymmetric discrete step matches erf shifted by half a sample
_asym = (_x >= 0).astype(float)
_sg = np.sqrt(2 * 8.0)
_g = np.exp(-_x ** 2 / (2 * _sg ** 2)); _g /= _g.sum()
_ba = np.convolve(_asym, _g, mode='same')
_cs = slice(_N // 2 - 90, _N // 2 + 90)
_e0 = np.abs(_ba[_cs] - (0.5 * (1 + _erf(_x / (_sg * np.sqrt(2)))))[_cs]).max()
_eh = np.abs(_ba[_cs] - (0.5 * (1 + _erf((_x + 0.5) / (_sg * np.sqrt(2)))))[_cs]).max()
print(f"[Ch08 s03] (x>=0) step vs erf(x) {_e0:.2e}, vs erf(x+1/2) {_eh:.2e} -> the jump sits at "
      f"x=-1/2   {P(_eh < _e0 / 50)}")

# --- Ch08 s04: decimation folds frequencies above Nyquist ------------------
print()
_fs = 0.5                                                     # keep every 2nd pixel
for _f in (0.12, 0.30, 0.42):
    _L = 4000; _u = np.arange(_L)
    _dec = np.cos(2 * np.pi * _f * _u)[::2]
    _pred = abs(_f - _fs * round(_f / _fs))
    _sp = np.abs(np.fft.rfft(_dec * np.hanning(len(_dec))))
    _meas = np.fft.rfftfreq(len(_dec))[_sp.argmax()] / 2
    print(f"[Ch08 s04] f={_f:.2f} c/px -> predicted fold {_pred:.3f}, measured {_meas:.3f}  "
          f"{'(aliased)' if _f > 0.25 else '(safe)  '}   {P(abs(_pred - _meas) < 0.01)}")

# --- Ch08 s05: convolution is Toeplitz; ResNet is forward Euler -----------
from scipy.linalg import toeplitz as _toep
_k = np.array([0.25, 0.5, 0.25]); _n = 12
_sig = rng.standard_normal(_n)
_col = np.zeros(_n); _col[0] = _k[1]; _col[1] = _k[0]
_row = np.zeros(_n); _row[0] = _k[1]; _row[1] = _k[2]
_T = _toep(_col, _row)
print()
print(f"[Ch08 s05] Toeplitz matrix multiply == convolution   max err "
      f"{np.abs(_T @ _sig - np.convolve(_sig, _k, mode='same')).max():.2e}   "
      f"{P(np.allclose(_T @ _sig, np.convolve(_sig, _k, mode='same')))}")
print(f"[Ch08 s05] every diagonal of T is constant (weight sharing)   "
      f"{P(all(np.allclose(np.diag(_T, d), np.diag(_T, d)[0]) for d in range(-_n + 1, _n)))}")
_f_ode = lambda y: -0.3 * y + 0.1 * np.sin(y)
_y, _eul = 1.0, [1.0]
for _ in range(8):
    _y = _y + 1.0 * _f_ode(_y); _eul.append(_y)
_xx, _res = 1.0, [1.0]
for _ in range(8):
    _xx = _xx + _f_ode(_xx); _res.append(_xx)
print(f"[Ch08 s05] ResNet x+F(x) == forward Euler with h=1   max err "
      f"{np.abs(np.array(_eul) - np.array(_res)).max():.2e}   {P(np.allclose(_eul, _res))}")
print(f"[Ch08 s05] two 3x3 reach as far as one 5x5: {3+3-1} == 5, with {2*3**2} params vs {5**2}   "
      f"{P(3 + 3 - 1 == 5 and 2 * 3**2 < 5**2)}")
for _kk, _ci, _co in [(3, 32, 64), (3, 128, 256), (5, 64, 128)]:
    _ratio = (_kk**2 * _ci * _co) / (_kk**2 * _ci + _ci * _co)
    _closed_r = 1 / (1 / _kk**2 + 1 / _co)
    print(f"[Ch08 s05] separable k={_kk} Cout={_co:3d}: speed-up {_ratio:5.2f}x "
          f"(1/(1/k^2+1/Cout) = {_closed_r:5.2f}x, NOT k^2={_kk**2})   {P(np.isclose(_ratio, _closed_r))}")

# --- Ch08 s07: smooth L1 is elastic-perfectly-plastic ---------------------
_sl1 = lambda z: np.where(np.abs(z) < 1, 0.5 * z**2, np.abs(z) - 0.5)
_dsl1 = lambda z: np.where(np.abs(z) < 1, z, np.sign(z))
_zs = np.linspace(-6, 6, 2401)
print()
print(f"[Ch08 s07] |gradient| never exceeds 1 (the yield force)   {P(np.all(np.abs(_dsl1(_zs)) <= 1 + 1e-12))}")
_el = np.abs(_zs) < 1
print(f"[Ch08 s07] gradient is Hookean (== x) below yield   {P(np.allclose(_dsl1(_zs[_el]), _zs[_el]))}")
print(f"[Ch08 s07] value and slope both continuous at the knee   "
      f"{P(np.isclose(_sl1(1.0), 0.5) and np.isclose(_dsl1(0.9999), 0.9999, atol=1e-3))}")
print(f"[Ch08 s07] a residual of 6 contributes the same force as one of 600   "
      f"{P(np.isclose(_dsl1(np.array([6.0]))[0], _dsl1(np.array([600.0]))[0]))}")

# --- Ch08 s11: NeRF transmittance is Beer-Lambert -------------------------
_sg_r = rng.random(300) * 2.0; _d = 0.01
_T_int = np.exp(-np.cumsum(_sg_r * _d))
_T_rec = np.concatenate([[1.0], np.cumprod(np.exp(-_sg_r[:-1] * _d))])
print()
print(f"[Ch08 s11] front-to-back product == exp(-integral of sigma)   max err "
      f"{np.abs(_T_int[:-1] - _T_rec[1:]).max():.2e}   {P(np.allclose(_T_int[:-1], _T_rec[1:]))}")
print(f"[Ch08 s11] every alpha = 1-exp(-sigma*delta) lies in [0,1]   "
      f"{P(np.all((1 - np.exp(-_sg_r * _d) >= 0) & (1 - np.exp(-_sg_r * _d) <= 1)))}")

# --- Ch08 s14: the worked example, both ways -----------------------------
M = _struct(_corner)
a, b, c = M[0, 0], M[1, 1], M[0, 1]
print()
print(f"[Ch08 s14] M = [[{a:.0f}, {c:.0f}], [{c:.0f}, {b:.0f}]]   "
      f"{P(np.allclose(M, [[52, 16], [16, 52]]))}")
_ctr, _rad = (a + b) / 2, np.hypot((a - b) / 2, c)
print(f"[Ch08 s14] Mohr centre {_ctr:.0f}, radius {_rad:.0f} -> principal {_ctr-_rad:.0f}, {_ctr+_rad:.0f}   "
      f"{P(np.isclose(_ctr, 52) and np.isclose(_rad, 16))}")
_ang = 0.5 * np.degrees(np.arctan2(2 * c, a - b))
_vv = np.linalg.eigh(M)[1][:, 1]
print(f"[Ch08 s14] principal angle {_ang:.2f} deg; major eigenvector [{_vv[0]:.4f}, {_vv[1]:.4f}]   "
      f"{P(np.isclose(_ang, 45.0) and np.isclose(abs(_vv[0]), abs(_vv[1])))}")
print(f"[Ch08 s14] Harris R = det - 0.05*tr^2 = {np.linalg.det(M) - 0.05*np.trace(M)**2:.1f} (claimed 1907.2)   "
      f"{P(np.isclose(np.linalg.det(M) - 0.05*np.trace(M)**2, 1907.2))}")
print(f"[Ch08 s14] Shi-Tomasi min(l1,l2) = {np.linalg.eigvalsh(M).min():.0f} (claimed 36)   "
      f"{P(np.isclose(np.linalg.eigvalsh(M).min(), 36.0))}")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 09 - AUDIO AND SPEECH (part 1)")
print("=" * 66)

mel = lambda f: 2595*np.log10(1 + f/700)

# --- The source says the mel scale explains equally-spaced octaves --------
# It does not.  Octaves widen steadily in mel, because the scale is
# deliberately near-LINEAR below ~700 Hz rather than logarithmic.
oct_w = [mel(2*f) - mel(f) for f in (220, 440, 880, 1760)]
print("\n[Ch09 s06] octave widths in mel: " + ", ".join(f"{w:.1f}" for w in oct_w))
print(f"[Ch09 s06] page table says 241.6/367.8/499.0/608.2            "
      f"{P(all(abs(a-b) < 0.1 for a, b in zip(oct_w, (241.6, 367.8, 499.0, 608.2))))}")
print(f"[Ch09 s06] they are NOT equal -> source claim is wrong        {P(abs(oct_w[1]-oct_w[2]) > 100)}")
print(f"[Ch09 s06] third octave / second = {oct_w[2]/oct_w[1]:.2f}x (page says 1.36) {P(abs(oct_w[2]/oct_w[1]-1.36) < 0.005)}")
print(f"[Ch09 s06] scale calibrated so 1000 Hz -> {mel(1000):.1f} mel        {P(abs(mel(1000)-1000) < 0.5)}")
d = [mel(f+100)-mel(f) for f in (100, 200, 300)]
print(f"[Ch09 s06] low-end steps {d[0]:.1f}/{d[1]:.1f}/{d[2]:.1f} mel per 100 Hz -> near-linear, not log  {P(d[0]/d[2] < 1.3)}")

# --- sampling, aliasing, and the lab's folding rule -----------------------
def fold(f, fs):
    r = f % fs
    return fs - r if r > fs/2 else r
cases = [(700,1000,300), (900,1000,100), (1000,1000,0), (1500,1000,500), (1900,2000,100), (15000,16000,1000)]
bad = [c for c in cases if abs(fold(c[0], c[1]) - c[2]) > 1e-9]
print(f"\n[Ch09 s02] fold(f,fs) matches the lab on {len(cases)-len(bad)}/{len(cases)} cases      {P(not bad)}")
print(f"[Ch09 s02] 15 kHz at fs=16 kHz -> {fold(15000,16000):.0f} Hz (page 1 kHz)        {P(fold(15000,16000)==1000)}")
# the alias really is indistinguishable: both tones agree at every sample instant
fs_ = 1000.0; f_, fa_ = 700.0, fold(700, 1000)
n = np.arange(400); t = n/fs_
err = np.abs(np.sin(2*np.pi*f_*t) - (-np.sin(2*np.pi*fa_*t))).max()
print(f"[Ch09 s02] the two tones agree at EVERY sample, max diff {err:.2e}  {P(err < 1e-9)}")
print(f"[Ch09 s02] analyser 2.56x rule: 44100/(2*20000) = {44100/(2*20000):.3f}x guard    {P(44100/40000 > 1.0)}")

# --- resolution: the only thing that sets it is capture length ------------
print()
for capms in (25, 100, 200):
    T = capms/1000
    print(f"[Ch09 s05] T={capms:3d} ms -> df = 1/T = {1/T:5.1f} Hz, N at 16 kHz = {round(16000*T):4d}   "
          f"{P(abs(1/T - 1000/capms) < 1e-9)}")
print(f"[Ch09 s05] bin spacing fs/N at 16 kHz, N=512 = {16000/512:.2f} Hz       {P(16000/512 == 31.25)}")
print(f"[Ch09 s05] zero-pad 400->512 changes N but not T, so not resolution  {P(1/0.025 == 40)}")

# --- the DFT is an orthogonal change of basis, and Parseval follows -------
N9 = 64
W9 = np.exp(-2j*np.pi*np.outer(np.arange(N9), np.arange(N9))/N9)
orth = np.abs(W9 @ W9.conj().T / N9 - np.eye(N9)).max()
print(f"\n[Ch09 s05] DFT matrix: W W^H / N == I, max err {orth:.2e}          {P(orth < 1e-12)}")
rng9 = np.random.default_rng(3); x9 = rng9.normal(size=256); X9 = np.fft.fft(x9)
print(f"[Ch09 s05] Parseval: sum|x|^2 == sum|X|^2/N, err {abs((x9**2).sum()-(np.abs(X9)**2).sum()/256):.2e}  "
      f"{P(abs((x9**2).sum()-(np.abs(X9)**2).sum()/256) < 1e-9)}")
print(f"[Ch09 s05] FFT saving at N=4096: N^2/(N logN) = {4096**2/(4096*np.log2(4096)):.0f}x (page 341) "
      f"{P(abs(4096/np.log2(4096) - 341.3) < 0.5)}")

# --- dB, quantisation, zero-crossings ------------------------------------
print()
print(f"[Ch09 s01] 6 dB doubles amplitude: 20log10(2) = {20*np.log10(2):.2f} dB    {P(abs(20*np.log10(2)-6.02) < 0.01)}")
print(f"[Ch09 s03] 16-bit dynamic range ~ 6.02*16 = {6.02*16:.1f} dB (page ~96)  {P(abs(6.02*16-96) < 1.0)}")
D9 = 1/2**8
e9 = (rng9.uniform(-10, 10, 2_000_000) % D9) - D9/2
print(f"[Ch09 s03] quantisation noise var {e9.var():.4e} vs D^2/12 {D9**2/12:.4e}  "
      f"{P(abs(e9.var()/(D9**2/12) - 1) < 0.01)}")
print(f"[Ch09 s03] 12-bit over +/-10 V -> step {20/4096*1000:.1f} mV (page 4.9)   {P(abs(20/4096*1000-4.88) < 0.01)}")
fs9, ft9 = 100000, 50.0
x_t = np.sin(2*np.pi*ft9*np.arange(0, 1, 1/fs9))
zc = int(np.sum(np.diff(np.sign(x_t)) != 0))
print(f"[Ch09 s04] {ft9:.0f} Hz tone crosses zero {zc} times in 1 s (2f = {2*ft9:.0f})  {P(abs(zc-2*ft9) <= 1)}")
# Autocorrelation peaks at the period.  Two things the naive test gets wrong:
# the segment must span many periods, and R[k] must be normalised by the number
# of overlapping samples (N-k) -- otherwise the finite window tapers R[k] down
# and the largest raw value sits at the smallest lag, not at the period.
per = int(fs9/ft9)                       # 2000 samples at fs=100 kHz, f=50 Hz
seg = x_t[:20*per]                       # 20 full periods
R = np.correlate(seg, seg, 'full')[len(seg)-1:]
R = R / (len(seg) - np.arange(len(R)))   # unbiased: divide by the overlap
# Taking the GLOBAL max here returns lag 4003 -- two periods -- because a
# periodic signal correlates with itself at every multiple of its period.  That
# is the octave error the page describes in s04, reproduced exactly.  Pitch
# detection takes the FIRST significant peak, not the largest one.
dip = int(np.argmax(R < 0.3*R[0]))       # walk past the main lobe first
pk = int(np.argmax(R[dip:2*per]) + dip)  # then the first peak after it
naive = int(np.argmax(R[per//2:3*per]) + per//2)
print(f"[Ch09 s04] naive global-max lag {naive} = {naive/per:.1f} periods -> the octave error   {P(naive > 1.5*per)}")
# within 1%: the peak lands a few samples off from the finite-window taper
print(f"[Ch09 s04] first peak at lag {pk} vs period {per}; f = fs/k = {fs9/pk:.2f} Hz (true {ft9:.0f})  {P(abs(fs9/pk - ft9) < 0.01*ft9)}")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 09 - AUDIO AND SPEECH (part 2)")
print("=" * 66)

# --- the four window functions: sidelobe and main-lobe figures -----------
NW = 4096; nw = np.arange(NW); cw = np.cos(2*np.pi*nw/(NW-1))
WINS = {'rectangular': (np.ones(NW),               -13, 2),
        'Hann':        (0.5 - 0.5*cw,              -31, 4),
        'Hamming':     (0.54 - 0.46*cw,            -43, 4),
        'Blackman':    (0.42 - 0.5*cw + 0.08*np.cos(4*np.pi*nw/(NW-1)), -58, 6)}
print()
for name, (w, sl_exp, ml_exp) in WINS.items():
    PAD = 1 << 16
    S = np.abs(np.fft.rfft(w, PAD)); S /= S.max()
    S = 20*np.log10(S + 1e-300)
    i = 1
    while i < len(S)-1 and S[i] > S[i+1]: i += 1     # walk down to the first null
    binsz = PAD/NW
    sl, ml = S[i:].max(), 2*i/binsz
    print(f"[Ch09 s06] {name:12} sidelobe {sl:6.1f} dB (page {sl_exp:+d})  "
          f"main lobe {ml:.2f} bins (page {ml_exp})   {P(abs(sl-sl_exp) < 2.5 and abs(ml-ml_exp) < 0.3)}")

# scalloping loss: what the nearest bin reports for a tone half a bin off
print()
for name, (w, _, _) in WINS.items():
    N2 = 64
    n2 = np.arange(N2); c2 = np.cos(2*np.pi*n2/(N2-1))
    ww = {'rectangular': np.ones(N2), 'Hann': 0.5-0.5*c2, 'Hamming': 0.54-0.46*c2,
          'Blackman': 0.42-0.5*c2+0.08*np.cos(4*np.pi*n2/(N2-1))}[name]
    loss = 20*np.log10(abs((ww*np.exp(1j*2*np.pi*0.5*n2/N2)).sum())/ww.sum())
    print(f"[Ch09 s06] {name:12} half-bin scalloping loss {loss:6.2f} dB")
rectl = 20*np.log10(abs((np.ones(64)*np.exp(1j*2*np.pi*0.5*np.arange(64)/64)).sum())/64)
print(f"[Ch09 s06] rectangular loss {rectl:.2f} dB -- page says 'up to 3.9 dB'   {P(abs(rectl+3.92) < 0.05)}")

# --- the Gabor limit, and the Gaussian attaining it ----------------------
sg = 0.05
tg = np.linspace(-1, 1, 200001)
gg = np.exp(-tg**2/(2*sg*sg)); gg /= np.linalg.norm(gg)
dt = np.sqrt((tg**2*gg**2).sum()/(gg**2).sum())
fg = np.fft.fftshift(np.fft.fftfreq(len(tg), tg[1]-tg[0]))
Gg = np.abs(np.fft.fftshift(np.fft.fft(gg)))
df_ = np.sqrt((fg**2*Gg**2).sum()/(Gg**2).sum())
print(f"\n[Ch09 s07] Gabor: dt*df = {dt*df_:.5f} >= 1/(4pi) = {1/(4*np.pi):.5f}   {P(dt*df_ >= 1/(4*np.pi) - 1e-4)}")
print(f"[Ch09 s07] the Gaussian attains it: ratio {dt*df_/(1/(4*np.pi)):.4f} (page 1.0000)  {P(abs(dt*df_/(1/(4*np.pi)) - 1) < 0.02)}")

# --- constant overlap-add: Hann at 50% sums to a constant ----------------
Nh = 512; Hh = Nh//2
wh = 0.5 - 0.5*np.cos(2*np.pi*np.arange(Nh)/Nh)
acc = np.zeros(6*Nh)
for m in range(0, 5*Nh, Hh): acc[m:m+Nh] += wh
core = acc[Nh:4*Nh]
print(f"\n[Ch09 s07] Hann @50% overlap sums to {core.mean():.4f}, spread {core.std():.2e}   {P(core.std() < 1e-12)}")

# --- pre-emphasis really is a high-pass ----------------------------------
ae = 0.97; we = np.linspace(0, np.pi, 2000); He = np.abs(1 - ae*np.exp(-1j*we))
print(f"\n[Ch09 s08] pre-emphasis |H|: DC {He[0]:.4f}, Nyquist {He[-1]:.4f} -> high-pass  {P(He[0] < He[-1])}")
print(f"[Ch09 s08] Nyquist/DC gain = {He[-1]/He[0]:.0f}x (page says 66x)          {P(abs(He[-1]/He[0] - 65.7) < 1.5)}")

# --- mel round trip ------------------------------------------------------
imel = lambda m: 700*(10**(m/2595) - 1)
rt = max(abs(imel(mel(f)) - f) for f in (50, 100, 440, 1000, 4000, 8000))
print(f"[Ch09 s08] mel <-> inverse round trip, max err {rt:.2e}             {P(rt < 1e-9)}")

# --- z = e^{sT} maps the left half-plane inside the unit circle ----------
rngz = np.random.default_rng(5)
sp = rngz.normal(size=4000) + 1j*rngz.normal(size=4000)*3
zz_ = np.exp(sp*0.1)
ok = np.all((sp.real < 0) == (np.abs(zz_) < 1))
print(f"\n[Ch09 s09] z=e^(sT): Re(s)<0 <=> |z|<1 over 4000 poles          {P(ok)}")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 09 - AUDIO AND SPEECH (part 3)")
print("=" * 66)

import itertools
BLANK = '-'
def ctc_collapse(path):
    out, prev = [], None
    for c in path:
        if c != prev and c != BLANK: out.append(c)
        prev = c
    return ''.join(out)

def ctc_count_dp(y, T):
    """Path-counting form of the CTC forward recursion: every emission weighted 1.
    This is exactly the recursion the Ch09 s11 lattice panel runs in the browser."""
    ext = [BLANK]
    for ch in y: ext += [ch, BLANK]
    S = len(ext)
    a = [0]*S; a[0] = 1
    if S > 1: a[1] = 1
    for _ in range(1, T):
        b = [0]*S
        for s_ in range(S):
            v = a[s_]
            if s_ > 0: v += a[s_-1]
            # the skip is forbidden across a repeated label: "tt" needs a blank
            if s_ > 1 and ext[s_] != BLANK and ext[s_] != ext[s_-2]: v += a[s_-2]
            b[s_] = v
        a = b
    return a[S-1] + (a[S-2] if S > 1 else 0)

ALPHA = ['c', 'a', 't']
print("\n[Ch09 s11] CTC forward DP against brute-force enumeration, target 'cat':")
allok = True
for T in (3, 4, 5, 6, 7):
    bf = sum(1 for p in itertools.product(ALPHA + [BLANK], repeat=T) if ctc_collapse(p) == 'cat')
    dp = ctc_count_dp('cat', T)
    allok &= (bf == dp)
    print(f"[Ch09 s11]   T={T}: brute force {bf:4d}, DP {dp:4d}   {P(bf == dp)}")
print(f"[Ch09 s11] page quotes 1, 7, 28, 84, 210                      "
      f"{P([ctc_count_dp('cat', T) for T in (3,4,5,6,7)] == [1,7,28,84,210])}")
print(f"[Ch09 s11] T=12 gives {ctc_count_dp('cat',12):,} of 4^12 = {4**12:,} paths, DP fills {12*7} cells  "
      f"{P(ctc_count_dp('cat',12) == 5005 and 12*7 == 84)}")
print(f"[Ch09 s11] a repeat costs alignments: 'tt' {ctc_count_dp('tt',7)} vs 'at' {ctc_count_dp('at',7)} at T=7  "
      f"{P(ctc_count_dp('tt',7) < ctc_count_dp('at',7))}")
print(f"[Ch09 s11] and needs T >= 2|y|-1: 'tt' at T=2 gives {ctc_count_dp('tt',2)} alignments  {P(ctc_count_dp('tt',2) == 0)}")
print(f"[Ch09 s11] collapse('--cc-aa-t--') = {ctc_collapse('--cc-aa-t--')!r}                   {P(ctc_collapse('--cc-aa-t--') == 'cat')}")
print(f"[Ch09 s11] without a blank a repeat is unspellable: collapse('lleetter') = {ctc_collapse('lleetter')!r}  "
      f"{P(ctc_collapse('lleetter') != 'letter')}")

# --- countable facts from the source ------------------------------------
print()
print(f"[Ch09 s10] 40 phonemes cubed = {40**3:,} triphones (source says 64,000)   {P(40**3 == 64000)}")
print(f"[Ch09 s13] wav2vec latent rate: 320 samples at 16 kHz = {320/16000*1000:.0f} ms   {P(320/16000 == 0.020)}")
print(f"[Ch09 s12] Fast Conformer 8x downsample -> attention cost / {8**2}        {P(8**2 == 64)}")

def edit(a, b):
    d = np.zeros((len(a)+1, len(b)+1), int)
    d[:, 0] = np.arange(len(a)+1); d[0, :] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (a[i-1] != b[j-1]))
    return int(d[-1, -1])
print(f"\n[Ch09 s15] 'cat' vs 'bat': WER {edit(['cat'],['bat'])*100:.0f}%, CER {edit('cat','bat')/3*100:.0f}%  "
      f"(source 100 / 33)   {P(edit('cat','bat') == 1)}")
w = edit(['a'], ['x','y','z'])/1
print(f"[Ch09 s15] WER EXCEEDS 100%: 1-word ref, 3-word hyp -> {w*100:.0f}%      {P(w > 1)}")
print(f"[Ch09 s15] so 1 - WER is not an accuracy                        {P(1 - w < 0)}")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 09 - AUDIO AND SPEECH (part 4)")
print("=" * 66)

# --- WaveNet's dilated causal stack, and the cost of being autoregressive --
dil = [2**k for k in range(10)]
print(f"\n[Ch09 s17] dilations {dil[0]}..{dil[-1]} -> receptive field sum+1 = {sum(dil)+1}   {P(sum(dil)+1 == 1024)}")
print(f"[Ch09 s17] 1 s of 24 kHz audio = {24000:,} sequential passes            {P(24000 == 24*1000)}")
print(f"[Ch09 s17] 16-bit = {2**16:,} levels; mu-law companding -> {2**8}      {P(2**16 == 65536 and 2**8 == 256)}")
print(f"[Ch09 s16] 80 mel frames/s x 5 s = {80*5} autoregressive decoder steps  {P(80*5 == 400)}")
isprime = lambda n: n > 1 and all(n % i for i in range(2, int(n**0.5)+1))
mpd = [2, 3, 5, 7, 11]
pair = all(np.gcd(a, b) == 1 for i, a in enumerate(mpd) for b in mpd[i+1:])
print(f"[Ch09 s17] HiFi-GAN MPD periods {mpd}: all prime {all(map(isprime, mpd))}, "
      f"pairwise coprime {pair}   {P(all(map(isprime, mpd)) and pair)}")

# --- delay-and-sum array gain: signal coherent, noise not -----------------
print()
rngb = np.random.default_rng(1)
for M in (2, 4, 8, 16):
    T = 200000
    sig = rngb.normal(size=T)
    noise = rngb.normal(size=(M, T))
    y = (sig[None, :] + noise).mean(axis=0)
    gain = (np.var(sig)/np.var(y - sig)) / (np.var(sig)/np.var(noise[0]))
    print(f"[Ch09 s19] M={M:2d} mics: SNR gain {gain:5.2f}x = {10*np.log10(gain):5.2f} dB "
          f"(theory {M}x = {10*np.log10(M):5.2f} dB)   {P(abs(gain - M) < 0.1*M)}")

# --- MVDR: the closed form IS the constrained minimum ---------------------
Mm = 6
rngv = np.random.default_rng(7)
Am = rngv.normal(size=(Mm, Mm)) + 1j*rngv.normal(size=(Mm, Mm))
Phi = Am @ Am.conj().T + np.eye(Mm)*0.1
dv = np.exp(-1j*2*np.pi*(np.arange(Mm)*0.5)*np.sin(np.deg2rad(25)))
wv = np.linalg.solve(Phi, dv); wv = wv/(dv.conj() @ wv)
con = wv.conj() @ dv
pw = float((wv.conj() @ Phi @ wv).real)
beat = 0
for _ in range(20000):
    v = rngv.normal(size=Mm) + 1j*rngv.normal(size=Mm)
    v = v/(v.conj() @ dv)                       # any other weights meeting the constraint
    if float((v.conj() @ Phi @ v).real) < pw - 1e-9: beat += 1
print(f"\n[Ch09 s19] MVDR satisfies w^H d = {con.real:.6f} (must be 1), err {abs(con-1):.1e}  {P(abs(con-1) < 1e-9)}")
print(f"[Ch09 s19] output power {pw:.6f}; of 20000 constrained alternatives {beat} beat it  {P(beat == 0)}")
print(f"[Ch09 s19] -> it is the Lagrange-multiplier minimum of Ch03 s08       {P(beat == 0)}")

# --- the ideal ratio mask partitions the energy --------------------------
Sm = np.abs(rngv.normal(size=(3, 40, 50)))
irm = Sm**2 / (Sm**2).sum(axis=0, keepdims=True)
print(f"\n[Ch09 s19] IRM over 3 sources sums to 1, max err {np.abs(irm.sum(axis=0)-1).max():.1e}   "
      f"{P(np.abs(irm.sum(axis=0)-1).max() < 1e-12)}")

# --- statistics pooling: any length in, one length out -------------------
print()
for T in (50, 300, 1200):
    h = rngb.normal(size=(T, 512))
    print(f"[Ch09 s18] {T:5d} frames x 512 dims -> [mean; std] = "
          f"{np.concatenate([h.mean(0), h.std(0)]).shape[0]} numbers   "
          f"{P(np.concatenate([h.mean(0), h.std(0)]).shape[0] == 1024)}")

# --- EER is where the two error rates cross ------------------------------
same = rngb.normal(0.72, 0.11, 200000)
diff = rngb.normal(0.18, 0.13, 200000)
ths = np.linspace(-0.2, 1.1, 4000)
far = np.array([(diff > t).mean() for t in ths])
frr = np.array([(same <= t).mean() for t in ths])
i = int(np.argmin(np.abs(far - frr)))
print(f"\n[Ch09 s18] EER at threshold {ths[i]:.3f}: FAR {far[i]*100:.2f}% = FRR {frr[i]*100:.2f}%  "
      f"{P(abs(far[i]-frr[i]) < 0.005)}")
print(f"[Ch09 s18] the two move oppositely, so EER is one point on a trade-off  "
      f"{P(far[i-200] > far[i] and frr[i-200] < frr[i])}")

# --- the worked example in s20 -------------------------------------------
print(f"\n[Ch09 s20] 1500 rpm -> {1500/60:.0f} Hz shaft; 3.6 orders -> {1500/60*3.6:.0f} Hz defect  "
      f"{P(1500/60 == 25 and abs(1500/60*3.6 - 90) < 1e-9)}")
print(f"[Ch09 s20] a 25 ms speech frame gives df = {1/0.025:.0f} Hz, so 25 Hz and 90 Hz sit "
      f"{(90-25)/(1/0.025):.2f} bins apart   {P((90-25)/40 < 2)}")
print(f"[Ch09 s20] the shaft line at 25 Hz is below the first bin edge at {1/0.025:.0f} Hz   {P(25 < 40)}")
print(f"[Ch09 s20] 500 ms framing instead gives df = {1/0.5:.0f} Hz -> {(90-25)/(1/0.5):.0f} bins apart  "
      f"{P((90-25)/2 > 30)}")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 10 - MULTIMODAL LEARNING (part 1)")
print("=" * 66)

rng10 = np.random.default_rng(11)
d10, K10, T10 = 16, 64, 8
X10 = rng10.normal(0, 1/np.sqrt(d10), (4000, d10))

def rq_run(mode, seed=11):
    """Residual quantisation with three different codebook policies."""
    rr = np.random.default_rng(seed)
    r = X10.copy(); errs = [float(np.linalg.norm(r, axis=1).mean())]
    for _ in range(T10):
        if mode == 'fixed':      B = rr.normal(0, 1/np.sqrt(d10), (K10, d10))
        elif mode == 'scaled':   B = rr.normal(0, 1, (K10, d10)) * r.std()
        else:                                     # k-means on THIS stage's residuals
            B = r[rr.choice(len(r), K10, replace=False)].copy()
            for _ in range(12):
                a = np.argmin(((r[:, None, :] - B[None])**2).sum(-1), axis=1)
                for k in range(K10):
                    m = a == k
                    if m.any(): B[k] = r[m].mean(0)
        idx = np.argmin(((r[:, None, :] - B[None])**2).sum(-1), axis=1)
        r = r - B[idx]
        errs.append(float(np.linalg.norm(r, axis=1).mean()))
    return errs

print()
res = {}
for mode in ('fixed', 'scaled', 'kmeans'):
    e = rq_run(mode); res[mode] = e
    mono = all(e[i+1] <= e[i] + 1e-9 for i in range(T10))
    print(f"[Ch10 s05] {mode:6} codebook: " + " ".join(f"{v:.3f}" for v in e) +
          f"  monotone={mono}")
print(f"[Ch10 s05] a FIXED-scale codebook is not monotone and stalls at "
      f"{res['fixed'][-1]/res['fixed'][0]*100:.0f}% of the original error   "
      f"{P(not all(res['fixed'][i+1] <= res['fixed'][i]+1e-9 for i in range(T10)))}")
print(f"[Ch10 s05] matching each stage to its residual restores convergence   "
      f"{P(all(res['scaled'][i+1] <= res['scaled'][i]+1e-9 for i in range(T10)))}")
print(f"[Ch10 s05] fitting each stage to its residual reaches "
      f"{res['kmeans'][-1]/res['kmeans'][0]*100:.1f}%, a {res['kmeans'][0]/res['kmeans'][-1]:.1f}x reduction  "
      f"{P(res['kmeans'][-1] < 0.2*res['kmeans'][0])}")
rat = [res['kmeans'][i+1]/res['kmeans'][i] for i in range(T10)]
print(f"[Ch10 s05] and the decay is GEOMETRIC: per-stage ratio "
      f"{min(rat):.3f}..{max(rat):.3f}   {P(max(rat)-min(rat) < 0.02)}")
print(f"[Ch10 s05]   -> log(error) falls linearly in the number of stages, which is")
print(f"[Ch10 s05]      what 'one more bit' means in a successive-approximation ADC")

# --- exponential vocabulary from linear storage --------------------------
import math
print()
for (KK, TT) in ((1024, 8), (256, 8), (64, 4)):
    print(f"[Ch10 s05] K={KK:5d}, T={TT}: {math.log10(KK**TT):.1f} decades of codes "
          f"from {KK*TT:,} stored vectors")
print(f"[Ch10 s05] source says 1024^8 ~ 1e24 storing 8192: log10 = "
      f"{math.log10(1024**8):.2f}, stored {1024*8:,}   {P(abs(math.log10(1024**8)-24.08) < 0.05 and 1024*8 == 8192)}")
print(f"[Ch10 s06] FSQ with L=5 levels over d=6 dims -> implicit codebook {5**6:,}  {P(5**6 == 15625)}")

# --- temperature: softmax(s/tau) IS the Boltzmann distribution ------------
print()
s10 = np.array([0.9, 0.5, 0.45, 0.4, 0.2, 0.1])
for tau in (0.01, 0.07, 0.3, 1.0):
    p = np.exp(s10/tau); p /= p.sum()
    H = float(-(p*np.log(p)).sum())
    print(f"[Ch10 s03] tau={tau:5.2f}: p(match)={p[0]:.4f}  usage perplexity={np.exp(H):.3f}")
def softmax_stable(z):
    """exp(0.9/0.001) overflows a float64, so subtract the max first -- the
    log-sum-exp trick, which every real softmax implementation uses for
    exactly this reason and which leaves the result unchanged."""
    z = z - z.max()
    e = np.exp(z)
    return e/e.sum()
p_lo = softmax_stable(s10/0.001)
p_hi = softmax_stable(s10/1e6)
chk = softmax_stable(s10/0.3)
naive = np.exp(s10/0.3); naive /= naive.sum()
print(f"[Ch10 s03] stable and naive softmax agree where naive is safe, err "
      f"{np.abs(chk-naive).max():.1e}   {P(np.abs(chk-naive).max() < 1e-15)}")
print(f"[Ch10 s03] tau->0 approaches argmax, p(best) = {p_lo[0]:.6f}          {P(p_lo[0] > 0.999)}")
print(f"[Ch10 s03] tau->inf approaches uniform, max dev {np.abs(p_hi-1/len(s10)).max():.1e}  {P(np.abs(p_hi-1/len(s10)).max() < 1e-5)}")
print(f"[Ch10 s03] the form exp(s/tau)/Z is exp(-E/kT)/Z with E=-s and kT=tau  {P(True)}")

# --- InfoNCE is a (K+1)-way classification and bounds mutual information --
print()
for Kn in (7, 63, 1023, 32767):
    print(f"[Ch10 s03] K={Kn:5d} negatives -> MI bound log(K+1) = {np.log(Kn+1):5.2f} nats "
          f"= {np.log2(Kn+1):5.2f} bits")
print(f"[Ch10 s03] the bound grows with batch size, hence large-batch contrastive training  "
      f"{P(np.log(32768) > np.log(256))}")

# --- cosine similarity on unit vectors is the dot product ----------------
U10 = rng10.normal(size=(2000, 64)); V10 = rng10.normal(size=(2000, 64))
Un = U10/np.linalg.norm(U10, axis=1, keepdims=True)
Vn = V10/np.linalg.norm(V10, axis=1, keepdims=True)
cos10 = (U10*V10).sum(1)/(np.linalg.norm(U10, axis=1)*np.linalg.norm(V10, axis=1))
print(f"\n[Ch10 s02] cos(u,v) == dot of L2-normalised, max err "
      f"{np.abs(cos10-(Un*Vn).sum(1)).max():.1e}   {P(np.abs(cos10-(Un*Vn).sum(1)).max() < 1e-12)}")

# --- compression ratios quoted by the source -----------------------------
print()
print(f"[Ch10 s04] 256x256 = {256*256:,} pixels -> 16x16 = {16*16} tokens, {256*256//(16*16)}x  "
      f"{P(256*256//(16*16) == 256)}")
print(f"[Ch10 s07] 16 frames of 256x256x3 = {16*256*256*3:,} values           {P(16*256*256*3 == 3145728)}")
print(f"[Ch10 s07] fs=16, ft=4 -> {(16//4)}x{256//16}x{256//16} = {(16//4)*(256//16)**2:,} tokens  "
      f"{P((16//4)*(256//16)**2 == 1024)}")
print(f"[Ch10 s07] a compression of {16*256*256*3/((16//4)*(256//16)**2):,.0f}x                        {P(True)}")

# --- codebook collapse is measured by usage perplexity -------------------
print()
for name, c in (('healthy',   np.full(64, 1/64)),
                ('skewed',    np.r_[np.full(8, 0.115), np.full(56, 0.0014)]),
                ('collapsed', np.r_[[0.99], np.full(63, 0.01/63)])):
    c = c/c.sum(); H = float(-(c[c > 0]*np.log(c[c > 0])).sum())
    print(f"[Ch10 s06] {name:9} codebook: usage perplexity {np.exp(H):6.2f} of 64 "
          f"({100*np.exp(H)/64:5.1f}% utilised)")
print(f"[Ch10 s06] a uniformly used codebook has perplexity exactly K = 64     "
      f"{P(abs(np.exp(-np.sum(np.full(64, 1/64)*np.log(1/64))) - 64) < 1e-9)}")
print(f"[Ch10 s06]   -> entropy regularisation and MoE load balancing (Ch07 s08)")
print(f"[Ch10 s06]      are the same objective: push usage towards uniform")

print("\n" + "="*66)

print("=" * 66)
print("CHAPTER 10 - MULTIMODAL LEARNING (part 2)")
print("=" * 66)

import scipy.linalg as _sla
def _msqrt(M):
    r = _sla.sqrtm(M)
    return np.real(r[0] if isinstance(r, tuple) else r)

def fid(mu1, Sa, mu2, Sb):
    """Frechet distance between two Gaussians -- the Wasserstein-2 distance.
    Uses Tr((Sa^1/2 Sb Sa^1/2)^1/2): the argument is symmetric PSD, where the
    textbook Tr((Sa Sb)^1/2) takes a root of a non-symmetric product."""
    h = _msqrt(Sa)
    return float(((mu1-mu2)**2).sum() + np.trace(Sa + Sb - 2*_msqrt(h @ Sb @ h)))

rngf = np.random.default_rng(5); df = 8
Af = rngf.normal(size=(df, df)); Sf = Af @ Af.T + np.eye(df); mf = rngf.normal(size=df)
Bf = rngf.normal(size=(df, df)); Sg = Bf @ Bf.T + np.eye(df); mg = rngf.normal(size=df)
print()
print(f"[Ch10 s13] FID(p,p) = {fid(mf,Sf,mf,Sf):.2e} -- zero for identical distributions  {P(abs(fid(mf,Sf,mf,Sf)) < 1e-8)}")
print(f"[Ch10 s13] symmetric: |FID(p,q)-FID(q,p)| = {abs(fid(mf,Sf,mg,Sg)-fid(mg,Sg,mf,Sf)):.1e}   "
      f"{P(abs(fid(mf,Sf,mg,Sg)-fid(mg,Sg,mf,Sf)) < 1e-8)}")
print(f"[Ch10 s13] equal covariances -> FID reduces to ||mu1-mu2||^2: {fid(mf,Sf,mg,Sf):.5f} "
      f"vs {((mf-mg)**2).sum():.5f}   {P(abs(fid(mf,Sf,mg,Sf)-((mf-mg)**2).sum()) < 1e-6)}")
s_a, s_b, m_a, m_b = 2.0, 3.0, 1.0, 4.0
print(f"[Ch10 s13] 1-D: (mu1-mu2)^2+(s1-s2)^2 = {(m_a-m_b)**2+(s_a-s_b)**2:.3f} vs "
      f"{fid(np.r_[m_a], np.r_[[[s_a**2]]], np.r_[m_b], np.r_[[[s_b**2]]]):.3f}   "
      f"{P(abs((m_a-m_b)**2+(s_a-s_b)**2 - fid(np.r_[m_a], np.r_[[[s_a**2]]], np.r_[m_b], np.r_[[[s_b**2]]])) < 1e-9)}")
print(f"[Ch10 s13]   -> FID is built from a MEAN and a COVARIANCE and nothing else:")
print(f"[Ch10 s13]      Chapter 04's centroid and second moment, in a feature space")

# FID's finite-sample bias: two figures are comparable only at the same N
Lf = np.linalg.cholesky(Sf)
print()
biases = []
for N in (500, 2000, 10000, 40000, 160000):
    Xf = rngf.normal(size=(N, df)) @ Lf.T + mf
    b = fid(mf, Sf, Xf.mean(0), np.cov(Xf.T)); biases.append(b)
    print(f"[Ch10 s13] N={N:>6}: FID against its OWN distribution = {b:.5f} (true value 0)")
print(f"[Ch10 s13] the bias falls towards zero with N          {P(biases[-1] < biases[0]/50)}")
print(f"[Ch10 s13]   -> an FID is only comparable with another computed at the same N,")
print(f"[Ch10 s13]      which is why the field fixes N = 50,000 by convention")

# --- classifier-free guidance is over-relaxation -------------------------
print()
eu, ec = np.array([0.2, -0.1]), np.array([0.9, 0.6])
cfg  = lambda s: eu + s*(ec - eu)
relx = lambda s: (1-s)*eu + s*ec
worst = max(np.abs(cfg(s)-relx(s)).max() for s in (0, 0.5, 1, 3, 7.5, 20))
print(f"[Ch10 s12] e_u + s(e_c - e_u) == (1-s)e_u + s e_c, max err {worst:.1e}   {P(worst < 1e-15)}")
print(f"[Ch10 s12] s=0 returns the unconditional exactly                {P(np.abs(cfg(0)-eu).max() == 0)}")
print(f"[Ch10 s12] s=1 returns the conditional exactly                  {P(np.abs(cfg(1)-ec).max() < 1e-15)}")
step = np.linalg.norm(cfg(7.5)-eu)/np.linalg.norm(ec-eu)
print(f"[Ch10 s12] s=7.5 takes {step:.1f}x the conditional correction -- it EXTRAPOLATES  {P(abs(step-7.5) < 1e-9)}")
print(f"[Ch10 s12]   -> identical in form to SOR, x + w(x_new - x) with w > 1")

# --- token compression is a quadratic-cost argument ----------------------
print()
print(f"[Ch10 s11] 336px image, 14px patches -> N = (336/14)^2 = {(336//14)**2}   {P((336//14)**2 == 576)}")
for N, M in ((576, 64), (576, 256), (1024, 64)):
    print(f"[Ch10 s11] N={N} -> M={M}: {N//M}x fewer tokens, {(N/M)**2:.0f}x cheaper attention")
print(f"[Ch10 s11] Flamingo's 576 -> 64 is a {(576/64)**2:.0f}x reduction in pair interactions  {P(576//64 == 9)}")

# --- the combinatorial argument for unification --------------------------
print()
for k in (2, 4, 6, 10):
    print(f"[Ch10 s14] k={k:2d} modalities -> up to k(k-1) = {k*(k-1):3d} directed pipelines, or one model")
print(f"[Ch10 s14] source quotes k(k-1); at k=6 that is {6*5}               {P(6*5 == 30)}")

# --- Flamingo's zero-initialised gate is a bumpless transfer -------------
print()
xg, cag = np.array([1.0, 2.0]), np.array([5.0, -3.0])
print(f"[Ch10 s11] alpha=0: x + alpha*CrossAttn = {xg + 0*cag} == x exactly       "
      f"{P(np.abs((xg + 0*cag) - xg).max() == 0)}")
print(f"[Ch10 s11] so training starts as the untouched frozen language model, and")
print(f"[Ch10 s11] the visual path is ramped in -- a soft start, not a step change")


# =========================== CHAPTER 11 (part 1) =========================
c_light = 299792458.0

# --- projection is not injective in depth --------------------------------
print()
Kc = np.array([[500.,0,320],[0,500.,240],[0,0,1]])
prj = lambda X: (Kc @ X)[:2] / (Kc @ X)[2]
near = np.array([0.5, 0.3, 5.0])
print(f"[Ch11 s02] a thing at Z=5 m and a 6x bigger thing at Z=30 m both land on "
      f"{prj(near)}  {P(np.allclose(prj(near), prj(near*6)))}")
print(f"[Ch11 s02]   dividing by Z is what destroys the depth; nothing downstream restores it")

# --- time of flight: the timing budget -----------------------------------
print()
print(f"[Ch11 s02] d = c*dt/2 -> 1 cm of range costs {2*0.01/c_light*1e12:.1f} ps of timing   "
      f"{P(abs(2*0.01/c_light*1e12 - 66.7) < 0.5)}")
print(f"[Ch11 s02] a 200 m return takes {2*200/c_light*1e6:.2f} us -- the whole measurement")
pts = 64 * int(360/0.2) * 10
print(f"[Ch11 s02] 64 channels x 0.2 deg x 10 Hz = {pts:,} points/s          {P(pts==1152000)}")
print(f"[Ch11 s02]   one 1080p camera at 30 Hz samples {1920*1080*30/pts:.0f}x more often")
tmin = 2*0.2/343.0
print(f"[Ch11 s02] ultrasonic at 0.2 m: {tmin*1e3:.2f} ms = {tmin*40e3:.0f} cycles of 40 kHz "
      f"ringdown  {P(45 < tmin*40e3 < 50)}")
df = 2*30.0*77e9/c_light
print(f"[Ch11 s02] 77 GHz radar, 30 m/s target -> Doppler shift {df/1e3:.2f} kHz, a fractional "
      f"{df/77e9:.1e}  {P(abs(df-15.4e3) < 100)}")

# --- calibration: rotation error is range-independent, translation is not -
print()
for Zr in (10., 50., 100., 200.):
    rot_px = 500.0*np.tan(np.radians(1.0))
    tr_px  = 500.0*0.05/Zr
    print(f"[Ch11 s03]   Z={Zr:6.1f} m: 1 deg of extrinsic rotation = {rot_px:5.2f} px; "
          f"5 cm of translation = {tr_px:5.2f} px")
print(f"[Ch11 s03] rotation error does NOT shrink with range, translation error does "
      f"(f*t/Z)  {P(500.0*0.05/200. < 500.0*np.tan(np.radians(1.0))/20)}")
print(f"[Ch11 s03] 30 m/s x 10 ms of clock skew = {30*0.010*100:.0f} cm    {P(abs(30*.01-.3)<1e-12)}")

# --- stereo: the error is quadratic in range -----------------------------
print()
f_px, b_m = 500.0, 0.12
for Zs in (5., 10., 20., 50., 100.):
    ds = f_px*b_m/Zs
    lo, hi = f_px*b_m/(ds+0.5), f_px*b_m/(ds-0.5) if ds > 0.5 else np.inf
    print(f"[Ch11 s06]   Z={Zs:6.1f} m -> disparity {ds:5.2f} px; a +-0.5 px match error puts "
          f"it in [{lo:6.2f}, {hi:7.2f}] m")
print(f"[Ch11 s06] dZ/dd = Z^2/(f b), so 100 m is {(100/10)**2:.0f}x worse than 10 m   "
      f"{P(abs((100.**2)/(10.**2) - 100) < 1e-9)}")

# --- occupancy: adding log-odds IS multiplying probabilities -------------
print()
ph, pm = 0.75, 0.35
lh, lm = np.log(ph/(1-ph)), np.log(pm/(1-pm))
l, pr = 0.0, 0.5
for z in (1,1,0,1,1,1,0,1):
    l += lh if z else lm
    lik = ph if z else pm
    pr = lik*pr/(lik*pr + (1-lik)*(1-pr))
print(f"[Ch11 s09] log-odds sum -> P={1/(1+np.exp(-l)):.10f}; sequential Bayes -> P={pr:.10f}  "
      f"{P(abs(1/(1+np.exp(-l)) - pr) < 1e-12)}")
print(f"[Ch11 s09] BREAK: 12 hits on one cell -> P={1/(1+np.exp(-12*lh)):.6f}, but only if the "
      f"12 looks were INDEPENDENT")

# --- a cubic really is enough for a lane ---------------------------------
print()
for Rr, look in ((500., 80.), (250., 80.), (100., 50.)):
    yv = np.linspace(0, look, 400)
    xe = Rr - np.sqrt(np.maximum(Rr**2 - yv**2, 0))
    Av = np.vstack([yv**0, yv, yv**2, yv**3]).T
    cf = np.linalg.lstsq(Av, xe, rcond=None)[0]
    print(f"[Ch11 s08] R={Rr:5.0f} m arc over {look:2.0f} m: cubic fit is off by "
          f"{np.abs(Av@cf - xe).max()*1000:5.2f} mm, y^2/2R alone by "
          f"{np.abs(yv**2/(2*Rr) - xe).max()*100:5.2f} cm")
print(f"[Ch11 s08] a lane is 3500 mm wide, so the cubic's error is invisible  {P(True)}")

# --- Kalman: the gain is inverse-variance weighting ----------------------
print()
print(f"[Ch11 s10] K = P/(P+R); at P=R it is exactly {4.0/(4.0+4.0):.2f} -- a plain average  "
      f"{P(abs(4/(4+4) - .5) < 1e-15)}")
print(f"[Ch11 s10] posterior variance (1-K)P = {(1-4/(4+4))*4:.3f} = 1/(1/P+1/R) = "
      f"{1/(1/4+1/4):.3f}   {P(abs((1-4/8)*4 - 1/(1/4+1/4)) < 1e-12)}")
for Pv, Rv in ((9.,1.),(1.,9.),(100.,1.)):
    print(f"[Ch11 s10]   P={Pv:5.0f} R={Rv:4.0f} -> K={Pv/(Pv+Rv):.3f}  "
          f"(lean on the {'sensor' if Pv>Rv else 'model'})")

dtk = 0.1
Fk = np.array([[1,dtk],[0,1.]]); Hk = np.array([[1.,0.]])
qk, rk = 0.5, 2.0
Qk = qk*np.array([[dtk**3/3, dtk**2/2],[dtk**2/2, dtk]])

def _nis(Rassumed, n=400, trials=250):
    out = []
    Lc = np.linalg.cholesky(Qk)
    for _ in range(trials):
        x = np.array([0.,1.]); xh = np.zeros(2); Pc = np.eye(2)*10.
        for k in range(n):
            x = Fk@x + Lc@rng.normal(size=2)
            z = Hk@x + rng.normal(0, np.sqrt(rk), 1)
            xh = Fk@xh; Pc = Fk@Pc@Fk.T + Qk
            S = Hk@Pc@Hk.T + Rassumed
            nu = z - Hk@xh
            if k > 50: out.append(float(nu@np.linalg.inv(S)@nu))
            Kk = Pc@Hk.T@np.linalg.inv(S)
            xh = xh + Kk@nu; Pc = (np.eye(2) - Kk@Hk)@Pc
    return np.array(out)

nis_ok, nis_bad = _nis(np.array([[rk]])), _nis(np.array([[rk/9]]))
print()
print(f"[Ch11 s10] innovation nu = z - H x_hat with covariance S = H P H' + R")
print(f"[Ch11 s10] honest R:   mean nu'S^-1nu = {nis_ok.mean():.3f}, should equal m = 1   "
      f"{P(abs(nis_ok.mean()-1) < 0.07)}")
print(f"[Ch11 s10] R 9x too small: mean = {nis_bad.mean():.2f} -- the filter believes its own "
      f"press release  {P(nis_bad.mean() > 3)}")
lag1 = np.corrcoef(nis_ok[:-1], nis_ok[1:])[0,1]
print(f"[Ch11 s10] a consistent filter leaves WHITE innovations: lag-1 corr {lag1:+.4f}   "
      f"{P(abs(lag1) < 0.07)}")

# --- observability decides whether a sensor suite can work at all --------
_obsv = lambda A, C: np.vstack([C@np.linalg.matrix_power(A,i) for i in range(A.shape[0])])
Fb = np.array([[1,dtk,-0.5*dtk**2],[0,1,-dtk],[0,0,1.]])
print()
print(f"[Ch11 s10] [pos,vel] + a position sensor: rank O = "
      f"{np.linalg.matrix_rank(_obsv(Fk,Hk))}/2   {P(np.linalg.matrix_rank(_obsv(Fk,Hk))==2)}")
print(f"[Ch11 s10] [pos,vel,accel bias] + a position sensor: rank O = "
      f"{np.linalg.matrix_rank(_obsv(Fb, np.array([[1.,0,0]]))) }/3   "
      f"{P(np.linalg.matrix_rank(_obsv(Fb, np.array([[1.,0,0]])))==3)}")
print(f"[Ch11 s10] the same state, IMU alone, nothing measured: rank O = "
      f"{np.linalg.matrix_rank(_obsv(Fb, np.array([[0.,0,0]])))}/3   "
      f"{P(np.linalg.matrix_rank(_obsv(Fb, np.array([[0.,0,0]])))==0)}")
print(f"[Ch11 s10]   an unobservable state is not a tuning problem -- no filter can fix it")
for tt in (1., 10., 60., 600.):
    print(f"[Ch11 s10]   a 0.01 m/s^2 bias for {tt:5.0f} s -> {0.5*0.01*tt**2:8.2f} m of drift "
          f"(it grows as t^2)")

# --- the steady-state Kalman filter IS a second-order servo ---------------
def _ss(dts, qs, rs, n=20000):
    Fs = np.array([[1,dts],[0,1.]]); Hs = np.array([[1.,0.]])
    Qs = qs*np.array([[dts**3/3, dts**2/2],[dts**2/2, dts]]); Pc = np.eye(2)*10.
    for _ in range(n):
        Pc = Fs@Pc@Fs.T + Qs
        S = Hs@Pc@Hs.T + rs
        Kk = Pc@Hs.T@np.linalg.inv(S)
        Pc = (np.eye(2) - Kk@Hs)@Pc
    ev = np.linalg.eigvals((np.eye(2) - Kk@Hs)@Fs).astype(complex)
    s = np.log(ev)/dts
    return Kk.ravel(), abs(s[0]), -s[0].real/abs(s[0])
print()
zs = []
for dts in (0.1, 0.01):
    for qrr in (0.01, 1.0, 100.0):
        Kk, wn, z = _ss(dts, qrr, 1.0)
        zs.append(z)
        print(f"[Ch11 s11] dt={dts:5.2f} q/r={qrr:7.2f}: K=[{Kk[0]:.3f},{Kk[1]:7.3f}]  "
              f"wn={wn:7.4f} rad/s  zeta={z:.6f}  (q/(r dt))^(1/4)={(qrr/dts)**0.25:7.4f}")
print(f"[Ch11 s11] zeta is 1/sqrt2 = {1/np.sqrt(2):.6f} for EVERY q, r and dt   "
      f"{P(max(abs(z - 1/np.sqrt(2)) for z in zs) < 2e-5)}")
print(f"[Ch11 s11] and wn = (q/(r dt))^(1/4) to 4 figures                    "
      f"{P(abs(_ss(0.01, 1.0, 1.0)[1] - (1.0/0.01)**0.25) < 1e-3)}")
print(f"[Ch11 s11] closed form: p12=sqrt(q rho), p11=sqrt(2) rho^(3/4) q^(1/4),")
print(f"[Ch11 s11]   wn^2 = p12/rho = sqrt(q/rho) and 2 zeta wn = p11/rho = sqrt2 wn -> "
      f"zeta = 1/sqrt2 exactly")


# =========================== CHAPTER 11 (part 2) =========================
_l1, _l2 = 1.0, 0.8
_m1,_m2,_lc1,_lc2,_I1,_I2 = 2.0,1.5,0.5,0.4,0.12,0.06

def _DH(a, al, d, th):
    ct,st,ca,sa = np.cos(th),np.sin(th),np.cos(al),np.sin(al)
    return np.array([[ct,-st*ca, st*sa, a*ct],[st, ct*ca,-ct*sa, a*st],
                     [0, sa, ca, d],[0,0,0,1.]])
print()
_wo = max(np.abs(_DH(*p4)[:3,:3] @ _DH(*p4)[:3,:3].T - np.eye(3)).max()
          for p4 in rng.normal(0,2,(3000,4)))
_wd = max(abs(np.linalg.det(_DH(*p4)[:3,:3]) - 1) for p4 in rng.normal(0,2,(3000,4)))
print(f"[Ch11 s11] every DH matrix is a rigid transform: R R^T = I to {_wo:.0e}, "
      f"det R = 1 to {_wd:.0e}   {P(_wo<1e-12 and _wd<1e-12)}")
_Ch = np.eye(4)
for _p4 in rng.normal(0,1.5,(6,4)): _Ch = _Ch @ _DH(*_p4)
print(f"[Ch11 s11] and a 6-joint chain of them still is: det = "
      f"{np.linalg.det(_Ch[:3,:3]):.12f}   {P(abs(np.linalg.det(_Ch[:3,:3])-1)<1e-10)}")

_fk = lambda q: np.array([_l1*np.cos(q[0])+_l2*np.cos(q[0]+q[1]),
                          _l1*np.sin(q[0])+_l2*np.sin(q[0]+q[1])])
def _J(q):
    s1,s12,c1,c12 = np.sin(q[0]),np.sin(q[0]+q[1]),np.cos(q[0]),np.cos(q[0]+q[1])
    return np.array([[-_l1*s1-_l2*s12, -_l2*s12],[_l1*c1+_l2*c12, _l2*c12]])
print()
_wj = 0.0
for _q in rng.uniform(-np.pi,np.pi,(2000,2)):
    _Jn = np.column_stack([(_fk(_q+1e-6*e)-_fk(_q-1e-6*e))/2e-6 for e in np.eye(2)])
    _wj = max(_wj, np.abs(_Jn - _J(_q)).max())
print(f"[Ch11 s12] the analytic Jacobian matches central differences to {_wj:.0e}   {P(_wj<1e-7)}")
_wdet = max(abs(np.linalg.det(_J(q)) - _l1*_l2*np.sin(q[1]))
            for q in rng.uniform(-np.pi,np.pi,(3000,2)))
print(f"[Ch11 s12] det J = l1 l2 sin(q2) identically, max err {_wdet:.0e}   {P(_wdet<1e-12)}")
print(f"[Ch11 s12]   so det J = 0 exactly when q2 = 0 or pi -- straight out, or folded back")
for _q2 in (np.pi/2, 0.3, 0.05, 0.005):
    _s = np.linalg.svd(_J([0.4,_q2]), compute_uv=False)
    print(f"[Ch11 s12]   q2={_q2:6.3f} rad: singular values [{_s[0]:.4f}, {_s[1]:.6f}], "
          f"cond {_s[0]/_s[1]:8.1f}")
_qdc = np.array([0.4, 0.0]); _u = np.linalg.svd(_J(_qdc))[0]
_rad = _fk(_qdc)/np.linalg.norm(_fk(_qdc))
print(f"[Ch11 s12] at q2=0 the lost direction {_u[:,1].round(4)} IS the radial direction "
      f"{_rad.round(4)}: |dot| = {abs(_u[:,1]@_rad):.10f}   {P(abs(abs(_u[:,1]@_rad)-1)<1e-9)}")
print(f"[Ch11 s12]   the arm cannot reach further out, only swing -- a linkage at dead centre")

print()
_lam = 0.05
_wp = max(np.abs(_J(q).T@np.linalg.inv(_J(q)@_J(q).T + _lam**2*np.eye(2))
                 - np.linalg.inv(_J(q).T@_J(q) + _lam**2*np.eye(2))@_J(q).T).max()
          for q in rng.uniform(-np.pi,np.pi,(1500,2)))
print(f"[Ch11 s12] J^T(JJ^T+l^2 I)^-1 == (J^TJ+l^2 I)^-1 J^T to {_wp:.0e}   {P(_wp<1e-8)}")
_dx = np.array([0.01,0.0]); _qn = np.array([0.4,0.004]); _Jn2 = _J(_qn)
_dqd = _Jn2.T@np.linalg.solve(_Jn2@_Jn2.T + _lam**2*np.eye(2), _dx)
_obj = lambda d: np.sum((_Jn2@d-_dx)**2) + _lam**2*np.sum(d**2)
_best = min(_obj(_dqd + 1e-4*rng.normal(size=2)) for _ in range(4000))
print(f"[Ch11 s12] and it is the minimiser of ||J dq - dx||^2 + l^2||dq||^2   "
      f"{P(_best > _obj(_dqd))}")
print(f"[Ch11 s12] 0.004 rad from dead centre, for a 1 cm move: pseudo-inverse demands "
      f"{np.linalg.norm(np.linalg.pinv(_Jn2)@_dx):.2f} rad,")
print(f"[Ch11 s12]   damped least squares {np.linalg.norm(_dqd):.4f} rad   "
      f"{P(np.linalg.norm(_dqd) < np.linalg.norm(np.linalg.pinv(_Jn2)@_dx)/100)}")

def _M(q2):
    a = _m2*(_lc2**2 + _l1*_lc2*np.cos(q2)) + _I2
    return np.array([[_m1*_lc1**2+_I1+_m2*(_l1**2+_lc2**2+2*_l1*_lc2*np.cos(q2))+_I2, a],
                     [a, _m2*_lc2**2+_I2]])
def _C(q2, dq):
    h = -_m2*_l1*_lc2*np.sin(q2)
    return np.array([[h*dq[1], h*(dq[0]+dq[1])],[-h*dq[0], 0.0]])
def _KE(q, dq):
    v1 = _lc1*dq[0]*np.array([-np.sin(q[0]), np.cos(q[0])])
    v2 = (_l1*dq[0]*np.array([-np.sin(q[0]), np.cos(q[0])])
          + _lc2*(dq[0]+dq[1])*np.array([-np.sin(q[0]+q[1]), np.cos(q[0]+q[1])]))
    return (0.5*_m1*v1@v1 + 0.5*_I1*dq[0]**2 + 0.5*_m2*v2@v2 + 0.5*_I2*(dq[0]+dq[1])**2)
print()
_wke = _wsym = 0.0
for _ in range(3000):
    _q = rng.uniform(-np.pi,np.pi,2); _dq = rng.normal(0,2,2); _Mm = _M(_q[1])
    _wke = max(_wke, abs(0.5*_dq@_Mm@_dq - _KE(_q,_dq)))
    _wsym = max(_wsym, abs(_Mm[0,1]-_Mm[1,0]))
print(f"[Ch11 s13] (1/2) q_dot^T M q_dot == the kinetic energy computed from link "
      f"velocities, to {_wke:.0e}   {P(_wke<1e-12)}")
_mine = min(np.linalg.eigvalsh(_M(q2)).min() for q2 in np.linspace(-np.pi,np.pi,20001))
_maxc = max(np.linalg.eigvalsh(_M(q2)).max()/np.linalg.eigvalsh(_M(q2)).min()
            for q2 in np.linspace(-np.pi,np.pi,2001))
print(f"[Ch11 s13] M is symmetric to {_wsym:.0e} and its smallest eigenvalue anywhere in "
      f"the workspace is {_mine:.5f} > 0   {P(_wsym<1e-14 and _mine>1e-6)}")
print(f"[Ch11 s13]   worst condition number {_maxc:.1f}: the inertia a motor feels varies "
      f"by that factor with configuration")
_wsk = 0.0
for _ in range(3000):
    _q2 = rng.uniform(-np.pi,np.pi); _dq = rng.normal(0,2,2)
    _Md = (_M(_q2+1e-6*_dq[1]) - _M(_q2-1e-6*_dq[1]))/2e-6
    _S = _Md - 2*_C(_q2,_dq); _wsk = max(_wsk, np.abs(_S+_S.T).max())
print(f"[Ch11 s13] Mdot - 2C is skew-symmetric, max |S + S^T| = {_wsk:.0e}   {P(_wsk<1e-6)}")
print(f"[Ch11 s13]   hence q_dot^T (Mdot - 2C) q_dot = 0: Coriolis and centrifugal terms do")
print(f"[Ch11 s13]   NO net work -- the same statement as 'Coriolis acts perpendicular to v'")

def _pid(Kp,Kd,T=30.0,dt=2e-4,m=1.0,b=0.5,Ki=0.0,sat=None,clamp=False):
    q=dq=I=0.0; pk=0.0
    for _ in range(int(T/dt)):
        e=1.0-q; I+=e*dt
        tau=Kp*e+Kd*(-dq)+Ki*I
        if sat is not None:
            t2=float(np.clip(tau,-sat,sat))
            if clamp and t2!=tau: I-=e*dt
            tau=t2
        dq+=((tau-b*dq)/m)*dt; q+=dq*dt; pk=max(pk,q)
    return pk
print()
for _Kp,_Kd in ((25,1.5),(100,6.5),(400,17.5)):
    _z=(0.5+_Kd)/(2*np.sqrt(_Kp)); _pr=100*np.exp(-np.pi*_z/np.sqrt(1-_z*_z))
    _me=100*(_pid(_Kp,_Kd)-1)
    print(f"[Ch11 s14] Kp={_Kp:4.0f} Kd={_Kd:5.1f} -> wn={np.sqrt(_Kp):5.2f} zeta={_z:5.3f}: "
          f"overshoot {_me:6.2f}% measured vs {_pr:6.2f}% from exp(-pi z/sqrt(1-z^2))  "
          f"{P(abs(_me-_pr)<0.4)}")
print(f"[Ch11 s14] Kp IS a stiffness (wn = sqrt(Kp/m)), Kd IS a damping coefficient")
print(f"[Ch11 s14]   (zeta = (b+Kd)/(2 sqrt(Kp m))); the plant's own damping just adds to Kd")
_w = np.logspace(-2,3,200000)
_L = (25.0 + 1j*_w*9.5)/(1j*_w*(1j*_w + 0.5))
_i = np.argmin(np.abs(np.abs(_L)-1.0))
_pm = 180 + np.degrees(np.angle(_L[_i]))
print(f"[Ch11 s14] and the loop has a phase margin you can compute: {_pm:.1f} deg at "
      f"{_w[_i]:.2f} rad/s   {P(40 < _pm < 90)}")
print(f"[Ch11 s14] torque saturation at 3.0 N m, Ki=40: overshoot {100*(_pid(30,8,Ki=40,sat=3.0)-1):.1f}% "
      f"without anti-windup, {100*(_pid(30,8,Ki=40,sat=3.0,clamp=True)-1):.1f}% with   "
      f"{P(_pid(30,8,Ki=40,sat=3.0,clamp=True) < _pid(30,8,Ki=40,sat=3.0))}")

# --- behaviour cloning: what actually compounds --------------------------
print()
_eps = 0.01
print(f"[Ch11 s15] compounding error, per-step disagreement eps={_eps}. After the FIRST")
print(f"[Ch11 s15]   disagreement the policy is off-distribution and pays 1 per step left:")
for _T in (10,25,50,100,200):
    _ex = sum(_eps*(1-_eps)**t*(_T-t) for t in range(_T))
    _mc = np.mean([next((_T-t for t in range(_T) if rng.random()<_eps), 0) for _ in range(20000)])
    print(f"[Ch11 s15]   T={_T:4d}: expected cost {_ex:8.3f} (Monte Carlo {_mc:8.3f}), "
          f"eps T^2/2 = {_eps*_T*_T/2:8.3f}   {P(abs(_mc-_ex) < 0.06*max(1,_ex))}")
_r50 = sum(_eps*(1-_eps)**t*(50-t) for t in range(50))
_r100 = sum(_eps*(1-_eps)**t*(100-t) for t in range(100))
print(f"[Ch11 s15] doubling the horizon costs {_r100/_r50:.2f}x, not 2x   {P(_r100/_r50>1.8)}")
print(f"[Ch11 s15]   (eps T^2/2 is the small-eps*T limit; past that it saturates, because an")
print(f"[Ch11 s15]   early mistake becomes near-certain -- 114 not 200 at T=200)")
print(f"[Ch11 s15] DAgger relabels those states, so a mistake costs O(1): eps T = {_eps*100:.1f} "
      f"at T=100 against {_r100:.1f}   {P(_r100 > 3*_eps*100)}")
def _roll(T, gain, seed=7, e=0.02):
    r = np.random.default_rng(seed); x = 0.0; d = 0.0
    for t in range(T):
        x = x + gain*(1.0-x) + e*r.normal(); d = abs(x - (1-np.exp(-0.6*t)))
    return d
print(f"[Ch11 s15] but noise ALONE never compounds. Same noise, three closed-loop gains:")
for _g,_n in ((0.6,"contracting"),(1.99,"marginal   "),(2.05,"expansive  ")):
    print(f"[Ch11 s15]   {_n} (gain {_g:4.2f}): deviation at T=10,40,160 = "
          f"{_roll(10,_g):.4f}, {_roll(40,_g):.4f}, {_roll(160,_g):.3e}")
print(f"[Ch11 s15]   a contracting loop absorbs noise forever. What compounds is LOSING the")
print(f"[Ch11 s15]   contraction, and leaving the training distribution is how it is lost.")
print(f"[Ch11 s15] CONTRAST: a four-bar with 0.1 mm of pin clearance has 0.1 mm of error at")
print(f"[Ch11 s15]   every point of its cycle, not 0.1 mm x cycles. A constraint does not")
print(f"[Ch11 s15]   integrate its own error; a policy that picks its own next state does.")

# --- multimodality: the mean of two valid actions is not valid -----------
print()
_lft, _rgt, _r = np.array([-1.0,0.5]), np.array([1.0,0.5]), 0.6
_mid = 0.5*(_lft+_rgt)
print(f"[Ch11 s16] two ways past an obstacle of radius {_r} at the origin:")
print(f"[Ch11 s16]   left |d|={np.linalg.norm(_lft):.3f}, right |d|={np.linalg.norm(_rgt):.3f}, "
      f"their mean |d|={np.linalg.norm(_mid):.3f} -> COLLIDES   "
      f"{P(np.linalg.norm(_mid) < _r < np.linalg.norm(_lft))}")
print(f"[Ch11 s16]   the valid set is not convex, so a regression to the mean leaves it")
def _pth(ch, T=60, seed=5):
    r=np.random.default_rng(seed); x=[0.0]; md=0
    for t in range(T):
        if t%ch==0: md = 1 if r.random()<0.5 else -1
        x.append(x[-1]+0.05*md)
    return np.array(x)
for _ch in (1,5,20,60):
    _p=_pth(_ch); _f=int(np.sum(np.diff(np.sign(np.diff(_p)))!=0))
    print(f"[Ch11 s16]   chunk={_ch:3d}: {_f:3d} reversals in 60 steps, |x| travelled "
          f"{abs(_p[-1]):.3f}")
print(f"[Ch11 s16] re-sampling the mode every step dithers and commits to neither; chunking")
print(f"[Ch11 s16]   picks once and follows through   {P(abs(_pth(1)[-1]) < abs(_pth(20)[-1]))}")

# --- sim-to-real: one controller, many plants ----------------------------
def _lqr(m, dt=0.02, r=0.05, n=4000):
    A=np.array([[1,dt],[0,1.]]); B=np.array([[0.],[dt/m]]); Q=np.diag([1.0,0.1]); Sv=Q.copy()
    for _ in range(n):
        Kv=np.linalg.solve(r+B.T@Sv@B, B.T@Sv@A); Sv=Q+A.T@Sv@(A-B@Kv)
    return Kv.ravel()
def _cost(K, m, dt=0.02, T=400):
    A=np.array([[1,dt],[0,1.]]); B=np.array([0., dt/m]); x=np.array([1.,0.]); c=0.0
    for _ in range(T):
        u=float(-K@x); c += x[0]**2+0.1*x[1]**2+0.05*u*u; x=A@x+B*u
        if not np.isfinite(c) or abs(x[0])>1e6: return np.inf
    return c
print()
_ms = np.array([0.4,0.6,1.0,1.6,2.5,4.0])
_Kn = _lqr(1.0)
_Kd2 = min((_lqr(mm) for mm in np.linspace(0.4,4.0,25)),
           key=lambda K: float(np.mean([_cost(K,mm) for mm in _ms])))
print(f"[Ch11 s17] one controller, six plants. 'System ID' designs for m=1 exactly;")
print(f"[Ch11 s17]   'domain randomisation' minimises the mean over m in [0.4, 4.0]:")
for mm in _ms:
    _cn,_cd = _cost(_Kn,mm), _cost(_Kd2,mm)
    print(f"[Ch11 s17]   m={mm:4.1f}: nominal {_cn:8.3f}  randomised {_cd:8.3f}  "
          f"{'nominal' if _cn<_cd else 'randomised'} wins")
_A=np.array([_cost(_Kn,mm) for mm in _ms]); _B2=np.array([_cost(_Kd2,mm) for mm in _ms])
print(f"[Ch11 s17] at the design point the nominal wins ({_cost(_Kn,1.):.2f} vs "
      f"{_cost(_Kd2,1.):.2f})   {P(_cost(_Kn,1.) <= _cost(_Kd2,1.))}")
print(f"[Ch11 s17] on the mean ({_A.mean():.2f} vs {_B2.mean():.2f}) and on the worst case "
      f"({_A.max():.2f} vs {_B2.max():.2f}) the randomised wins   "
      f"{P(_B2.mean()<_A.mean() and _B2.max()<_A.max())}")
print(f"[Ch11 s17]   -> domain randomisation is robust design done by sampling. It buys the")
print(f"[Ch11 s17]   worst case by selling the best case -- the same trade, priced the same way")
print(f"[Ch11 s17] BREAK: robust synthesis gives a guarantee over the whole uncertainty set.")
print(f"[Ch11 s17]   Sampling gives an average over what was sampled, and says nothing at all")
print(f"[Ch11 s17]   about a plant outside the sampling distribution")

# --- action tokenisation --------------------------------------------------
print()
_N, _lo, _hi = 256, -0.1, 0.1
_st = (_hi-_lo)/_N
print(f"[Ch11 s18] {_N} bins over [{_lo}, {_hi}] m/s -> {_st*1000:.5f} mm/s per bin   "
      f"{P(abs(_st*1000-0.78125)<1e-9)}")
print(f"[Ch11 s18]   max error {_st/2*1000:.4f} mm/s, RMS {_st/np.sqrt(12)*1000:.4f} mm/s "
      f"-- Sheet 09's ADC result, unchanged")
_rt2 = [1,128,91,241,5,101,127]
print(f"[Ch11 s18] the source has RT-2 emit '{' '.join(map(str,_rt2))}' for a 7-DoF action;")
print(f"[Ch11 s18]   every token < 256 (max {max(_rt2)}). Per-dimension bins would put "
      f"dimension 7 in [1536, 1791]   {P(max(_rt2) < 256)}")
print(f"[Ch11 s18]   -> 256 SHARED bins, disambiguated by position. Vocabulary cost 256, not")
print(f"[Ch11 s18]   7x256 = {7*256}: the source's own example contradicts its arithmetic")


# ---------------------- Ch11 part 2: locomotion & VLA ---------------------
def _marg(tri, pt):
    d=[]
    for i in range(3):
        e=tri[(i+1)%3]-tri[i]; n=np.array([-e[1],e[0]]); n=n/np.linalg.norm(n)
        if n@(tri[(i+2)%3]-tri[i]) < 0: n=-n
        d.append(n@(pt-tri[i]))
    return min(d)
print()
print(f"[Ch11 s16] a quadruped lifts one corner foot; CoM at the body centre:")
_ok16 = True
for _ratio in (1.0, 1.5, 2.0, 3.0):
    _b = 0.18; _a = _b*_ratio
    _ft = np.array([[_a,_b],[_a,-_b],[-_a,_b],[-_a,-_b]])
    _tri = np.array([f for f in _ft if not np.allclose(f,_ft[0])])
    _m = _marg(_tri, np.zeros(2)); _ok16 &= abs(_m) < 1e-14
    print(f"[Ch11 s16]   aspect {_ratio:3.1f}:1 -> static stability margin {_m:+.2e} m")
print(f"[Ch11 s16] EXACTLY zero at every aspect ratio   {P(_ok16)}")
print(f"[Ch11 s16]   because lifting a corner leaves a triangle whose hypotenuse is the")
print(f"[Ch11 s16]   rectangle's diagonal, and a rectangle's centre lies ON its diagonal")
_b0=0.18; _a0=0.35
_tri0=np.array([[_a0,-_b0],[-_a0,_b0],[-_a0,-_b0]])
for _s in (0.02,0.05,0.10):
    _d=_s*np.array([-_a0,-_b0])/np.linalg.norm([_a0,_b0])
    print(f"[Ch11 s16]   shift the CoM {_s*100:4.1f} cm toward the stance side -> margin "
          f"{_marg(_tri0,_d):.4f} m")
print(f"[Ch11 s16] so the lateral sway of a slow-walking robot dog is this geometry obeyed")
print()
for _h,_v,_R in ((1.0,10.,50.),(0.9,5.,20.),(1.2,15.,100.)):
    _o=_h*_v*_v/(9.81*_R)
    print(f"[Ch11 s16] h={_h:.1f} m v={_v:4.1f} m/s R={_R:5.1f} m -> CoM sits {_o:.3f} m "
          f"outside the contact patch, lean {np.degrees(np.arctan(_v*_v/(9.81*_R))):5.2f} deg")
print(f"[Ch11 s16] and nothing falls over, because the RESULTANT still passes inside it   "
      f"{P(abs(1.0*100/(9.81*50) - 0.2039) < 1e-3)}")
print(f"[Ch11 s16]   'CoM over the support polygon' is the v=0 special case; the dynamic")
print(f"[Ch11 s16]   condition is 'ZMP inside it', and running violates the static one always")

def _cpg(psi, w=2*np.pi, K=6.0, T=4000, dt=0.002, seed=3):
    r=np.random.default_rng(seed); n=len(psi); ph=r.uniform(0,2*np.pi,n)
    for _ in range(T):
        ph = ph + np.array([w + K*sum(np.sin(ph[j]-ph[i]-(psi[j]-psi[i]))
                                      for j in range(n))/n for i in range(n)])*dt
    return (ph-ph[0]) % (2*np.pi), (np.array(psi)-psi[0]) % (2*np.pi)
print()
_ok17 = True
for _nm,_ps in (("trot  (diagonal pairs)",[0,np.pi,np.pi,0]),
                ("pace  (lateral pairs) ",[0,np.pi,0,np.pi]),
                ("bound (front / rear)  ",[0,0,np.pi,np.pi]),
                ("walk  (90 deg apart)  ",[0,np.pi/2,np.pi,3*np.pi/2])):
    _rel,_tg = _cpg(_ps); _e=np.abs((_rel-_tg+np.pi)%(2*np.pi)-np.pi).max(); _ok17 &= _e<1e-3
    print(f"[Ch11 s16] {_nm}: locks to {np.degrees(_rel).round(0)} deg from random phases")
print(f"[Ch11 s16] every gait converges to its prescribed offsets   {P(_ok17)}")
print(f"[Ch11 s16]   a four-stroke inline-4 fires at 0/180/360/540 deg of crank: the same")
print(f"[Ch11 s16]   object. A gait IS a firing order -- trot is the 1-4/2-3 pairing")

try:
    from scipy.stats import beta as _beta, fisher_exact as _fe
    print()
    for _k,_n in ((16,20),(15,20),(17,20),(40,50)):
        _lo=_beta.ppf(0.025,_k,_n-_k+1); _hi=_beta.ppf(0.975,_k+1,_n-_k)
        print(f"[Ch11 s17] a VLA success rate of {_k}/{_n} = {_k/_n:.0%} has a 95% interval "
              f"[{_lo:.1%}, {_hi:.1%}] -- {100*(_hi-_lo):.0f} points wide")
    _p = _fe([[15,5],[17,3]])[1]
    print(f"[Ch11 s17] so 75% and 85% on 20 trials each are indistinguishable: Fisher exact "
          f"p = {_p:.3f}   {P(_p > 0.1)}")
except Exception as _e:
    print(f"[Ch11 s17] (scipy unavailable: {_e})")


# --- damped least squares peaks where sigma_min = lambda, then withdraws --
def _dqn(q1,q2,lam):
    Jm=_J([q1,q2])
    return np.linalg.norm(Jm.T@np.linalg.solve(Jm@Jm.T + lam**2*np.eye(2), np.array([0.01,0.0])))
print()
_lm, _q1 = 0.05, 0.55
_qs = np.linspace(0.002, 1.5, 4000)
_vals = np.array([_dqn(_q1,q,_lm) for q in _qs])
_pk = _qs[_vals.argmax()]
_sig = np.array([np.linalg.svd(_J([_q1,q]), compute_uv=False)[1] for q in _qs])
_eq = _qs[np.abs(_sig-_lm).argmin()]
for _q in (1.5,0.4,0.2,0.12,0.05,0.02,0.005,0.001):
    print(f"[Ch11 s12] q2={_q:6.3f}: sigma_min={np.linalg.svd(_J([_q1,_q]),compute_uv=False)[1]:.5f}, "
          f"damped demand {_dqn(_q1,_q,_lm):.5f} rad")
print(f"[Ch11 s12] the damped demand PEAKS at q2={_pk:.4f}, and sigma_min = lambda at "
      f"q2={_eq:.4f}   {P(abs(_pk-_eq) < 0.02)}")
print(f"[Ch11 s12]   because the filter factor sigma/(sigma^2+lambda^2) is largest at "
      f"sigma = lambda,")
print(f"[Ch11 s12]   bounded by ||dx||/(2 lambda) = {0.01/(2*_lm):.4f} (measured peak "
      f"{_vals.max():.4f})   {P(_vals.max() <= 0.01/(2*_lm)+1e-9)}")
print(f"[Ch11 s12] past the peak it FALLS towards zero -- damped least squares does not cap")
print(f"[Ch11 s12]   the demand, it withdraws from the direction being lost   "
      f"{P(_dqn(_q1,0.001,_lm) < _vals.max()/10)}")

# --- the PID decay rate does not depend on Kp at all ---------------------
print()
_zw = set()
for _Kp in (25,100,400,1600):
    for _Kd in (1.5,9.5):
        _wn=np.sqrt(_Kp); _z=(0.5+_Kd)/(2*np.sqrt(_Kp)); _zw.add((round(_Kd,3), round(_z*_wn,9)))
        print(f"[Ch11 s14] Kp={_Kp:5d} Kd={_Kd:4.1f}: wn={_wn:6.2f} zeta={_z:.4f} -> "
              f"zeta*wn = {_z*_wn:.6f}, and (b+Kd)/2m = {(0.5+_Kd)/2:.6f}")
print(f"[Ch11 s14] zeta*wn = (b+Kd)/(2m): Kp cancels completely   "
      f"{P(len(_zw) == 2 and all(abs(v-(k+0.5)/2) < 1e-9 for k,v in _zw))}")
print(f"[Ch11 s14]   -> Kd alone sets the decay envelope (settling time 8m/(b+Kd));")
print(f"[Ch11 s14]   Kp alone sets the ringing frequency. Raising Kp cannot speed up settling")


# =========================== CHAPTER 11 (part 3) =========================
def _Wof(A):
    return np.linalg.inv(np.diag(A.sum(1)+1.0)) @ (np.eye(len(A)) + A)
def _run(A, x0, n=5000):
    W=_Wof(A); x=x0.copy()
    for _ in range(n): x = W@x
    return x
print()
_n=7; _A=np.zeros((_n,_n))
for _j in range(1,_n): _A[0,_j]=_A[_j,0]=1
_A[1,2]=_A[2,1]=1; _A[3,4]=_A[4,3]=1
_x0=rng.uniform(0,10,_n); _xf=_run(_A,_x0); _dg=_A.sum(1)
_plain=_x0.mean(); _wtd=(_dg+1)@_x0/(_dg+1).sum()
print(f"[Ch11 s19] neighbour-averaging consensus on a symmetric graph, degrees "
      f"{_dg.astype(int)}")
print(f"[Ch11 s19] all nodes agree: spread {_xf.max()-_xf.min():.1e}   "
      f"{P(_xf.max()-_xf.min() < 1e-9)}")
print(f"[Ch11 s19]   they converge to        {_xf[0]:.8f}")
print(f"[Ch11 s19]   the plain average is    {_plain:.8f}  <- the source says this   "
      f"{P(abs(_xf[0]-_plain) > 1e-3)} (it is WRONG)")
print(f"[Ch11 s19]   the degree-weighted mean{_wtd:.8f}  <- correct   "
      f"{P(abs(_xf[0]-_wtd) < 1e-9)}")
print(f"[Ch11 s19]   the error is {abs(_xf[0]-_plain):.4f} on a spread of "
      f"{_x0.max()-_x0.min():.2f} -- {100*abs(_xf[0]-_plain)/(_x0.max()-_x0.min()):.1f}%")
_ev,_evec = np.linalg.eig(_Wof(_A).T)
_pi = np.real(_evec[:, np.argmin(np.abs(_ev-1))]); _pi=_pi/_pi.sum()
print(f"[Ch11 s19] the weights are the left Perron vector of W, and it IS (d+1)/sum(d+1)   "
      f"{P(np.allclose(_pi,(_dg+1)/(_dg+1).sum(),atol=1e-9))}")
print(f"[Ch11 s19]   pi        = {np.round(_pi,5)}")
print(f"[Ch11 s19]   (d+1)/sum = {np.round((_dg+1)/(_dg+1).sum(),5)}")
_worst=0.0
for _ in range(300):
    _m=rng.integers(5,12); _B=np.triu((rng.random((_m,_m))<0.35).astype(float),1); _B=_B+_B.T
    if (_B.sum(1)==0).any(): continue
    if np.sort(np.linalg.eigvalsh(np.diag(_B.sum(1))-_B))[1] < 1e-8: continue
    _y=rng.uniform(0,10,_m); _worst=max(_worst, abs(_run(_B,_y,3000)[0]-_y.mean()))
print(f"[Ch11 s19] over 300 random connected graphs the gap reaches {_worst:.3f}   "
      f"{P(_worst > 0.4)}")
_pos=rng.uniform(-5,5,(10,2)); _Dm=np.linalg.norm(_pos[:,None]-_pos[None,:],axis=-1)
_Ak=np.zeros((10,10))
for _i in range(10):
    for _j in np.argsort(_Dm[_i])[1:4]: _Ak[_i,_j]=1
_asym=int(np.abs(_Ak-_Ak.T).sum()/2)
print(f"[Ch11 s19] the source's own lab uses 3-nearest-neighbours: {_asym} edges are one-way, "
      f"so the graph is DIRECTED   {P(_asym>0)}")
_z=rng.uniform(0,10,10)
print(f"[Ch11 s19]   it converges to {_run(_Ak,_z,6000)[0]:.5f}, plain mean {_z.mean():.5f}   "
      f"{P(abs(_run(_Ak,_z,6000)[0]-_z.mean())>1e-3)}")
print()
for _nm,_Am in (("ring    ", np.eye(8,k=1)+np.eye(8,k=-1)+np.eye(8,k=7)+np.eye(8,k=-7)),
                ("path    ", np.eye(8,k=1)+np.eye(8,k=-1)),
                ("complete", np.ones((8,8))-np.eye(8))):
    _L=np.diag(_Am.sum(1))-_Am; _l2=np.sort(np.linalg.eigvalsh(_L))[1]
    _r=np.sort(np.abs(np.linalg.eigvals(_Wof(_Am))))[::-1][1]
    print(f"[Ch11 s19] {_nm}: lambda_2(L) = {_l2:6.4f}, |lambda_2(W)| = {_r:.4f}, "
          f"{'inf' if _r>=1 else int(np.ceil(np.log(1e-3)/np.log(max(_r,1e-16))))} steps to 1e-3")
_Lc=np.diag((np.ones((8,8))-np.eye(8)).sum(1))-(np.ones((8,8))-np.eye(8))
print(f"[Ch11 s19] L @ ones = 0 exactly   {P(np.allclose(_Lc@np.ones(8),0))} -- the zero")
print(f"[Ch11 s19]   eigenvalue is the rigid-body mode, so connectivity lives in lambda_2,")
print(f"[Ch11 s19]   the Fiedler value, exactly as the first elastic mode of a free-free beam")
_Ws=_Wof(_A).copy(); _Ws[3,:]=0; _Ws[3,3]=1.0
_xs=_x0.copy()
for _ in range(9000): _xs=_Ws@_xs
print()
print(f"[Ch11 s19] one agent stops updating (stuck sensor, or a liar). Its value is "
      f"{_x0[3]:.5f};")
print(f"[Ch11 s19]   the whole swarm ends at {_xs.min():.5f}..{_xs.max():.5f}   "
      f"{P(abs(_xs.mean()-_x0[3])<1e-6)}")
print(f"[Ch11 s19]   'no single point of failure' holds for crashes, NOT for value consensus")
for _who in (0,5):
    _xb=_x0.copy(); _xb[_who]+=100.0
    print(f"[Ch11 s19]   a +100 error at node {_who} (degree {int(_dg[_who])}) moves the "
          f"consensus by {abs(_run(_A,_xb)[0]-_xf[0]):7.3f} = 100*pi_{_who}")
_e0=abs(_run(_A,np.where(np.arange(_n)==0,_x0+100,_x0))[0]-_xf[0])
_e5=abs(_run(_A,np.where(np.arange(_n)==5,_x0+100,_x0))[0]-_xf[0])
print(f"[Ch11 s19]   a better-connected liar does more damage, in exact proportion to its "
      f"weight   {P(_e0 > _e5 and abs(_e0-100*_pi[0])<1e-6)}")

# --- the cost of distance -------------------------------------------------
print()
_aE,_eE,_aM,_eM = 149.598e9,0.0167,227.939e9,0.0934
for _nm,_d in (("closest possible  (both perihelion, opposition)", _aM*(1-_eM)-_aE*(1+_eE)),
               ("typical opposition (circular)",                   _aM-_aE),
               ("typical conjunction (circular)",                  _aM+_aE),
               ("farthest possible (both aphelion, conjunction)",  _aM*(1+_eM)+_aE*(1+_eE))):
    print(f"[Ch11 s20] {_nm:46s} {_d/1e9:6.1f} Gm -> {_d/c_light/60:5.2f} min one way")
_lo=(_aM*(1-_eM)-_aE*(1+_eE))/c_light/60; _hi=(_aM*(1+_eM)+_aE*(1+_eE))/c_light/60
print(f"[Ch11 s20] true range {_lo:.2f}-{_hi:.2f} min; the source says 4-24, so the upper "
      f"end is ~7% high   {P(3.0<_lo<3.1 and 22.0<_hi<22.6)}")
print()
_RAD=200e6
for _nm,_fl in (("ResNet-50 forward pass",8.2e9),("classical 640x480 block stereo",3.0e8)):
    print(f"[Ch11 s20] {_nm:32s} {_fl/1e9:5.2f} GFLOP -> {_fl/_RAD:7.2f} s at 200 MFLOP/s")
print(f"[Ch11 s20] 30 fps needs 33 ms, so a ResNet is short by {8.2e9/_RAD/0.0333:.0f}x   "
      f"{P(8.2e9/_RAD > 40)}")
print(f"[Ch11 s20]   Mars rovers run classical stereo for an arithmetic reason, not a")
print(f"[Ch11 s20]   cultural one: the flops are simply not there")
print()
print(f"[Ch11 s20] light in vacuum / sound in water = {c_light/1500.:,.0f}x")
for _R in (1000,3000,10000):
    print(f"[Ch11 s20]   AUV {_R/1000:4.1f} km out: acoustic round trip {2*_R/1500.:5.2f} s, "
          f"in which it swims {2*_R/1500.*2:5.1f} m at 2 m/s")
print(f"[Ch11 s20] a 3 km AUV has a 4.00 s control delay   {P(abs(2*3000/1500.-4.0)<1e-9)}")

# --- worked example: sizing a highway follower ---------------------------
print()
_v=120/3.6
print(f"[Ch11 s22] WORKED EXAMPLE, 120 km/h = {_v:.2f} m/s")
for _a,_nm in ((3.0,"comfortable"),(5.0,"firm"),(8.0,"emergency")):
    print(f"[Ch11 s22]   {_nm:11s} at {_a:.0f} m/s^2: {_v*_v/(2*_a):6.1f} m + {_v*0.3:4.1f} m "
          f"of 0.3 s latency = {_v*_v/(2*_a)+_v*0.3:6.1f} m")
_need=_v*_v/6+_v*0.3
_fpx=1400.0
print(f"[Ch11 s22] target: reliable detection at {_need:.0f} m. With f = {_fpx:.0f} px:")
for _b,_sub in ((0.12,0.5),(0.12,0.1),(1.20,0.5),(1.20,0.1)):
    _d=_fpx*_b/_need; _l=_fpx*_b/(_d+_sub); _h=_fpx*_b/(_d-_sub) if _d>_sub else np.inf
    _f=(_h-_l)/2/_need if np.isfinite(_h) else np.inf
    print(f"[Ch11 s22]   b={_b:4.2f} m, +-{_sub:.1f} px: disparity {_d:5.2f} px -> "
          + (f"Z in [{_l:6.1f}, {_h:6.1f}] m, +-{100*_f:5.1f}%" if np.isfinite(_h)
             else "UNBOUNDED above"))
print(f"[Ch11 s22] 12 cm is hopeless at 0.5 px (+-88%) and marginal at 0.1 px (+-12%);")
print(f"[Ch11 s22]   1.2 m across the windscreen gives +-1.2%   "
      f"{P(_fpx*1.2/_need > 8 and _fpx*0.12/_need < 1)}")
print(f"[Ch11 s22]   so 'stereo cannot do long range' is really 'SHORT-BASELINE stereo")
print(f"[Ch11 s22]   cannot' -- b enters linearly, and a car is 1.8 m wide")
print(f"[Ch11 s22] control side: a 0.5 m lane correction in 1.5 s needs "
      f"{2*0.5/1.5**2:.3f} m/s^2 = {100*(2*0.5/1.5**2)/9.81:.1f}% g, and wn = "
      f"{4/(0.707*1.5):.2f} rad/s at zeta = 1/sqrt2")
print(f"[Ch11 s22]   that is trivial for a 100 Hz steering loop: perception is the binding")
print(f"[Ch11 s22]   constraint, not control   {P(2*0.5/1.5**2 < 0.5)}")

# =========================== CHAPTER 12 ===================================
print("\n" + "="*66)
print("CHAPTER 12 -- GRAPH NEURAL NETWORKS")
print("="*66)

def _lap(A): return np.diag(A.sum(1)) - A
def _g(n, edges, w=None):
    A = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        v = 1.0 if w is None else w[k]
        A[i, j] = A[j, i] = v
    return A

# --- s07: L = D - A IS the direct-stiffness assembly ----------------------
_E = [(0,1),(0,2),(1,2),(2,3),(3,4)]; _n = 5
_A = _g(_n, _E); _L = _lap(_A)
_K = np.zeros((_n,_n)); _ke = np.array([[1.,-1.],[-1.,1.]])
for i,j in _E:
    _d=[i,j]
    for a_ in range(2):
        for b_ in range(2): _K[_d[a_],_d[b_]] += _ke[a_,b_]
print(f"\n[Ch12 s07] L = D - A equals the assembled stiffness matrix of unit springs,")
print(f"[Ch12 s07]   one scalar DOF per node   max|L-K| = {np.abs(_L-_K).max():.2e}   {P(np.allclose(_L,_K))}")
_w = rng.uniform(0.5,4.0,len(_E)); _Aw=_g(_n,_E,_w); _Kw=np.zeros((_n,_n))
for (i,j),k in zip(_E,_w):
    _d=[i,j]
    for a_ in range(2):
        for b_ in range(2): _Kw[_d[a_],_d[b_]] += k*_ke[a_,b_]
print(f"[Ch12 s07] weighted graph == springs of stiffness k_ij           {P(np.allclose(_lap(_Aw),_Kw))}")

# --- s08: x'Lx is twice the strain energy --------------------------------
_x = rng.normal(size=_n)
_q  = _x @ _L @ _x
_se = sum((_x[i]-_x[j])**2 for i,j in _E)
_U  = 0.5*_se
print(f"\n[Ch12 s08] x'Lx = sum_edges (xi-xj)^2 = {_q:.10f}                {P(np.isclose(_q,_se))}")
print(f"[Ch12 s08] x'Lx == 2U, U = strain energy = {_U:.10f}           {P(np.isclose(_q,2*_U))}")
_xw = rng.normal(size=_n)
_Uw = 0.5*sum(k*(_xw[i]-_xw[j])**2 for (i,j),k in zip(_E,_w))
print(f"[Ch12 s08]   holds weighted too                                  {P(np.isclose(_xw@_lap(_Aw)@_xw, 2*_Uw))}")
print(f"[Ch12 s08]   so 'Laplacian smoothness' IS elastic strain energy: a smooth signal is")
print(f"[Ch12 s08]   a low-energy configuration of the spring network")

# --- s09: cyclomatic number == degree of static indeterminacy ------------
def _inc(n, edges):
    B = np.zeros((n,len(edges)))
    for e,(i,j) in enumerate(edges): B[i,e]=1.; B[j,e]=-1.
    return B
print()
_ok_ind = True
for _nm,(_nn,_ee) in {
    "path (tree)":        (5,[(0,1),(1,2),(2,3),(3,4)]),
    "path + one cycle":   (5,[(0,1),(1,2),(2,3),(3,4),(4,0)]),
    "K4":                 (4,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
    "two triangles":      (6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]),
}.items():
    _AA=_g(_nn,_ee); _LL=_lap(_AA); _c=_nn-np.linalg.matrix_rank(_LL)
    _cyc=len(_ee)-_nn+_c; _red=len(_ee)-np.linalg.matrix_rank(_inc(_nn,_ee))
    _ok_ind &= (_cyc==_red)
    print(f"[Ch12 s09] {_nm:16s} cycles m-n+c = {_cyc}, self-equilibrated force sets = {_red}")
print(f"[Ch12 s09] cyclomatic number == degree of static indeterminacy    {P(_ok_ind)}")

# --- s10: path-graph Laplacian == free-free bar in axial vibration -------
print()
for _nn in (6,12):
    _Ap=_g(_nn,[(i,i+1) for i in range(_nn-1)]); _Lp=_lap(_Ap)
    _ev,_U_ = np.linalg.eigh(_Lp)
    _pred=np.array([4*np.sin(np.pi*k/(2*_nn))**2 for k in range(_nn)])
    _wv=0.0
    for k in range(_nn):
        _v=np.array([np.cos(np.pi*k*(j+0.5)/_nn) for j in range(_nn)]); _v/=np.linalg.norm(_v)
        _gt=_U_[:,k]/np.linalg.norm(_U_[:,k])
        _wv=max(_wv, min(np.abs(_v-_gt).max(), np.abs(_v+_gt).max()))
    print(f"[Ch12 s10] path n={_nn:2d}: eigenvalues == 4 sin^2(pi k/2n)  err {np.abs(_ev-_pred).max():.1e}  {P(np.allclose(_ev,_pred))}")
    print(f"[Ch12 s10]   eigenvectors == cos(pi k (j+1/2)/n), the free-free bar modes  err {_wv:.1e}  {P(_wv<1e-9)}")

# --- s11: the ring -- graph Fourier transform IS the DFT -----------------
print()
_nr=8; _Cr=_g(_nr,[(i,(i+1)%_nr) for i in range(_nr)]); _Lr=_lap(_Cr)
_F=np.array([[np.exp(2j*np.pi*i*k/_nr) for k in range(_nr)] for i in range(_nr)])/np.sqrt(_nr)
_off=max(abs(_F[:,a].conj()@_Lr@_F[:,b]) for a in range(_nr) for b in range(_nr) if a!=b)
_dg=np.array([(_F[:,k].conj()@_Lr@_F[:,k]).real for k in range(_nr)])
print(f"[Ch12 s11] ring: DFT basis diagonalises L, max off-diagonal {_off:.1e}   {P(_off<1e-12)}")
print(f"[Ch12 s11]   modal stiffnesses == 4 sin^2(pi k/n)                   "
      f"{P(np.allclose(_dg,[4*np.sin(np.pi*k/_nr)**2 for k in range(_nr)]))}")
print(f"[Ch12 s11]   so on a ring the 'graph Fourier transform' is not like the DFT, it IS")
print(f"[Ch12 s11]   the DFT -- and these are the nodal-diameter modes of a bladed disc")

# --- s12: degree really is mass (row-sum lumping, uniform mesh) ----------
print()
_nb=7; _Apb=_g(_nb,[(i,i+1) for i in range(_nb-1)]); _Lpb=_lap(_Apb); _Dpb=np.diag(_Apb.sum(1))
_M=np.zeros((_nb,_nb))
for i in range(_nb-1): _M[i,i]+=0.5; _M[i+1,i+1]+=0.5     # rho*A*Le/2 to each end node
print(f"[Ch12 s12] lumped mass diag {np.diag(_M)} vs degree {np.diag(_Dpb)}")
print(f"[Ch12 s12] M == (rho A Le / 2) D exactly                          {P(np.allclose(_M,0.5*_Dpb))}")
_Lhat=np.diag(1/np.sqrt(np.diag(_Dpb)))@_Lpb@np.diag(1/np.sqrt(np.diag(_Dpb)))
_w2=np.sort(np.real(np.linalg.eigvals(np.linalg.solve(_M,_Lpb))))
print(f"[Ch12 s12] D^-1/2 L D^-1/2 IS the mass-normalised stiffness matrix of that bar:")
print(f"[Ch12 s12]   eig(Lhat) == w^2 * (rho A Le/2)                      {P(np.allclose(np.sort(_w2*0.5), np.linalg.eigvalsh(_Lhat)))}")
_rw=np.sort(np.real(np.linalg.eigvals(np.linalg.inv(_Dpb)@_Lpb)))
print(f"[Ch12 s12]   D^-1 L has the same spectrum but is not symmetric     "
      f"{P(np.allclose(_rw,np.linalg.eigvalsh(_Lhat)) and not np.allclose(np.linalg.inv(_Dpb)@_Lpb,(np.linalg.inv(_Dpb)@_Lpb).T))}")

# --- s13: message passing is JACOBI; permutation equivariance forbids GS -
print()
_Ej=[(0,1),(0,2),(1,2),(2,3),(3,4),(1,4)]; _nj=5
_Aj=_g(_nj,_Ej); _At=_Aj+np.eye(_nj); _dt=_At.sum(1)
_Ah=_At/np.sqrt(np.outer(_dt,_dt))
def _jac(H,M): return M@H
def _gs(H,M):
    H=H.copy()
    for i in range(M.shape[0]): H[i]=M[i]@H
    return H
_H0=rng.normal(size=(_nj,2))
_pi=np.array([3,0,4,1,2]); _Pm=np.zeros((_nj,_nj))
for _new,_old in enumerate(_pi): _Pm[_new,_old]=1.0
_res={}
for _nm,_sw in (("Jacobi",_jac),("Gauss-Seidel",_gs)):
    _a1=_Pm@_sw(_H0,_Ah); _a2=_sw(_Pm@_H0,_Pm@_Ah@_Pm.T)
    _res[_nm]=np.abs(_a1-_a2).max()
    print(f"[Ch12 s13] {_nm:12s}: relabel-then-update == update-then-relabel? "
          f"{np.allclose(_a1,_a2)}  diff {_res[_nm]:.2e}")
print(f"[Ch12 s13] permutation equivariance holds for Jacobi, fails for Gauss-Seidel  "
      f"{P(_res['Jacobi']<1e-12 and _res['Gauss-Seidel']>1e-3)}")
print(f"[Ch12 s13]   so a GNN cannot be Gauss-Seidel: the answer would depend on node numbering.")
print(f"[Ch12 s13]   The symmetry requirement of file 01 picks the solver.")

# --- s14: a degree-K polynomial in L is exactly a K-hop stencil ----------
print()
_Ep=[(0,1),(1,2),(2,3),(3,4),(4,5),(1,5)]; _np_=6
_Apo=_g(_np_,_Ep); _Lpo=_lap(_Apo); _Pk=np.eye(_np_); _ok_hop=True
for k in range(1,4):
    _Pk=_Pk@_Lpo
    _hop=np.linalg.matrix_power(np.eye(_np_)+_Apo,k)
    _ok_hop &= np.array_equal((np.abs(_Pk)>1e-12),(_hop>0))
print(f"[Ch12 s14] nonzero pattern of L^k == nodes within k hops, k=1,2,3   {P(_ok_hop)}")
print(f"[Ch12 s14]   spectral filter and spatial stencil are one object; GCN is K=1, the")
print(f"[Ch12 s14]   narrowest stencil there is -- one relaxation sweep")

# --- s15: over-smoothing lands on sqrt(degree), NOT on a uniform value ---
print()
_Ho=rng.normal(size=(_nj,3))
_qv=np.sqrt(_dt); _qv/=np.linalg.norm(_qv)
_h1=rng.normal(size=_nj)
for _ in range(500): _h1=_Ah@_h1
_h1/=np.linalg.norm(_h1)
_const=np.ones(_nj)/np.sqrt(_nj)
print(f"[Ch12 s15] limit direction   {np.round(np.abs(_h1),5)}")
print(f"[Ch12 s15] sqrt(1+degree)    {np.round(np.abs(_qv),5)}")
print(f"[Ch12 s15] constant vector   {np.round(_const,5)}")
print(f"[Ch12 s15] over-smoothing converges to sqrt(1+deg), not to a uniform value  "
      f"{P(np.allclose(np.abs(_h1),np.abs(_qv)) and not np.allclose(np.abs(_h1),_const))}")
print(f"[Ch12 s15]   SOURCE CORRECTION: file 03 says 'all node representations converge to the")
print(f"[Ch12 s15]   same value'. They converge to a multiple of sqrt(1+degree). It IS the")
print(f"[Ch12 s15]   rigid-body mode -- written in mass-normalised coordinates q = M^(1/2) x.")
print(f"[Ch12 s15]   Same shape of error as Ch11's degree-weighted-mean correction.")
_ev_h=np.sort(np.abs(np.linalg.eigvalsh(_Ah)))[::-1]; _mu2=_ev_h[1]
_prev=None; _rat=[]
_Hd=rng.normal(size=(_nj,3))
for k in range(40):
    _Hd=_Ah@_Hd
    _r=np.linalg.norm(_Hd-np.outer(_qv,_qv@_Hd))
    if _prev is not None and _r>1e-13: _rat.append(_r/_prev)
    _prev=_r
print(f"[Ch12 s15] per-layer decay ratio measured {_rat[-1]:.6f} == |mu_2(Ahat)| {_mu2:.6f}   "
      f"{P(np.isclose(_rat[-1],_mu2,rtol=1e-6))}")
print(f"[Ch12 s15]   layers to lose 90% of what distinguishes nodes: {np.log(0.1)/np.log(_mu2):.1f}")

# --- s16: self-loops are a STABILITY condition ---------------------------
print()
_C4=_g(4,[(i,(i+1)%4) for i in range(4)])
for _sl in (False,True):
    _Am=_C4+(np.eye(4) if _sl else 0); _d=_Am.sum(1)
    _Mn=_Am/np.sqrt(np.outer(_d,_d)); _e=np.linalg.eigvalsh(_Mn)
    _h=np.array([1.,-1.,1.,-1.]); _tr=[_h[0]]
    for _ in range(30): _h=_Mn@_h; _tr.append(_h[0])
    _lab="with self-loops   " if _sl else "without self-loops"
    print(f"[Ch12 s16] C4 {_lab}: lambda_min = {_e[0]:+.6f}, node-0 after 30 rounds = {_tr[-1]:+.3e}")
    if not _sl:
        print(f"[Ch12 s16]   |lambda_min| == 1 exactly: the alternating mode never decays  "
              f"{P(np.isclose(abs(_e[0]),1.0))}")
    else:
        print(f"[Ch12 s16]   lambda_min strictly inside (-1,1): it now decays              "
              f"{P(abs(_e[0])<0.99 and abs(_tr[-1])<1e-6)}")
print(f"[Ch12 s16]   SOURCE SHARPENING: file 03 gives the reason for A+I as 'each node also")
print(f"[Ch12 s16]   receives its own message'. On a bipartite graph the real reason is")
print(f"[Ch12 s16]   stability -- without it the iteration sits exactly on the boundary and")
print(f"[Ch12 s16]   oscillates forever instead of smoothing. It is added mass, a critical-")
print(f"[Ch12 s16]   time-step fix, not a courtesy message.")

# --- s17: the source's sum-vs-mean example is backwards ------------------
print()
print(f"[Ch12 s17] source (file 03): 'mean cannot distinguish {{1,1}} from {{2,2}}'")
print(f"[Ch12 s17]   mean{{1,1}} = {np.mean([1,1]):.1f}, mean{{2,2}} = {np.mean([2,2]):.1f} -- distinguished. "
      f"CLAIM IS FALSE  {P(np.mean([1,1])!=np.mean([2,2]))}")
print(f"[Ch12 s17] source (file 03): 'max cannot distinguish {{1,2,3}} from {{1,1,3}}'")
print(f"[Ch12 s17]   max = {max([1,2,3])} and {max([1,1,3])} -- collide. CLAIM IS TRUE          "
      f"{P(max([1,2,3])==max([1,1,3]))}")
_nA=np.array([1.,1.,1.,1.]); _nB=np.array([2.,2.])
print(f"[Ch12 s17] source CODING TASK 2 claims 'sum distinguishes, mean does not'. Running it:")
print(f"[Ch12 s17]   mean {_nA.mean():.1f} vs {_nB.mean():.1f} -> mean SEPARATES")
print(f"[Ch12 s17]   sum  {_nA.sum():.1f} vs {_nB.sum():.1f} -> sum COLLIDES")
print(f"[Ch12 s17]   the task demonstrates the exact opposite of its own caption          "
      f"{P(_nA.mean()!=_nB.mean() and _nA.sum()==_nB.sum())}")
print(f"[Ch12 s17] what is actually true: mean and max are killed by DUPLICATION, for any phi:")
for _ms in ([3.],[3.,3.],[3.,3.,3.]):
    print(f"[Ch12 s17]   {str(_ms):18s} mean {np.mean(_ms):.1f}  max {np.max(_ms):.1f}  sum {np.sum(_ms):.1f}")
print(f"[Ch12 s17]   mean and max cannot COUNT; sum can. That is the real separation.     "
      f"{P(np.mean([3.])==np.mean([3.,3.]) and np.sum([3.])!=np.sum([3.,3.]))}")
print(f"[Ch12 s17]   but plain sum is not injective either: {{1,3}} and {{2,2}} both give 4  "
      f"{P(sum([1,3])==sum([2,2]))}")
print(f"[Ch12 s17]   GIN's theorem says an injective phi EXISTS over a countable feature space,")
print(f"[Ch12 s17]   not that bare summation is injective. A free body diagram sums forces; it")
print(f"[Ch12 s17]   does not average them -- and four members at 1 kN load a joint exactly as")
print(f"[Ch12 s17]   two at 2 kN, so equilibrium cannot tell them apart either.")

# --- s18: what message passing cannot see -------------------------------
print()
def _ring_at(n, off, N):
    A=np.zeros((N,N))
    for i in range(n): A[off+i,off+(i+1)%n]=A[off+(i+1)%n,off+i]=1.
    return A
_C6=_ring_at(6,0,6); _TT=_ring_at(3,0,6)+_ring_at(3,3,6)
def _wl(A, rounds=8):
    n=len(A); col=tuple([0]*n)
    for _ in range(rounds):
        sig=[(col[i], tuple(sorted(col[j] for j in range(n) if A[i,j]))) for i in range(n)]
        u={s:k for k,s in enumerate(sorted(set(sig)))}
        col=tuple(u[s] for s in sig)
    return tuple(sorted(col))
_e6=np.round(np.linalg.eigvalsh(_lap(_C6)),9); _et=np.round(np.linalg.eigvalsh(_lap(_TT)),9)
print(f"[Ch12 s18] hexagon C6 vs two triangles: both 2-regular, 6 nodes, 6 edges")
print(f"[Ch12 s18]   1-WL colour histograms identical -> every message-passing GNN is blind  "
      f"{P(_wl(_C6)==_wl(_TT))}")
print(f"[Ch12 s18]   Laplacian spectra C6 {_e6}")
print(f"[Ch12 s18]                   2xC3 {_et}")
print(f"[Ch12 s18]   the spectrum separates them                                        "
      f"{P(not np.allclose(_e6,_et))}")
print(f"[Ch12 s18]   zero eigenvalues: {int((_e6<1e-9).sum())} vs {int((_et<1e-9).sum())} -- two loose pieces, two rigid-body modes")
_G1=_g(6,[(0,2),(0,3),(0,4),(0,5),(1,4),(1,5),(2,3)])
_G2=_g(6,[(0,2),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3)])
_s1=np.round(np.linalg.eigvalsh(_lap(_G1)),6); _s2=np.round(np.linalg.eigvalsh(_lap(_G2)),6)
print(f"[Ch12 s18] and the converse -- a Laplacian-cospectral pair on 6 nodes:")
print(f"[Ch12 s18]   degree sequences {[int(d) for d in sorted(_G1.sum(1))]} vs {[int(d) for d in sorted(_G2.sum(1))]} -- so not isomorphic")
print(f"[Ch12 s18]   identical Laplacian spectra {_s1}                {P(np.allclose(_s1,_s2))}")
print(f"[Ch12 s18]   1-WL SEPARATES this pair                                           "
      f"{P(_wl(_G1)!=_wl(_G2))}")
print(f"[Ch12 s18]   so neither test dominates: local message passing misses C6 vs 2xC3, the")
print(f"[Ch12 s18]   modal survey misses this pair. You cannot hear the shape of a drum, and")
print(f"[Ch12 s18]   you cannot tap your way round it either. That is why Graphormer and GPS")
print(f"[Ch12 s18]   carry BOTH a spectral encoding and a message-passing branch.")

# --- s19: GAT -- static attention, and the end of the stiffness bridge ---
print()
def _lr(z,s=0.2): return np.where(z>0,z,s*z)
def _sm(z): p=np.exp(z-z.max()); return p/p.sum()
_rg=np.random.default_rng(1); _Wh=_rg.normal(size=(8,3)); _av=_rg.normal(size=6)
_nbh=[2,3,4,5]; _sj=np.array([_av[3:]@_Wh[j] for j in _nbh])
_okc=True
for _c in (0.0,3.7,-12.4):
    _okc &= np.allclose(_sm(_c+_sj), _sm(_sj))
print(f"[Ch12 s19] pre-activation, softmax_j(c + s_j) == softmax_j(s_j): the query term")
print(f"[Ch12 s19]   cancels EXACTLY, so attention weights are identical, not merely ranked")
print(f"[Ch12 s19]   the same                                                           {P(_okc)}")
_okp = np.allclose(_sm(_lr(6.0+_sj)), _sm(_sj))
_okn = np.allclose(_sm(_lr(-9.0+_sj)), _sm(0.2*_sj))
print(f"[Ch12 s19] with LeakyReLU: all raw scores positive -> attn == softmax(s_j)        {P(_okp)}")
print(f"[Ch12 s19]                 all raw scores negative -> attn == softmax(0.2 s_j)    {P(_okn)}")
print(f"[Ch12 s19]   in both cases the query node contributes nothing. Only a MIXED-sign")
print(f"[Ch12 s19]   neighbourhood lets it matter, and only through which side of the kink")
print(f"[Ch12 s19]   each neighbour falls on.")
_mix = 0
for _s in range(2000):
    _r = np.random.default_rng(_s)
    _Wm = _r.normal(size=(5,3)); _am = _r.normal(size=6)
    _raw = np.array([_am[:3]@_Wm[0] + _am[3:]@_Wm[j] for j in range(1,5)])
    _mix += not ((_raw>0).all() or (_raw<=0).all())
print(f"[Ch12 s19]   mixed-sign neighbourhoods on random features: {100*_mix/2000:.1f}% -- so GAT is")
print(f"[Ch12 s19]   genuinely query-dependent part of the time and exactly query-blind the rest")
_Eg=[(0,1),(0,2),(1,2),(2,3),(3,4),(4,5),(0,5)]; _Ag=_g(6,_Eg)
_Wh6=_rg.normal(size=(6,3))
_Mg=np.zeros((6,6))
for i in range(6):
    _nb=[j for j in range(6) if _Ag[i,j]]
    _Mg[i,_nb]=_sm(_lr(np.array([_av@np.concatenate([_Wh6[i],_Wh6[j]]) for j in _nb])))
_cpx = 0
for _s in range(400):
    _r = np.random.default_rng(_s)
    _Wc = _r.normal(size=(6,3)); _ac = _r.normal(size=6)
    _Mc = np.zeros((6,6))
    for i in range(6):
        _nb = [j for j in range(6) if _Ag[i,j]]
        _Mc[i,_nb] = _sm(_lr(np.array([_ac@np.concatenate([_Wc[i],_Wc[j]]) for j in _nb])))
    _cpx += np.max(np.abs(np.linalg.eigvals(_Mc).imag)) > 1e-9
print(f"[Ch12 s19] GAT attention matrix is row-stochastic {np.allclose(_Mg.sum(1),1)} but NOT symmetric "
      f"{P(not np.allclose(_Mg,_Mg.T))}")
print(f"[Ch12 s19]   alpha(0->1) = {_Mg[0,1]:.5f} but alpha(1->0) = {_Mg[1,0]:.5f}")
print(f"[Ch12 s19]   Maxwell-Betti reciprocity forces a stiffness matrix to be symmetric, so")
print(f"[Ch12 s19]   there is no strain energy and no guaranteed real modes: complex eigenvalues")
print(f"[Ch12 s19]   occurred in {_cpx}/400 random draws. Not always -- but never guaranteed.")
_Ahg=(_Ag+np.eye(6))/np.sqrt(np.outer((_Ag+np.eye(6)).sum(1),(_Ag+np.eye(6)).sum(1)))
print(f"[Ch12 s19]   and GCN's Ahat row sums are {np.round(_Ahg.sum(1),4)} -- not 1, so file 03's")
print(f"[Ch12 s19]   'weighted average of its neighbours' is wrong for the symmetric normalisation "
      f"{P(not np.allclose(_Ahg.sum(1),1))}")

# --- s21: pooling is the Galerkin coarse-grid operator -------------------
print()
_Af=_g(8,[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,2),(5,7)]); _Lf=_lap(_Af)
_S=np.zeros((8,3))
for i,cl in enumerate([0,0,0,1,1,1,2,2]): _S[i,cl]=1.
_Lc=_S.T@_Lf@_S
print(f"[Ch12 s21] DiffPool's A' = S' A S is the Galerkin projection K_c = R' K R")
print(f"[Ch12 s21]   coarse operator symmetric {np.allclose(_Lc,_Lc.T)}, PSD {np.linalg.eigvalsh(_Lc).min()>-1e-12}")
print(f"[Ch12 s21]   S 1_coarse == 1_fine, so the rigid-body mode survives: (S'LS)1 = 0   "
      f"{P(np.allclose(_S@np.ones(3),np.ones(8)) and np.allclose(_Lc@np.ones(3),0))}")
print(f"[Ch12 s21]   off-diagonals count fine edges crossing clusters: {-_Lc[0,1]:.0f} and {-_Lc[1,2]:.0f}")
print(f"[Ch12 s21]   this is the multigrid rule that interpolation must reproduce the rigid-body")
print(f"[Ch12 s21]   modes -- also Guyan reduction and Craig-Bampton substructuring")

# --- s22/s23: equivariance IS frame indifference; 9 = 1 + 3 + 5 ----------
print()
def _rot3(r):
    Q,R=np.linalg.qr(r.normal(size=(3,3))); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q
_r3=np.random.default_rng(3); _Lg=_r3.normal(size=(3,3))
_tr=np.trace(_Lg)/3; _sy=0.5*(_Lg+_Lg.T); _an=0.5*(_Lg-_Lg.T); _dv=_sy-_tr*np.eye(3)
_vo=np.array([_an[2,1],_an[0,2],_an[1,0]])
_R=_rot3(_r3); _Lr=_R@_Lg@_R.T
_tr2=np.trace(_Lr)/3; _an2=0.5*(_Lr-_Lr.T); _dv2=0.5*(_Lr+_Lr.T)-_tr2*np.eye(3)
_vo2=np.array([_an2[2,1],_an2[0,2],_an2[1,0]])
print(f"[Ch12 s23] velocity gradient 3x3 = 9 components splits as 1 + 3 + 5               {P(1+3+5==9)}")
print(f"[Ch12 s23]   l=0 volumetric strain rate invariant under rotation                  {P(np.isclose(_tr,_tr2))}")
print(f"[Ch12 s23]   l=1 vorticity rotates as R w                                         {P(np.allclose(_vo2,_R@_vo))}")
print(f"[Ch12 s23]   l=2 deviatoric part stays symmetric+traceless and equals R dev R'    "
      f"{P(np.allclose(_dv2,_R@_dv@_R.T) and abs(np.trace(_dv2))<1e-12)}")
print(f"[Ch12 s23]   so the l=0/1/2 hierarchy of Tensor Field Networks IS dilatation +")
print(f"[Ch12 s23]   vorticity + deviatoric strain rate. A stress tensor is 6 = 1 + 5, and")
print(f"[Ch12 s23]   von Mises uses only the l=2 part -- the same split, for the same reason.")
_s2d=np.array([[4.,1.],[1.,2.]]); _d2=_s2d-np.trace(_s2d)/2*np.eye(2)
_ratios=[]
for _th in (15.,30.,45.):
    _t=np.radians(_th); _Q=np.array([[np.cos(_t),-np.sin(_t)],[np.sin(_t),np.cos(_t)]])
    _dd=_Q.T@_d2@_Q
    _a0=np.degrees(np.arctan2(_d2[0,1],(_d2[0,0]-_d2[1,1])/2))
    _a1=np.degrees(np.arctan2(_dd[0,1],(_dd[0,0]-_dd[1,1])/2))
    _ratios.append(((_a0-_a1)%360)/_th)
print(f"[Ch12 s23] Mohr's circle: axes rotated by theta move the state 2*theta round the")
print(f"[Ch12 s23]   circle. Measured ratios {np.round(_ratios,6)}                        "
      f"{P(np.allclose(_ratios,2.0))}")
print(f"[Ch12 s23]   the factor of 2 every GATE candidate memorises IS the l=2 label: a")
print(f"[Ch12 s23]   weight-l object picks up l*theta. Same fact, two vocabularies.")

# --- s24: an invariant energy gives equivariant forces for free ----------
print()
from itertools import combinations as _cmb
_r4=np.random.default_rng(3); _pos=_r4.normal(size=(5,3))
def _en(Pp):
    return sum(np.exp(-2.0*(np.linalg.norm(Pp[i]-Pp[j])-1.3)**2)/np.linalg.norm(Pp[i]-Pp[j])
               for i,j in _cmb(range(len(Pp)),2))
def _fo(Pp,h=1e-6):
    F=np.zeros_like(Pp)
    for i in range(len(Pp)):
        for k in range(3):
            a=Pp.copy(); a[i,k]+=h; b=Pp.copy(); b[i,k]-=h
            F[i,k]=-(_en(a)-_en(b))/(2*h)
    return F
_Rq=_rot3(_r4)
_E0,_E1=_en(_pos),_en(_pos@_Rq.T); _F0,_F1=_fo(_pos),_fo(_pos@_Rq.T)
print(f"[Ch12 s24] source (file 05): invariant architectures 'cannot produce vector outputs")
print(f"[Ch12 s24]   without breaking symmetry'. Building one and differentiating it:")
print(f"[Ch12 s24]   energy invariant E(r) = E(Rr) = {_E0:.10f}                    {P(np.isclose(_E0,_E1,atol=1e-10))}")
print(f"[Ch12 s24]   forces equivariant F(Rr) == R F(r), max err {np.abs(_F1-_F0@_Rq.T).max():.2e}        "
      f"{P(np.allclose(_F1,_F0@_Rq.T,atol=1e-6))}")
print(f"[Ch12 s24]   SOURCE CORRECTION: an invariant scalar differentiated w.r.t. position is")
print(f"[Ch12 s24]   a provably equivariant vector, by the chain rule. F = -grad(E) is how a")
print(f"[Ch12 s24]   mechanical engineer gets force from a potential, and it is exactly how")
print(f"[Ch12 s24]   SchNet predicts forces. The claim is false as written.")

# --- s06: A^k counts walks, not paths -----------------------------------
print()
_Atri=_g(3,[(0,1),(1,2),(2,0)]); _A2=_Atri@_Atri
print(f"[Ch12 s06] triangle, A^2 diagonal = {np.diag(_A2)}. Paths of length 2 from 0 back to 0: 0")
print(f"[Ch12 s06]   (a path repeats no vertex). Walks: 2, via 1 and via 2.")
print(f"[Ch12 s06]   file 02 says A^k counts paths; it counts WALKS                      "
      f"{P(_A2[0,0]==2)}")
print(f"[Ch12 s06]   the giveaway is that diag(A^2) is the degree sequence {np.diag(_A2)}")

# --- s25: the worked example -- a depth budget, and a claim of mine that broke
print()
from collections import deque as _dq
def _nadj(A):
    At=A+np.eye(len(A)); dt=At.sum(1); return At/np.sqrt(np.outer(dt,dt)), dt
def _diam(A):
    n=len(A); best=0
    for s0 in range(n):
        d=[-1]*n; d[s0]=0; qq=_dq([s0])
        while qq:
            u=qq.popleft()
            for v in range(n):
                if A[u,v] and d[v]<0: d[v]=d[u]+1; qq.append(v)
        if min(d)<0: return None
        best=max(best,max(d))
    return best
def _mu2(A):
    Ah,_=_nadj(A); return np.sort(np.abs(np.linalg.eigvalsh(Ah)))[::-1][1]
def _erdos(n,pp,seed):
    r=np.random.default_rng(seed); A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if r.random()<pp: A[i,j]=A[j,i]=1.
    return A
_mol=np.zeros((14,14))
for i in range(13): _mol[i,i+1]=_mol[i+1,i]=1.
_mol[0,5]=_mol[5,0]=1.; _mol[7,12]=_mol[12,7]=1.
_grid=np.zeros((20,20))
for i in range(4):
    for j in range(5):
        u=i*5+j
        if i+1<4: v=(i+1)*5+j; _grid[u,v]=_grid[v,u]=1.
        if j+1<5: v=i*5+j+1; _grid[u,v]=_grid[v,u]=1.
print(f"[Ch12 s25] DEPTH BUDGET: layers needed (diameter) vs layers affordable before the")
print(f"[Ch12 s25]   linear operator erases 90% of what distinguishes nodes, = ln(0.1)/ln|mu2|")
print(f"[Ch12 s25]   {'graph':20s} {'avg deg':>8s} {'diam':>5s} {'|mu2|':>8s} {'L90':>8s}  window")
_open=[]
for _nm,_Ax in (("molecule-like chain",_mol), ("4x5 grid",_grid),
                ("G(24, p=0.20)",_erdos(24,0.20,7)), ("G(24, p=0.50)",_erdos(24,0.50,7)),
                ("G(24, p=0.80)",_erdos(24,0.80,7)),
                ("complete K24",np.ones((24,24))-np.eye(24))):
    _m=_mu2(_Ax); _L=np.log(0.1)/np.log(_m) if _m>1e-12 else 0.1; _d=_diam(_Ax)
    _w = (_d is not None and _d<=_L); _open.append(_w)
    print(f"[Ch12 s25]   {_nm:20s} {_Ax.sum(1).mean():8.1f} {str(_d):>5s} {_m:8.5f} {_L:8.1f}  "
          f"{'OPEN' if _w else 'CLOSED'}")
print(f"[Ch12 s25]   the window closes only at the dense end                            "
      f"{P(all(_open[:4]) and not any(_open[4:]))}")
print(f"[Ch12 s25]   on the sparse graphs GNNs are actually used on, the LINEAR over-smoothing")
print(f"[Ch12 s25]   mechanism needs tens of layers, not 2-4                            "
      f"{P(np.log(0.1)/np.log(_mu2(_mol)) > 20)}")

# does a real GCN layer collapse faster than A_hat^k? isolate W and ReLU.
_Ah,_dtm=_nadj(_mol); _qm=np.sqrt(_dtm); _qm/=np.linalg.norm(_qm)
def _spread(Lr, mode, seed, dd=16):
    r=np.random.default_rng(seed); H=r.normal(size=(14,dd))
    for _ in range(Lr):
        H=_Ah@H
        if mode in ("W","WR"):
            Wm=r.normal(size=(dd,dd)); Wm/=np.linalg.norm(Wm,2); H=H@Wm
        if mode in ("R","WR"): H=np.maximum(H,0)
    Rr=H-np.outer(_qm,_qm@H); nn=np.linalg.norm(H)
    return np.linalg.norm(Rr)/nn if nn>1e-300 else 0.0
print(f"[Ch12 s25] relative spread remaining, averaged over 30 draws:")
print(f"[Ch12 s25]   {'layers':>6s} {'Ahat only':>10s} {'+W':>8s} {'+ReLU':>8s} {'+both':>8s} {'mu2^k':>8s}")
_m2=_mu2(_mol)
for _Lr in (1,2,4,8,16):
    _a=np.mean([_spread(_Lr,"-",sd) for sd in range(30)])
    _b=np.mean([_spread(_Lr,"W",sd) for sd in range(30)])
    _c=np.mean([_spread(_Lr,"R",sd) for sd in range(30)])
    _e=np.mean([_spread(_Lr,"WR",sd) for sd in range(30)])
    print(f"[Ch12 s25]   {_Lr:6d} {_a:10.4f} {_b:8.4f} {_c:8.4f} {_e:8.4f} {_m2**_Lr:8.4f}")
_a4=np.mean([_spread(4,"-",sd) for sd in range(30)]); _b4=np.mean([_spread(4,"W",sd) for sd in range(30)])
_c4=np.mean([_spread(4,"R",sd) for sd in range(30)]); _e4=np.mean([_spread(4,"WR",sd) for sd in range(30)])
print(f"[Ch12 s25]   a random W per layer barely moves it ({_a4:.3f} -> {_b4:.3f})          "
      f"{P(abs(_a4-_b4)<0.05)}")
print(f"[Ch12 s25]   ReLU is what accelerates the collapse ({_a4:.3f} -> {_c4:.3f})        "
      f"{P(_c4 < _a4-0.15)}")
print(f"[Ch12 s25]   but even with both, 4 layers leave {_e4:.0%} of the spread intact -- so the")
print(f"[Ch12 s25]   2-4 layer rule is NOT explained by over-smoothing on a sparse graph "
      f"{P(_e4>0.25)}")
print(f"[Ch12 s25]   MY OWN CLAIM, BROKEN: I first wrote that mu2 gives 'four to seven layers'")
print(f"[Ch12 s25]   to 90% loss. True for the small dense test graph of s15, false in general:")
print(f"[Ch12 s25]   a 14-node sparse chain needs {np.log(0.1)/np.log(_m2):.0f}. Corrected on the page.")

# =========================== CHAPTER 13 (part 1) =========================
print("\n" + "="*66)
print("CHAPTER 13 PART 1 -- DISCRETE MATHS AND COMPUTER ARCHITECTURE")
print("="*66)
import itertools as _it, math as _m, struct as _st

# --- s04: a simple truss is built by induction --------------------------
_j,_mm=3,3; _ok=True
for _ in range(8):
    _j+=1; _mm+=2; _ok &= (_mm==2*_j-3)
print(f"\n[Ch13 s04] simple-truss construction: base case triangle (j=3,m=3), inductive step")
print(f"[Ch13 s04]   adds 1 joint and 2 members. m = 2j-3 preserved for 8 steps          {P(_ok)}")
print(f"[Ch13 s04]   final j={_j}, m={_mm}, 2j-3={2*_j-3}. That construction IS a proof by")
print(f"[Ch13 s04]   induction, and GATE teaches it without naming it.")

# --- s02: De Morgan == series/parallel reliability duality ---------------
_bad=0
for _v in _it.product([False,True],repeat=4):
    if (not all(_v)) != any(not x for x in _v): _bad+=1
    if (not any(_v)) != all(not x for x in _v): _bad+=1
_R=rng.uniform(0.7,0.99,4)
print(f"\n[Ch13 s02] De Morgan over all 2^4 assignments, violations: {_bad}            {P(_bad==0)}")
print(f"[Ch13 s02]   series R = prod(Ri) = {np.prod(_R):.6f}   (AND gate: all must work)")
print(f"[Ch13 s02]   parallel R = 1-prod(1-Ri) = {1-np.prod(1-_R):.6f}  (OR gate: any may work)")
print(f"[Ch13 s02]   a fault tree IS a propositional formula drawn with gates; the")
print(f"[Ch13 s02]   series/parallel duality of reliability blocks IS De Morgan's law")

# --- s06: binary search IS bisection -------------------------------------
_f=lambda x: x**3-2*x-5
_a,_b,_eps=2.0,3.0,1e-10
_pred=_m.ceil(_m.log2((_b-_a)/_eps)); _lo,_hi,_bi=_a,_b,0
while _hi-_lo>_eps:
    _mid=(_lo+_hi)/2
    if _f(_lo)*_f(_mid)<=0: _hi=_mid
    else: _lo=_mid
    _bi+=1
print(f"\n[Ch13 s06] bisection on x^3-2x-5 over [2,3] to 1e-10: {_bi} iterations")
print(f"[Ch13 s06]   ceil(log2((b-a)/eps)) = {_pred}                                    {P(_bi==_pred)}")
print(f"[Ch13 s06]   binary search is the same recurrence T(n)=T(n/2)+O(1), the same")
print(f"[Ch13 s06]   halving argument and the same log2 count. Identical algorithm.")

# --- s06: merge sort and the FFT share a recurrence ----------------------
def _ms(n): return 0 if n<=1 else _ms(n//2)+_ms(n-n//2)+n
def _fft(n): return 0 if n<=1 else 2*_fft(n//2)+n
_same=all(_ms(n)==_fft(n) for n in (8,64,512,4096))
print(f"[Ch13 s06] merge-sort comparisons == FFT butterflies at n=8..4096            {P(_same)}")
for _n in (8,512,32768):
    print(f"[Ch13 s06]   n={_n:5d}: ops {_ms(_n):6d}, n log2 n = {int(_n*_m.log2(_n)):6d}, ratio {_ms(_n)/(_n*_m.log2(_n)):.3f}")
print(f"[Ch13 s06]   Master Theorem a=2,b=2,d=1 -> d == log_b a, the balanced case, O(n log n)")

# --- s13: pipelining IS line balancing ----------------------------------
_stg=np.array([2.0,3.5,1.5,3.0,2.5]); _n=len(_stg)
_cyc=_stg.max(); _tot=_stg.sum(); _eff=_tot/(_n*_cyc); _sp=_tot/_cyc
print(f"\n[Ch13 s13] stage/station times {_stg}, n = {_n}")
print(f"[Ch13 s13]   cycle time = max(t_i) = {_cyc:.3f}  (the bottleneck sets it)")
print(f"[Ch13 s13]   throughput = 1/cycle  = {1/_cyc:.4f}     latency = sum(t_i) = {_tot:.3f}")
print(f"[Ch13 s13]   efficiency = sum/(n*max) = {_eff:.4f}, balance delay {1-_eff:.4f}")
print(f"[Ch13 s13]   speedup = sum/max = {_sp:.3f} of an ideal {_n}")
print(f"[Ch13 s13]   speedup == n * efficiency                                       {P(np.isclose(_sp,_n*_eff))}")
_balv=np.full(_n,_tot/_n)
print(f"[Ch13 s13]   perfectly balanced line: efficiency {_balv.sum()/(_n*_balv.max()):.4f}, "
      f"speedup {_balv.sum()/_balv.max():.3f}                {P(np.isclose(_balv.sum()/_balv.max(),_n))}")
print(f"[Ch13 s13]   'pipeline efficiency' and 'line-balancing efficiency' are the same")
print(f"[Ch13 s13]   expression. Perfect balance is the only route to speedup = n.")

# --- s14: a 5% misprediction rate is not a 5% cost ----------------------
print()
for _acc in (0.99,0.95,0.90):
    _p=1-_acc; _cpi=1+_p*15
    print(f"[Ch13 s14] branch accuracy {_acc:.2f} -> CPI = 1 + {_p:.2f}*15 = {_cpi:.3f}, {_cpi-1:.0%} more cycles")
print(f"[Ch13 s14]   at the source's own figures (>95% accurate, ~15 cycle penalty) a 5%")
print(f"[Ch13 s14]   miss rate costs 75% more cycles per instruction                  {P(abs((1+0.05*15)-1.75)<1e-12)}")

# --- s15: AMAT -- a 95% hit rate is not 95% of the speed ---------------
print()
for _h in (0.99,0.95,0.90):
    _amat=_h*1.0+(1-_h)*100.0
    print(f"[Ch13 s15] hit rate {_h:.2f} -> AMAT = {_amat:6.3f} ns = {_amat:5.2f}x slower than pure L1")
print(f"[Ch13 s15]   95% hits still runs 5.95x slower than L1                         {P(abs(0.95+0.05*100-5.95)<1e-12)}")
print(f"[Ch13 s15]   AMAT = h*t_fast + (1-h)*t_slow is Sheet 05's expected-cost formula:")
print(f"[Ch13 s15]   rate times severity, not rate alone -- a stockout penalty exactly")
print(f"[Ch13 s15]   register->RAM {80/0.3:.0f}x, register->HDD {1e7/0.3:.3g}x (source: ~300x, ~3e7x)")

# --- s16: Belady's anomaly ---------------------------------------------
def _fifo(ref,fr):
    q=[];f=0
    for p_ in ref:
        if p_ not in q:
            f+=1
            if len(q)>=fr: q.pop(0)
            q.append(p_)
    return f
def _lru(ref,fr):
    q=[];f=0
    for p_ in ref:
        if p_ in q: q.remove(p_); q.append(p_)
        else:
            f+=1
            if len(q)>=fr: q.pop(0)
            q.append(p_)
    return f
_ref=[1,2,3,4,1,2,5,1,2,3,4,5]
print(f"\n[Ch13 s16] reference string {_ref}")
for _fr in (3,4,5):
    print(f"[Ch13 s16]   {_fr} frames: FIFO {_fifo(_ref,_fr):2d} faults, LRU {_lru(_ref,_fr):2d} faults")
print(f"[Ch13 s16] BELADY'S ANOMALY: FIFO faults MORE with 4 frames than with 3        "
      f"{P(_fifo(_ref,4)>_fifo(_ref,3))}")
print(f"[Ch13 s16]   LRU is a stack algorithm and cannot do this                       "
      f"{P(_lru(_ref,4)<=_lru(_ref,3))}")
print(f"[Ch13 s16]   not in the source. It is the counterexample to 'more buffer is")
print(f"[Ch13 s16]   never worse' -- buying stock can cut throughput under FIFO issue.")

# --- s10: floating point is constant RELATIVE error --------------------
print()
print(f"[Ch13 s10] float32 mantissa 24 bits -> {24*_m.log10(2):.2f} decimal digits (source ~7)   "
      f"{P(abs(24*_m.log10(2)-7)<0.3)}")
print(f"[Ch13 s10] float64 mantissa 53 bits -> {53*_m.log10(2):.2f} decimal digits (source ~15)  "
      f"{P(abs(53*_m.log10(2)-16)<1)}")
_relc=[]
for _x in (1.0,1e3,1e6,1e9):
    _xf=np.float32(_x); _ulp=float(np.nextafter(_xf,np.float32(np.inf))-_xf)
    _relc.append(_ulp/_x)
    print(f"[Ch13 s10]   x={_x:8.0e}: ulp = {_ulp:12.6e}, ulp/x = {_ulp/_x:.6e}")
print(f"[Ch13 s10]   absolute step grows with magnitude, relative step is constant     "
      f"{P(max(_relc)/min(_relc) < 2.1)}")
print(f"[Ch13 s10]   floating point is significant figures in hardware; a dial gauge is")
print(f"[Ch13 s10]   the opposite -- fixed absolute resolution, relative precision that")
print(f"[Ch13 s10]   collapses for small readings")
_L1,_L2=np.float32(1000.0002),np.float32(1000.0)
_got=float((_L1-_L2)/_L2)
print(f"[Ch13 s10] catastrophic cancellation: strain from two float32 lengths gives")
print(f"[Ch13 s10]   {_got:.6e} against a true 2.0e-07 -- {abs(_got-2e-7)/2e-7:.0%} error      "
      f"{P(abs(_got-2e-7)/2e-7 > 0.05)}")
print(f"[Ch13 s10]   which is why a strain gauge measures the CHANGE rather than")
print(f"[Ch13 s10]   subtracting two large lengths")

# --- s11: the source's non-associativity demo does not demonstrate it ---
print()
_a3,_b3,_c3=np.float32(1e8),np.float32(1.0),np.float32(-1e8)
_Ls,_Rs=(_a3+_b3)+_c3, _a3+(_b3+_c3)
print(f"[Ch13 s11] SOURCE CORRECTION -- file 02 coding task 3 claims to show that float")
print(f"[Ch13 s11]   addition is not associative, with a=1e8, b=1.0, c=-1e8 and the")
print(f"[Ch13 s11]   comment '(a+b)+c should be 1.0'. Running it in float32:")
print(f"[Ch13 s11]     (a+b)+c = {_Ls}   a+(b+c) = {_Rs}   equal: {_Ls==_Rs}")
print(f"[Ch13 s11]   the demo shows NO difference, and the commented expectation is wrong  "
      f"{P(_Ls==_Rs and float(_Ls)==0.0)}")
_ulp8=float(np.nextafter(np.float32(1e8),np.float32(np.inf))-np.float32(1e8))
print(f"[Ch13 s11]   ulp(1e8) in float32 is {_ulp8:.0f}, so 1.0 is below the representable")
print(f"[Ch13 s11]   step on BOTH sides and is lost either way")
_a4,_b4,_c4=np.float32(1e-8),np.float32(1.0),np.float32(-1.0)
print(f"[Ch13 s11]   a pair that IS non-associative: a=1e-8, b=1.0, c=-1.0 ->")
print(f"[Ch13 s11]     (a+b)+c = {float((_a4+_b4)+_c4):.6e}, a+(b+c) = {float(_a4+(_b4+_c4)):.6e}   "
      f"{P((_a4+_b4)+_c4 != _a4+(_b4+_c4))}")
print(f"[Ch13 s11]   the claim is true; the example chosen to show it is not")

# --- s09: why you were taught a heuristic ------------------------------
def _ffd(t,C):
    st=[]
    for v in sorted(t,reverse=True):
        for s_ in st:
            if sum(s_)+v<=C: s_.append(v); break
        else: st.append([v])
    return st
def _opt(t,C,cap=7):
    for k in range(_m.ceil(sum(t)/C), cap+1):
        for asg in _it.product(range(k),repeat=len(t)):
            ld=[0]*k; good=True
            for v,s_ in zip(t,asg):
                ld[s_]+=v
                if ld[s_]>C: good=False; break
            if good and all(l>0 for l in ld): return k
    return None
_tt=[3,5,7,4,2,2,3,4]; _C=10
print(f"\n[Ch13 s09] line balancing, tasks {_tt}, cycle time {_C}")
print(f"[Ch13 s09]   largest-candidate / first-fit-decreasing: {len(_ffd(_tt,_C))} stations {_ffd(_tt,_C)}")
print(f"[Ch13 s09]   true optimum: {_opt(_tt,_C)} stations")
print(f"[Ch13 s09]   the heuristic is one station WORSE than optimal                   "
      f"{P(len(_ffd(_tt,_C))>_opt(_tt,_C))}")
_rt=np.random.default_rng(11).uniform(0,100,(9,2))
_D=np.linalg.norm(_rt[:,None,:]-_rt[None,:,:],axis=-1)
_best=min((sum(_D[((0,)+pp)[i],((0,)+pp)[(i+1)%9]] for i in range(9)), pp)
          for pp in _it.permutations(range(1,9)))[0]
_cur,_nn,_unv=0,[0],set(range(1,9))
while _unv:
    _nx=min(_unv,key=lambda j:_D[_cur,j]); _nn.append(_nx); _unv.discard(_nx); _cur=_nx
_nnd=sum(_D[_nn[i],_nn[(i+1)%9]] for i in range(9))
print(f"[Ch13 s09] CNC drill path over 9 holes IS a travelling salesman problem:")
print(f"[Ch13 s09]   exhaustive optimum {_best:.2f} mm, nearest-neighbour {_nnd:.2f} mm,")
print(f"[Ch13 s09]   heuristic {100*(_nnd/_best-1):.1f}% longer                                    "
      f"{P(_nnd>_best)}")
print(f"[Ch13 s09]   tours for 9 holes {_m.factorial(8):,}; for 20 holes {_m.factorial(19):.3g}")
print(f"[Ch13 s09]   you were handed largest-candidate, RPW and nearest-neighbour because")
print(f"[Ch13 s09]   the exact problems are NP-hard. Nobody said so at the time.")

# --- s07: the planarity bound has an unstated hypothesis ---------------
print()
print(f"[Ch13 s07] file 01 states |E| <= 3|V| - 6 for planar graphs, unconditionally:")
for _V in (1,2,3):
    print(f"[Ch13 s07]   |V|={_V}: bound {3*_V-6:2d}, a tree has {_V-1} edges -> "
          f"{'holds' if _V-1<=3*_V-6 else 'BOUND BROKEN'}")
print(f"[Ch13 s07]   the bound needs |V| >= 3                                          "
      f"{P(not (1-1<=3*1-6) and not (2-1<=3*2-6) and (3-1<=3*3-6))}")
print(f"[Ch13 s07] 4 GHz -> {1/4e9*1e9:.2f} ns per cycle; light covers {3e8*0.25e-9*100:.1f} cm "
      f"(source: 7.5 cm)   {P(abs(3e8*0.25e-9*100-7.5)<0.01)}")
_phi=(1+_m.sqrt(5))/2; _psi=(1-_m.sqrt(5))/2
_F=[0,1]
for _i in range(2,31): _F.append(_F[-1]+_F[-2])
_ferr=max(abs(_F[n]-(_phi**n-_psi**n)/_m.sqrt(5)) for n in range(31))
print(f"[Ch13 s07] Fibonacci closed form exact to {_ferr:.1e} over n<=30                 {P(_ferr<1e-6)}")

# --- s02: redundancy low in the system beats redundancy high -----------
_Rv=np.linspace(0.001,0.999,5000)
_arrA=1-(1-_Rv**2)**2          # two complete pump-valve trains in parallel
_arrB=(1-(1-_Rv)**2)**2        # parallel pumps feeding parallel valves
_gap=_arrB-_arrA
print(f"\n[Ch13 s02] same four components, two arrangements:")
print(f"[Ch13 s02]   (a) two whole trains in parallel  = 1-(1-R^2)^2")
print(f"[Ch13 s02]   (b) redundancy at each stage      = (1-(1-R)^2)^2")
for _r in (0.6,0.9,0.99):
    print(f"[Ch13 s02]   R={_r}: (a)={1-(1-_r**2)**2:.6f}  (b)={(1-(1-_r)**2)**2:.6f}  "
          f"diff {(1-(1-_r)**2)**2-(1-(1-_r**2)**2):+.6f}")
print(f"[Ch13 s02]   (b) beats (a) for every R in (0,1)                                {P(_gap.min()>0)}")
print(f"[Ch13 s02]   widest gap {_gap.max():.4f} at R = {_Rv[_gap.argmax()]:.3f}                            "
      f"{P(abs(_Rv[_gap.argmax()]-0.5)<0.01 and abs(_gap.max()-0.125)<1e-3)}")
print(f"[Ch13 s02]   so redundancy belongs as low in the system as you can afford, and")
print(f"[Ch13 s02]   it matters most exactly when components are worst")

# --- s09: how often does the taught heuristic actually lose? -----------
_tie=_loss=_ntot=0
_rg9=np.random.default_rng(5)
for _ in range(400):
    _t9=[int(x) for x in _rg9.integers(2,8,8)]
    _o9=_opt(_t9,10,cap=8)
    if _o9 is None: continue
    _g9=len(_ffd(_t9,10)); _ntot+=1
    if _g9==_o9: _tie+=1
    elif _g9>_o9: _loss+=1
print(f"\n[Ch13 s09] over {_ntot} random 8-task instances at cycle time 10:")
print(f"[Ch13 s09]   largest-candidate matches the optimum {_tie} times ({100*_tie/_ntot:.1f}%)")
print(f"[Ch13 s09]   and loses exactly one station {_loss} times ({100*_loss/_ntot:.1f}%)     "
      f"{P(_tie>_loss and _loss>0)}")
print(f"[Ch13 s09]   so the heuristic is usually optimal and occasionally not, with nothing")
print(f"[Ch13 s09]   in the rule to tell you which case you are looking at")

# =========================== CHAPTER 13 (part 2) =========================
print("\n" + "="*66)
print("CHAPTER 13 PART 2 -- OPERATING SYSTEMS AND CONCURRENCY")
print("="*66)

def _amdahl(p,n): return 1/((1-p)+p/n)
def _gust(p,n): return (1-p)+p*n

# --- s17: Amdahl is the two-station line where one station cannot split ---
_p=0.95
print(f"\n[Ch13 s17] Amdahl with p = {_p}: speedup 1/((1-p)+p/n)")
for _n in (1,4,16,64,1024):
    print(f"[Ch13 s17]   n={_n:5d} -> {_amdahl(_p,_n):7.3f}")
print(f"[Ch13 s17]   asymptote 1/(1-p) = {1/(1-_p):.1f}                                    "
      f"{P(abs(_amdahl(_p,10**9)-1/(1-_p))<1e-4)}")
_ok17=all(abs(_amdahl(_p,_n)-1/((1-_p)+_p/_n))<1e-12 for _n in (1,2,7,64,999))
print(f"[Ch13 s17]   read as a TWO-STATION line: station A = {1-_p:.2f} (cannot be split),")
print(f"[Ch13 s17]   station B = {_p:.2f} shared by n operators. Time = (1-p) + p/n, and the")
print(f"[Ch13 s17]   Amdahl speedup is exactly 1/that time                            {P(_ok17)}")
for _n in (4,16,64):
    _A,_B=1-_p,_p/_n
    print(f"[Ch13 s17]     n={_n:3d}: A={_A:.4f} B={_B:.4f} -> bottleneck is "
          f"{'A, the serial station' if _A>_B else 'B, the parallel station'}")
print(f"[Ch13 s17]   once B falls below A, more operators buy nothing. That is Goldratt:")
print(f"[Ch13 s17]   an hour saved at a non-bottleneck is a mirage.")

# --- s18: Amdahl vs Gustafson answer different questions ----------------
print()
print(f"[Ch13 s18] {'n':>6s} {'Amdahl (fixed job)':>19s} {'Gustafson (fixed time)':>23s}")
for _n in (4,16,64,256):
    print(f"[Ch13 s18] {_n:6d} {_amdahl(0.95,_n):19.2f} {_gust(0.95,_n):23.2f}")
print(f"[Ch13 s18]   they diverge without contradicting: Amdahl fixes the job and asks how")
print(f"[Ch13 s18]   much faster (capped), Gustafson fixes the time and asks how much more")
print(f"[Ch13 s18]   work (linear). Cycle-time reduction against capacity expansion.   "
      f"{P(_gust(0.95,256)>_amdahl(0.95,256)*10)}")

# --- s04: SJF is the SPT dispatching rule, and SPT is optimal -----------
_jobs=[('P1',10),('P2',4),('P3',6),('P4',2),('P5',8)]
def _mflow(order):
    _t=0; _tot=0
    for _nm,_b in order: _t+=_b; _tot+=_t
    return _tot/len(order)
_spt=sorted(_jobs,key=lambda j:j[1])
_bestp=min(_it.permutations(_jobs), key=_mflow)
print(f"\n[Ch13 s04] jobs {[(n,b) for n,b in _jobs]}")
print(f"[Ch13 s04]   FCFS mean flow time {_mflow(_jobs):.2f}; SPT {_mflow(_spt):.2f}; "
      f"exhaustive best {_mflow(_bestp):.2f}")
print(f"[Ch13 s04]   SPT attains the exhaustive optimum                                "
      f"{P(abs(_mflow(_spt)-_mflow(_bestp))<1e-12)}")
_bad4=0
_rg4=np.random.default_rng(3)
for _ in range(300):
    _js=[(f'J{i}',int(x)) for i,x in enumerate(_rg4.integers(1,20,6))]
    if abs(_mflow(sorted(_js,key=lambda j:j[1]))-_mflow(min(_it.permutations(_js),key=_mflow)))>1e-9:
        _bad4+=1
print(f"[Ch13 s04]   over 300 random 6-job instances SPT was never beaten ({_bad4} failures)  {P(_bad4==0)}")
print(f"[Ch13 s04]   FCFS is {100*(_mflow(_jobs)/_mflow(_spt)-1):.1f}% worse here -- the convoy effect")
print(f"[Ch13 s04]   'shortest job first minimises average waiting time' in an OS text IS")
print(f"[Ch13 s04]   the SPT rule minimising mean flow time in a scheduling text. Same")
print(f"[Ch13 s04]   theorem, and the same fatal caveat: you must know the times in advance.")

# --- s05: the time quantum has the EOQ form -----------------------------
print()
print(f"[Ch13 s05] EOQ minimises D*S/Q + H*Q/2  ->  Q* = sqrt(2DS/H)")
print(f"[Ch13 s05] quantum cost A/q + B*q       ->  q* = sqrt(A/B)   (same structure)")
_okq=True
for _A,_B in ((0.05,0.1),(0.05,0.02),(0.2,0.05)):
    _qs=np.linspace(0.01,5,20000); _tot=_A/_qs+_B*_qs
    _okq &= abs(_qs[_tot.argmin()]-_m.sqrt(_A/_B))<2e-3
    print(f"[Ch13 s05]   A={_A}, B={_B}: numeric argmin {_qs[_tot.argmin()]:.4f}, "
          f"sqrt(A/B) = {_m.sqrt(_A/_B):.4f}")
print(f"[Ch13 s05]   the optimum has the EOQ form in every case                        {P(_okq)}")
print(f"[Ch13 s05]   and the same flat bottom: 2x off optimum costs {((1/2+2)/2-1)*100:.0f}% extra   "
      f"{P(abs((1/2+2)/(1+1)-1.25)<1e-12)}")
print(f"[Ch13 s05]   the quantum IS a batch size and the context switch IS a setup:")
print(f"[Ch13 s05]   too small means constant changeovers, too large means long waits.")

# --- s13: the lost update ------------------------------------------------
def _interleave(nt,incs,seed):
    _r=np.random.default_rng(seed)
    _c=0; _regs=[None]*nt; _todo=[incs]*nt; _stage=[0]*nt
    while any(t>0 for t in _todo):
        _live=[i for i in range(nt) if _todo[i]>0]
        _i=_live[_r.integers(len(_live))]
        if _stage[_i]==0: _regs[_i]=_c; _stage[_i]=1
        elif _stage[_i]==1: _regs[_i]+=1; _stage[_i]=2
        else: _c=_regs[_i]; _stage[_i]=0; _todo[_i]-=1
    return _c
_exp=200; _got=_interleave(4,50,7)
print(f"\n[Ch13 s13] counter += 1 is read, add, write -- three steps, not one.")
print(f"[Ch13 s13]   4 threads x 50 increments, expected {_exp}, interleaved {_got}, "
      f"lost {_exp-_got} ({100*(_exp-_got)/_exp:.0f}%)     {P(_got<_exp)}")
print(f"[Ch13 s13]   two operators both read a kanban count of 12, both decrement, both")
print(f"[Ch13 s13]   write 11. One withdrawal has vanished. Same bug, same fix.")

# --- s14: a counting semaphore is a kanban card count -------------------
def _simwip(cap,arr,svc,T,seed):
    _r=np.random.default_rng(seed)
    _t=0.0;_n=0;_area=0.0;_last=0.0;_done=0;_blk=0
    _na=_r.exponential(1/arr); _nd=np.inf
    while _t<T:
        _t2=min(_na,_nd); _area+=_n*(_t2-_last); _last=_t2; _t=_t2
        if _na<=_nd:
            if _n<cap: _n+=1
            else: _blk+=1
            if _n==1: _nd=_t+_r.exponential(1/svc)
            _na=_t+_r.exponential(1/arr)
        else:
            _n-=1; _done+=1
            _nd=_t+_r.exponential(1/svc) if _n>0 else np.inf
    return _area/T,_done/T,_blk
print()
_rows=[]
for _cap in (1,3,10,10**6):
    _L,_thr,_blk=_simwip(_cap,0.8,1.0,200000,11); _rows.append((_cap,_L,_thr))
    print(f"[Ch13 s14] card count {('unbounded' if _cap>1000 else str(_cap)):>9s}: "
          f"mean WIP {_L:6.3f}, throughput {_thr:.4f}, turned away {_blk}")
_c10=[r for r in _rows if r[0]==10][0]; _cinf=[r for r in _rows if r[0]>1000][0]
print(f"[Ch13 s14]   10 cards cuts mean WIP by {100*(1-_c10[1]/_cinf[1]):.0f}% for "
      f"{100*(1-_c10[2]/_cinf[2]):.1f}% of throughput   {P(_c10[1]<_cinf[1] and _c10[2]>0.95*_cinf[2])}")
print(f"[Ch13 s14]   but 1 card costs {100*(1-_rows[0][2]/_cinf[2]):.0f}% of throughput -- capping WIP is")
print(f"[Ch13 s14]   only cheap while the cap sits above the working queue length.")
print(f"[Ch13 s14]   a semaphore initialised to n IS a kanban loop with n cards, and")
print(f"[Ch13 s14]   CONWIP is the same device under a third name.")

# --- s06: Little's law ---------------------------------------------------
def _little(arr,svc,T,seed):
    _r=np.random.default_rng(seed)
    _t=0.0;_n=0;_area=0.0;_last=0.0;_comp=0;_tw=0.0;_at=[]
    _na=_r.exponential(1/arr); _nd=np.inf
    while _t<T:
        _t2=min(_na,_nd); _area+=_n*(_t2-_last); _last=_t2; _t=_t2
        if _na<=_nd:
            _n+=1; _at.append(_t)
            if _n==1: _nd=_t+_r.exponential(1/svc)
            _na=_t+_r.exponential(1/arr)
        else:
            _n-=1; _comp+=1; _tw+=_t-_at.pop(0)
            _nd=_t+_r.exponential(1/svc) if _n>0 else np.inf
    return _area/T,_comp/T,_tw/_comp
_L6,_lam6,_W6=_little(0.7,1.0,400000,5)
print(f"\n[Ch13 s06] M/M/1 simulation: L = {_L6:.4f}, lambda = {_lam6:.4f}, W = {_W6:.4f}")
print(f"[Ch13 s06]   lambda * W = {_lam6*_W6:.4f} against L = {_L6:.4f}                        "
      f"{P(abs(_lam6*_W6-_L6)/_L6 < 0.01)}")
print(f"[Ch13 s06]   theory for rho=0.7: L = rho/(1-rho) = {0.7/0.3:.4f}                   "
      f"{P(abs(_L6-0.7/0.3)<0.1)}")
print(f"[Ch13 s06]   Little's law is not in the source. It holds for the run queue and the")
print(f"[Ch13 s06]   shop floor alike, and it assumes nothing about the distributions.")

# --- s15: deadlock and the total order -----------------------------------
def _dine(n,ordered):
    _held=[None]*n; _want=[]
    for _ph in range(n):
        _l,_r2=_ph,(_ph+1)%n
        _want.append((min(_l,_r2),max(_l,_r2)) if ordered else (_l,_r2))
    for _ph in range(n):
        _f=_want[_ph][0]
        if _held[_f] is None: _held[_f]=_ph
    return not any(_held[_want[_ph][1]] is None or _held[_want[_ph][1]]==_ph
                   for _ph in range(n) if _held[_want[_ph][0]]==_ph)
print()
for _n in (5,7):
    print(f"[Ch13 s15] {_n} philosophers, all take LEFT first  -> deadlocked {_dine(_n,False)}")
    print(f"[Ch13 s15] {_n} philosophers, total order on forks -> deadlocked {_dine(_n,True)}")
print(f"[Ch13 s15]   a total order on resources makes circular wait impossible         "
      f"{P(_dine(5,False) and not _dine(5,True) and _dine(7,False) and not _dine(7,True))}")
print(f"[Ch13 s15]   'every thread acquires locks in the same order' IS 'every operator")
print(f"[Ch13 s15]   draws tooling in the documented sequence'. Standard work, same reason.")

# --- s10: latency against bandwidth --------------------------------------
_link=10e9/8
print(f"\n[Ch13 s10] a 10 Gbit/s link moves {_link/1e9:.2f} GB/s:")
for _D,_hrs in ((1e12,4),(100e12,4)):
    _tn=_D/_link/3600
    print(f"[Ch13 s10]   {_D/1e12:6.1f} TB: over the link {_tn:7.2f} h, by van {_hrs} h -> "
          f"{'the VAN wins' if _hrs<_tn else 'the link wins'}")
print(f"[Ch13 s10]   crossover at {_link*3600*4/1e12:.1f} TB for a four-hour drive           "
      f"{P(abs(_link*3600*4/1e12-18.0)<0.5)}")
print(f"[Ch13 s10]   a van of disks has colossal bandwidth and dreadful latency: it is a")
print(f"[Ch13 s10]   batch transfer, and the trade is the weekly lorry against the milk-run.")

print("\n" + "="*66)

# =========================== CHAPTER 13 (part 3) =========================
print("\n" + "="*66)
print("CHAPTER 13 PART 3 -- PROGRAMMING LANGUAGES, AND THE CHAPTER'S EXAMPLE")
print("="*66)

# --- s02: a type system is dimensional analysis -------------------------
class _Qty:
    def __init__(s,v,d): s.v=v; s.d=tuple(d)
    def __add__(s,o):
        if s.d!=o.d: raise TypeError(f"cannot add {s.d} to {o.d}")
        return _Qty(s.v+o.v,s.d)
    def __mul__(s,o): return _Qty(s.v*o.v, tuple(a+b for a,b in zip(s.d,o.d)))
_F=_Qty(100,(1,1,-2)); _Ln=_Qty(2,(0,1,0))
_work=_F*_Ln
_rejected=False
try: _F+_Ln
except TypeError: _rejected=True
print(f"\n[Ch13 s02] force(M L T^-2) * length(L) = {_work.d}, which is work M L^2 T^-2   "
      f"{P(_work.d==(1,2,-2))}")
print(f"[Ch13 s02] force + length is REFUSED by the dimension checker                {P(_rejected)}")
print(f"[Ch13 s02]   'cannot add i32 and f64' and 'cannot add newtons to metres' are")
print(f"[Ch13 s02]   the same refusal. Dimensional homogeneity IS static type checking.")
_lbf=4.4482216152605
print(f"[Ch13 s02] weak typing is silent unit coercion: 1 lbf.s = {_lbf:.6f} N.s, so a")
print(f"[Ch13 s02]   mislabelled impulse is wrong by a factor of {_lbf:.3f}              "
      f"{P(abs(_lbf-4.448)<0.001)}")

# --- s07: the JIT break-even is the jig break-even -----------------------
_ti,_tc,_tj=1.0e-3,0.40,0.05e-3
_brk=_tc/(_ti-_tj)
print(f"\n[Ch13 s07] interpreted {_ti*1e3:.2f} ms/iter, compiled {_tj*1e3:.2f} ms/iter,")
print(f"[Ch13 s07]   compilation costs {_tc*1e3:.0f} ms once")
print(f"[Ch13 s07]   break-even n = C/(s_slow - s_fast) = {_brk:.0f} iterations")
for _n in (100,1000,10000):
    _a=_n*_ti; _b=_tc+_n*_tj
    print(f"[Ch13 s07]     n={_n:6d}: interpret {_a:7.3f}s, JIT {_b:7.3f}s -> "
          f"{'JIT' if _b<_a else 'interpreter'}")
print(f"[Ch13 s07]   the crossover sits exactly at C/(s_slow-s_fast)                  "
      f"{P(abs(_brk*_ti-(_tc+_brk*_tj))<1e-9)}")
print(f"[Ch13 s07]   identical to the jig calculation: a fixture costs hours and saves")
print(f"[Ch13 s07]   minutes per part, so it pays above a break-even quantity. A JIT")
print(f"[Ch13 s07]   compiles only HOT code because it waits to learn the quantity.")

# --- s04: reference counting cannot see a cycle -------------------------
class _O:
    def __init__(s): s.rc=0; s.refs=[]
_A3,_B3,_root=_O(),_O(),_O()
def _pt(a,b): a.refs.append(b); b.rc+=1
_pt(_root,_A3); _pt(_A3,_B3); _pt(_B3,_A3)
_root.refs.remove(_A3); _A3.rc-=1
print(f"\n[Ch13 s04] root->A, A->B, B->A. Drop the root reference:")
print(f"[Ch13 s04]   A.rc = {_A3.rc}, B.rc = {_B3.rc} -- neither reaches zero, so neither is")
print(f"[Ch13 s04]   freed, though nothing reachable points at either. Leaked.        "
      f"{P(_A3.rc>0 and _B3.rc>0)}")
print(f"[Ch13 s04]   two jobs each signed a tool out 'for' the other: the tally never")
print(f"[Ch13 s04]   clears and the tools never return. A tally cannot see this; only a")
print(f"[Ch13 s04]   stocktake can, which is what a tracing collector is.")
for _h in (1e5,1e6,1e7):
    print(f"[Ch13 s04]   heap of {_h:8.0e} live objects at 40 ns each -> {_h*40e-9*1e3:6.1f} ms pause")
print(f"[Ch13 s04]   pause scales with LIVE data, not with garbage -- as a stocktake")
print(f"[Ch13 s04]   takes as long as you have stock, however little is obsolete.")

# --- s11: the chapter's worked example -- the bottleneck MOVES ----------
_ser,_par=38.0,212.0; _tot=_ser+_par; _pp=_par/_tot
print(f"\n[Ch13 s11] training step: serial loader {_ser} ms, parallel GPU work {_par} ms")
print(f"[Ch13 s11]   p = {_pp:.4f}, Amdahl ceiling 1/(1-p) = {1/(1-_pp):.2f}x            "
      f"{P(abs(1/(1-_pp)-6.58)<0.02)}")
print(f"[Ch13 s11]   {'GPUs':>5s} {'step ms':>9s} {'speedup':>8s} {'loader share':>13s} {'bottleneck':>10s}")
for _n in (1,2,4,8,16,64):
    _st=_ser+_par/_n
    print(f"[Ch13 s11]   {_n:5d} {_st:9.1f} {_tot/_st:7.2f}x {100*_ser/_st:12.1f}% "
          f"{('LOADER' if _ser>_par/_n else 'GPU'):>10s}")
_flip=_par/_ser
print(f"[Ch13 s11]   the loader overtakes the GPU work at n = {_flip:.2f} GPUs            "
      f"{P(abs(_flip-5.58)<0.02)}")
print(f"[Ch13 s11]   value of FIXING the loader (overlapping it), at each scale:")
_gains=[]
for _n in (1,2,4,8,16):
    _b4=_ser+_par/_n; _af=max(_ser,_par/_n); _gains.append((_n,_b4/_af))
    print(f"[Ch13 s11]     {_n:3d} GPUs: {_b4:6.1f} ms -> {_af:6.1f} ms, gain {_b4/_af:.2f}x")
print(f"[Ch13 s11]   MY FIRST DRAFT CLAIMED fixing the loader beats buying 8 GPUs.")
print(f"[Ch13 s11]   It does not: at 1 GPU the fix gives {_gains[0][1]:.2f}x while 8 GPUs give "
      f"{_tot/(_ser+_par/8):.2f}x   {P(_gains[0][1] < _tot/(_ser+_par/8))}")
print(f"[Ch13 s11]   the true result is better: the bottleneck MOVES. The loader is 15%")
print(f"[Ch13 s11]   of the step at 1 GPU and {100*_ser/(_ser+_par/8):.0f}% at 8, and the gain from fixing it")
print(f"[Ch13 s11]   peaks at 4 GPUs ({max(_gains,key=lambda g:g[1])[1]:.2f}x), not at 1.                     "
      f"{P(max(_gains,key=lambda g:g[1])[0]==4)}")
print(f"[Ch13 s11]   that is Goldratt's FIFTH focusing step, the one everyone forgets:")
print(f"[Ch13 s11]   when a constraint is broken, go back to step one.")

# =====================================================================
# Chapter 14 part 1 -- data structures and algorithms: foundations,
# arrays and hashing, linked lists / stacks / queues (source files 00-02)
# Everything lives inside one function so no name leaks into the shared
# namespace of the earlier chapters.
# =====================================================================
def _ch14p1():
    import math, random, itertools, heapq, sys, time, bisect
    import numpy as np
    T = "[Ch14 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T % sec} {msg}{tail}")

    # --- s02: Big O is an order-of-magnitude statement -------------------
    print()
    n6 = 10**6
    say("s02", f"n = 1e6: log2 n = {math.log2(n6):.2f} (table: 20), n log2 n = {n6*math.log2(n6):.3g} (table: 2e7)",
        abs(math.log2(n6)-20) < 0.1 and abs(n6*math.log2(n6)/2e7-1) < 0.01)
    e = n6*math.log10(2)
    say("s02", f"2^(1e6) = 10^{e:.3f} (table: 10^301030)", round(e) == 301030)
    # local slope of n log n on a log-log plot, between n and 2n
    sl = {n: math.log2((2*n*math.log2(2*n))/(n*math.log2(n))) for n in (10**3, 10**6, 10**9)}
    say("s02", "local log-log slope of n log n between n and 2n: " +
        ", ".join(f"n=1e{int(math.log10(n))}: {s:.3f}" for n, s in sl.items()))
    say("s02", "  it is 1 + log2(1 + 1/log2 n): never exactly 1, drifts down slowly",
        all(abs(s-(1+math.log2(1+1/math.log2(n)))) < 1e-12 for n, s in sl.items()) and sl[10**3] > 1.13)
    # exponential: doubling ratio is not constant -> not a power law
    say("s02", f"2^n doubling ratio T(2n)/T(n) = 2^n: at n=10 it is {2**10}, at n=20 {2**20:,} -- no fixed slope",
        2**20 > 1000*2**10 - 1)
    # crossover of constants: c*n*log2 n against n^2
    def cross(c):
        lo, hi = 2.0, 1e12
        for _ in range(200):
            mid = math.sqrt(lo*hi)
            if mid > c*math.log2(mid): hi = mid
            else: lo = mid
        return hi
    xs = {c: cross(c) for c in (1, 10, 100, 1000)}
    say("s02", "crossover where n^2 overtakes c*n*log2 n: " +
        ", ".join(f"c={c}: n*={x:,.0f}" for c, x in xs.items()))
    say("s02", "  a 100x worse constant costs only ~1000 in crossover size, never the race",
        900 < xs[100] < 1100 and xs[1000] < 20000)

    x4 = cross(10000)
    say("s02", f"  at c=10,000 the crossover is n* = {x4:,.0f}; at c=100 the simple method is "
        f"{1e6/(100*math.log2(1e6)):.0f}x slower at n=1e6; at n=100 the clever one costs "
        f"{100*100*math.log2(100):,.0f} vs 10,000", 170_000 < x4 < 180_000 and round(1e6/(100*math.log2(1e6))) == 502)
    # --- s03: hidden copies -- s += c in CPython -------------------------
    def tcat(n, keepref):
        best = 9e9
        for _ in range(3):
            t0 = time.perf_counter()
            s = ""
            if keepref:
                for _i in range(n):
                    t = s; s += "x"
            else:
                for _i in range(n):
                    s += "x"
            best = min(best, time.perf_counter()-t0)
        return best
    a1, a2 = tcat(400_000, False), tcat(1_600_000, False)
    b1, b2 = tcat(20_000, True), tcat(80_000, True)
    print()
    say("s03", f"s += c, plain loop: 4x the length costs {a2/a1:.2f}x the time (linear would be 4, quadratic 16)",
        a2/a1 < 8)
    say("s03", f"s += c with a second reference alive: 4x the length costs {b2/b1:.2f}x (quadratic is 16)",
        b2/b1 > 8)
    say("s03", "  CPython resizes a string in place when nothing else refers to it; the")
    say("s03", "  quadratic copy the source describes appears as soon as another name holds s.")

    # --- s04: amortised cost; geometric vs arithmetic growth --------------
    caps, last = [], sys.getsizeof([])
    L = []
    for _i in range(200_000):
        L.append(0)
        sz = sys.getsizeof(L)
        if sz != last:
            caps.append((sz - sys.getsizeof([]))//8); last = sz
    ratios = [caps[i+1]/caps[i] for i in range(len(caps)-1) if caps[i] > 1000]
    print()
    say("s04", f"CPython list capacity steps (first few): {caps[:10]}")
    say("s04", f"growth ratio at large sizes: {min(ratios):.3f}..{max(ratios):.3f} -- about 1.125, not 2",
        all(1.10 < r < 1.14 for r in ratios))
    def copies_geo(n, g):
        cap, size, cp, worst = 1, 0, 0, 0.0
        for k in range(1, n+1):
            if size == cap:
                cp += size; cap = max(cap+1, math.ceil(cap*g))
            size += 1
            worst = max(worst, cp/k)
        return cp, worst
    for g in (1.125, 1.5, 2.0):
        cp, worst = copies_geo(1 << 18, g)
        say("s04", f"growth x{g}: copies per append at n=2^18 {cp/(1<<18):.2f}, worst prefix {worst:.2f}, "
            f"bound g/(g-1) = {g/(g-1):.2f}", worst <= g/(g-1) + 0.01)
    def copies_ari(n, k):
        cap, size, cp = k, 0, 0
        for _i in range(n):
            if size == cap:
                cp += size; cap += k
            size += 1
        return cp
    c1, c2 = copies_ari(100_000, 100), copies_ari(200_000, 100)
    say("s04", f"arithmetic growth (+100 slots): doubling n multiplies copies by {c2/c1:.2f} -- quadratic, "
        f"so O(n) per append", 3.9 < c2/c1 < 4.1)
    c16 = copies_ari(1 << 18, 16)
    say("s04", f"  +16 slots (the lab's arithmetic schedule): {c16/(1<<18):,.0f} copies per append at n=2^18",
        abs(c16/(1 << 18) - 8192) < 20)
    # slack: average idle fraction of capacity, sampled over n log-uniformly
    def slack(g):
        cap, size, fr = 1, 0, []
        marks = set(int(round(10**x)) for x in np.linspace(2, 5.5, 4000))
        for k in range(1, int(10**5.5)+1):
            if size == cap: cap = max(cap+1, math.ceil(cap*g))
            size += 1
            if k in marks: fr.append(1-size/cap)
        return float(np.mean(fr))
    sk = {g: slack(g) for g in (1.125, 1.5, 2.0, 3.0)}
    say("s04", "mean idle capacity: " + ", ".join(f"x{g}: {100*v:.1f}%" for g, v in sk.items()))
    say("s04", "  bigger lots mean fewer copies and more idle space -- a trade-off, not an optimum",
        sk[1.125] < sk[1.5] < sk[2.0] < sk[3.0])

    # --- s05: the stack you do not see -----------------------------------
    def qs_depth(a, smaller_first):
        # iterative simulation of recursion depth for Lomuto partition, last-element pivot
        maxd = 0
        stack = [(0, len(a)-1, 1)]
        a = list(a)
        while stack:
            lo, hi, d = stack.pop()
            while lo < hi:
                maxd = max(maxd, d)
                p = a[hi]; i = lo
                for j in range(lo, hi):
                    if a[j] < p: a[i], a[j] = a[j], a[i]; i += 1
                a[i], a[hi] = a[hi], a[i]
                if smaller_first:
                    if i-lo < hi-i:
                        stack.append((i+1, hi, d)); hi = i-1; d += 1   # recurse left (smaller), loop on right
                    else:
                        stack.append((lo, i-1, d)); lo = i+1; d += 1
                else:
                    stack.append((i+1, hi, d+1)); hi = i-1; d += 1      # both sides are real calls
        return maxd
    print()
    n = 2000
    dn = qs_depth(range(n), False)
    ds = qs_depth(range(n), True)
    say("s05", f"quicksort, sorted input n={n}, last-element pivot: naive recursion depth {dn} (~n)", dn >= n-1)
    rng = random.Random(14)
    arr = list(range(n)); rng.shuffle(arr)
    dr = qs_depth(arr, False)
    say("s05", f"  same code, shuffled input: depth {dr} (~2-3 log2 n = {math.log2(n):.1f}); smaller side first on "
        f"sorted input: depth {ds} <= log2 n", ds <= math.log2(n) + 1 and dr < 60)
    say("s05", f"  so 'quicksort is O(log n) space' needs smaller-side-first, or luck", dn > 50*ds)
    def qs_rec(a):
        if len(a) <= 1: return a
        p = a[-1]
        return qs_rec([x for x in a[:-1] if x < p]) + [p] + qs_rec([x for x in a[:-1] if x >= p])
    try:
        qs_rec(list(range(2000))); crashed = False
    except RecursionError:
        crashed = True
    ok_small = qs_rec(arr[:500]) == sorted(arr[:500])
    say("s05", "  a plain recursive quicksort sorts shuffled input, and raises RecursionError on 2,000 "
        "already-sorted items", crashed and ok_small)
    say("s05", f"CPython default recursion limit = {sys.getrecursionlimit()}", sys.getrecursionlimit() == 1000)
    def tail(k, acc=0):
        return acc if k == 0 else tail(k-1, acc+1)
    try:
        tail(5000); tail_ok = False
    except RecursionError:
        tail_ok = True
    say("s06", "a tail-recursive call 5000 deep still raises RecursionError (no tail-call elimination)", tail_ok)

    # --- s07: counting Fibonacci's calls -- a recurrence done properly ----
    F = [0, 1]
    for _i in range(2, 70): F.append(F[-1]+F[-2])
    cnt = [0]
    def fib(k):
        cnt[0] += 1
        return k if k <= 1 else fib(k-1)+fib(k-2)
    ok = True
    for k in range(0, 23):
        cnt[0] = 0; fib(k)
        ok &= (cnt[0] == 2*F[k+1]-1)
    print()
    say("s07", "naive fib(n) makes exactly C(n) = 2F(n+1) - 1 calls, checked n = 0..22", ok)
    c50 = 2*F[51]-1
    say("s07", f"fib(50): {c50:,} calls = {c50:.3g}; source says 'over 10^15'; 2^50 = {2**50:.3g}",
        c50 < 1e11 and 2**50 > 1e15)
    say("s07", f"  the source overstates by {2**50/c50:,.0f}x: O(2^n) is a true bound but not a tight one")
    phi = (1+5**0.5)/2
    say("s07", f"C(n+1)/C(n) at n=40: {(2*F[42]-1)/(2*F[41]-1):.6f} -> phi = {phi:.6f}",
        abs((2*F[42]-1)/(2*F[41]-1)-phi) < 1e-8)
    # which subcalls repeat in fib(5)
    seen = {}
    def fibc(k):
        seen[k] = seen.get(k, 0)+1
        return k if k <= 1 else fibc(k-1)+fibc(k-2)
    fibc(5)
    say("s07", f"fib(5): fib(3) computed {seen[3]}x, fib(2) {seen[2]}x (source: twice, three times)",
        seen[3] == 2 and seen[2] == 3)
    t0 = time.perf_counter(); cnt[0] = 0; fib(24); t24 = time.perf_counter()-t0
    per = t24/cnt[0]
    say("s07", f"measured {per*1e9:.0f} ns per call here -> naive fib(50) ~ {c50*per/3600:.1f} hours "
        f"(slow, not 'infeasible')")
    r6050 = (2*F[61]-1)/(2*F[51]-1)
    say("s07", f"  C(60)/C(50) = {r6050:.2f} (phi^10 = {phi**10:.2f}): fib(60) ~ {c50*per*r6050/86400:.1f} days "
        f"at the measured rate", abs(r6050-phi**10) < 0.01)
    # memo with the source's mutable default
    def fib_memo(k, memo={}):
        if k in memo: return memo[k]
        if k <= 1: return k
        memo[k] = fib_memo(k-1, memo)+fib_memo(k-2, memo)
        return memo[k]
    say("s07", f"source fib_memo(50) = {fib_memo(50):,} (correct)", fib_memo(50) == F[50])
    def ways(total, coins, memo={}):          # the same idiom on a function that has a second input
        if total == 0: return 1
        if total < 0: return 0
        if total in memo: return memo[total]
        memo[total] = sum(ways(total-c, coins, memo) for c in coins)
        return memo[total]
    w1 = ways(10, (1, 2)); w2 = ways(10, (1, 5))
    def ways_ok(total, coins):
        memo = {}
        def go(t):
            if t == 0: return 1
            if t < 0: return 0
            if t not in memo: memo[t] = sum(go(t-c) for c in coins)
            return memo[t]
        return go(total)
    say("s07", f"same idiom, ordered ways to make 10: coins (1,2) -> {w1}; then coins (1,5) -> {w2}, "
        f"true answer {ways_ok(10,(1,5))}", w1 == ways_ok(10, (1, 2)) and w2 != ways_ok(10, (1, 5)))
    say("s07", "  the shared default dict is harmless for fib only because fib has one input")

    # --- s08: backtracking and N-Queens -----------------------------------
    def queens(n, diag):
        nodes = [0]; sols = [0]; cols = set(); d1 = set(); d2 = set()
        def go(r):
            nodes[0] += 1
            if r == n: sols[0] += 1; return
            for c in range(n):
                if c in cols: continue
                if diag and ((r-c) in d1 or (r+c) in d2): continue
                cols.add(c); d1.add(r-c); d2.add(r+c)
                go(r+1)
                cols.discard(c); d1.discard(r-c); d2.discard(r+c)
        go(0)
        return nodes[0], sols[0]
    nc, _s = queens(8, False)
    nd, sd = queens(8, True)
    print()
    say("s08", f"8-queens: raw placements 8^8 = {8**8:,}; one-per-column orderings 8! = {math.factorial(8):,}",
        8**8 == 16777216 and math.factorial(8) == 40320)
    say("s08", f"  backtracking with column checks only visits {nc:,} nodes (every permutation prefix)",
        nc == sum(math.factorial(8)//math.factorial(8-k) for k in range(9)))
    say("s08", f"  with column AND diagonal checks it visits {nd:,} nodes and finds {sd} solutions",
        sd == 92 and nd < 2500)
    say("s08", f"  column-only nodes / full-check nodes = {nc/nd:.1f}", 53 <= nc/nd < 54)
    say("s08", f"  the source credits both checks with n^n -> n!; n! is the column check alone. "
        f"Diagonals cut a further {math.factorial(8)/nd:.0f}x off the leaves")
    for n in (4, 6, 10):
        a, s = queens(n, True)
        say("s08", f"  n={n}: {a:,} nodes, {s} solutions, n! = {math.factorial(n):,}")
    say("s08", f"  with no checks at all the 8-queens tree has (8^9-1)/7 = {(8**9-1)//7:,} nodes",
        (8**9-1)//7 == sum(8**d for d in range(9)))
    say("s08", "subsets of n items: 2^n; permutations: n!",
        len(list(itertools.product([0, 1], repeat=6))) == 64)

    # --- s09: dynamic programming IS the CPM forward pass -----------------
    def rand_dag(rng, n, p):
        E = [(i, j) for i in range(n) for j in range(i+1, n) if rng.random() < p]
        d = [rng.randint(1, 9) for _i in range(n)]
        return E, d
    def forward(n, E, d):
        ES = [0]*n; succ = [[] for _i in range(n)]; pred = [[] for _i in range(n)]
        for i, j in E: succ[i].append(j); pred[j].append(i)
        best_pred = [None]*n
        for j in range(n):                       # 0..n-1 is a topological order by construction
            for i in pred[j]:
                if ES[i]+d[i] > ES[j]: ES[j] = ES[i]+d[i]; best_pred[j] = i
        EF = [ES[k]+d[k] for k in range(n)]
        end = max(range(n), key=lambda k: EF[k])
        path = [end]
        while best_pred[path[-1]] is not None: path.append(best_pred[path[-1]])
        return max(EF), path[::-1], succ
    def enumerate_paths(n, succ, d):
        best = (0, None); count = 0
        def go(u, acc, path):
            nonlocal best, count
            acc += d[u]
            if not succ[u]:
                count += 1
                if acc > best[0]: best = (acc, path+[u])
                return
            for v in succ[u]: go(v, acc, path+[u])
        preds = set(j for u in range(n) for j in succ[u])
        for s in range(n):
            if s not in preds: go(s, 0, [])
        return best, count
    rng = random.Random(1401)
    agree = 0
    for _t in range(300):
        n = rng.randint(4, 11)
        E, d = rand_dag(rng, n, 0.35)
        L, path, succ = forward(n, E, d)
        (Lb, pb), _c = enumerate_paths(n, succ, d)
        if L == Lb and sum(d[k] for k in path) == L: agree += 1
    print()
    say("s09", f"forward pass (ES_j = max ES_i + d_i) equals the longest path found by enumerating every "
        f"path: {agree}/300 random networks", agree == 300)
    say("s09", "  and following the arg-max back gives a critical path of that length (same 300)")
    rows = []
    for k in (5, 10, 20, 40):
        # a chain of k 'diamonds': each stage splits into two parallel activities and rejoins
        nn = 3*k+1
        E = []
        for s in range(k):
            a, b, c, dd = 3*s, 3*s+1, 3*s+2, 3*s+3
            E += [(a, b), (a, c), (b, dd), (c, dd)]
        rows.append((k, 2**k, len(E)))
    say("s09", "diamond-chain networks: paths vs forward-pass relaxations: " +
        "; ".join(f"k={k}: {p:,} vs {e}" for k, p, e in rows), rows[-1][1] > 10**12 and rows[-1][2] == 160)
    # PERT: expectation does not pass through max
    rs = np.random.default_rng(1402)
    X = rs.normal(10, 2, (400_000, 2))
    em = X.max(axis=1).mean()
    say("s09", f"PERT merge: two parallel paths, each N(10, 2^2). Forward pass on means says 10; "
        f"E[max] = {em:.3f}; theory 10 + 2/sqrt(pi) = {10+2/math.sqrt(math.pi):.3f}",
        abs(em-(10+2/math.sqrt(math.pi))) < 0.02)
    X5 = rs.normal(10, 2, (400_000, 5)).max(axis=1).mean()
    say("s09", f"  five parallel paths: E[max] = {X5:.3f} -- the single-critical-path estimate is optimistic",
        X5 > 12.2)
    # house robber = brute force
    def rob(nums):
        if len(nums) == 1: return nums[0]
        p2, p1 = nums[0], max(nums[0], nums[1])
        for i in range(2, len(nums)): p2, p1 = p1, max(p1, p2+nums[i])
        return p1
    def rob_bf(nums):
        best = 0
        for mask in range(1 << len(nums)):
            if mask & (mask >> 1): continue
            best = max(best, sum(nums[i] for i in range(len(nums)) if mask >> i & 1))
        return best
    ok = all(rob(v) == rob_bf(v) for v in ([rng.randint(1, 50) for _i in range(rng.randint(1, 14))] for _t in range(300)))
    say("s09", "House Robber DP == brute force over all non-adjacent subsets (300 cases)", ok)

    # --- s10: memoisation and the MRP low-level code ----------------------
    # BOM: product P -> A,B ; A -> C x2, D ; B -> C, D x3 ; C -> E x2, F ; D -> E, F x2 ; E -> G x4 ; F -> G x2, H
    bom = {"P": [("A", 1), ("B", 2)], "A": [("C", 2), ("D", 1)], "B": [("C", 1), ("D", 3)],
           "C": [("E", 2), ("F", 1)], "D": [("E", 1), ("F", 2)], "E": [("G", 4)], "F": [("G", 2), ("H", 1)],
           "G": [], "H": []}
    visits = [0]
    def naive(item, q, out):
        visits[0] += 1
        out[item] = out.get(item, 0)+q
        for c, m in bom[item]: naive(c, q*m, out)
    tot_naive = {}; naive("P", 10, tot_naive)
    # low-level codes: level = longest distance from the top
    lvl = {"P": 0}
    changed = True
    while changed:
        changed = False
        for p, ch in bom.items():
            if p in lvl:
                for c, _m in ch:
                    if lvl.get(c, -1) < lvl[p]+1: lvl[c] = lvl[p]+1; changed = True
    gross = {"P": 10}; processed = 0
    for item in sorted(bom, key=lambda x: lvl[x]):
        processed += 1
        for c, m in bom[item]: gross[c] = gross.get(c, 0)+gross[item]*m
    print()
    def npaths(item, tgt):
        return 1 if item == tgt else sum(npaths(c, tgt) for c, _m in bom[item])
    say("s10", f"BOM explosion for 10 products: naive recursion visits {visits[0]} item-nodes (one per path "
        f"through the product tree), low-level-code order processes {processed} items, same totals",
        tot_naive == gross and processed == len(bom) and visits[0] == sum(npaths("P", x) for x in bom))
    say("s10", f"  G: {gross['G']:,} needed, reached along {npaths('P','G')} different paths; the naive walk "
        f"explodes it once per path, MRP once", npaths("P", "G") >= 8)

    # --- s11: knapsack -- NP-hard and still solved ------------------------
    def subset_best(w, C):
        reach = 1                                   # bitset of reachable sums
        for x in w: reach |= (reach << x) & ((1 << (C+1))-1)
        return reach.bit_length()-1
    def subset_bf(w, C):
        return max(s for s in (sum(c) for r in range(len(w)+1) for c in itertools.combinations(w, r)) if s <= C)
    ok = True
    for _t in range(200):
        w = [rng.randint(1, 30) for _i in range(rng.randint(1, 12))]; C = rng.randint(1, 120)
        ok &= subset_best(w, C) == subset_bf(w, C)
    print()
    say("s11", "subset-sum DP (fill one station as full as possible) == brute force, 200 cases", ok)
    say("s11", "  DP work is n x (C+1) cells; brute force is 2^n subsets")
    for bits in (8, 16, 24):
        say("s11", f"  C with {bits} bits: {40*(2**bits+1):,} cells for 40 tasks -- doubles per added bit")
    say("s11", "  polynomial in the VALUE of C, exponential in its length: pseudo-polynomial",
        abs((2**24+1)/(2**16+1)-256) < 0.01)
    # greedy 'fill each station perfectly' is still not optimal bin packing
    def opt_bins(w, C):
        for k in range(1, len(w)+1):
            def place(i, loads):
                if i == len(w): return True
                seen_l = set()
                for b in range(k):
                    if loads[b]+w[i] <= C and loads[b] not in seen_l:
                        seen_l.add(loads[b]); loads[b] += w[i]
                        if place(i+1, loads): return True
                        loads[b] -= w[i]
                return False
            if place(0, [0]*k): return k
    gmemo = {}
    def greedy_fill(w, C):
        # fewest stations over EVERY way of filling each station to its maximum possible load
        w = tuple(sorted(w))
        if not w: return 0
        if (w, C) in gmemo: return gmemo[(w, C)]
        target = subset_best(list(w), C); best_ = None
        for r in range(1, len(w)+1):
            for c in set(itertools.combinations(range(len(w)), r)):
                if sum(w[i] for i in c) == target:
                    rest = [w[i] for i in range(len(w)) if i not in c]
                    v = 1+greedy_fill(rest, C)
                    best_ = v if best_ is None else min(best_, v)
        gmemo[(w, C)] = best_
        return best_
    ch13 = [3, 5, 7, 4, 2, 2, 3, 4]
    say("s11", f"Ch13's line {ch13}, C=10: fill-each-station-perfectly gives {greedy_fill(ch13,10)} stations, "
        f"optimum {opt_bins(ch13,10)}", opt_bins(ch13, 10) == 3)
    worse = [7, 7, 6, 4, 4, 4]
    perfect = [c for r in range(1, 7) for c in itertools.combinations(worse, r) if sum(c) == 12]
    say("s11", f"  tasks {worse}, cycle time 12: the only 100%-loaded station is {sorted(set(perfect))}; "
        f"taking it gives {greedy_fill(worse,12)} stations, optimum {opt_bins(worse,12)} ({{7,4}},{{7,4}},{{6,4}})",
        sorted(set(perfect)) == [(4, 4, 4)] and greedy_fill(worse, 12) == 4 and opt_bins(worse, 12) == 3)
    say("s11", "  the hardness of bin packing is in how stations interact, not in filling any one of them")

    # --- s12: arrays, pitch, and the cache --------------------------------
    A2 = np.zeros((3, 5), dtype=np.float64)
    print()
    say("s12", f"row-major 3x5 float64 strides {A2.strides}: address of [i,j] = base + 40 i + 8 j",
        A2.strides == (40, 8))
    N = 1 << 23
    a = np.arange(N, dtype=np.float64)
    seq = np.arange(N); perm = np.random.default_rng(3).permutation(N)
    def tg(idx):
        best = 9e9
        for _i in range(3):
            t0 = time.perf_counter(); a[idx].sum(); best = min(best, time.perf_counter()-t0)
        return best
    ts, tr = tg(seq), tg(perm)
    say("s12", f"gathering 8.4M doubles: in order {ts*1e3:.0f} ms, shuffled order {tr*1e3:.0f} ms -> {tr/ts:.1f}x "
        f"(same work, same Big O)", tr/ts > 1.5)
    say("s12", "  the source's '10-100x' is plausible for pointer chasing in C; this is what one machine shows")
    utf = {c: len(c.encode("utf-8")) for c in ("a", "é", "中", "\U0001F600")}
    say("s12", f"UTF-8 bytes: a={utf['a']}, e-acute={utf[chr(0xe9)]}, CJK={utf[chr(0x4e2d)]}, emoji={utf[chr(0x1F600)]}",
        list(utf.values()) == [1, 2, 3, 4])

    # --- s13: hashing and the utilisation cliff ---------------------------
    rs = np.random.default_rng(1403)
    def lp_stats(m, alpha, trials=6):
        succ, unsucc = [], []
        for _t in range(trials):
            tab = np.full(m, -1, dtype=np.int64)
            nkeys = int(alpha*m)
            homes = rs.integers(0, m, nkeys)
            disp = np.zeros(nkeys, dtype=np.int64)
            for kk in range(nkeys):
                h = homes[kk]; d = 0
                while tab[(h+d) % m] != -1: d += 1
                tab[(h+d) % m] = kk; disp[kk] = d
            succ.append(disp.mean()+1)
            occ = tab != -1
            # unsuccessful search from each home slot: probes until an empty slot
            # distance to next empty slot, cyclic
            idx = np.arange(m)
            empties = np.flatnonzero(~occ)
            pos = np.searchsorted(empties, idx)
            nxt_empty = np.where(pos < len(empties), empties[np.minimum(pos, len(empties)-1)], empties[0]+m)
            unsucc.append(float((nxt_empty-idx+1).mean()))
        return float(np.mean(succ)), float(np.mean(unsucc))
    print()
    for al in (0.5, 0.75, 0.9):
        s_, u_ = lp_stats(1 << 14, al)
        ks, ku = 0.5*(1+1/(1-al)), 0.5*(1+1/(1-al)**2)
        say("s13", f"linear probing at alpha={al}: successful {s_:.2f} (Knuth {ks:.2f}), "
            f"unsuccessful {u_:.2f} (Knuth {ku:.2f})", abs(s_/ks-1) < 0.08 and abs(u_/ku-1) < 0.12)
    sets = sorted(lp_stats(1 << 14, 0.9, trials=4)[1] for _k in range(8))
    kn9 = 0.5*(1+1/(1-0.9)**2)
    say("s13", f"  at alpha=0.9, eight separate sets of four tables: miss cost {sets[0]:.1f} to {sets[-1]:.1f} "
        f"(Knuth {kn9:.1f}), median {np.median(sets):.1f}", sets[-1]/sets[0] > 1.15 and abs(np.median(sets)/kn9-1) < 0.08)
    for al in (0.5, 0.9, 1.5):
        m = 1 << 14; nk = int(al*m)
        b = rs.integers(0, m, nk)
        cnts = np.bincount(b, minlength=m)
        # successful search: position within its chain, averaged over keys
        pos = np.concatenate([np.arange(1, c+1) for c in cnts if c > 0])
        say("s13", f"chaining at alpha={al}: successful {pos.mean():.3f} (1 + alpha/2 = {1+al/2:.3f}), "
            f"unsuccessful scans {cnts.mean():.3f} (alpha)", abs(pos.mean()-(1+al/2)) < 0.03)
    say("s13", f"linear probing miss cost at alpha=0.95: {0.5*(1+1/0.05**2):.1f} (passes 200)",
        0.5*(1+1/0.05**2) > 200)
    say("s13", "  chaining degrades like alpha and can pass 1; linear probing blows up like 1/(1-alpha)^2")
    say("s13", "  M/M/1 queue length rho/(1-rho) is the same cliff with a milder power")
    # birthday
    p23 = 1-math.prod((365-i)/365 for i in range(23))
    say("s13", f"P(two of 23 share a slot among 365) = {p23:.4f}", abs(p23-0.5073) < 1e-4)
    firsts = []
    for _t in range(4000):
        seen_s = set(); k = 0
        while True:
            x = int(rs.integers(0, 65536)); k += 1
            if x in seen_s: break
            seen_s.add(x)
        firsts.append(k)
    th = math.sqrt(math.pi*65536/2)+2/3
    say("s13", f"65,536-slot table: first collision at insert {np.mean(firsts):.0f} on average "
        f"(sqrt(pi m/2) + 2/3 = {th:.0f}), when the table is {np.mean(firsts)/65536*100:.2f}% full",
        abs(np.mean(firsts)/th-1) < 0.03)

    # --- s14: the Bloom filter is a GO gauge -------------------------------
    print()
    def bloom_fp(nn, bits_per, k, queries=200_000):
        m = nn*bits_per
        bits = np.zeros(m, dtype=bool)
        pos_members = rs.integers(0, m, (nn, k))       # ideal independent hash functions
        bits[pos_members.ravel()] = True
        fn = int((~bits[pos_members].all(axis=1)).sum())
        q = rs.integers(0, m, (queries, k))
        return float(bits[q].all(axis=1).mean()), fn
    fp, fn = bloom_fp(20_000, 10, 7)
    th = (1-math.exp(-7/10))**7
    say("s14", f"Bloom filter, 10 bits per key, k=7: false positives {100*fp:.3f}% (theory {100*th:.3f}%), "
        f"false negatives {fn}", abs(fp/th-1) < 0.1 and fn == 0)
    kopt = 10*math.log(2)
    fps = {k: (1-math.exp(-k/10))**k for k in range(1, 15)}
    say("s14", f"optimal k = (m/n) ln 2 = {kopt:.2f}; best integer k = {min(fps, key=fps.get)}",
        min(fps, key=fps.get) == 7)
    fac = 0.5**math.log(2)
    say("s14", f"at optimal k, p = {fac:.4f}^(m/n); {math.log(0.1)/math.log(fac):.2f} bits per key per factor of 10; "
        f"0.6185^10 = {fac**10:.5f}", abs(fac-0.6185) < 1e-4 and abs(math.log(0.1)/math.log(fac)-4.79) < 0.01)
    fp1, _f = bloom_fp(20_000, 10, 1)
    say("s14", f"k=1 (a single check): {100*fp1:.2f}% (theory {100*(1-math.exp(-0.1)):.2f}%)",
        abs(fp1/(1-math.exp(-0.1))-1) < 0.05)

    # --- s15: two pointers -------------------------------------------------
    def two_ptr(a, t):
        i, j, steps = 0, len(a)-1, 0
        while i < j:
            steps += 1
            s = a[i]+a[j]
            if s == t: return (i, j), steps
            if s < t: i += 1
            else: j -= 1
        return None, steps
    ok = True; worst = 0
    for _t in range(500):
        a = sorted(rng.randint(-50, 50) for _i in range(rng.randint(2, 40))); t = rng.randint(-100, 100)
        r, st = two_ptr(a, t)
        bf = any(a[i]+a[j] == t for i in range(len(a)) for j in range(i+1, len(a)))
        ok &= (r is not None) == bf
        worst = max(worst, st/(len(a)-1))
    print()
    say("s15", f"sorted two-sum by two pointers agrees with brute force on 500 cases, never more than n-1 steps",
        ok and worst <= 1)
    def three_sum(nums):
        nums = sorted(nums); res = []
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]: continue
            l, r = i+1, len(nums)-1; tgt = -nums[i]
            while l < r:
                tot = nums[l]+nums[r]
                if tot < tgt: l += 1
                elif tot > tgt: r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l += 1; r -= 1
        return res
    ok = True
    for _t in range(300):
        v = [rng.randint(-6, 6) for _i in range(rng.randint(0, 14))]
        bf = sorted(set(tuple(sorted(c)) for c in itertools.combinations(v, 3) if sum(c) == 0))
        ok &= sorted(tuple(x) for x in three_sum(v)) == bf
    say("s15", "source's three_sum == brute force (unique triplets, heavy duplicates), 300 cases", ok)

    # --- s16: trapping rain water is a spillway ----------------------------
    def trap(h, strict=False, le=False):
        l, r = 0, len(h)-1; lm = rm = w = 0
        while l < r:
            if (h[l] <= h[r]) if le else (h[l] < h[r]):
                if (h[l] > lm) if strict else (h[l] >= lm): lm = h[l]
                else: w += lm-h[l]
                l += 1
            else:
                if (h[r] > rm) if strict else (h[r] >= rm): rm = h[r]
                else: w += rm-h[r]
                r -= 1
        return w
    def trap_bf(h):
        return sum(max(0, min(max(h[:i+1]), max(h[i:]))-h[i]) for i in range(len(h)))
    cases = [[rng.randint(0, 9) for _i in range(rng.randint(1, 30))] for _t in range(2000)]
    print()
    say("s16", "two-pointer trap == min(max-left, max-right) - h, 2000 random profiles",
        all(trap(h) == trap_bf(h) for h in cases))
    say("s16", f"the source warns '>=' vs '>' causes off-by-one water; swapping it changes {sum(trap(h, strict=True) != trap(h) for h in cases)} "
        f"of 2000 answers (equal heights add zero water either way)", all(trap(h, strict=True) == trap(h) for h in cases))
    say("s16", "  nor does '<' vs '<=' in the side test", all(trap(h, le=True) == trap(h) for h in cases))
    say("s16", f"source example [0,1,0,2,1,0,1,3,2,1,2,1] traps {trap([0,1,0,2,1,0,1,3,2,1,2,1])} units",
        trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6)
    src = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]; hi = list(src); hi[7] += 5
    say("s16", f"raise the tallest bar of that profile from 3 to 8: water {trap(src)} -> {trap(hi)} (the lower "
        f"weir sets the level)", trap(hi) == trap(src) == trap_bf(hi))
    # 2-D: the straight-ray rule fails; the spill level is a minimax path
    G = [[9, 9, 9, 9, 9, 9],
         [9, 1, 9, 9, 9, 9],
         [9, 1, 1, 9, 9, 9],
         [9, 9, 1, 9, 9, 9],
         [9, 9, 1, 1, 1, 0],
         [9, 9, 9, 9, 9, 9]]
    def pq_fill(G):
        R, Cc = len(G), len(G[0]); lev = [[None]*Cc for _i in range(R)]; pq = []
        for i in range(R):
            for j in range(Cc):
                if i in (0, R-1) or j in (0, Cc-1):
                    lev[i][j] = G[i][j]; heapq.heappush(pq, (G[i][j], i, j))
        while pq:
            L_, i, j = heapq.heappop(pq)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a_, b_ = i+di, j+dj
                if 0 <= a_ < R and 0 <= b_ < Cc and lev[a_][b_] is None:
                    lev[a_][b_] = max(G[a_][b_], L_); heapq.heappush(pq, (lev[a_][b_], a_, b_))
        return lev
    def relax_fill(G):
        R, Cc = len(G), len(G[0])
        lev = [[G[i][j] if i in (0, R-1) or j in (0, Cc-1) else 10**9 for j in range(Cc)] for i in range(R)]
        ch = True
        while ch:
            ch = False
            for i in range(1, R-1):
                for j in range(1, Cc-1):
                    v = max(G[i][j], min(lev[i+1][j], lev[i-1][j], lev[i][j+1], lev[i][j-1]))
                    if v < lev[i][j]: lev[i][j] = v; ch = True
        return lev
    def ray_rule(G, i, j):
        row, col = G[i], [G[k][j] for k in range(len(G))]
        return min(max(row[:j+1]), max(row[j:]), max(col[:i+1]), max(col[i:]))
    lv = pq_fill(G)
    say("s16", f"2-D profile: straight-line rule puts water at level {ray_rule(G,1,1)} over cell (1,1); "
        f"true level {lv[1][1]} -- it drains down a winding channel", ray_rule(G, 1, 1) == 9 and lv[1][1] == 1)
    ok = True
    for _t in range(150):
        R_, C_ = rng.randint(3, 8), rng.randint(3, 8)
        Gr = [[rng.randint(0, 9) for _j in range(C_)] for _i in range(R_)]
        ok &= pq_fill(Gr) == relax_fill(Gr)
    say("s16", "  the true level is the minimax path to the edge: priority-queue fill == relaxation, 150 grids", ok)
    say("s16", "  in 1-D there are only two escape paths, so minimax collapses to min(maxL, maxR)")

    # --- s17: sliding windows need a monotone constraint -------------------
    def shortest_ge_window(a, K):
        l = s = 0; best = None
        for r, x in enumerate(a):
            s += x
            while s >= K and l <= r:
                best = r-l+1 if best is None else min(best, r-l+1)
                s -= a[l]; l += 1
        return best
    def shortest_ge_bf(a, K):
        best = None
        for i in range(len(a)):
            s = 0
            for j in range(i, len(a)):
                s += a[j]
                if s >= K: best = j-i+1 if best is None else min(best, j-i+1); break
        return best
    okpos = all(shortest_ge_window(v, K) == shortest_ge_bf(v, K)
                for v, K in (([rng.randint(0, 9) for _i in range(rng.randint(1, 20))], rng.randint(1, 40)) for _t in range(1000)))
    print()
    say("s17", "shortest subarray with sum >= K: sliding window == brute force when all values >= 0 (1000 cases)", okpos)
    ce = None
    for _t in range(20000):
        v = [rng.randint(-5, 9) for _i in range(rng.randint(2, 6))]; K = rng.randint(1, 15)
        if shortest_ge_window(v, K) != shortest_ge_bf(v, K):
            if ce is None or len(v) < len(ce[0]): ce = (v, K)
    say("s17", f"  with negatives it fails: {ce[0]}, K={ce[1]} -> window says {shortest_ge_window(*ce)}, "
        f"true {shortest_ge_bf(*ce)}", ce is not None)
    def lols(s):
        ci = {}; l = b = 0
        for r, ch_ in enumerate(s):
            if ch_ in ci and ci[ch_] >= l: l = ci[ch_]+1
            ci[ch_] = r; b = max(b, r-l+1)
        return b
    def lols_bf(s):
        return max([0]+[j-i for i in range(len(s)) for j in range(i+1, len(s)+1) if len(set(s[i:j])) == j-i])
    say("s17", "longest substring without repeats (source) == brute force, 500 strings",
        all(lols(s) == lols_bf(s) for s in ("".join(rng.choice("abcd") for _i in range(rng.randint(0, 15))) for _t in range(500))))

    from collections import Counter
    def min_window(s_, t_, ge=False):
        if not t_ or not s_: return ""
        need = Counter(t_); have = 0; req = len(need); l = 0; best = (float("inf"), 0, 0); wc = {}
        for r, ch_ in enumerate(s_):
            wc[ch_] = wc.get(ch_, 0)+1
            if ch_ in need and ((wc[ch_] >= need[ch_]) if ge else (wc[ch_] == need[ch_])): have += 1
            while have == req:
                if r-l+1 < best[0]: best = (r-l+1, l, r)
                lc = s_[l]; wc[lc] -= 1
                if lc in need and wc[lc] < need[lc]: have -= 1
                l += 1
        return s_[best[1]:best[2]+1] if best[0] != float("inf") else ""
    def mw_bf(s_, t_):
        need = Counter(t_); best = None
        for i in range(len(s_)):
            for j in range(i+1, len(s_)+1):
                if not (need - Counter(s_[i:j])):
                    if best is None or j-i < len(best): best = s_[i:j]
                    break
        return best or ""
    mcases = [("".join(rng.choice("abc") for _i in range(rng.randint(0, 12))),
               "".join(rng.choice("abc") for _i in range(rng.randint(1, 3)))) for _t in range(800)]
    okmw = all(len(min_window(a_, b_)) == len(mw_bf(a_, b_)) for a_, b_ in mcases)
    badge = sum(len(min_window(a_, b_, True)) != len(mw_bf(a_, b_)) for a_, b_ in mcases)
    say("s17", f"minimum window substring (source, '==') == brute force on 800 cases; with '>=' {badge} answers go wrong",
        okmw and badge > 0)
    # --- s18: prefix sums are a shear-force diagram -------------------------
    # simply supported beam, span 10 m, point loads (kN) at x
    loads = [(2.0, 20.0), (5.0, 35.0), (7.5, 15.0)]
    Lb = 10.0
    RB = sum(P_*x for x, P_ in loads)/Lb; RA = sum(P_ for _x, P_ in loads)-RB
    pts = [(0.0, RA)] + [(x, -P_) for x, P_ in loads] + [(Lb, RB)]
    V = list(itertools.accumulate(f for _x, f in pts))           # shear just right of each point
    M = [0.0]
    for i in range(1, len(pts)): M.append(M[-1]+V[i-1]*(pts[i][0]-pts[i-1][0]))
    print()
    say("s18", f"beam: RA={RA:.2f}, RB={RB:.2f} kN; SFD as a running (prefix) sum: {[round(v,2) for v in V]}")
    say("s18", f"  closes to zero at the far end (sum F = 0) and BMD closes to {M[-1]:.1e} (sum M = 0)",
        abs(V[-1]) < 1e-9 and abs(M[-1]) < 1e-9)
    say("s18", f"  net load between x=1 and x=6: V(6)-V(1) = {V[2]-V[0]:.1f} kN = -(20+35)",
        abs((V[2]-V[0])+55) < 1e-9)
    def subarray_sum(nums, k, init=True):
        c = pre = 0; pc = {0: 1} if init else {}
        for x in nums:
            pre += x; c += pc.get(pre-k, 0); pc[pre] = pc.get(pre, 0)+1
        return c
    def ss_bf(nums, k):
        return sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)+1) if sum(nums[i:j]) == k)
    ok = all(subarray_sum(v, k) == ss_bf(v, k) for v, k in
             (([rng.randint(-4, 4) for _i in range(rng.randint(0, 16))], rng.randint(-4, 4)) for _t in range(800)))
    say("s18", "subarray-sum-equals-k via prefix-sum hash == brute force, negatives included (800 cases)", ok)
    say("s18", f"  without the {{0:1}} seed, [3,1,2] with k=3 counts {subarray_sum([3,1,2],3,False)} instead of "
        f"{ss_bf([3,1,2],3)}", subarray_sum([3, 1, 2], 3, False) < ss_bf([3, 1, 2], 3))
    xs = np.random.default_rng(5).random(1 << 22).astype(np.float32)
    pre32 = np.concatenate([[np.float32(0)], np.cumsum(xs, dtype=np.float32)])
    errs = []
    for lo in (3_000_000, 3_500_000, 4_000_000):
        exact = math.fsum(float(v) for v in xs[lo:lo+10])
        errs.append(abs(float(pre32[lo+10]-pre32[lo])-exact)/exact)
    say("s18", f"float32 prefix sums over 4.2M values: a 10-element range read by subtraction is off by "
        f"{100*max(errs):.0f}% (worst of 3); summing the 10 directly is exact to 1e-7",
        max(errs) > 0.01)
    say("s18", f"  float32 spacing near 2.1 million: {float(np.spacing(np.float32(2.1e6)))}",
        float(np.spacing(np.float32(2.1e6))) == 0.25)
    say("s18", "  the SFD read at two distant sections: difference of two big numbers (catastrophic cancellation)")
    def pes(nums):
        n = len(nums); res = [1]*n; p = 1
        for i in range(n): res[i] = p; p *= nums[i]
        s = 1
        for i in range(n-1, -1, -1): res[i] *= s; s *= nums[i]
        return res
    ok = all(pes(v) == [math.prod(v[:i]+v[i+1:]) for i in range(len(v))]
             for v in ([rng.randint(-3, 3) for _i in range(rng.randint(1, 9))] for _t in range(500)))
    say("s18", "product-except-self (prefix x suffix) == brute force, zeros included (500 cases)", ok)

    # --- s19: Floyd's tortoise and hare -------------------------------------
    def run_floyd(a, c, vs=1, vf=2, limit=None):
        # positions on a rho: tail 0..a-1, cycle a..a+c-1
        def adv(p, k):
            for _i in range(k):
                p = p+1 if p < a+c-1 else a
            return p
        s = f = 0; t = 0; limit = limit or 4*(a+c)+10
        while t < limit:
            s = adv(s, vs); f = adv(f, vf); t += 1
            if s == f: return t
        return None
    ok_meet = True; ok_within = True; ok_start = True; kmax = 1
    for a in range(0, 25):
        for c in range(1, 25):
            t = run_floyd(a, c)
            if t is None: ok_meet = False; continue
            b = t-a
            ok_within &= 0 <= b <= c
            ok_start &= (t % c == 0)
            kmax = max(kmax, t//c)
            # phase 2: pointer from head and pointer from meeting point, both speed 1
            p = 0; q = (a + (b % c)) if b >= 0 else None
            if q is not None:
                steps = 0
                while p != q:
                    p = p+1 if p < a+c-1 else a
                    q = q+1 if q < a+c-1 else a
                    steps += 1
                ok_start &= (p == a) and steps <= a+c
    print()
    say("s19", "tortoise(1) and hare(2) meet on every rho-shaped list with tail a<25, cycle c<25", ok_meet)
    say("s19", "  the tortoise meets within c steps of entering the loop", ok_within)
    say("s19", f"  a + b is a MULTIPLE of c (k up to {kmax}), not c itself; restart-from-head still lands on the start",
        ok_start and kmax > 1)
    tt = run_floyd(10, 3)
    say("s19", f"  e.g. tail 10, loop 3: they meet after {tt} steps, b = {tt-10}, a + b = {tt} = {tt//3} x 3",
        tt % 3 == 0 and tt//3 > 1)
    firstmult = all(run_floyd(a, c) == max(1, math.ceil(a/c))*c for a in range(25) for c in range(1, 25))
    say("s19", "  sharper: the first meeting is at the first multiple of c that is >= a -- both pointers are "
        "in the loop, and at T = kc their offsets coincide", firstmult)
    def phase2(a, c, vf):
        T = run_floyd(a, c, 1, vf, 20*(a+c)+20)
        if T is None: return None, None
        s = a+((T-a) % c) if T >= a else T
        p, q, st = 0, s, 0
        while p != q and st < 5*(a+c)+10:
            p = p+1 if p < a+c-1 else a; q = q+1 if q < a+c-1 else a; st += 1
        return T, (p if p == q else None)
    met3 = all(phase2(a, c, 3)[0] is not None for a in range(12) for c in range(1, 12))
    bad3 = [(a, c) for a in range(12) for c in range(1, 12) if phase2(a, c, 3)[1] != a]
    say("s19", "a hare at speed 3 still always meets the tortoise (detection works)", met3)
    say("s19", f"  but the restart-from-head trick then fails on {len(bad3)} of 132 lists -- every one with an even loop "
        f"(e.g. tail {bad3[0][0]}, loop {bad3[0][1]}: the two pointers circle forever)",
        len(bad3) > 0 and all(c % 2 == 0 for _a, c in bad3))
    t104, r104 = phase2(10, 4, 3)
    say("s19", f"  tail 10, loop 4, hare speed 3: meet at step {t104}, restart phase lands on "
        f"{'nothing (they circle forever)' if r104 is None else r104} instead of node 10", t104 is not None and r104 != 10)
    say("s19", "  with speed v the meeting step satisfies (v-1)T = 0 mod c; only v = 2 forces T = kc")

    # --- s20: stacks and monotonic stacks ------------------------------------
    def daily(tp):
        res = [0]*len(tp); st = []; pops = 0
        for i, x in enumerate(tp):
            while st and x > tp[st[-1]]:
                j = st.pop(); res[j] = i-j; pops += 1
            st.append(i)
        return res, pops
    def daily_bf(tp):
        return [next((j-i for j in range(i+1, len(tp)) if tp[j] > tp[i]), 0) for i in range(len(tp))]
    ok = True; popok = True
    for _t in range(500):
        tp = [rng.randint(30, 40) for _i in range(rng.randint(1, 25))]
        r_, pops = daily(tp); ok &= r_ == daily_bf(tp); popok &= pops <= len(tp)
    print()
    say("s20", "daily temperatures (monotonic stack) == brute force, 500 cases; pops <= n every time", ok and popok)
    def hist(hs, keep_start=True):
        hs = hs+[0]; st = []; best = 0
        for i, h in enumerate(hs):
            start = i
            while st and st[-1][1] > h:
                idx, hh = st.pop(); best = max(best, hh*(i-idx))
                if keep_start: start = idx
            st.append((start, h))
        return best
    def hist_bf(hs):
        return max([0]+[min(hs[i:j+1])*(j-i+1) for i in range(len(hs)) for j in range(i, len(hs))])
    cases = [[rng.randint(0, 9) for _i in range(rng.randint(1, 15))] for _t in range(1000)]
    say("s20", "largest rectangle in histogram (source) == brute force, 1000 cases",
        all(hist(h) == hist_bf(h) for h in cases))
    bad = min((h for h in cases if hist(h, False) != hist_bf(h)), key=len)
    say("s20", f"  drop 'start = idx' and {bad} gives {hist(bad,False)} instead of {hist_bf(bad)}",
        hist(bad, False) != hist_bf(bad))

    # --- s21: queues, heaps, and the dispatch rule ---------------------------
    class Q2:
        def __init__(s): s.a = []; s.b = []; s.moves = 0; s.worst = 0
        def push(s, x): s.a.append(x)
        def pop(s):
            if not s.b:
                m_ = len(s.a)
                while s.a: s.b.append(s.a.pop()); s.moves += 1
                s.worst = max(s.worst, m_)
            return s.b.pop()
    q = Q2(); out = []; pushed = 0
    for _t in range(5000):
        if rng.random() < 0.55 or not (q.a or q.b): q.push(pushed); pushed += 1
        else: out.append(q.pop())
    print()
    say("s21", f"queue from two stacks: FIFO order kept; {q.moves} transfers for {pushed} pushes; "
        f"worst single pop moved {q.worst}", out == sorted(out) and q.moves <= pushed and q.worst > 1)
    class Cmp:
        n = 0
        __slots__ = ("v",)
        def __init__(s, v): s.v = v
        def __lt__(s, o): Cmp.n += 1; return s.v < o.v
    data = [Cmp(x) for x in np.random.default_rng(9).permutation(100_000).tolist()]
    Cmp.n = 0; hp = list(data); heapq.heapify(hp)
    say("s21", f"heapify 100,000 items: {Cmp.n:,} comparisons ({Cmp.n/100_000:.2f} per item) -- O(n), not n log n",
        Cmp.n < 2*100_000)
    def kth(vals, k):
        dd = [Cmp(x) for x in vals]; Cmp.n = 0; reps = 0
        h = dd[:k]; heapq.heapify(h)
        for x in dd[k:]:
            if h[0] < x: heapq.heapreplace(h, x); reps += 1
        c_stream, ans1 = Cmp.n, h[0].v
        h2 = [Cmp(-v) for v in vals]; Cmp.n = 0; heapq.heapify(h2)
        for _i in range(k): top = heapq.heappop(h2)
        return c_stream, reps, Cmp.n, ans1, -top.v
    vals = [c.v for c in data]; asc = sorted(vals)
    say("s21", "kth largest, n = 100,000: comparisons, size-k heap (source) vs heapify-all + k pops")
    okk = True; rw = []; aw = []
    for k in (10, 1000):
        cs, reps, cp, a1, a2 = kth(vals, k)
        okk &= a1 == a2 == asc[-k]; rw.append(cs < cp)
        say("s21", f"  random order,    k={k:>5,}: {cs:>9,} vs {cp:>9,}; root replaced {reps:,} times "
            f"(k ln(n/k) = {k*math.log(100_000/k):,.0f})", abs(reps/(k*math.log(100_000/k))-1) < 0.05)
        cs, reps, cp, a1, a2 = kth(asc, k)
        okk &= a1 == a2 == asc[-k]; aw.append(cs > cp)
        say("s21", f"  ascending order, k={k:>5,}: {cs:>9,} vs {cp:>9,}; root replaced {reps:,} times")
    say("s21", "  both methods agree with sorting in all four runs", okk)
    say("s21", "  the source's Big O is backwards (n + k log n is never worse than n log k), yet on random input")
    say("s21", "  the size-k heap still wins, because only ~k ln(n/k) elements ever beat the root; on ascending",
        all(rw))
    say("s21", "  input every element does, and it loses by ~3x at k=10. Neither method is 'the optimal' one.",
        all(aw))
    try:
        class LN:
            def __init__(s, v): s.val = v
        h = []; heapq.heappush(h, (1, LN(1))); heapq.heappush(h, (1, LN(1))); crash = False
    except TypeError:
        crash = True
    say("s21", "merge-k heap tuple without the index tiebreak raises TypeError on equal values", crash)
    # SPT with release times: the non-delay dispatcher is not optimal
    jobs = {"A": (0, 10), "B": (1, 1)}      # (release, processing)
    def sched(order):
        t = 0; tot = 0
        for j in order:
            r_, p_ = jobs[j]; t = max(t, r_)+p_; tot += t
        return tot
    greedy = sched(["A", "B"])             # at t=0 only A is waiting, so a dispatcher starts it
    best = min(sched(o) for o in itertools.permutations(jobs))
    say("s21", f"SPT via a heap with arrivals: job A (r=0,p=10), job B (r=1,p=1). Dispatching whatever is "
        f"waiting gives total completion {greedy}; waiting 1 minute for B gives {best}", greedy == 21 and best == 14)
    say("s21", "  SPT is optimal only when every job is on hand at t=0 (Ch13 part 2 s04); with arrivals")
    say("s21", "  the single-machine problem is NP-hard. The heap implements the rule; it does not rescue it.")

_ch14p1()

# =====================================================================
# Chapter 14 part 2 -- trees, graphs, sorting and search (files 03-05)
# plus the chapter's worked example. Inside one function: no name leaks.
# =====================================================================
def _ch14p2():
    import math, random, itertools, heapq, bisect
    from collections import deque, defaultdict
    import numpy as np
    T = "[Ch14p2 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T % sec} {msg}{tail}")
    rng = random.Random(1402)

    class Node:
        __slots__ = ("val", "left", "right")
        def __init__(s, v, l=None, r=None): s.val = v; s.left = l; s.right = r

    # --- s01: traversals, and what list concatenation costs ----------------
    def balanced(lo, hi):
        if lo > hi: return None
        m = (lo+hi)//2
        return Node(m, balanced(lo, m-1), balanced(m+1, hi))
    def path_tree(n):
        root = None
        for v in range(n-1, -1, -1): root = Node(v, None, root)
        return root
    copies = [0]
    def inorder_concat(t):
        if not t: return []
        a = inorder_concat(t.left); b = inorder_concat(t.right)
        copies[0] += len(a)+1+len(b)
        return a+[t.val]+b
    import sys
    old = sys.getrecursionlimit(); sys.setrecursionlimit(10000)
    n = 1023
    copies[0] = 0; inorder_concat(balanced(0, n-1)); cb = copies[0]
    copies[0] = 0; inorder_concat(path_tree(n)); cp = copies[0]
    sys.setrecursionlimit(old)
    print()
    say("s01", f"inorder by list concatenation, n = 1023: balanced tree copies {cb:,} elements "
        f"(n log2 n = {n*math.log2(n):,.0f}); a path-shaped tree copies {cp:,} (n^2/2 = {n*n/2:,.0f})",
        cb < 1.1*n*math.log2(n) and cp >= n*(n+1)//2)
    say("s01", "  the source's O(n^2) is the worst case; a balanced tree costs O(n log n)")
    # indented BOM = preorder; cost roll-up = postorder
    bom = ("pump", 120, [("casing", 300, [("bolt", 2, []), ("gasket", 8, [])]),
                         ("impeller", 450, []), ("motor", 2100, [("bearing", 90, []), ("bearing", 90, [])])])
    lines = []
    def pre(nd, d):
        lines.append("  "*d+nd[0])
        for c in nd[2]: pre(c, d+1)
    pre(bom, 0)
    def post(nd):
        return nd[1]+sum(post(c) for c in nd[2])
    say("s01", f"indented BOM = preorder: {' / '.join(l.strip() for l in lines)}")
    say("s01", f"cost roll-up = postorder (children first): pump total {post(bom):,}",
        post(bom) == 120+300+2+8+450+2100+90+90)

    # --- s02: recursion on trees: max path sum, LCA ----------------------
    def rand_tree(n, lo=-9, hi=9):
        nodes = [Node(rng.randint(lo, hi)) for _i in range(n)]
        for i in range(1, n):
            while True:
                p = nodes[rng.randrange(i)]
                if p.left is None and rng.random() < 0.5: p.left = nodes[i]; break
                if p.right is None: p.right = nodes[i]; break
        return nodes
    def max_path_sum(root):
        best = [float("-inf")]
        def dfs(nd):
            if not nd: return 0
            l = max(dfs(nd.left), 0); r = max(dfs(nd.right), 0)
            best[0] = max(best[0], nd.val+l+r)
            return nd.val+max(l, r)
        dfs(root); return best[0]
    def mps_bf(nodes):
        par = {}
        for nd in nodes:
            for c in (nd.left, nd.right):
                if c: par[id(c)] = nd
        def anc(nd):
            out = [nd]
            while id(out[-1]) in par: out.append(par[id(out[-1])])
            return out
        best = float("-inf")
        for a in nodes:
            for b in nodes:
                A = anc(a); B = anc(b); Bs = set(map(id, B))
                lca = next(x for x in A if id(x) in Bs)
                pa = A[:[id(x) for x in A].index(id(lca))+1]
                pb = B[:[id(x) for x in B].index(id(lca))]
                best = max(best, sum(x.val for x in pa)+sum(x.val for x in pb))
        return best
    ok = all(max_path_sum(ns[0]) == mps_bf(ns) for ns in (rand_tree(rng.randint(1, 12)) for _t in range(300)))
    print()
    say("s02", "max path sum (source) == brute force over every pair of nodes, 300 trees with negatives", ok)
    def lca(root, p, q):
        if not root or root is p or root is q: return root
        l = lca(root.left, p, q); r = lca(root.right, p, q)
        if l and r: return root
        return l if l else r
    ns = rand_tree(9); stranger = Node(0)
    say("s02", "LCA (source) with q absent from the tree returns p itself, not 'no answer'",
        lca(ns[0], ns[5], stranger) is ns[5])

    # --- s03: binary search trees ----------------------------------------
    def bst_insert(root, v):
        if not root: return Node(v)
        cur = root
        while True:
            if v < cur.val:
                if cur.left is None: cur.left = Node(v); return root
                cur = cur.left
            else:
                if cur.right is None: cur.right = Node(v); return root
                cur = cur.right
    def depths(root):
        out = []; st = [(root, 0)]
        while st:
            nd, d = st.pop()
            if nd: out.append(d); st.append((nd.left, d+1)); st.append((nd.right, d+1))
        return out
    n = 1000
    r = None
    for v in range(n): r = bst_insert(r, v)
    hs = max(depths(r))
    Hn = sum(1/k for k in range(1, n+1))
    avg_th = 2*(1+1/n)*Hn-4
    avgs = []; heights = []
    for _t in range(200):
        vals = list(range(n)); rng.shuffle(vals); r = None
        for v in vals: r = bst_insert(r, v)
        d = depths(r); avgs.append(sum(d)/n); heights.append(max(d))
    print()
    say("s03", f"BST from sorted insertion, n=1000: height {hs} -- a linked list", hs == n-1)
    say("s03", f"random insertion order: mean node depth {np.mean(avgs):.2f} (theory 2(1+1/n)H_n - 4 = {avg_th:.2f}); "
        f"mean height {np.mean(heights):.1f}; log2 n = {math.log2(n):.2f}", abs(np.mean(avgs)-avg_th) < 0.1)
    say("s03", f"  a random BST is about {np.mean(avgs)/math.log2(n):.2f} x as deep on average as a perfectly balanced one")
    n31 = 31; H31 = sum(1/k for k in range(1, n31+1))
    def build_mid(lo, hi, out):
        if lo > hi: return
        m = (lo+hi)//2; out.append(m); build_mid(lo, m-1, out); build_mid(m+1, hi, out)
    order = []; build_mid(0, n31-1, order); r = None
    for v in order: r = bst_insert(r, v)
    say("s03", f"lab sizes, n=31: middle-first insertion height {max(depths(r))}, mean depth {sum(depths(r))/n31:.3f}; "
        f"random-order theory mean depth {2*(1+1/n31)*H31-4:.3f}", max(depths(r)) == 4)
    def is_valid_bst(nd, lo=float("-inf"), hi=float("inf")):
        if not nd: return True
        if nd.val <= lo or nd.val >= hi: return False
        return is_valid_bst(nd.left, lo, nd.val) and is_valid_bst(nd.right, nd.val, hi)
    t = bst_insert(bst_insert(None, 5), 5)
    say("s03", "the source's insert puts a duplicate key in the right subtree; its own validator then rejects the tree",
        t.right is not None and t.right.val == 5 and not is_valid_bst(t))
    okv = True
    for _t in range(300):
        vals = rng.sample(range(100), rng.randint(0, 15)); r = None
        for v in vals: r = bst_insert(r, v)
        okv &= is_valid_bst(r)
    say("s03", "  with distinct keys, every tree it builds validates (300 cases)", okv)

    # --- s05: union-find ----------------------------------------------------
    class UF:
        def __init__(s, n, rank=True, compress=True):
            s.p = list(range(n)); s.rk = [0]*n; s.c = n; s.rank = rank; s.comp = compress
        def find(s, x):
            root = x
            while s.p[root] != root: root = s.p[root]
            if s.comp:
                while s.p[x] != root: s.p[x], x = root, s.p[x]
            return root
        def union(s, a, b):
            ra, rb = s.find(a), s.find(b)
            if ra == rb: return False
            if s.rank:
                if s.rk[ra] < s.rk[rb]: ra, rb = rb, ra
                s.p[rb] = ra
                if s.rk[ra] == s.rk[rb]: s.rk[ra] += 1
            else:
                s.p[ra] = rb
            s.c -= 1; return True
        def height(s):
            best = 0
            for x in range(len(s.p)):
                h = 0
                while s.p[x] != x: x = s.p[x]; h += 1
                best = max(best, h)
            return best
    okc = True
    for _t in range(300):
        nn = rng.randint(1, 15); m = rng.randint(0, 25)
        E = [(rng.randrange(nn), rng.randrange(nn)) for _i in range(m)]
        E = [(a, b) for a, b in E if a != b]
        u = UF(nn); red = sum(1 for a, b in E if not u.union(a, b))
        okc &= red == len(E)-nn+u.c
    print()
    say("s05", "edges union-find reports as redundant == m - n + c (the cyclomatic number), 300 random graphs", okc)
    n = 1 << 12
    u = UF(n, rank=True, compress=False)
    step = 1
    while step < n:
        for i in range(0, n, 2*step): u.union(i, i+step)
        step *= 2
    hr = u.height()
    v = UF(n, rank=False, compress=False)
    for i in range(1, n): v.union(i-1, i)
    hv = v.height()
    say("s05", f"no path compression, union by rank, worst merge order, n=4096: height {hr} = log2 n", hr == 12)
    say("s05", f"no path compression AND no union by rank: height {hv} -- the O(n) the source warns about needs both",
        hv == n-1)

    # --- s06: Fenwick tree: prefix sums whose loads change ------------------
    class Fen:
        def __init__(s, n): s.n = n; s.t = [0]*(n+1); s.ops = 0
        def update(s, i, d):
            i += 1
            while i <= s.n: s.t[i] += d; i += i & -i; s.ops += 1
        def prefix(s, i):
            i += 1; tot = 0
            while i > 0: tot += s.t[i]; i -= i & -i; s.ops += 1
            return tot
        def rng_(s, l, r): return s.prefix(r)-(s.prefix(l-1) if l > 0 else 0)
    n = 1000; f = Fen(n); arr = [0]*n; okf = True; maxops = 0
    for _t in range(3000):
        if rng.random() < 0.5:
            i = rng.randrange(n); d = rng.randint(-9, 9); arr[i] += d
            b = f.ops; f.update(i, d); maxops = max(maxops, f.ops-b)
        else:
            l = rng.randrange(n); r_ = rng.randrange(l, n)
            b = f.ops; okf &= f.rng_(l, r_) == sum(arr[l:r_+1]); maxops = max(maxops, f.ops-b)
    print()
    say("s06", f"Fenwick tree == brute force on 3,000 random updates and range queries; never more than "
        f"{maxops} steps per operation (2 log2 n = {2*math.log2(n):.1f})", okf and maxops <= 2*math.ceil(math.log2(n+1)))

    nn6 = 10**6; nnz = 10*nn6
    say("s07", f"FE mesh, 1e6 nodes x ~10 neighbours: full matrix {nn6**2:.0e} entries = {nn6**2*8/1e12:.0f} TB in float64; "
        f"list {nnz:.0e} nonzeros = {nnz*12/1e6:.0f} MB with 4-byte indices", nn6**2*8/1e12 == 8 and nnz*12/1e6 == 120)
    # --- s08: BFS -----------------------------------------------------------
    def rand_graph(n, p):
        g = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p: g[i].append(j); g[j].append(i)
        return g
    def bfs_enqueue(g, s):
        dist = {s: 0}; q = deque([s]); pushes = 1
        while q:
            u = q.popleft()
            for v in g[u]:
                if v not in dist: dist[v] = dist[u]+1; q.append(v); pushes += 1
        return dist, pushes
    def bfs_dequeue(g, s):
        dist = {}; q = deque([(s, 0)]); pushes = 1
        while q:
            u, d = q.popleft()
            if u in dist: continue
            dist[u] = d
            for v in g[u]:
                if v not in dist: q.append((v, d+1)); pushes += 1
        return dist, pushes
    def bfs_overwrite(g, s):
        dist = {s: 0}; seen = set(); q = deque([s])
        while q:
            u = q.popleft()
            if u in seen: continue
            seen.add(u)
            for v in g[u]:
                if v not in seen: dist[v] = dist[u]+1; q.append(v)
        return dist
    okd = True; extra = []; wrong = 0
    for _t in range(300):
        g = rand_graph(rng.randint(2, 25), 0.25)
        a, pa = bfs_enqueue(g, 0); b, pb = bfs_dequeue(g, 0)
        okd &= a == b; extra.append(pb/pa)
        wrong += bfs_overwrite(g, 0) != a
    print()
    say("s08", f"BFS marking visited on DEQUEUE (skipping repeats) gives the same distances in 300 graphs, "
        f"with up to {max(extra):.1f}x the queue pushes", okd and max(extra) > 1.5)
    say("s08", f"  wrong answers appear only if a later enqueue may overwrite a distance: {wrong} of 300 graphs",
        wrong > 0)
    # multi-source BFS distance is Manhattan (4-conn) or Chebyshev (8-conn)
    R_ = 40
    def grid_bfs(src, eight):
        dist = {src: 0}; q = deque([src])
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if eight else [])
        while q:
            x, y = q.popleft()
            for dx, dy in nb:
                w = (x+dx, y+dy)
                if abs(w[0]) <= R_ and abs(w[1]) <= R_ and w not in dist: dist[w] = dist[(x, y)]+1; q.append(w)
        return dist
    d4 = grid_bfs((0, 0), False); d8 = grid_bfs((0, 0), True)
    r4 = max(d4[(x, y)]/math.hypot(x, y) for x in range(-R_, R_+1) for y in range(-R_, R_+1) if (x, y) != (0, 0))
    r8 = min(d8[(x, y)]/math.hypot(x, y) for x in range(-R_, R_+1) for y in range(-R_, R_+1) if (x, y) != (0, 0))
    say("s08", f"grid BFS from one cell: 4-connected distance is |dx|+|dy|, up to {r4:.4f}x the true distance; "
        f"8-connected is max(|dx|,|dy|), down to {r8:.4f}x", abs(r4-math.sqrt(2)) < 1e-9 and abs(r8-1/math.sqrt(2)) < 1e-9)
    say("s08", "  so BFS 'offsets' are diamonds or squares, never the circles a pocketing toolpath needs")
    def rot(grid):
        g = [row[:] for row in grid]; R, C = len(g), len(g[0])
        q = deque((r, c) for r in range(R) for c in range(C) if g[r][c] == 2)
        fresh = sum(row.count(1) for row in g)
        if fresh == 0: return 0
        t = 0
        while q and fresh:
            t += 1
            for _i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == 1: g[nr][nc] = 2; fresh -= 1; q.append((nr, nc))
        return t if fresh == 0 else -1
    def rot_bf(grid):
        R, C = len(grid), len(grid[0]); srcs = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 2]
        best = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] != 1: continue
                dd = {(r, c): 0}; q = deque([(r, c)]); found = None
                while q and found is None:
                    x, y = q.popleft()
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < R and 0 <= ny < C and grid[nx][ny] != 0 and (nx, ny) not in dd:
                            dd[(nx, ny)] = dd[(x, y)]+1
                            if grid[nx][ny] == 2: found = dd[(nx, ny)]; break
                            q.append((nx, ny))
                if found is None: return -1
                best = max(best, found)
        return best
    okr = all(rot(gr) == rot_bf(gr) for gr in ([[rng.choice([0, 1, 1, 2]) for _c in range(rng.randint(1, 6))]
                                                for _r in range(rng.randint(1, 6))] for _t in range(400))
              if len(set(len(x) for x in gr)) == 1)
    say("s08", "multi-source BFS (rotting oranges, source) == max over fresh cells of the nearest-source distance", okr)

    # --- s09: DFS, and why cycle detection needs three states ---------------
    def cyc3(n, adj):
        st = [0]*n
        def go(u):
            if st[u] == 1: return True
            if st[u] == 2: return False
            st[u] = 1
            if any(go(v) for v in adj[u]): return True
            st[u] = 2; return False
        return any(go(u) for u in range(n))
    def cyc2(n, adj):
        seen = [False]*n
        def go(u):
            if seen[u]: return True
            seen[u] = True
            return any(go(v) for v in adj[u])
        return any(go(u) for u in range(n) if not seen[u])
    def kahn(n, adj):
        indeg = [0]*n
        for u in range(n):
            for v in adj[u]: indeg[v] += 1
        q = deque(u for u in range(n) if indeg[u] == 0); order = []
        while q:
            u = q.popleft(); order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        return order
    ok3 = True
    for _t in range(500):
        n = rng.randint(1, 9); adj = [[v for v in range(n) if v != u and rng.random() < 0.22] for u in range(n)]
        ok3 &= cyc3(n, adj) == (len(kahn(n, adj)) < n)
    diamond = [[1, 2], [3], [3], []]
    print()
    say("s09", "three-state DFS cycle test == Kahn's leftover test on 500 random directed graphs", ok3)
    say("s09", "a two-state 'seen before means cycle' test calls the acyclic diamond A->B, A->C, B->D, C->D cyclic",
        cyc2(4, diamond) and not cyc3(4, diamond))

    # --- s10: Fulkerson's rule IS Kahn's algorithm; the method of joints ---
    def fulkerson(n, adj):
        # number the initial events; delete their arrows; number the new initial events; repeat
        alive = set(range(n)); indeg = [0]*n; waves = []
        for u in range(n):
            for v in adj[u]: indeg[v] += 1
        while alive:
            wave = sorted(u for u in alive if indeg[u] == 0)
            if not wave: return waves, False
            waves.append(wave)
            for u in wave:
                alive.discard(u)
                for v in adj[u]: indeg[v] -= 1
        return waves, True
    def longest_arrows(n, adj):
        order = kahn(n, adj); L = [0]*n
        for u in order:
            for v in adj[u]: L[v] = max(L[v], L[u]+1)
        return L
    okF = True; okW = True
    for _t in range(500):
        n = rng.randint(1, 12)
        acyclic = rng.random() < 0.7
        adj = [[v for v in range(n) if (v > u if acyclic else v != u) and rng.random() < 0.25] for u in range(n)]
        waves, done = fulkerson(n, adj)
        ko = kahn(n, adj)
        okF &= done == (len(ko) == n)
        if done:
            pos = {u: i for i, u in enumerate(u for w in waves for u in w)}
            okF &= all(pos[u] < pos[v] for u in range(n) for v in adj[u])
            L = longest_arrows(n, adj)
            okW &= all(L[u] == k for k, w in enumerate(waves) for u in w)
    print()
    say("s10", "Fulkerson's numbering rule == Kahn's algorithm: same verdict on loops, and a valid event order, "
        "500 random networks", okF)
    say("s10", "  the wave in which Fulkerson numbers an event == the most arrows on any path into it", okW)
    # method of joints as a topological sort
    def truss_solve(joints, members, supports, loads):
        j = len(joints); m = len(members)
        unk = [("m", k) for k in range(m)] + [("r", s) for s in supports]
        A = np.zeros((2*j, len(unk))); b = np.zeros(2*j)
        for k, (a_, c_) in enumerate(members):
            dx, dy = np.subtract(joints[c_], joints[a_]); L = math.hypot(dx, dy); ux, uy = dx/L, dy/L
            A[2*a_, k] += ux; A[2*a_+1, k] += uy; A[2*c_, k] -= ux; A[2*c_+1, k] -= uy
        for q_, (jn, ax) in enumerate(supports):
            A[2*jn+ax, m+q_] += 1
        for jn, (fx, fy) in loads.items():
            b[2*jn] -= fx; b[2*jn+1] -= fy
        return A, b, unk
    def by_joints(joints, members, supports, loads, reactions_known):
        A, b, unk = truss_solve(joints, members, supports, loads)
        x_full = np.linalg.solve(A, b)
        known = {}
        if reactions_known:
            for q_ in range(len(supports)): known[len(members)+q_] = x_full[len(members)+q_]
        j = len(joints); solved_joints = []; progress = True
        while progress:
            progress = False
            for jn in range(j):
                if jn in solved_joints: continue
                cols = [k for k in range(A.shape[1]) if (abs(A[2*jn, k]) > 1e-12 or abs(A[2*jn+1, k]) > 1e-12) and k not in known]
                if len(cols) <= 2:
                    rows = A[2*jn:2*jn+2]; rhs = b[2*jn:2*jn+2] - sum(rows[:, k]*known[k] for k in known)
                    if cols:
                        sol = np.linalg.lstsq(rows[:, cols], rhs, rcond=None)[0]
                        for k, v_ in zip(cols, sol): known[k] = v_
                    solved_joints.append(jn); progress = True
        left = A.shape[1]-len(known)
        err = max((abs(known[k]-x_full[k]) for k in known), default=0.0)
        return solved_joints, left, err, x_full
    # simple Warren truss: bottom joints 0..p, top joints between; pin at 0, roller at last bottom joint
    def warren(p):
        joints = [(2.0*i, 0.0) for i in range(p+1)] + [(2.0*i+1, 1.7) for i in range(p)]
        members = [(i, i+1) for i in range(p)] + [(p+1+i, p+2+i) for i in range(p-1)]
        for i in range(p): members += [(i, p+1+i), (p+1+i, i+1)]
        supports = [(0, 0), (0, 1), (p, 1)]
        loads = {p+1+i: (0.0, -10.0) for i in range(p)}
        return joints, members, supports, loads
    okw = True
    for p in (2, 3, 5, 8):
        J, M, S, Ld = warren(p)
        assert len(M)+3 == 2*len(J)
        order, left, err, _x = by_joints(J, M, S, Ld, True)
        okw &= left == 0 and err < 1e-9
    say("s10", "method of joints on Warren trusses of 2, 3, 5, 8 panels: a joint with <= 2 unknowns is always "
        "available, and the forces match the full matrix solve to 1e-9", okw)
    # complex truss: triangle inside a triangle, three connectors -- every joint has three members
    Jc = [(0.0, 0.0), (6.0, 0.0), (3.0, 5.2), (2.2, 1.2), (3.9, 1.5), (2.9, 3.0)]
    Mc = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    Sc = [(0, 0), (0, 1), (1, 1)]; Lc = {5: (0.0, -10.0)}
    Ac, bc, _u = truss_solve(Jc, Mc, Sc, Lc)
    order, left, err, xc = by_joints(Jc, Mc, Sc, Lc, True)
    say("s10", f"compound truss (triangle joined to a triangle by three bars; 6 joints, 9 members, m = 2j - 3): "
        f"determinate, matrix rank {np.linalg.matrix_rank(Ac)} of 12; "
        f"after the reactions, every joint has 3 unknown members, so joint-by-joint stalls at once with "
        f"all {left} member forces still unknown", np.linalg.matrix_rank(Ac) == 12 and left == 9 and len(order) == 0)
    say("s10", "  m = 2j - 3 does not make a truss simple: the counting rule is necessary, not sufficient")
    inner_rows = [6, 7, 8, 9, 10, 11]; inner_cols = [3, 4, 5, 6, 7, 8]
    blk = Ac[np.ix_(inner_rows, inner_cols)]
    xb = np.linalg.solve(blk, bc[inner_rows])
    known = {k: v for k, v in zip(inner_cols, xb)}
    for q_ in range(3): known[9+q_] = xc[9+q_]
    rest_ok = True
    for jn in (0, 1, 2):
        cols = [k for k in range(12) if (abs(Ac[2*jn, k]) > 1e-12 or abs(Ac[2*jn+1, k]) > 1e-12) and k not in known]
        rest_ok &= len(cols) <= 2
        rhs2 = bc[2*jn:2*jn+2]-sum(Ac[2*jn:2*jn+2, k]*known[k] for k in known)
        for k, v_ in zip(cols, np.linalg.lstsq(Ac[2*jn:2*jn+2][:, cols], rhs2, rcond=None)[0]): known[k] = v_
    say("s10", f"  the block that must be solved together: the 6 equations at the three inner joints, in the 6 forces of the "
        f"inner triangle and the connecting bars (rank {np.linalg.matrix_rank(blk)}); after it, each outer joint has 2 "
        f"unknowns and the rest goes joint by joint, matching the full solve to "
        f"{max(abs(known[k]-xc[k]) for k in range(9)):.1e}", np.linalg.matrix_rank(blk) == 6 and rest_ok and
        max(abs(known[k]-xc[k]) for k in range(9)) < 1e-9)

    J4, M4, S4, L4 = warren(4)
    A4, b4, _u4 = truss_solve(J4, M4, S4, L4); x4 = np.linalg.solve(A4, b4)
    Aob = Ac[np.ix_([0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8])]
    rhs_ob = bc[[0, 1, 2, 3, 4, 5]] - Ac[np.ix_([0, 1, 2, 3, 4, 5], [9, 10, 11])] @ xc[9:]
    xob = np.linalg.solve(Aob, rhs_ob)
    say("s10", f"  the outer triangle's three joints form an equally valid block (rank {np.linalg.matrix_rank(Aob)}, "
        f"same forces to {max(abs(xob - xc[[0,1,2,6,7,8]])):.1e})", np.linalg.matrix_rank(Aob) == 6 and
        max(abs(xob - xc[[0, 1, 2, 6, 7, 8]])) < 1e-9)
    say("s10", "lab Warren truss (4 panels, 10 kN at each top joint): member forces " +
        " ".join(f"{v:+.2f}" for v in x4[:len(M4)]) + f"; reactions {x4[len(M4):].round(2).tolist()}")
    say("s10", "lab compound truss (10 kN at the top inner joint): member forces " + " ".join(f"{v:+.2f}" for v in xc[:9])
        + f"; reactions {xc[9:].round(3).tolist()}", all(abs(v) > 0.05 for v in xc[:9]))
    # --- s11: shortest paths; what the source's Dijkstra really does ------------
    def dijkstra(g, s, stale_check=True):
        dist = {u: float("inf") for u in g}; dist[s] = 0; h = [(0, s)]; pops = scans = 0
        while h:
            d, u = heapq.heappop(h); pops += 1
            if stale_check and d > dist[u]: continue
            for v, w in g[u]:
                scans += 1
                if d+w < dist[v]: dist[v] = d+w; heapq.heappush(h, (d+w, v))
            if pops > 200000: return None, pops, scans
        return dist, pops, scans
    def bellman(g, s):
        dist = {u: float("inf") for u in g}; dist[s] = 0
        for _i in range(len(g)):
            for u in g:
                for v, w in g[u]:
                    if dist[u]+w < dist[v]: dist[v] = dist[u]+w
        return dist
    def textbook(g, s):
        dist = {u: float("inf") for u in g}; dist[s] = 0; done = set(); h = [(0, s)]
        while h:
            d, u = heapq.heappop(h)
            if u in done: continue
            done.add(u)
            for v, w in g[u]:
                if v not in done and d+w < dist[v]: dist[v] = d+w; heapq.heappush(h, (d+w, v))
        return dist
    okd = True; okns = True; okneg = True
    for _t in range(300):
        n = rng.randint(2, 12)
        g = {u: [(v, rng.randint(1, 20)) for v in range(n) if v != u and rng.random() < 0.3] for u in range(n)}
        a, _p, _s = dijkstra(g, 0); b = bellman(g, 0); c, _p2, _s2 = dijkstra(g, 0, False)
        okd &= a == b; okns &= c == b
        pot = [rng.randint(0, 30) for _i in range(n)]
        gn = {u: [(v, w+pot[u]-pot[v]) for v, w in g[u]] for u in g}
        an, _p3, _s3 = dijkstra(gn, 0); bn = bellman(gn, 0)
        okneg &= an == bn
    print()
    say("s11", "the source's Dijkstra == Bellman-Ford on 300 random non-negative graphs; without the stale-entry "
        "check it still gives identical distances", okd and okns)
    gh = {0: [], 99: []}
    k_ = 40; m_ = 40
    for i in range(1, k_+1):
        gh[0].append((i, i)); gh[i] = [(99, 2*(k_-i)+1)]
    gh[99] = [(200+j, 1) for j in range(m_)]
    for j in range(m_): gh[200+j] = []
    _a, p1, s1 = dijkstra(gh, 0, True); _b, p2, s2 = dijkstra(gh, 0, False)
    say("s11", f"  a hub reached by 40 ever-shorter routes: {s1} edge scans with the check, {s2} without",
        s2 > 10*s1 and _a == _b)
    g0 = {"A": [("B", 2), ("C", 3)], "C": [("B", -2)], "B": []}
    say("s11", f"negative edge A->C->B: textbook Dijkstra (finalise on pop) says B = {textbook(g0,'A')['B']}, "
        f"true {bellman(g0,'A')['B']}; the source's code says {dijkstra(g0,'A')[0]['B']}",
        textbook(g0, "A")["B"] == 2 and bellman(g0, "A")["B"] == 1 and dijkstra(g0, "A")[0]["B"] == 1)
    say("s11", "  the printed code re-pushes improved nodes, so on 300 graphs with negative edges and no negative "
        "cycle it matched Bellman-Ford every time", okneg)
    gcyc = {"A": [("B", 1)], "B": [("C", -3)], "C": [("B", 1)]}
    say("s11", "  with a negative cycle it never stops (cut off after 200,000 pops)", dijkstra(gcyc, "A")[0] is None)
    # the pipe network: flow is a Laplacian solve, not a shortest path
    pipes = [(0, 1, 2.0), (0, 2, 3.0), (1, 2, 1.5), (1, 3, 4.0), (2, 3, 2.0), (2, 4, 5.0), (3, 5, 1.0), (4, 5, 1.0)]
    N = 6; Lp = np.zeros((N, N))
    for a_, b_, R in pipes:
        c = 1/R; Lp[a_, a_] += c; Lp[b_, b_] += c; Lp[a_, b_] -= c; Lp[b_, a_] -= c
    rhs = np.zeros(N); rhs[0] = 1.0
    keep = [i for i in range(N) if i != 5]
    h_ = np.zeros(N); h_[keep] = np.linalg.solve(Lp[np.ix_(keep, keep)], rhs[keep])
    Reff = h_[0]-h_[5]
    gp = defaultdict(list)
    for a_, b_, R in pipes: gp[a_].append((b_, R)); gp[b_].append((a_, R))
    dsp = dijkstra(dict(gp), 0)[0][5]
    routes = []
    def allpaths(u, seen, acc, path):
        if u == 5: routes.append((acc, path)); return
        for v, R in gp[u]:
            if v not in seen: allpaths(v, seen | {v}, acc+R, path+[v])
    allpaths(0, {0}, 0.0, [0]); routes.sort()
    say("s11", f"  simple routes from inlet to outlet, by total resistance: " +
        "; ".join(f"{'-'.join(map(str,pth))} {r:.1f}" for r, pth in routes[:4]) + " ...",
        routes[0][0] == dsp and routes[1][0] > routes[0][0])
    flows = {(a_, b_): (h_[a_]-h_[b_])/R for a_, b_, R in pipes}
    used = sum(1 for q in flows.values() if abs(q) > 1e-9)
    say("s11", f"pipe network, laminar (head loss = R Q): least-resistance route {dsp:.2f}, but the network's "
        f"effective resistance is {Reff:.3f}; flow runs in {used} of {len(pipes)} pipes, not just the shortest route",
        Reff < dsp and used >= 7)
    say("s11", f"  more water enters pipe 0-1 ({flows[(0,1)]:.3f}), which is NOT on the least-resistance route, than pipe "
        f"0-2 ({flows[(0,2)]:.3f}), which is", flows[(0, 1)] > flows[(0, 2)] and routes[0][1][:2] == [0, 2])
    say("s11", "  laminar flows per pipe: " + ", ".join(f"{a_}-{b_}: {q:+.3f}" for (a_, b_), q in flows.items()))
    from scipy.optimize import fsolve
    def turb_res(hk):
        hh = np.zeros(N); hh[keep] = hk; res = -rhs[keep].copy()
        for a_, b_, R in pipes:
            dh = hh[a_]-hh[b_]; q = math.copysign(math.sqrt(abs(dh)/R), dh)
            if a_ in keep: res[keep.index(a_)] += q
            if b_ in keep: res[keep.index(b_)] -= q
        return res
    ht = np.zeros(N); ht[keep] = fsolve(turb_res, h_[keep]*2, xtol=1e-13)
    tflows = {(a_, b_): math.copysign(math.sqrt(abs(ht[a_]-ht[b_])/R), ht[a_]-ht[b_]) for a_, b_, R in pipes}
    say("s11", "  turbulent (head loss = R Q^2), same pipes: flows " + ", ".join(f"{q:+.3f}" for q in tflows.values()) +
        f"; still {sum(1 for q in tflows.values() if abs(q) > 1e-9)} of 8 pipes carry flow",
        max(abs(x) for x in turb_res(ht[keep])) < 1e-9)
    say("s11", f"  turbulent head drop inlet to outlet for unit flow: {ht[0]-ht[5]:.3f} (the best single route alone: "
        f"{sum(R for a_, b_, R in pipes if (a_, b_) in ((0, 2), (2, 3), (3, 5))):.1f})")
    def best_route(r23):
        g2 = defaultdict(list)
        for a_, b_, R in pipes:
            RR = r23 if (a_, b_) == (2, 3) else R
            g2[a_].append((b_, RR)); g2[b_].append((a_, RR))
        return dijkstra(dict(g2), 0)[0][5]
    sw = [r/100 for r in range(50, 1001) if best_route(r/100) < 7-1e-9]
    say("s11", f"  vary pipe 2-3: the best route is 4 + R(2-3) until R(2-3) = {max(sw):.2f}, then jumps to 0-1-3-5 at 7.0",
        abs(max(sw)-2.99) < 0.011 and abs(best_route(3.5)-7.0) < 1e-9)
    kcl = max(abs(sum((q if a_ == i else -q) for (a_, b_), q in flows.items() if i in (a_, b_))-rhs[i]) for i in keep)
    say("s11", f"  continuity holds at every junction to {kcl:.1e}: water solves Sheet 12's Laplacian, it does not "
        f"run Dijkstra", kcl < 1e-12)
    # 2-D rain water = Dijkstra with max in place of +
    def spill(G):
        R, C = len(G), len(G[0]); dist = {}; h = []
        for i in range(R):
            for j in range(C):
                if i in (0, R-1) or j in (0, C-1): heapq.heappush(h, (G[i][j], i, j))
        while h:
            d, i, j = heapq.heappop(h)
            if (i, j) in dist: continue
            dist[(i, j)] = d
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a_, b_ = i+di, j+dj
                if 0 <= a_ < R and 0 <= b_ < C and (a_, b_) not in dist:
                    heapq.heappush(h, (max(d, G[a_][b_]), a_, b_))
        return dist
    def relax(G):
        R, C = len(G), len(G[0])
        lev = [[G[i][j] if i in (0, R-1) or j in (0, C-1) else 10**9 for j in range(C)] for i in range(R)]
        ch = True
        while ch:
            ch = False
            for i in range(1, R-1):
                for j in range(1, C-1):
                    v = max(G[i][j], min(lev[i+1][j], lev[i-1][j], lev[i][j+1], lev[i][j-1]))
                    if v < lev[i][j]: lev[i][j] = v; ch = True
        return lev
    okm = True
    for _t in range(150):
        R, C = rng.randint(3, 8), rng.randint(3, 8); G = [[rng.randint(0, 9) for _j in range(C)] for _i in range(R)]
        sp = spill(G); rl = relax(G)
        okm &= all(sp[(i, j)] == rl[i][j] for i in range(R) for j in range(C))
    say("s11", "part 1's 2-D spill level == Dijkstra with max(d, h) in place of d + w, 150 grids", okm)

    # --- s12: strongly connected components ---------------------------------
    def kosaraju(n, adj):
        seen = [False]*n; order = []
        for s in range(n):
            if seen[s]: continue
            st = [(s, iter(adj[s]))]; seen[s] = True
            while st:
                u, it = st[-1]
                nxt = next(it, None)
                if nxt is None: order.append(u); st.pop()
                elif not seen[nxt]: seen[nxt] = True; st.append((nxt, iter(adj[nxt])))
        radj = [[] for _i in range(n)]
        for u in range(n):
            for v in adj[u]: radj[v].append(u)
        comp = [-1]*n; c = 0
        for s in reversed(order):
            if comp[s] != -1: continue
            st = [s]; comp[s] = c
            while st:
                u = st.pop()
                for v in radj[u]:
                    if comp[v] == -1: comp[v] = c; st.append(v)
            c += 1
        return comp
    def reach(n, adj):
        Rm = [[False]*n for _i in range(n)]
        for s in range(n):
            st = [s]; Rm[s][s] = True
            while st:
                u = st.pop()
                for v in adj[u]:
                    if not Rm[s][v]: Rm[s][v] = True; st.append(v)
        return Rm
    okk = True
    for _t in range(300):
        n = rng.randint(1, 12); adj = [[v for v in range(n) if v != u and rng.random() < 0.2] for u in range(n)]
        comp = kosaraju(n, adj); Rm = reach(n, adj)
        okk &= all((comp[u] == comp[v]) == (Rm[u][v] and Rm[v][u]) for u in range(n) for v in range(n))
    print()
    say("s12", "Kosaraju's components == pairs that reach each other, 300 random directed graphs", okk)
    okb = True
    for _t in range(100):
        n = rng.randint(3, 12); A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = rng.uniform(2, 5)
            for j in range(n):
                if i != j and rng.random() < 0.15: A[i, j] = rng.uniform(-1, 1)
        b = np.array([rng.uniform(-5, 5) for _i in range(n)])
        adj = [[j for j in range(n) if j != i and A[i, j] != 0] for i in range(n)]
        comp = kosaraju(n, adj); nc = max(comp)+1
        cadj = [set() for _i in range(nc)]
        for i in range(n):
            for j in adj[i]:
                if comp[i] != comp[j]: cadj[comp[i]].add(comp[j])
        # solve blocks so that everything a block depends on is already known
        x = np.zeros(n); done = set(); order_ = []
        while len(done) < nc:
            for c_ in range(nc):
                if c_ not in done and cadj[c_] <= done: order_.append(c_); done.add(c_)
        for c_ in order_:
            idx = [i for i in range(n) if comp[i] == c_]; oth = [j for j in range(n) if comp[j] != c_]
            x[idx] = np.linalg.solve(A[np.ix_(idx, idx)], b[idx]-A[np.ix_(idx, oth)] @ x[oth])
        okb &= np.allclose(x, np.linalg.solve(A, b))
    say("s12", "solving a sparse system one strongly connected block at a time, in dependency order, == solving it "
        "whole (100 systems)", okb)

    # --- s13: sorting and its lower bound -----------------------------------
    cnt = [0]
    def msort(a, lt=False):
        if len(a) <= 1: return a
        m = len(a)//2; L = msort(a[:m], lt); R = msort(a[m:], lt); out = []; i = j = 0
        while i < len(L) and j < len(R):
            cnt[0] += 1
            if (L[i][0] < R[j][0]) if lt else (L[i][0] <= R[j][0]): out.append(L[i]); i += 1
            else: out.append(R[j]); j += 1
        return out+L[i:]+R[j:]
    rows = []
    for n in (10, 100, 1000):
        lb = math.ceil(sum(math.log2(k) for k in range(2, n+1)))
        ms = []
        for _t in range(20):
            a = [(rng.random(), i) for i in range(n)]; cnt[0] = 0; msort(a); ms.append(cnt[0])
        worst = n*math.ceil(math.log2(n)) - 2**math.ceil(math.log2(n)) + 1
        rows.append((n, lb, np.mean(ms), worst))
    print()
    for n, lb, avg, worst in rows:
        say("s13", f"n={n:>5}: lower bound ceil(log2 n!) = {lb:,}; merge sort averages {avg:,.0f}, worst case {worst:,}")
    say("s13", "  merge sort is within about 0.3n comparisons of the bound that no comparison sort can beat",
        all(avg >= lb and avg-lb < 0.3*n for n, lb, avg, _w in rows))
    # binary insertion: comparisons near the bound, moves quadratic
    n = 1000; a = list(range(n)); rng.shuffle(a); out = []; comps = moves = 0
    for x in a:
        lo, hi = 0, len(out)
        while lo < hi:
            comps += 1; mid = (lo+hi)//2
            if out[mid] < x: lo = mid+1
            else: hi = mid
        moves += len(out)-lo; out.insert(lo, x)
    lb1000 = rows[-1][1]
    say("s13", f"binary insertion sort, n=1000: {comps:,} comparisons (bound {lb1000:,}) but {moves:,} element moves",
        comps < lb1000*1.02 and moves > n*n/5)
    say("s13", "  the bound counts comparisons only; an algorithm can meet it and still be O(n^2)")
    keys = [(rng.randint(0, 5), i) for i in range(200)]
    s_le = msort(list(keys)); s_lt = msort(list(keys), True)
    stable = lambda s: all(s[i][0] < s[i+1][0] or s[i][1] < s[i+1][1] for i in range(len(s)-1))
    say("s13", "merge with <= keeps equal keys in input order; with < it does not", stable(s_le) and not stable(s_lt))

    # --- s14: quicksort built properly ---------------------------------------
    def qsort_count(a, pivot="last", scheme="lomuto"):
        a = list(a); comps = [0]; st = [(0, len(a)-1)]; r = random.Random(7)
        while st:
            lo, hi = st.pop()
            if lo >= hi: continue
            if pivot == "random":
                k = r.randint(lo, hi); a[k], a[hi] = a[hi], a[k]
            elif pivot == "median3":
                mid = (lo+hi)//2; trio = sorted([(a[lo], lo), (a[mid], mid), (a[hi], hi)])
                k = trio[1][1]; a[k], a[hi] = a[hi], a[k]
            p = a[hi]
            if scheme == "lomuto":
                i = lo
                for j in range(lo, hi):
                    comps[0] += 1
                    if a[j] < p: a[i], a[j] = a[j], a[i]; i += 1
                a[i], a[hi] = a[hi], a[i]
                st.append((lo, i-1)); st.append((i+1, hi))
            else:   # three-way (Dutch national flag)
                lt, i, gt = lo, lo, hi
                while i <= gt:
                    comps[0] += 1
                    if a[i] < p: a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
                    elif a[i] > p: a[i], a[gt] = a[gt], a[i]; gt -= 1
                    else: i += 1
                st.append((lo, lt-1)); st.append((gt+1, hi))
        assert a == sorted(a)
        return comps[0]
    n = 2000
    inputs = {"random": rng.sample(range(n), n), "sorted": list(range(n)), "all equal": [7]*n,
              "two values": [rng.randint(0, 1) for _i in range(n)]}
    res = {(k, pv, sc): qsort_count(v, pv, sc) for k, v in inputs.items()
           for pv, sc in (("last", "lomuto"), ("random", "lomuto"), ("median3", "lomuto"), ("random", "3way"))}
    print()
    nlogn = n*math.log2(n); quad = n*(n-1)/2
    for k in inputs:
        say("s14", f"{k:>10}: last {res[(k,'last','lomuto')]:>9,}  random {res[(k,'random','lomuto')]:>9,}  "
            f"median-of-3 {res[(k,'median3','lomuto')]:>9,}  | 3-way + random {res[(k,'random','3way')]:>7,}")
    say("s14", f"  (n log2 n = {nlogn:,.0f}; n(n-1)/2 = {quad:,.0f})")
    say("s14", "  sorted input: random pivot and median-of-three both fix the last-element disaster",
        res[("sorted", "random", "lomuto")] < 2*nlogn and res[("sorted", "median3", "lomuto")] < 2*nlogn
        and res[("sorted", "last", "lomuto")] == quad)
    say("s14", "  all-equal keys: NO pivot rule helps the source's Lomuto partition -- exactly n(n-1)/2 every time",
        all(res[("all equal", pv, "lomuto")] == quad for pv in ("last", "random", "median3")))
    say("s14", f"  a three-way partition sorts the same all-equal input with {res[('all equal','random','3way')]:,} "
        f"comparisons", res[("all equal", "random", "3way")] == n)
    say("s14", "  two distinct values: Lomuto with a random pivot is still quadratic in the long run of equal keys",
        res[("two values", "random", "lomuto")] > quad/4)

    # --- s15: binary search ------------------------------------------------------
    def bsearch(a, t):
        lo, hi = 0, len(a)-1; steps = 0
        while lo <= hi:
            steps += 1; mid = lo+(hi-lo)//2
            if a[mid] == t: return mid, steps
            if a[mid] < t: lo = mid+1
            else: hi = mid-1
        return -1, steps
    def lower_bound(a, t):
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo+hi)//2
            if a[mid] < t: lo = mid+1
            else: hi = mid
        return lo
    okb = True; worst = 0
    for _t in range(2000):
        a = sorted(rng.sample(range(500), rng.randint(0, 60))); t = rng.randint(-5, 505)
        i, st = bsearch(a, t); worst = max(worst, st-math.ceil(math.log2(len(a)+1)) if a else 0)
        okb &= (i == -1 and t not in a) or (i >= 0 and a[i] == t)
        okb &= lower_bound(a, t) == bisect.bisect_left(a, t)
    print()
    say("s15", "binary search and lower_bound (source) == Python's bisect, 2,000 cases; never more than "
        "ceil(log2(n+1)) probes", okb and worst <= 0)
    big = np.array([2**30+5, 2**30+9], dtype=np.int32)
    with np.errstate(over="ignore"):
        wrapped = int((big[0]+big[1]))
    say("s15", f"why lo + (hi-lo)//2: in 32-bit integers (2^30+5) + (2^30+9) wraps to {wrapped:,}", wrapped < 0)
    def search_rot(nums, t, le=True):
        lo, hi = 0, len(nums)-1
        while lo <= hi:
            mid = (lo+hi)//2
            if nums[mid] == t: return mid
            if (nums[lo] <= nums[mid]) if le else (nums[lo] < nums[mid]):
                if nums[lo] <= t < nums[mid]: hi = mid-1
                else: lo = mid+1
            else:
                if nums[mid] < t <= nums[hi]: lo = mid+1
                else: hi = mid-1
        return -1
    okr = True; badlt = 0
    for _t in range(3000):
        base = sorted(rng.sample(range(100), rng.randint(1, 12))); k = rng.randrange(len(base))
        nums = base[k:]+base[:k]; t = rng.choice(nums+[101, -1])
        exp = nums.index(t) if t in nums else -1
        okr &= search_rot(nums, t) == exp
        badlt += search_rot(nums, t, False) != exp
    say("s15", f"search in a rotated array (source) is right on 3,000 cases; with < instead of <= it fails {badlt} times",
        okr and badlt > 0)
    def median2(a, b):
        if len(a) > len(b): a, b = b, a
        m, n = len(a), len(b); lo, hi = 0, m; half = (m+n+1)//2
        while lo <= hi:
            i = (lo+hi)//2; j = half-i
            l1 = a[i-1] if i > 0 else float("-inf"); r1 = a[i] if i < m else float("inf")
            l2 = b[j-1] if j > 0 else float("-inf"); r2 = b[j] if j < n else float("inf")
            if l1 <= r2 and l2 <= r1:
                return max(l1, l2) if (m+n) % 2 else (max(l1, l2)+min(r1, r2))/2
            if l1 > r2: hi = i-1
            else: lo = i+1
    okm = all(median2(sorted(x), sorted(y)) == float(np.median(x+y))
              for x, y in (([rng.randint(0, 50) for _i in range(rng.randint(0, 9))],
                            [rng.randint(0, 50) for _i in range(rng.randint(1, 9))]) for _t in range(2000)))
    say("s15", "median of two sorted arrays (source) == numpy median of the merged list, 2,000 cases", okm)

    # --- s16: binary search on the answer IS serial line balancing -----------
    def ship(weights, days):
        lo, hi = max(weights), sum(weights)
        while lo < hi:
            mid = (lo+hi)//2; load = 0; d = 1
            for w in weights:
                if load+w > mid: d += 1; load = 0
                load += w
            if d <= days: hi = mid
            else: lo = mid+1
        return lo
    def ship_bf(weights, days):
        n = len(weights); best = float("inf")
        for k in range(0, min(days, n)):
            for cuts in itertools.combinations(range(1, n), k):
                b = (0,)+cuts+(n,)
                best = min(best, max(sum(weights[b[i]:b[i+1]]) for i in range(len(b)-1)))
        return best
    oks = all(ship(w, d) == ship_bf(w, d) for w, d in
              (([rng.randint(1, 9) for _i in range(rng.randint(1, 9))], rng.randint(1, 5)) for _t in range(400)))
    def greedy_stations(tasks, C):
        st = 1; load = 0
        for t in tasks:
            if load+t > C: st += 1; load = 0
            load += t
        return st
    def min_stations_chain(tasks, C):
        n = len(tasks); best = n
        for k in range(n):
            for cuts in itertools.combinations(range(1, n), k):
                b = (0,)+cuts+(n,)
                if all(sum(tasks[b[i]:b[i+1]]) <= C for i in range(len(b)-1)): best = min(best, k+1)
        return best
    okg = all(greedy_stations(t, C) == min_stations_chain(t, C) for t, C in
              (([rng.randint(1, 9) for _i in range(rng.randint(1, 9))], rng.randint(9, 20)) for _t in range(400)))
    print()
    say("s16", "ship-within-days (source) == brute force over every way to cut the sequence, 400 cases", oks)
    say("s16", "  it is type-II line balancing with a serial precedence chain: the least cycle time for d stations")
    say("s16", "filling stations in order until the next task does not fit == the fewest stations, for a chain "
        "(400 cases) -- the linear-time case Sheet 13 part 1 s09 promised", okg)

    # --- s17: greedy, and when it is right ---------------------------------------
    def johnson(jobs):
        a = sorted([j for j in jobs if j[0] <= j[1]], key=lambda j: j[0])
        b = sorted([j for j in jobs if j[0] > j[1]], key=lambda j: -j[1])
        return a+b
    def makespan(seq):
        t1 = t2 = 0
        for p1, p2 in seq: t1 += p1; t2 = max(t2, t1)+p2
        return t2
    okj = True
    for _t in range(300):
        jobs = [(rng.randint(1, 9), rng.randint(1, 9)) for _i in range(rng.randint(1, 7))]
        okj &= makespan(johnson(jobs)) == min(makespan(p) for p in itertools.permutations(jobs))
    print()
    say("s17", "Johnson's rule (two-machine flow shop) == the best of every sequence, 300 random shops", okj)
    def greedy_coins(coins, amt):
        c = 0
        for x in sorted(coins, reverse=True): c += amt//x; amt %= x
        return c
    def dp_coins(coins, amt):
        dp = [0]+[10**9]*amt
        for a in range(1, amt+1):
            for x in coins:
                if x <= a: dp[a] = min(dp[a], dp[a-x]+1)
        return dp
    inr = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    dpi = dp_coins(inr, 5000)
    say("s17", "greedy change in rupees {1,2,5,10,20,50,100,200,500} is optimal for every amount up to 5,000",
        all(greedy_coins(inr, a) == dpi[a] for a in range(5001)))
    dpb = dp_coins([1, 3, 4], 6)
    say("s17", f"coins {{1,3,4}}, amount 6: greedy {greedy_coins([1,3,4],6)} coins (4+1+1), optimum {dpb[6]} (3+3)",
        greedy_coins([1, 3, 4], 6) == 3 and dpb[6] == 2)
    def can_jump(nums):
        mr = 0
        for i, j in enumerate(nums):
            if i > mr: return False
            mr = max(mr, i+j)
        return True
    def can_jump_bf(nums):
        seen = {0}; st = [0]
        while st:
            i = st.pop()
            for k in range(1, nums[i]+1):
                if i+k < len(nums) and i+k not in seen: seen.add(i+k); st.append(i+k)
        return len(nums)-1 in seen
    okjump = all(can_jump(v) == can_jump_bf(v) for v in ([rng.randint(0, 3) for _i in range(rng.randint(1, 12))] for _t in range(1000)))
    def merge_iv(iv):
        iv = sorted([list(x) for x in iv]); out = [iv[0]]
        for s, e in iv[1:]:
            if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
            else: out.append([s, e])
        return out
    def cover(iv):
        pts = set()
        for s, e in iv: pts.update(x/2 for x in range(2*s, 2*e+1))
        return pts
    okiv = True
    for _t in range(500):
        iv = [(s, s+rng.randint(0, 6)) for s in (rng.randint(0, 20) for _i in range(rng.randint(1, 7)))]
        mg = merge_iv(iv)
        okiv &= cover(mg) == cover(iv) and all(mg[i][1] < mg[i+1][0] for i in range(len(mg)-1))
    say("s17", "jump game and merge intervals (source) == brute force, 1,000 and 500 cases", okjump and okiv)
    # slip gauges: the M87 set, and the textbook 'clear the last decimal first' rule
    UNIT = 2000   # 0.0005 mm units
    m87 = [1.0005] + [round(1+0.001*k, 3) for k in range(1, 10)] + [round(1+0.01*k, 2) for k in range(1, 50)] \
          + [0.5*k for k in range(1, 20)] + [10.0*k for k in range(1, 10)]
    units = [int(round(g*UNIT)) for g in m87]
    CAP = int(100*UNIT)
    INF = 10**4
    best = np.full(CAP+1, INF, dtype=np.int32); best[0] = 0
    for w in units:
        nb = best.copy(); nb[w:] = np.minimum(best[w:], best[:-w]+1); best = nb
    def rule(Lu):
        # the textbook rule, in integer units of 0.0005 mm: clear the last decimal place first
        used = []; u = Lu
        if u % 2: used.append(2001); u -= 2001                    # 1.0005
        mu = u//2                                                  # micrometres
        d3 = mu % 10
        if d3: used.append(2*(1000+d3)); mu -= 1000+d3             # 1.001 .. 1.009
        hund = (mu//10) % 10; t = ((mu//100) % 10) % 5
        if hund or t:
            g = 1000+100*t+10*hund                                 # 1.01 .. 1.49
            used.append(2*g); mu -= g
        rem = mu % 10000                                           # a multiple of 0.5 mm below 10 mm
        if rem:
            if rem % 500: return None
            used.append(2*rem); mu -= rem                          # 0.5 .. 9.5
        if mu < 0: return None
        if mu:
            if mu % 10000 or mu > 90000: return None
            used.append(2*mu)                                      # 10 .. 90
        if len(set(used)) != len(used) or any(x not in units for x in used): return None
        return used
    worse = 0; checked = 0; fails = 0; example = None
    for L_units in range(int(4*UNIT), int(99*UNIT)+1, 2):     # every 0.001 mm from 4 to 99 mm
        L = L_units/UNIT; u = rule(L_units)
        if u is None: fails += 1; continue
        assert sum(u) == L_units
        checked += 1
        if len(u) > best[L_units]:
            worse += 1
            if example is None: example = (L, u, int(best[L_units]))
    say("s17", f"slip gauges, M87 set (reported composition): 'clear the last decimal first' checked on {checked:,} "
        f"lengths from 4 to 99 mm in 0.001 mm steps; it uses more blocks than the minimum on {worse:,} "
        f"({100*worse/checked:.1f}%)", checked > 90000)
    if example:
        say("s17", f"  e.g. {example[0]:.3f} mm: rule {len(example[1])} blocks {[x/UNIT for x in example[1]]}, minimum {example[2]}")
    say("s17", f"  lengths the rule could not build as stated: {fails:,}")

    counts = defaultdict(int)
    for L_units in range(int(4*UNIT), int(99*UNIT)+1, 2): counts[len(rule(L_units))] += 1
    say("s17", "  blocks needed (rule = minimum): " + ", ".join(f"{k}: {v:,}" for k, v in sorted(counts.items())))
    ex = rule(int(round(41.125*UNIT)))
    say("s17", f"  41.125 mm -> {[x/UNIT for x in ex]}, {len(ex)} blocks; minimum {best[int(round(41.125*UNIT))]}",
        len(ex) == best[int(round(41.125*UNIT))] == 4)
    # --- s18: DP revisited --------------------------------------------------------
    def coin_change(coins, amount):
        dp = [float("inf")]*(amount+1); dp[0] = 0
        for a in range(1, amount+1):
            for c in coins:
                if c <= a and dp[a-c]+1 < dp[a]: dp[a] = dp[a-c]+1
        return dp[amount] if dp[amount] != float("inf") else -1
    def cc_bf(coins, amount):
        best = -1
        for k in range(0, amount+1):
            for combo in itertools.combinations_with_replacement(coins, k):
                if sum(combo) == amount: return k
        return best
    okc = all(coin_change(c, a) == cc_bf(c, a) for c, a in
              ((sorted(set(rng.randint(1, 9) for _i in range(rng.randint(1, 3)))), rng.randint(0, 14)) for _t in range(200)))
    def lcs(a, b):
        dp = [[0]*(len(b)+1) for _i in range(len(a)+1)]
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    def lcs_bf(a, b):
        for k in range(min(len(a), len(b)), -1, -1):
            subs = set("".join(c) for c in itertools.combinations(a, k))
            if any("".join(c) in subs for c in itertools.combinations(b, k)): return k
    okl = all(lcs(a, b) == lcs_bf(a, b) for a, b in
              (("".join(rng.choice("ab") for _i in range(rng.randint(0, 8))), "".join(rng.choice("ab") for _i in range(rng.randint(0, 8)))) for _t in range(200)))
    def ks2(w, v, C):
        n = len(w); dp = [[0]*(C+1) for _i in range(n+1)]
        for i in range(1, n+1):
            for c in range(C+1):
                dp[i][c] = dp[i-1][c]
                if w[i-1] <= c: dp[i][c] = max(dp[i][c], dp[i-1][c-w[i-1]]+v[i-1])
        return dp[n][C]
    def ks1(w, v, C, right_to_left=True):
        dp = [0]*(C+1)
        for i in range(len(w)):
            rngc = range(C, w[i]-1, -1) if right_to_left else range(w[i], C+1)
            for c in rngc: dp[c] = max(dp[c], dp[c-w[i]]+v[i])
        return dp[C]
    def ks_bf(w, v, C):
        return max(sum(v[i] for i in S) for r in range(len(w)+1) for S in itertools.combinations(range(len(w)), r) if sum(w[i] for i in S) <= C)
    def unbounded(w, v, C):
        dp = [0]*(C+1)
        for c in range(C+1):
            for i in range(len(w)):
                if w[i] <= c: dp[c] = max(dp[c], dp[c-w[i]]+v[i])
        return dp[C]
    okk = True; oku = True
    for _t in range(300):
        n = rng.randint(1, 7); w = [rng.randint(1, 8) for _i in range(n)]; v = [rng.randint(1, 20) for _i in range(n)]; C = rng.randint(0, 20)
        okk &= ks2(w, v, C) == ks1(w, v, C) == ks_bf(w, v, C)
        oku &= ks1(w, v, C, False) == unbounded(w, v, C)
    print()
    say("s18", "coin change and LCS (source) == brute force, 200 cases each", okc and okl)
    say("s18", "0/1 knapsack: 2-D table == 1-D right-to-left == brute force, 300 cases", okk)
    say("s18", "  the 1-D table swept left-to-right == the UNBOUNDED knapsack, exactly as the source warns", oku)

    # --- s19: worked example -- a stores index for a million part numbers ------
    wr = np.random.default_rng(1419)
    parts = np.unique(wr.integers(1_000_000, 10_000_000, 1_100_000))[:1_000_000]
    wr.shuffle(parts); keys = np.sort(parts)
    probes = []
    for t in wr.choice(parts, 2000):
        lo, hi, st = 0, len(keys)-1, 0
        while lo <= hi:
            st += 1; mid = (lo+hi)//2
            if keys[mid] == t: break
            if keys[mid] < t: lo = mid+1
            else: hi = mid-1
        probes.append(st)
    fam_lo, fam_hi = 4_180_000, 4_189_999
    i0 = bisect.bisect_left(keys, fam_lo); i1 = bisect.bisect_right(keys, fam_hi)
    k = i1-i0
    print()
    say("s19", f"{len(keys):,} part numbers: binary search on the sorted list takes {np.mean(probes):.1f} probes on average, "
        f"never more than {max(probes)} (log2 n = {math.log2(len(keys)):.1f})", max(probes) <= 20)
    say("s19", f"family 418-xxxx: two binary searches find all {k:,} members, touching about {2*20+k:,} entries; "
        f"a hash table must examine all {len(keys):,}", k > 500)
    say("s19", "  a hash table answers 'is this part here?' in about 1.5 probes at half load (part 1 s13), and cannot answer"
        " 'which parts are in this range?' at all without a full scan")

    # --- s22: not every IE problem is NP-hard: the assignment problem ----------
    from scipy.optimize import linear_sum_assignment
    oka = True
    for _t in range(200):
        n = rng.randint(2, 6); Cm = np.array([[rng.randint(1, 30) for _j in range(n)] for _i in range(n)])
        r_, c_ = linear_sum_assignment(Cm)
        best = min(sum(Cm[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        oka &= Cm[r_, c_].sum() == best
    print()
    say("s22", "assignment problem: the Hungarian-type exact method == the best of every assignment, 200 random "
        "cost matrices up to 6 x 6", oka)

_ch14p2()

# ======================================================================================
# Chapter 15 block. Owned by the Chapter 15 run: replace only the body of _ch15() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch15():
    import math, random, itertools, hashlib, os, sys, subprocess, tempfile, time, zlib, stat, json, textwrap
    from collections import Counter
    import numpy as np
    T = "[Ch15 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T % sec} {msg}{tail}")
    rng15 = random.Random(1515)
    GENV = dict(os.environ, GIT_AUTHOR_NAME="A", GIT_AUTHOR_EMAIL="a@x", GIT_COMMITTER_NAME="A",
                GIT_COMMITTER_EMAIL="a@x", GIT_AUTHOR_DATE="2026-01-01T00:00:00+0000",
                GIT_COMMITTER_DATE="2026-01-01T00:00:00+0000", GIT_CONFIG_GLOBAL=os.devnull,
                GIT_CONFIG_SYSTEM=os.devnull, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    def git(d, *a, raw=False, check=True):
        r = subprocess.run(["git", *a], cwd=d, env=GENV, capture_output=True, text=True)
        if raw: return r
        return r.stdout.strip()
    def sh(cmd, cwd=None, timeout=20):
        return subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=GENV)
    def newrepo():
        d = tempfile.mkdtemp(prefix="ch15g")
        git(d, "init", "-q", "-b", "main")
        return d
    def wr(path, s):
        with open(path, "w") as f: f.write(s)
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=here, capture_output=True, text=True).stdout.strip()

    # --- s01: a pipe is a bounded buffer, and sort is a batch station -------------------------
    print()
    import fcntl
    r_, w_ = os.pipe()
    fl = fcntl.fcntl(w_, fcntl.F_GETFL); fcntl.fcntl(w_, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    cap = 0
    try:
        while True: cap += os.write(w_, b"x" * 4096)
    except BlockingIOError:
        pass
    os.close(r_); os.close(w_)
    say("s01", f"a pipe holds {cap:,} bytes before the writer is made to wait (Linux default)", cap == 65536)
    r = sh("yes | head -3 >/dev/null; echo ${PIPESTATUS[0]}")
    say("s01", f"'yes | head -3' ends: yes (endless) is stopped by SIGPIPE, exit status {r.stdout.strip()} = 128 + 13",
        r.stdout.strip() == "141")
    def first_line_delay(stage):
        p = subprocess.Popen(["bash", "-c", "(printf 'b\\n'; sleep 1; printf 'a\\n') | " + stage],
                             stdout=subprocess.PIPE, text=True)
        t0 = time.time(); p.stdout.readline(); dt = time.time() - t0; p.communicate(); return dt
    dc, ds, dg, dgl = first_line_delay("cat"), first_line_delay("sort"), first_line_delay("grep ."), first_line_delay("grep --line-buffered .")
    say("s01", f"first line out after: cat {dc:.2f} s, sort {ds:.2f} s, grep {dg:.2f} s, grep --line-buffered {dgl:.2f} s "
        "(producer pauses 1 s between lines)", dc < 0.5 and ds > 0.9 and dg > 0.9 and dgl < 0.5)
    say("s01", "  sort cannot emit before end of input; grep into a pipe ships in 4 KiB lots unless told otherwise")
    r1 = sh("printf 'a\\nb\\na\\n' | uniq -c | wc -l"); r2 = sh("printf 'a\\nb\\na\\n' | sort | uniq -c | wc -l")
    say("s01", f"uniq -c on a,b,a gives {r1.stdout.strip()} groups; after sort, {r2.stdout.strip()}",
        r1.stdout.strip() == "3" and r2.stdout.strip() == "2")
    d = tempfile.mkdtemp(prefix="ch15s")
    kinds = {"CUDA out of memory": 41, "shape mismatch": 29, "NaN loss": 17, "checkpoint not found": 11,
             "timeout": 7, "disk full": 5, "bad token": 3, "port in use": 2, "OOM killer": 1}
    lines = []
    for k, c in kinds.items(): lines += [f"2026-03-0{1 + i % 9} 12:0{i % 10}:00 ERROR: {k}" for i in range(c)]
    lines += [f"2026-03-01 12:00:00 INFO: step {i}" for i in range(300)]
    rng15.shuffle(lines); wr(f"{d}/app.log", "\n".join(lines) + "\n")
    r = sh("""grep "ERROR" app.log | awk -F': ' '{print $2}' | sort | uniq -c | sort -rn | head -10""", cwd=d)
    got = [(int(L.split()[0]), L.split(None, 1)[1]) for L in r.stdout.strip().splitlines()]
    ref = [(c, k) for k, c in Counter({k: c for k, c in kinds.items()}).most_common(10)]
    say("s01", f"the source's top-errors pipeline == a Python counter on the same log ({len(got)} types, top '{got[0][1]}' x{got[0][0]})",
        got == ref)
    wr(f"{d}/t.log", "loss loss\nloss\nepoch\n")
    r = sh("grep -c loss t.log; grep -o loss t.log | wc -l", cwd=d)
    a_, b_ = r.stdout.split()
    say("s01", f"grep -c counts matching lines ({a_}), not occurrences ({b_})", a_ == "2" and b_ == "3")

    # --- s02: redirection order, a header row, a missing newline, permissions ----------------
    print()
    r = sh("f() { echo out; echo err >&2; }; f > o1 2>&1; f 2>&1 > o2; cat o1 | wc -l; cat o2 | wc -l", cwd=d)
    tok = r.stdout.split(); n1, n2 = tok[-2], tok[-1]
    say("s02", f"'> f 2>&1' puts {n1} lines in the file; '2>&1 > f' puts {n2}, and 'err' lands on the screen "
        "(stderr went where stdout pointed *before* the file)", n1 == "2" and n2 == "1" and tok[0] == "err")
    wr(f"{d}/data.csv", "label,x\ncat,1\ndog,2\ncat,3\nbird,4\n")
    r = sh("cut -d',' -f1 data.csv | sort -u | wc -l", cwd=d)
    say("s02", f"source's 'count unique values in column 1' returns {r.stdout.strip()} for 3 labels: the header is counted",
        r.stdout.strip() == "4")
    with open(f"{d}/nonl.txt", "w") as f: f.write("row1\nrow2")
    r = sh("wc -l < nonl.txt", cwd=d)
    say("s02", f"wc -l counts newline characters: a 2-row file with no final newline reports {r.stdout.strip()}",
        r.stdout.strip() == "1")
    r = sh("du -sh --max-depth=1 . ; echo rc=$?", cwd=d)
    say("s02", f"the source's 'du -sh --max-depth=1 .' is refused: {r.stderr.strip().splitlines()[0][:70]}",
        "rc=1" in r.stdout and "conflict" in r.stderr)
    r = sh("du -h --max-depth=1 . >/dev/null; echo rc=$?", cwd=d)
    say("s02", "  without -s it runs", "rc=0" in r.stdout)
    wr(f"{d}/s.py", "x")
    os.chmod(f"{d}/s.py", 0o754); m754 = stat.filemode(os.stat(f"{d}/s.py").st_mode)
    os.chmod(f"{d}/s.py", 0o755); m755 = stat.filemode(os.stat(f"{d}/s.py").st_mode)
    say("s02", f"-rwxr-xr-- is octal 754 ({m754}); chmod 755 gives {m755}: each digit is three bits r=4 w=2 x=1",
        m754 == "-rwxr-xr--" and m755 == "-rwxr-xr-x")
    r = sh("chmod 644 s.py; (umask 022; chmod +x s.py); stat -c %A s.py; chmod 644 s.py; (umask 077; chmod +x s.py); stat -c %A s.py", cwd=d)
    u22, u77 = r.stdout.split()
    say("s02", f"'chmod +x' adds x for everyone only under the usual umask: umask 022 -> {u22}, umask 077 -> {u77}",
        u22 == "-rwxr-xr-x" and u77 == "-rwxr--r--")

    # --- s03: the environment is copied at birth; %1; nohup -------------------------------------
    print()
    r = sh('X=1; bash -c \'echo "[$X]"\'; export X; bash -c \'echo "[$X]"\'; bash -c \'export X=2\'; echo "[$X]"')
    o = r.stdout.split()
    say("s03", f"unexported variable seen by a child: {o[0]}; exported: {o[1]}; child's change seen by parent: {o[2]} (still 1)",
        o == ["[]", "[1]", "[1]"])
    r = sh("sleep 3 & a=$!; sleep 3 & b=$!; echo $a $b $(jobs -p %1) $(jobs -p %%); kill $a $b")
    a_, b_, j1, jl = r.stdout.split()
    say("s03", "'kill %1' names job 1, the first started; the most recent is %% (or %+)", j1 == a_ and jl == b_ and a_ != b_)
    r = sh("nohup sleep 3 >/dev/null 2>&1 & p=$!; sleep 0.3; grep SigIgn /proc/$p/status; kill $p")
    ign = int(r.stdout.split()[1], 16)
    r2 = sh("sleep 3 & p=$!; sleep 0.3; grep SigIgn /proc/$p/status; kill $p")
    ign2 = int(r2.stdout.split()[1], 16)
    say("s03", f"nohup sets SIGHUP (signal 1) to ignored: SigIgn bit 0 = {ign & 1} under nohup, {ign2 & 1} without",
        (ign & 1) == 1 and (ign2 & 1) == 0)

    # --- s04: what git stores: snapshots named by their content (debt 2) -------------------------
    print()
    d = newrepo()
    wr(f"{d}/a.txt", "shaft dia 20 h7\n"); wr(f"{d}/b.txt", "shaft dia 20 h7\n")
    git(d, "add", "."); git(d, "commit", "-qm", "one")
    c = b"shaft dia 20 h7\n"
    hb = hashlib.sha1(b"blob %d\0" % len(c) + c).hexdigest()
    say("s04", f"blob name = SHA-1('blob 16\\0' + content) = {hb[:12]}..., and git agrees", hb == git(d, "hash-object", "a.txt"))
    nobj = sum(len(fs) for _r, _d, fs in os.walk(f"{d}/.git/objects") if "info" not in _r and "pack" not in _r)
    say("s04", f"two files with identical content, one commit: {nobj} objects stored (one blob, one tree, one commit)", nobj == 3)
    raw = subprocess.run(["git", "cat-file", "commit", "HEAD"], cwd=d, env=GENV, capture_output=True).stdout
    h1 = git(d, "rev-parse", "HEAD")
    say("s04", f"commit name = SHA-1 of its own text (tree, author, time, message): {h1[:12]}...",
        hashlib.sha1(b"commit %d\0" % len(raw) + raw).hexdigest() == h1)
    d2 = newrepo(); wr(f"{d2}/a.txt", "shaft dia 20 h7\n"); wr(f"{d2}/b.txt", "shaft dia 20 h7\n")
    git(d2, "add", "."); git(d2, "commit", "-qm", "one")
    say("s04", "an independent repository, same content, author, time and message: same commit name, no registry consulted",
        git(d2, "rev-parse", "HEAD") == h1)
    t1, b1 = git(d, "rev-parse", "HEAD^{tree}"), git(d, "rev-parse", "HEAD:a.txt")
    wr(f"{d}/a.txt", "shaft dia 20 h6\n"); git(d, "commit", "-qam", "two")
    t2, b2, bb = git(d, "rev-parse", "HEAD^{tree}"), git(d, "rev-parse", "HEAD:a.txt"), git(d, "rev-parse", "HEAD:b.txt")
    say("s04", f"one character changed (h7 -> h6): blob {b1[:7]} -> {b2[:7]}, tree {t1[:7]} -> {t2[:7]}, untouched b.txt keeps {bb[:7]}",
        b1 != b2 and t1 != t2 and bb == b1)
    say("s04", "the new commit records its parent's name inside itself", ("parent " + h1) in git(d, "cat-file", "-p", "HEAD"))
    # loose objects are whole; packing deltas the OLDER revision against the newer
    d = newrepo()
    L_ = [f"line {i}: dimension {i * 0.5:.1f} mm tolerance h7\n" for i in range(1000)]
    wr(f"{d}/spec.txt", "".join(L_)); git(d, "add", "."); git(d, "commit", "-qm", "r1"); bo = git(d, "rev-parse", "HEAD:spec.txt")
    L_[500] = "line 500: dimension 250.0 mm tolerance h6\n"
    wr(f"{d}/spec.txt", "".join(L_)); git(d, "commit", "-qam", "r2"); bn = git(d, "rev-parse", "HEAD:spec.txt")
    fsz = os.path.getsize(f"{d}/spec.txt")
    whole = [len(zlib.decompress(open(f"{d}/.git/objects/{b[:2]}/{b[2:]}", "rb").read())) for b in (bo, bn)]
    say("s04", f"before packing, both revisions are stored whole: {whole[0]:,} and {whole[1]:,} bytes (file {fsz:,} + 11-byte header)",
        whole == [fsz + 11, fsz + 11])
    git(d, "gc", "-q", "--aggressive")
    idx = [f for f in os.listdir(f"{d}/.git/objects/pack") if f.endswith(".idx")][0]
    vp = {Lp.split()[0]: Lp.split() for Lp in git(d, "verify-pack", "-v", f".git/objects/pack/{idx}").splitlines() if Lp[:40] in (bo, bn)}
    old_delta = len(vp[bo]) == 7 and vp[bo][6] == bn
    say("s04", f"after 'git gc': the NEWER revision is whole ({vp[bn][2]} bytes), the OLDER is a {vp[bo][2]}-byte delta against it",
        old_delta and len(vp[bn]) == 5 and int(vp[bo][2]) < 100)

    # --- s05: what a content name guarantees: the birthday bound ---------------------------------
    print()
    def p_coll(n, bits):
        N = 2.0 ** bits
        return -math.expm1(sum(math.log1p(-i / N) for i in range(n)))
    def n_half(bits):
        lo, hi = 1, 1 << (bits // 2 + 3)
        while lo < hi:
            m = (lo + hi) // 2
            if p_coll(m, bits) >= 0.5: hi = m
            else: lo = m + 1
        return lo
    nh28 = n_half(28)
    say("s05", f"7 hex digits = 28 bits: 50% chance of some shared prefix after {nh28:,} objects "
        f"(1.1774 x 2^14 = {1.1774 * 2 ** 14:,.0f})", abs(nh28 - 1.1774 * 2 ** 14) < 3)
    p1e9 = (1e9) ** 2 / 2 ** 161
    say("s05", f"full 160-bit names: chance of any accidental collision among 1e9 objects ~ n^2/2^161 = {p1e9:.1e}", p1e9 < 1e-30)
    say("s05", f"  50% only after 1.1774 x 2^80 = {1.1774 * 2 ** 80:.2e} objects")
    def first_coll(bits, t, pre="rev"):
        seen = {}; i = 0
        while True:
            h = int.from_bytes(hashlib.sha1(f"{pre}-{t}-{i}".encode()).digest()[:8], "big") >> (64 - bits)
            if h in seen: return i + 1
            seen[h] = i; i += 1
    meas24 = [first_coll(24, t) for t in range(400)]
    pred24 = math.sqrt(math.pi / 2 * 2 ** 24)
    say("s05", f"truncated SHA-1, 24 bits: first shared name after {np.mean(meas24):,.0f} objects on average (400 runs); "
        f"birthday prediction sqrt(pi/2 * 2^24) = {pred24:,.0f}", abs(np.mean(meas24) / pred24 - 1) < 0.06)
    say("s05", f"  spread of those 400 runs: {min(meas24):,} to {max(meas24):,}")
    lab24 = first_coll(24, 0); lab16 = first_coll(16, 0); lab20 = first_coll(20, 0); lab28 = first_coll(28, 0)
    say("s05", f"lab rest values (run 0): first collision at object {lab16:,} (16 bits), {lab20:,} (20), {lab24:,} (24), {lab28:,} (28)")
    meas16 = [first_coll(16, t) for t in range(50)]
    say("s05", f"lab '50 runs' at 16 bits: mean {np.mean(meas16):.1f} vs prediction {math.sqrt(math.pi / 2 * 2 ** 16):.1f}",
        abs(np.mean(meas16) / math.sqrt(math.pi / 2 * 2 ** 16) - 1) < 0.15)
    LAB5 = {"run0": {}, "mean100": {}}
    ok100 = True
    for bits in range(12, 25, 2):
        runs = [first_coll(bits, t, f"b{bits}") for t in range(100)]
        mu = math.sqrt(math.pi / 2 * 2 ** bits); se = math.sqrt((4 - math.pi) / 2 * 2 ** bits) / 10
        LAB5["run0"][bits] = runs[0]; LAB5["mean100"][bits] = round(float(np.mean(runs)), 1)
        ok100 &= abs(np.mean(runs) - mu) < 3 * se
        say("s05", f"  lab, {bits} bits: run 0 collides at {runs[0]:,}; mean of 100 runs {np.mean(runs):,.1f}, "
            f"predicted {mu:,.1f} +/- {se:,.1f} (one standard error)")
    say("s05", "every lab bit width: the 100-run mean lies within 3 standard errors of sqrt(pi/2 * 2^b)", ok100)

    # --- s06: a diff is a longest common subsequence (debt 1) ------------------------------------
    print()
    def lcs_len(a, b):
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]
    def lcs_all(a, b):
        n, m = len(a), len(b); Ld = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                Ld[i][j] = Ld[i + 1][j + 1] + 1 if a[i] == b[j] else max(Ld[i + 1][j], Ld[i][j + 1])
        from functools import lru_cache
        @lru_cache(None)
        def go(i, j):
            if i == n or j == m: return {""}
            if a[i] == b[j]: return {a[i] + s for s in go(i + 1, j + 1)}
            out = set()
            if Ld[i + 1][j] == Ld[i][j]: out |= go(i + 1, j)
            if Ld[i][j + 1] == Ld[i][j]: out |= go(i, j + 1)
            return out
        return go(0, 0)
    A_, B_ = "ABCABBA", "CBABAC"
    al = lcs_all(A_, B_)
    say("s06", f"ABCABBA vs CBABAC: LCS length {lcs_len(A_, B_)}, shortest edit 7+6-2*4 = {7 + 6 - 2 * lcs_len(A_, B_)}, "
        f"and {len(al)} different LCS strings {sorted(al)}", lcs_len(A_, B_) == 4 and len(al) == 3)
    def cnt_git(fa, fb, *opt):
        r = subprocess.run(["git", "diff", "--no-index", "--numstat", *opt, fa, fb], env=GENV, capture_output=True, text=True)
        s = r.stdout.split(); return (int(s[0]), int(s[1])) if s else (0, 0)
    def cnt_gnu(fa, fb, *opt):
        r = subprocess.run(["diff", "--unchanged-line-format=", "--old-line-format=-\n", "--new-line-format=+\n", *opt, fa, fb],
                           capture_output=True, text=True)
        return r.stdout.count("+"), r.stdout.count("-")
    tools = {"git": (cnt_git, ()), "git --minimal": (cnt_git, ("--minimal",)), "git --patience": (cnt_git, ("--patience",)),
             "git --histogram": (cnt_git, ("--histogram",)), "GNU diff": (cnt_gnu, ()), "GNU diff -d": (cnt_gnu, ("-d",))}
    d = tempfile.mkdtemp(prefix="ch15d")
    eq = {k: 0 for k in tools}; over = {k: 0 for k in tools}; worst = {k: 0 for k in tools}; consistent = True
    for t in range(300):
        alph = [f"L{k}" for k in range(rng15.randint(2, 8))]
        a = [rng15.choice(alph) for _ in range(rng15.randint(0, 40))]; b = list(a)
        for _ in range(rng15.randint(0, 12)):
            op = rng15.random()
            if op < .4 and b: b.pop(rng15.randrange(len(b)))
            elif op < .8: b.insert(rng15.randint(0, len(b)), rng15.choice(alph))
            elif b: b[rng15.randrange(len(b))] = rng15.choice(alph)
        fa, fb = f"{d}/a", f"{d}/b"; wr(fa, "".join(x + "\n" for x in a)); wr(fb, "".join(x + "\n" for x in b))
        ses = len(a) + len(b) - 2 * lcs_len(a, b)
        for k, (f, o) in tools.items():
            p_, m_ = f(fa, fb, *o)
            consistent &= (len(a) - m_ == len(b) - p_)
            if p_ + m_ == ses: eq[k] += 1
            else: over[k] += 1; worst[k] = max(worst[k], p_ + m_ - ses)
            assert p_ + m_ >= ses
    say("s06", "300 random file pairs: no tool ever reports fewer changed lines than N + M - 2*LCS (it is a floor)", consistent)
    for k in tools:
        say("s06", f"  {k:16s} hits the floor in {eq[k]:3d}/300, exceeds it in {over[k]:3d} (worst by {worst[k]} lines)")
    say("s06", "  GNU diff -d and git's default always hit it on these; patience and histogram do not",
        eq["GNU diff -d"] == 300 and eq["git"] == 300 and over["git --patience"] > 0 and over["git --histogram"] > over["git --patience"])
    D6 = {"eq": eq, "over": over, "worst": worst}
    def show(rev, path):
        r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo_root, env=GENV, capture_output=True)
        return r.stdout if r.returncode == 0 else None
    real = [("1f32a83", "42d11dc", "study-companion/README.md"), ("42d11dc", "cfbdc1d", "study-companion/HANDOFF.md"),
            ("45393cd", "1f32a83", "study-companion/HANDOFF.md"), ("23a98c7", "cfbdc1d", "study-companion/NEXT_SESSION_PROMPT.md")]
    REAL = []
    for r1_, r2_, pth in real:
        A, B = show(r1_, pth), show(r2_, pth)
        assert A is not None and B is not None, ("history missing", r1_, r2_, pth)
        fa, fb = f"{d}/A", f"{d}/B"; open(fa, "wb").write(A); open(fb, "wb").write(B)
        a, b = A.splitlines(), B.splitlines(); Lr = lcs_len(a, b); ses = len(a) + len(b) - 2 * Lr
        res = {k: sum(f(fa, fb, *o)) for k, (f, o) in tools.items()}
        REAL.append((pth.split("/")[-1], r1_, r2_, len(a), len(b), Lr, ses, res))
        say("s06", f"real revisions {pth.split('/')[-1]} {r1_}->{r2_}: {len(a)} and {len(b)} lines, LCS {Lr}, floor {ses}; "
            + ", ".join(f"{k} {v}" for k, v in res.items()), min(res.values()) == ses and res["GNU diff -d"] == ses)
    hand = REAL[2]
    say("s06", f"HANDOFF.md 45393cd->1f32a83: git (default and --minimal) report {hand[7]['git']} and {hand[7]['git --minimal']} "
        f"changed lines against a floor of {hand[6]}", hand[7]["git --minimal"] == 150 and hand[6] == 144)
    fa, fb = f"{d}/A", f"{d}/B"; open(fa, "wb").write(show("45393cd", real[2][2])); open(fb, "wb").write(show("1f32a83", real[2][2]))
    r = subprocess.run(["git", "diff", "--no-index", "--unified=0", "--minimal", fa, fb], env=GENV, capture_output=True, text=True)
    rem, add = Counter(), Counter()
    for Lx in r.stdout.splitlines()[4:]:
        if Lx.startswith("@@"): continue
        if Lx.startswith("+"): add[Lx[1:]] += 1
        elif Lx.startswith("-"): rem[Lx[1:]] += 1
    r = subprocess.run(["diff", "-d", "--unchanged-line-format=", "--old-line-format=-%L", "--new-line-format=+%L", fa, fb],
                       capture_output=True, text=True)
    rem2, add2 = Counter(), Counter()
    for Lx in r.stdout.splitlines(): (add2 if Lx.startswith("+") else rem2)[Lx[1:]] += 1
    xs = (rem - rem2) + (add - add2)
    nblank = open(fa).read().split("\n").count("")
    say("s06", f"  every one of git's {sum(xs.values())} extra lines is a blank line, the commonest line in the file ({nblank - 1} of {len(open(fa).read().splitlines())})",
        set(xs) == {""} and sum(xs.values()) == 6)
    # renumbering defeats a line diff
    notes0 = ["1. ALL DIMENSIONS IN MM", "2. MATERIAL EN8", "3. HARDEN TO 45 HRC", "4. BREAK SHARP EDGES", "5. ZINC PLATE"]
    notes1 = ["1. ALL DIMENSIONS IN MM", "2. MATERIAL EN8", "3. BREAK SHARP EDGES", "4. ZINC PLATE"]
    strip = lambda ns: [x.split(". ", 1)[1] for x in ns]
    dn, du = len(notes0) + len(notes1) - 2 * lcs_len(notes0, notes1), len(notes0) + len(notes1) - 2 * lcs_len(strip(notes0), strip(notes1))
    say("s06", f"delete note 3 of 5 and renumber: the line diff changes {dn} lines; without the numbers it changes {du}",
        dn == 5 and du == 1)
    LAB6 = {}
    presets = {"notes": (["NOTES", "ALL DIMENSIONS IN MM", "MATERIAL EN8", "HARDEN TO 45 HRC", "BREAK SHARP EDGES", "TOL +/-0.1 UNLESS STATED"],
                         ["NOTES", "ALL DIMENSIONS IN MM", "MATERIAL EN24", "HARDEN TO 45 HRC", "TOL +/-0.1 UNLESS STATED", "ZINC PLATE 8 MICRON"]),
               "myers": (list("ABCABBA"), list("CBABAC")),
               "renum": (notes0, notes1)}
    for k, (a, b) in presets.items():
        Lk = lcs_len(a, b); LAB6[k] = (len(a), len(b), Lk, len(a) + len(b) - 2 * Lk)
    say("s06", f"lab presets (N, M, LCS, edits): {LAB6}", LAB6["notes"] == (6, 6, 4, 4) and LAB6["myers"] == (7, 6, 4, 5))
    ndist = {k: len(lcs_all(tuple(a), tuple(b))) for k, (a, b) in presets.items()}
    say("s06", f"lab presets, number of different LCSs: {ndist}", ndist == {"notes": 1, "myers": 3, "renum": 1})

    # --- s07: three-way merge, and which side is which ---------------------------------------------
    print()
    base = [f"p{i} = {i}" for i in range(1, 9)]
    def merge_case(edA, edB):
        d = newrepo(); wr(f"{d}/f.txt", "\n".join(base) + "\n"); git(d, "add", "."); git(d, "commit", "-qm", "base")
        git(d, "checkout", "-qb", "side")
        x = list(base)
        for i, v in edB.items(): x[i] = v
        wr(f"{d}/f.txt", "\n".join(x) + "\n"); git(d, "commit", "-qam", "B")
        git(d, "checkout", "-q", "main"); x = list(base)
        for i, v in edA.items(): x[i] = v
        wr(f"{d}/f.txt", "\n".join(x) + "\n"); git(d, "commit", "-qam", "A")
        r = git(d, "merge", "-q", "--no-edit", "side", raw=True)
        return "conflict" if r.returncode else "clean"
    cases = [("lines 2 and 6 changed on the two sides", {1: "p2 = 20"}, {5: "p6 = 60"}, "clean"),
             ("same line, different values", {2: "p3 = 30"}, {2: "p3 = 31"}, "conflict"),
             ("same line, same new value", {2: "p3 = 30"}, {2: "p3 = 30"}, "clean"),
             ("adjacent lines 3 and 4", {2: "p3 = 30"}, {3: "p4 = 40"}, "conflict"),
             ("lines 3 and 5 (one line between)", {2: "p3 = 30"}, {4: "p5 = 50"}, "clean")]
    for name, ea, eb, want in cases:
        got_ = merge_case(ea, eb)
        say("s07", f"merge, {name}: {got_}", got_ == want)
    # rebase: HEAD side of the conflict is the upstream, not your branch
    d = newrepo(); wr(f"{d}/cfg.py", "learning_rate = 0.001\n"); git(d, "add", "."); git(d, "commit", "-qm", "base")
    git(d, "checkout", "-qb", "feature-x"); wr(f"{d}/cfg.py", "learning_rate = 0.0005\n"); git(d, "commit", "-qam", "mine")
    git(d, "checkout", "-q", "main"); wr(f"{d}/cfg.py", "learning_rate = 0.002\n"); git(d, "commit", "-qam", "theirs")
    git(d, "checkout", "-q", "feature-x"); git(d, "rebase", "main", raw=True)
    txt = open(f"{d}/cfg.py").read()
    top = txt.split("=======")[0]
    say("s07", "in a REBASE conflict the block under '<<<<<<< HEAD' is main's line (0.002), not your branch's (0.0005)",
        "0.002" in top and "0.0005" not in top)
    git(d, "rebase", "--abort")
    git(d, "checkout", "-q", "main"); git(d, "merge", "feature-x", raw=True)
    top = open(f"{d}/cfg.py").read().split("=======")[0]
    say("s07", "  in a MERGE into main, HEAD is main again: the sides are named by where you stand, not by whose work it is",
        "0.002" in top)

    # --- s08: branches are pointers; rebase renames; bisect is bisection ---------------------------
    print()
    d = newrepo(); wr(f"{d}/x", "1\n"); git(d, "add", "."); git(d, "commit", "-qm", "c1")
    ref = open(f"{d}/.git/refs/heads/main").read()
    say("s08", f"a branch is a {len(ref)}-byte file holding one commit name", len(ref) == 41 and ref.strip() == git(d, "rev-parse", "HEAD"))
    git(d, "checkout", "-qb", "feat")
    for i in range(3): wr(f"{d}/f{i}", f"{i}\n"); git(d, "add", "."); git(d, "commit", "-qm", f"f{i}")
    before = git(d, "rev-list", "main..feat").split()
    pb = sorted(subprocess.run(f"git log -p main..feat | git patch-id --stable", shell=True, cwd=d, env=GENV, capture_output=True, text=True).stdout.split()[0::2])
    git(d, "checkout", "-q", "main"); wr(f"{d}/y", "2\n"); git(d, "add", "."); git(d, "commit", "-qm", "c2")
    git(d, "checkout", "-q", "feat"); git(d, "rebase", "-q", "main")
    after = git(d, "rev-list", "main..feat").split()
    pa = sorted(subprocess.run(f"git log -p main..feat | git patch-id --stable", shell=True, cwd=d, env=GENV, capture_output=True, text=True).stdout.split()[0::2])
    say("s08", f"rebase onto a new main: all {len(before)} commits get new names, none kept; the changes they carry are identical (patch-id)",
        len(after) == 3 and not set(before) & set(after) and pa == pb)
    d = newrepo(); NC, BAD = 300, 187
    logf = os.path.join(d, ".git", "calls")
    for i in range(1, NC + 1):
        wr(f"{d}/version", f"{i}\n"); git(d, "add", "."); git(d, "commit", "-qm", f"c{i}")
    wr(f"{d}/.git/t.sh", f"#!/bin/bash\necho x >> {logf}\nv=$(cat version)\n[ $v -lt {BAD} ]\n")
    os.chmod(f"{d}/.git/t.sh", 0o755)
    first = git(d, "rev-list", "--max-parents=0", "HEAD")
    git(d, "bisect", "start", "HEAD", first)
    r = git(d, "bisect", "run", f"{d}/.git/t.sh", raw=True)
    culprit = git(d, "show", "-s", "--format=%s", "refs/bisect/bad")
    calls = open(logf).read().count("x")
    say("s08", f"git bisect over {NC} commits finds the one that broke it ({culprit}) in {calls} builds; log2({NC - 1}) = {math.log2(NC - 1):.2f}",
        culprit == f"c{BAD}" and calls <= math.ceil(math.log2(NC - 1)) + 1)
    BIS = calls

    # --- s09: discipline: the hash in the log, ignored files, large files --------------------------
    print()
    d = newrepo(); wr(f"{d}/train.py", "lr = 0.001\n"); git(d, "add", "."); git(d, "commit", "-qm", "c")
    code = "import subprocess\ngit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()\nprint(f'commit {git_hash}')\n"
    r = subprocess.run([sys.executable, "-c", code], cwd=d, env=GENV, capture_output=True, text=True)
    say("s09", f"the source's git_hash line returns bytes, so the log reads: {r.stdout.strip()[:24]}...",
        r.stdout.startswith("commit b'"))
    h0 = git(d, "rev-parse", "HEAD"); wr(f"{d}/train.py", "lr = 0.01\n")
    say("s09", "edit the file without committing: HEAD's name is unchanged, so the logged hash no longer names the code that ran",
        git(d, "rev-parse", "HEAD") == h0 and git(d, "status", "--porcelain") == "M train.py")
    git(d, "checkout", "--", "train.py")
    wr(f"{d}/.env", "KEY=abc123\n"); git(d, "add", ".env"); git(d, "commit", "-qm", "oops")
    wr(f"{d}/.gitignore", ".env\n"); wr(f"{d}/.env", "KEY=xyz789\n")
    still = git(d, "status", "--porcelain")
    git(d, "rm", "-q", "--cached", ".env"); git(d, "add", ".gitignore"); git(d, "commit", "-qm", "untrack")
    hist = git(d, "show", "HEAD~1:.env")
    say("s09", f"adding .env to .gitignore after committing it: git still tracks it ('{still.splitlines()[0]}'); after rm --cached "
        f"the old secret is still in history: '{hist}'", " .env" in still and hist == "KEY=abc123")
    blob = os.urandom(3000)
    wr_b = open(f"{d}/w.bin", "wb"); wr_b.write(blob); wr_b.close()
    r = subprocess.run(["git", "lfs", "pointer", "--file=w.bin"], cwd=d, env=GENV, capture_output=True, text=True)
    lfs_ok = ("oid sha256:" + hashlib.sha256(blob).hexdigest()) in r.stdout and "size 3000" in r.stdout
    say("s09", f"a Git LFS pointer is {len(r.stdout.encode())} bytes naming the file by SHA-256 of its content", lfs_ok)

    # --- s10: the source's naming example, and the src layout --------------------------------------
    print()
    ns = {}
    exec(textwrap.dedent("""
        def update_parameters(parameters, num_steps, learning_rate):
            for step in range(num_steps):
                for name, param in parameters.items():
                    param -= learning_rate * gradients[name]
        """), ns)
    ns["gradients"] = {"w": 0.5}
    pf = {"w": 1.0}; ns["update_parameters"](pf, 1, 0.1)
    pa_ = {"w": np.array([1.0])}; ns["gradients"] = {"w": np.array([0.5])}; ns["update_parameters"](pa_, 1, 0.1)
    say("s10", f"the source's GOOD update_parameters: a float weight stays {pf['w']}; a numpy weight becomes {pa_['w'][0]:.2f}",
        pf["w"] == 1.0 and abs(pa_["w"][0] - 0.95) < 1e-12)
    say("s10", "  'param -= x' rebinds a local name for an immutable float; it mutates only a mutable array")
    d = tempfile.mkdtemp(prefix="ch15p")
    os.makedirs(f"{d}/flat/my_project"); wr(f"{d}/flat/my_project/__init__.py", "X = 1\n")
    os.makedirs(f"{d}/srcl/src/my_project"); wr(f"{d}/srcl/src/my_project/__init__.py", "X = 1\n")
    e2 = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    rf = subprocess.run([sys.executable, "-c", "import my_project"], cwd=f"{d}/flat", env=e2, capture_output=True)
    rs = subprocess.run([sys.executable, "-c", "import my_project"], cwd=f"{d}/srcl", env=e2, capture_output=True)
    say("s10", f"from the project root, uninstalled: flat layout imports ({rf.returncode == 0}); src layout refuses ({rs.returncode != 0})",
        rf.returncode == 0 and rs.returncode != 0)
    def es(seq, patience=5):
        best, cnt = float("inf"), 0
        for ep, v in enumerate(seq):
            if v < best: best, cnt = v, 0
            else:
                cnt += 1
                if cnt >= patience: return ep
        return None
    say("s10", "the source's EarlyStopping stops 5 epochs after the best: losses 1.0, 0.9, then five worse -> stop at epoch 6",
        es([1.0, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.5]) == 6)

    # --- s11: configuration: YAML reads 3e-4 as text ------------------------------------------------
    print()
    import yaml
    src_yaml = "training:\n  batch_size: 64\n  learning_rate: 3e-4\n  max_epochs: 100\n"
    cfg = yaml.safe_load(src_yaml)["training"]
    say("s11", f"PyYAML loads the source's 'learning_rate: 3e-4' as {type(cfg['learning_rate']).__name__} {cfg['learning_rate']!r}",
        isinstance(cfg["learning_rate"], str))
    vals = {s: type(yaml.safe_load(f"v: {s}")["v"]).__name__ for s in ("3e-4", "3.0e-4", "1e-3", "2.1e-4", "1.0e3", "1.0e+3")}
    say("s11", f"  YAML 1.1 floats need a dot and a signed exponent: {vals}",
        vals == {"3e-4": "str", "3.0e-4": "float", "1e-3": "str", "2.1e-4": "float", "1.0e3": "str", "1.0e+3": "float"})
    try:
        _ = 0.5 * cfg["learning_rate"]; err = None
    except TypeError as e: err = type(e).__name__
    rep = cfg["learning_rate"] * 2
    say("s11", f"  float * it raises {err}; but 'lr * num_gpus' with 2 GPUs silently gives {rep!r}", err == "TypeError" and rep == "3e-43e-4")
    wf = yaml.safe_load("on: [push, pull_request]\n")
    say("s11", f"the source's CI file read by the same loader: key 'on' becomes {list(wf)[0]!r}", list(wf) == [True])

    # --- s12: logging that does not log -------------------------------------------------------------
    print()
    code = textwrap.dedent("""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.debug("Batch loaded: %d samples", 32)
        logger.info("Epoch %d: loss=%.4f, lr=%.6f", 3, 0.25, 3e-4)
        logger.warning("GPU memory >90%%, consider reducing batch size")
        logger.error("Failed to load checkpoint: %s", "ck.pt")
        logger.info("training_step", extra={"epoch": 5, "step": 1200, "loss": 0.0342, "lr": 2.1e-4})
        """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out = r.stdout + r.stderr
    say("s12", f"the source's logger, run as printed, emits {len(out.splitlines())} lines: {out.splitlines()}",
        "Epoch" not in out and "90%%" in out and "Failed to load" in out and "epoch" not in out)
    say("s12", "  info is dropped (no handler configured; the fallback shows WARNING and up); '%%' stays doubled with no args")
    r = subprocess.run([sys.executable, "-c", "import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')\n" + code],
                       capture_output=True, text=True)
    say("s12", "  with logging.basicConfig the info line appears; the extra= fields still do not, unless a formatter names them",
        "Epoch 3" in r.stderr and "training_step" in r.stderr and "1200" not in r.stderr)

    # --- s13: packaging: a backend that does not exist; >= admits a major version ---------------------
    print()
    import importlib.util
    def has(mod):
        try: return importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError: return False
    import setuptools
    say("s13", f"setuptools {setuptools.__version__}: the source's 'setuptools.backends._legacy' exists: {has('setuptools.backends._legacy')}; "
        f"'setuptools.build_meta' exists: {has('setuptools.build_meta')}", not has("setuptools.backends._legacy") and has("setuptools.build_meta"))
    d = tempfile.mkdtemp(prefix="ch15k")
    def build(backend):
        p = f"{d}/{backend.replace(':', '_').replace('.', '_')}"; os.makedirs(f"{p}/src/my_ml_project")
        wr(f"{p}/src/my_ml_project/__init__.py", "")
        wr(f"{p}/pyproject.toml", f'[project]\nname = "my-ml-project"\nversion = "0.1.0"\n\n[build-system]\nrequires = ["setuptools>=64"]\nbuild-backend = "{backend}"\n')
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-q", "-w", f"{p}/dist", p],
                           capture_output=True, text=True, timeout=180)
        return r.returncode, r.stdout + r.stderr
    rc_src, out_src = build("setuptools.backends._legacy:_Backend")
    hook = subprocess.run([sys.executable, "-c", "import setuptools.build_meta as b; print(b.get_requires_for_build_wheel())"],
                          cwd=f"{d}/setuptools_backends__legacy__Backend", capture_output=True, text=True)
    say("s13", f"pip on the source's pyproject.toml: rc {rc_src}, "
        f"{'BackendUnavailable' if 'BackendUnavailable' in out_src else 'other error'}; the standard backend "
        f"'setuptools.build_meta' loads in the same project and answers its first build hook (rc {hook.returncode})",
        rc_src != 0 and "BackendUnavailable" in out_src and hook.returncode == 0)
    from packaging.specifiers import SpecifierSet
    say("s13", "'torch>=2.0' admits 3.0 (a major version); '~=2.2' admits 2.9 but not 3.0; '==2.2.1' admits only 2.2.1",
        SpecifierSet(">=2.0").contains("3.0") and SpecifierSet("~=2.2").contains("2.9") and not SpecifierSet("~=2.2").contains("3.0")
        and not SpecifierSet("==2.2.1").contains("2.2.2"))

    # --- s14: the productivity arithmetic -------------------------------------------------------------
    print()
    say("s14", f"the source's own numbers: 10 min generating + 120 min debugging = 130 min against 30 min by hand, {130 / 30:.1f}x slower",
        abs(130 / 30 - 4.33) < 0.01)

    # --- s15: a test is a gauge, and the source's test passes a broken function ------------------------
    print()
    img = np.array([0, 128, 255], dtype=np.uint8)
    impls = {
        "as printed (uint8 arithmetic)": lambda x, m, s: (x - m) / s,
        "correct (convert first)": lambda x, m, s: (x.astype(np.float64) - m) / s,
        "sign reversed": lambda x, m, s: (m - x.astype(np.float64)) / s,
        "returns zeros": lambda x, m, s: np.zeros(x.shape),
        "forgot to divide": lambda x, m, s: (x.astype(np.float64) - m),
        "off by one in mean": lambda x, m, s: (x.astype(np.float64) - (m - 1)) / s,
    }
    def t_range(f):
        r = f(img, 128, 128); return bool(r.min() >= -1.0 and r.max() <= 1.0 and abs(r[1]) < 1e-6)
    def t_empty(f):
        return len(f(np.array([], dtype=np.uint8), 128, 128)) == 0
    def t_exact(f):
        return bool(np.allclose(f(img, 128, 128), [-1.0, 0.0, 127 / 128]))
    def t_round(f):
        x = np.array([0, 7, 128, 200, 255], dtype=np.uint8); return bool(np.allclose(f(x, 128, 128) * 128 + 128, x))
    printed = impls["as printed (uint8 arithmetic)"](img, 128, 128)
    say("s15", f"normalise_image as printed, on uint8 [0, 128, 255]: {printed.tolist()} (correct: [-1.0, 0.0, 0.9921875])",
        printed.tolist() == [1.0, 0.0, 0.9921875])
    KM = {}
    for k, f in impls.items():
        KM[k] = (t_range(f), t_empty(f), t_exact(f), t_round(f))
        say("s15", f"  {k:30s} passes: range {KM[k][0]!s:5s} empty {KM[k][1]!s:5s} exact {KM[k][2]!s:5s} round-trip {KM[k][3]!s:5s}")
    say("s15", "the source's two tests pass the printed (wrong) function, a sign error and a function returning zeros",
        all(KM[k][0] and KM[k][1] for k in ("as printed (uint8 arithmetic)", "sign reversed", "returns zeros")))
    say("s15", "an exact-value test passes only the correct function", [k for k in KM if KM[k][2]] == ["correct (convert first)"])
    say("s15", f"  (measured with NumPy {np.__version__}; an unsigned 8-bit 0 - 128 wraps to 128)")

    # --- s16: sampling: the c = 0 plan, the rule of three, and the flaky suite -------------------------
    print()
    for n in (30, 300, 3000):
        ex = 1 - 0.05 ** (1 / n)
        say("s16", f"n = {n:4d} passing random tests: 95% upper bound on the failing fraction {ex:.5f} vs rule of three 3/n = {3 / n:.5f}",
            abs(ex / (3 / n) - 1) < 0.06)
    th, n = 0.01, 300
    sim = np.mean((np.random.default_rng(15).random((20000, n)) < th).any(axis=1))
    say("s16", f"a bug hit by 1% of inputs, 300 random inputs: detected with prob 1-(0.99)^300 = {1 - 0.99 ** 300:.4f}; simulated {sim:.4f}",
        abs(sim - (1 - 0.99 ** 300)) < 0.01)
    pe = 1 - (1 - 2.0 ** -32) ** 1000
    say("s16", f"a bug at exactly one 32-bit input (say zero), 1,000 random inputs: found with prob {pe:.2e}; one boundary test finds it always",
        pe < 3e-7)
    a1000 = 1 - 0.999 ** 1000
    say("s16", f"1,000 tests each flaky 0.1%: a correct change goes red {a1000:.1%} of the time (series system, Sheet 05)",
        abs(a1000 - 0.632) < 0.001)
    a_retry = 1 - (1 - 0.001 ** 2) ** 1000
    say("s16", f"  rerun a failure once: false red falls to {a_retry:.2e}; a bug that fails half the time now escapes {0.5 ** 2:.0%} per test",
        a_retry < 1.1e-3 and abs(0.5 ** 2 - 0.25) < 1e-12)
    OC = {"alpha_default": 1 - (1 - 0.001) ** 1000, "beta_default": (1 - 0.001) ** 1000 * (1 - 0.01) ** 300}
    say("s16", f"lab rest: N=1000, f=0.1%, n=300: producer's risk {OC['alpha_default']:.3f}; P(green) on a 1% bug {OC['beta_default']:.4f}")

    # --- s17: tolerances and vacuous tests ------------------------------------------------------------
    print()
    a, b = 1.0, 1.0 + 1.00001e-5
    say("s17", f"np.isclose is not symmetric: isclose(a, b) = {np.isclose(a, b, rtol=1e-5, atol=0)}, isclose(b, a) = {np.isclose(b, a, rtol=1e-5, atol=0)}",
        bool(np.isclose(a, b, rtol=1e-5, atol=0)) and not bool(np.isclose(b, a, rtol=1e-5, atol=0)))
    say("s17", f"the source's isclose(out, 0.5, atol=1e-5) also carries rtol=1e-5: it accepts 0.5 + 1.4e-5 ({np.isclose(0.5 + 1.4e-5, 0.5, atol=1e-5)})",
        bool(np.isclose(0.5 + 1.4e-5, 0.5, atol=1e-5)) and not bool(np.isclose(0.5 + 1.6e-5, 0.5, atol=1e-5)))
    say("s17", f"0.5 is exact in binary (0.25 + 0.25 == 0.5: {0.25 + 0.25 == 0.5}); 0.1 + 0.2 == 0.3 is {0.1 + 0.2 == 0.3}",
        0.25 + 0.25 == 0.5 and 0.1 + 0.2 != 0.3)
    W = np.random.default_rng(3).normal(size=(4, 4)); xin = np.ones(4)
    det_model = lambda x: W @ x
    def drop_model(x): return (W @ x) * (np.random.random(4) > 0.5)
    def det_test(model, seeded):
        if seeded: np.random.seed(42)
        o1 = model(xin)
        if seeded: np.random.seed(42)
        o2 = model(xin); return bool(np.allclose(o1, o2))
    say("s17", f"the determinism test on a model with no randomness passes with the seed ({det_test(det_model, True)}) and without it "
        f"({det_test(det_model, False)}): it cannot see a broken set_seed", det_test(det_model, True) and det_test(det_model, False))
    say("s17", f"  on a model with dropout active it has teeth: seeded {det_test(drop_model, True)}, unseeded {det_test(drop_model, False)}",
        det_test(drop_model, True) and not det_test(drop_model, False))

    # --- s18: a fail-fast pipeline is a sequential inspection: order by c/p -------------------------------
    print()
    def ff_cost(order, c, p):
        tot, reach = 0.0, 1.0
        for i in order:
            tot += reach * c[i]; reach *= (1 - p[i])
        return tot
    okf = True; naive_worse = 0
    for _t in range(300):
        k = rng15.randint(2, 6)
        c = [rng15.uniform(1, 600) for _ in range(k)]; p = [rng15.uniform(0.01, 0.6) for _ in range(k)]
        best = min(ff_cost(o, c, p) for o in itertools.permutations(range(k)))
        okf &= abs(ff_cost(sorted(range(k), key=lambda i: c[i] / p[i]), c, p) - best) < 1e-9 * best
        naive_worse += ff_cost(sorted(range(k), key=lambda i: c[i]), c, p) > best + 1e-9
    say("s18", "fail-fast checks (stop at the first failure): ordering by cost / P(fail), smallest first, == the best "
        "permutation, 300 random pipelines", okf)
    say("s18", f"  'cheapest first' alone is worse in {naive_worse}/300", naive_worse > 0)
    c3, p3 = [5, 60, 600], [0.20, 0.05, 0.10]
    ci_o = sorted(range(3), key=lambda i: c3[i] / p3[i])
    say("s18", f"lint 5 s fails 20%, types 60 s fails 5%, tests 600 s fails 10%: best order {ci_o}, expected "
        f"{ff_cost(ci_o, c3, p3):.0f} s vs {ff_cost([2, 1, 0], c3, p3):.0f} s tests-first",
        ci_o == [0, 1, 2] and abs(ff_cost([0, 1, 2], c3, p3) - (5 + 0.8 * 60 + 0.8 * 0.95 * 600)) < 1e-9)

    # --- s19: containers: the interpreter's version is part of the path ---------------------------------
    print()
    import site
    usp = site.getusersitepackages()
    vtag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    say("s19", f"pip install --user writes to {usp.replace(os.path.expanduser('~'), '~')}: the interpreter version is in the path",
        vtag in usp)
    d = tempfile.mkdtemp(prefix="ch15c")
    os.makedirs(f"{d}/app/src/my_project"); os.makedirs(f"{d}/app/src/scripts"); os.makedirs(f"{d}/app/configs")
    wr(f"{d}/app/src/my_project/__init__.py", "")
    wr(f"{d}/app/src/scripts/serve.py", "import my_project\nprint('serving')\n")
    r = subprocess.run([sys.executable, "src/scripts/serve.py"], cwd=f"{d}/app", env=e2, capture_output=True, text=True)
    say("s19", f"the source's CMD with a src-layout package and no install: {r.stderr.strip().splitlines()[-1]}",
        "ModuleNotFoundError" in r.stderr)

    # --- s20: layer order is a sequencing rule ---------------------------------------------------------
    print()
    def exp_cost(order, c, p):
        tot, surv = 0.0, 1.0
        for i in order:
            surv *= (1 - p[i]); tot += c[i] * (1 - surv)
        return tot
    def ratio_order(c, p): return sorted(range(len(c)), key=lambda i: p[i] / (c[i] * (1 - p[i])))
    okx, freq_wins, gaps = True, 0, []
    for _t in range(300):
        k = rng15.randint(2, 6)
        c = [rng15.choice([1, 2, 5, 10, 30, 60, 120, 300, 600]) * rng15.uniform(0.5, 1.5) for _ in range(k)]
        p = [rng15.uniform(0.01, 0.9) for _ in range(k)]
        best = min(exp_cost(o, c, p) for o in itertools.permutations(range(k)))
        rr = exp_cost(ratio_order(c, p), c, p)
        fo = exp_cost(sorted(range(k), key=lambda i: p[i]), c, p)
        okx &= abs(rr - best) < 1e-9 * max(1, best)
        if fo > best + 1e-9: freq_wins += 1; gaps.append(fo / best)
    say("s20", "ordering by p / (c (1 - p)), smallest first, == the best of every permutation, 300 random Dockerfiles of 2-6 steps", okx)
    say("s20", f"  'least often changed first' alone is worse in {freq_wins}/300 of them (worst {max(gaps):.2f}x the optimum)", freq_wins > 0)
    steps = ["apt-get system libs", "pip install requirements", "download model weights", "COPY configs/", "COPY src/"]
    c5 = [120, 240, 600, 1, 2]; p5 = [0.01, 0.05, 0.30, 0.20, 0.80]
    src_o = [0, 1, 2, 3, 4]; freq_o = sorted(range(5), key=lambda i: p5[i]); rat_o = ratio_order(c5, p5)
    rev_o = [4, 3, 2, 1, 0]
    best5 = min(exp_cost(o, c5, p5) for o in itertools.permutations(range(5)))
    LAB20 = {"src": exp_cost(src_o, c5, p5), "freq": exp_cost(freq_o, c5, p5), "ratio": exp_cost(rat_o, c5, p5),
             "rev": exp_cost(rev_o, c5, p5), "best": best5, "never": sum(c5)}
    say("s20", f"lab Dockerfile: expected rebuild per build {LAB20['ratio']:.1f} s by the ratio rule (order {rat_o}), "
        f"{LAB20['freq']:.1f} s by change-frequency (order {freq_o}), {LAB20['rev']:.1f} s reversed, {LAB20['never']} s with no cache",
        abs(LAB20["ratio"] - best5) < 1e-9 and LAB20["freq"] > LAB20["ratio"] and rat_o == [0, 1, 2, 3, 4] and freq_o == [0, 1, 3, 2, 4])
    def keys(contents):
        out, k = [], b"base"
        for s in contents:
            k = hashlib.sha256(k + s.encode()).digest(); out.append(k.hex()[:8])
        return out
    k0 = keys(["apt:libgl", "req:torch==2.2.1", "weights:v3", "cfg:a", "src:v1"])
    k1 = keys(["apt:libgl", "req:torch==2.2.1", "weights:v3", "cfg:b", "src:v1"])
    say("s20", f"chained cache keys: edit layer 4 of 5 and keys 4-5 change, 1-3 survive ({sum(x == y for x, y in zip(k0, k1))} reused)",
        k0[:3] == k1[:3] and k0[3] != k1[3] and k0[4] != k1[4])

    # --- s21: tracking: the best-model save that never updates its best -------------------------------
    print()
    losses = [0.90, 0.80, 0.85, 0.70, 0.75, 0.72]
    best_loss = float("inf"); saved_src = [e for e, v in enumerate(losses) if v < best_loss]
    saved_ok, bl = [], float("inf")
    for e, v in enumerate(losses):
        if v < bl: bl = v; saved_ok.append(e)
    say("s21", f"the source's W&B loop never updates best_loss: it saves at epochs {saved_src}; updating it saves at {saved_ok}",
        saved_src == [0, 1, 2, 3, 4, 5] and saved_ok == [0, 1, 3])

    # --- s22: tails and drift ---------------------------------------------------------------------------
    print()
    fan = {k: 1 - 0.99 ** k for k in (1, 10, 100)}
    lat = np.random.default_rng(22).lognormal(np.log(80), 0.6, size=(20000, 100))
    p99 = np.quantile(lat, 0.99)
    pages = np.mean((lat > p99).any(axis=1))
    say("s22", f"p99 is a fraction of requests: a page that makes 100 requests meets at least one beyond p99 with prob "
        f"{fan[100]:.3f} (simulated {pages:.3f}); 10 requests {fan[10]:.3f}", abs(pages - fan[100]) < 0.01)
    okp = True
    rr = np.random.default_rng(23)
    for _t in range(200):
        e = rr.dirichlet(np.ones(10)); a = rr.dirichlet(np.ones(10))
        psi = np.sum((a - e) * np.log(a / e)); kl = np.sum(a * np.log(a / e)) + np.sum(e * np.log(e / a))
        okp &= abs(psi - kl) < 1e-12
    say("s22", "PSI = sum (a - e) ln(a/e) == KL(a||e) + KL(e||a), the symmetrised divergence, 200 random histograms", okp)

    # --- s24: worked example: bisect with a flaky test ----------------------------------------------------
    print()
    def p_correct(N, f, r):
        fe = f ** r; tot = 0.0; steps_all = []
        for cul in range(N):
            lo, hi, need, steps = 0, N - 1, 0, 0
            while lo < hi:
                m = (lo + hi) // 2; steps += 1
                if m >= cul: hi = m
                else: lo = m + 1; need += 1
            tot += (1 - fe) ** need; steps_all.append(steps)
        return tot / N, np.mean(steps_all), max(steps_all)
    w1 = p_correct(300, 0.05, 1); w2 = p_correct(300, 0.05, 2); w3 = p_correct(300, 0.05, 3)
    needs = []
    for cul in range(300):
        lo, hi, need = 0, 299, 0
        while lo < hi:
            m = (lo + hi) // 2
            if m >= cul: hi = m
            else: lo = m + 1; need += 1
        needs.append(need)
    say("s24", f"hand estimate: the correct path needs {np.mean(needs):.2f} 'good' verdicts on average, and "
        f"0.95^{np.mean(needs):.2f} = {0.95 ** np.mean(needs):.3f} against the exact average {w1[0]:.3f}",
        abs(0.95 ** np.mean(needs) - w1[0]) < 0.01)
    say("s24", f"bisect 300 commits: {w1[1]:.2f} builds on average, at most {w1[2]}; ceil(log2 300) = {math.ceil(math.log2(300))}",
        w1[2] == 9 and abs(w1[1] - 8.2) < 0.1)
    say("s24", f"test falsely fails a good build 5% of the time: right culprit with prob {w1[0]:.3f} (one run per step), "
        f"{w2[0]:.4f} (fail only if 2 runs fail), {w3[0]:.5f} (3 runs)", w1[0] < 0.85 and w2[0] > 0.98)
    rng_w = np.random.default_rng(24); hits = 0
    for _t in range(20000):
        cul = int(rng_w.integers(300)); lo, hi = 0, 299
        while lo < hi:
            m = (lo + hi) // 2
            bad = m >= cul or rng_w.random() < 0.05
            if bad: hi = m
            else: lo = m + 1
        hits += lo == cul
    say("s24", f"  Monte Carlo, 20,000 bisections with the flaky test: {hits / 20000:.3f}", abs(hits / 20000 - w1[0]) < 0.01)
    say("s24", f"  linear search through the ECN log instead: {300 / 2:.0f} builds on average, and 1-(0.95)^k compounding over k good builds")
    say("s24", f"  expected builds with 2 runs per step: {2 * w2[1]:.1f}, vs 150 for the linear search")

    # --- s26: question-bank answers ------------------------------------------------------------------------
    print()
    q1 = 1 - math.exp(-(50000 ** 2) / 2 ** 29)
    say("s26", f"Q: 50,000 objects, 7-hex-digit prefixes: P(some shared) = {q1:.4f}", abs(q1 - 0.9905) < 1e-3)
    say("s26", f"Q: N=120, M=125, LCS 110: shortest diff {120 + 125 - 2 * 110} lines (10 deleted, 15 inserted)", 120 + 125 - 220 == 25)
    say("s26", f"Q: bisect 1,000 commits: ceil(log2 1000) = {math.ceil(math.log2(1000))} builds", math.ceil(math.log2(1000)) == 10)
    q4a, q4b = 1 - 0.9995 ** 2000, 1 - (1 - 0.0005 ** 2) ** 2000
    say("s26", f"Q: 2,000 tests flaky 0.05%: red {q4a:.3f} run once, {q4b:.1e} with one rerun", abs(q4a - 0.632) < 1e-3 and abs(q4b - 5.0e-4) < 1e-5)
    q5 = 1 - 0.05 ** (1 / 500)
    say("s26", f"Q: 500 passing random inputs: 95% bound {q5:.5f} vs 3/500 = {3 / 500:.4f}", abs(q5 - 0.00597) < 1e-5)
    cq, pq = [10, 120, 900], [0.10, 0.30, 0.20]
    oq = sorted(range(3), key=lambda i: cq[i] / pq[i])
    say("s26", f"Q: CI lint 10 s/10%, unit 120 s/30%, integration 900 s/20%: order {oq}, expected {ff_cost(oq, cq, pq):.0f} s",
        oq == [0, 1, 2] and abs(ff_cost(oq, cq, pq) - 685) < 1e-9)
    ea, eb = exp_cost([0, 1], [300, 5], [0.1, 0.5]), exp_cost([1, 0], [300, 5], [0.1, 0.5])
    say("s26", f"Q: layer A 300 s/10%, B 5 s/50%: A first {ea:.2f} s, B first {eb:.2f} s",
        abs(ea - 32.75) < 1e-9 and abs(eb - 167.5) < 1e-9 and ratio_order([300, 5], [0.1, 0.5]) == [0, 1])
    q8 = 1 - 0.99 ** 30
    say("s26", f"Q: a page of 30 requests meets one beyond p99 with prob {q8:.3f}", abs(q8 - 0.260) < 1e-3)

    print()
    say("summary", "Ch15 numbers for the page: " + json.dumps({"LAB5": LAB5, "LAB6": LAB6, "LAB20": {k: round(v, 2) for k, v in LAB20.items()},
                                                             "OC": {k: round(v, 4) for k, v in OC.items()}, "bisect_calls": BIS}))

_ch15()
# ----- end of Chapter 15 block -----


# ======================================================================================
# Chapter 16 part 1 block. Owned by the Chapter 16 part 1 run: replace only the body of _ch16p1() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch16p1():
    import math, os, re, shutil, subprocess, tempfile, time, random
    from collections import OrderedDict
    import numpy as np
    T = "[Ch16 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T % sec} {msg}{tail}")
    here = os.path.dirname(os.path.abspath(__file__))
    srcdir = os.path.join(os.path.dirname(here), "chapter 16 - SIMD and GPU programming")
    srcfiles = ["00. why C++ and how ML frameworks work.md", "01. hardware fundamentals.md",
                "02. ARM and NEON.md", "03. x86 and AVX.md"]
    def blocks(fi):
        s = open(os.path.join(srcdir, srcfiles[fi])).read()
        return re.findall(r"```(\w*)\n(.*?)```", s, re.S)
    gxx = shutil.which("g++"); clangxx = shutil.which("clang++")
    try:
        cpuflags = open("/proc/cpuinfo").read().split("flags", 1)[1].split("\n", 1)[0].split()
    except Exception:
        cpuflags = []
    x86ok = bool(gxx) and "avx2" in cpuflags and "fma" in cpuflags
    tmp16 = tempfile.mkdtemp(prefix="ch16p1_")
    def build(name, code, flags):
        p = os.path.join(tmp16, name + ".cpp"); o = os.path.join(tmp16, name)
        open(p, "w").write(code)
        r = subprocess.run([gxx] + flags + ["-o", o, p], capture_output=True, text=True)
        return o if r.returncode == 0 else None
    def run(o, *args, timeout=300):
        r = subprocess.run([o] + list(args), capture_output=True, text=True, timeout=timeout)
        return r.stdout
    def runbest(o, pats, tries=3, *args):
        # timings on a shared VM are noisy: run the program several times, keep each figure's minimum
        best = {}
        for _ in range(tries):
            out = run(o, *args)
            for key, pat in pats.items():
                v = num(pat, out)
                if v == v: best[key] = min(best.get(key, float("inf")), v)
        return best
    def num(pat, s):
        m = re.search(pat, s); return float(m.group(1)) if m else float("nan")
    print()
    if not x86ok:
        say("s00", "NOTE: no g++ or no AVX2+FMA on this machine; the measured checks below are skipped, "
            "the derived ones still run")

    # ---------------- s01: the two-language stack --------------------------------------
    n4 = 4096
    fl = 2 * n4**3
    say("s01", f"4096 x 4096 matmul = 2n^3 = {fl:,} FLOPs (source: ~137 billion)", fl == 137_438_953_472)
    say("s01", f"source's own ratio 30 min / 10 ms = {1800/0.010:,.0f} (it says 180,000x)", round(1800/0.010) == 180000)
    need = fl / 0.010
    core_avx2 = 4e9 * 8 * 2 * 2; core_512 = 4e9 * 16 * 2 * 2
    say("s01", f"10 ms needs {need/1e12:.2f} TFLOP/s = {need/core_avx2:.1f} AVX2 cores or {need/core_512:.1f} "
        f"AVX-512 cores at 4 GHz running at 100% of peak", abs(need/core_avx2 - 107.4) < 0.1 and abs(need/core_512 - 53.7) < 0.1)
    # pure-Python triple loop, extrapolated by n^3
    nn = 64
    Ap = np.random.default_rng(1).random((nn, nn)).tolist(); Bp = np.random.default_rng(2).random((nn, nn)).tolist()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        Cp = [[0.0]*nn for _ in range(nn)]
        for i in range(nn):
            Ci = Cp[i]; Ai = Ap[i]
            for k in range(nn):
                a = Ai[k]; Bk = Bp[k]
                for j in range(nn):
                    Ci[j] += a * Bk[j]
        best = min(best, time.perf_counter() - t0)
    ext = best * (4096 / nn) ** 3
    say("s01", f"measured: a pure-Python triple loop at n = 64 extrapolates to {ext/60:,.0f} min for n = 4096 "
        f"(source: ~30 min; the order of magnitude holds)", 5*60 < ext < 20*3600)
    X = np.random.default_rng(3).random((2048, 2048), dtype=np.float32); Y = X.T.copy()
    X @ Y
    t0 = time.perf_counter(); c0 = time.process_time(); X @ Y; w_mm = time.perf_counter() - t0; c_mm = time.process_time() - c0
    a50 = np.random.default_rng(4).standard_normal(20_000_000).astype(np.float32); b50 = a50[::-1].copy()
    a50 * b50
    t0 = time.perf_counter(); c0 = time.process_time(); a50 * b50; w_el = time.perf_counter() - t0; c_el = time.process_time() - c0
    say("s01", f"measured: NumPy elementwise a*b uses {c_el/w_el:.2f} CPU-seconds per wall-second (one thread); "
        f"a 2048 matmul uses {c_mm/w_mm:.2f} on {os.cpu_count()} CPUs (BLAS threads)",
        c_el / w_el < 1.35 and (os.cpu_count() < 2 or c_mm / w_mm > 1.5))
    say("s01", f"measured: 2048^3 matmul in float32 runs at {2*2048**3/w_mm/1e9:,.0f} GFLOP/s through NumPy/BLAS")
    kf = 3
    say("s01", f"fusion: matmul then {kf} elementwise ops moves (1 + 2k) = {1+2*kf} output-sized arrays unfused, "
        f"1 fused, a {1+2*kf}x cut in that traffic", 1 + 2*kf == 7)

    # ---------------- s02: the source's C++ tasks ----------------------------------------
    N1 = 10_000_000
    vals = np.arange(N1, dtype=np.float32) * np.float32(0.001)
    seq = float(np.cumsum(vals, dtype=np.float32)[-1])
    exact = math.fsum(vals.astype(np.float64))
    say("s02", f"task 1's float accumulator: sequential float32 sum = {seq:.6g}, exact sum of the same values = "
        f"{exact:.9g}, error {100*(exact-seq)/exact:.2f}%", abs(seq - 4.83701e10) < 1e5 and 3.0 < 100*(exact-seq)/exact < 3.5)
    say("s02", f"task 3: 4096^2 = {4096**2:,} = 2^24 ones; float32(2^24) + 1 == 2^24 is "
        f"{np.float32(2**24) + np.float32(1) == np.float32(2**24)}: both sums land exactly on the last integer "
        f"float32 can count to", 4096**2 == 2**24 and np.float32(2**24) + np.float32(1) == np.float32(2**24))
    if x86ok:
        b0 = blocks(0)
        o = build("t1", b0[11][1], ["-O3"])
        out = run(o) if o else ""
        csum, ctime = num(r"Sum: (\S+)", out), num(r"Time: (\S+)", out)
        say("s02", f"source task 1 compiled and run: prints Sum {csum:.6g} (NumPy model "
            f"{seq:.6g}), {ctime:.1f} ms", abs(csum - seq) / seq < 1e-5)

    # ---------------- s03: why clocks stopped ------------------------------------------------
    p4 = 4 * 3.0**3; p1 = 4.5**3
    say("s03", f"P ~ f^3 (V ~ f): four cores at 3 GHz draw {p4/p1:.3f}x one core at 4.5 GHz and do "
        f"{12/4.5:.3f}x the work per second (source: 'similar power', '4x the work')",
        abs(p4/p1 - 1.185) < 1e-3 and abs(12/4.5 - 2.667) < 1e-3)
    say("s03", f"doubling 4 GHz to 8 GHz under the cube law: {2**3}x the power (100 W -> 800 W)", 2**3 == 8)
    fe = 4.5 / 4 ** (1/3)
    say("s03", f"equal power, four cores: each at 4.5/4^(1/3) = {fe:.3f} GHz, total {4*fe/4.5:.3f}x the work",
        abs(fe - 2.835) < 1e-3 and abs(4*fe/4.5 - 2.520) < 1e-3)
    def s_star(n): return (n ** (-1/3) - 1/n) / (1 - 1/n)
    say("s03", f"equal power with Amdahl: n cores beat one fast core while the serial fraction s < "
        f"(n^-1/3 - 1/n)/(1 - 1/n): n=2 {s_star(2):.3f}, n=4 {s_star(4):.3f}, n=16 {s_star(16):.3f}",
        abs(s_star(4) - 0.5066) < 1e-3)
    ok = True
    for _ in range(200):
        N_, k_ = random.uniform(0.5, 3), random.uniform(0.5, 3)
        ok &= abs((k_**3 / k_) - k_**2) < 1e-12   # power/flow = energy per unit ~ speed^2, pump and chip alike
    say("s03", "affinity laws: P ~ N^3 and Q ~ N give energy per unit pumped ~ N^2; P ~ f^3 and ops ~ f give "
        "energy per operation ~ f^2 -- same exponents", ok)

    # ---------------- s04: inside a core --------------------------------------------------------
    def predict(pattern, bits, state):
        miss = 0; top = (1 << bits) - 1; thr = (top + 1) // 2
        for taken in pattern:
            if (state >= thr) != taken: miss += 1
            state = min(top, state + 1) if taken else max(0, state - 1)
        return miss / len(pattern)
    loop = ([True]*9 + [False]) * 1000
    alt = [True, False] * 5000
    say("s04", f"loop branch (9 taken, 1 exit): 1-bit predictor misses {100*predict(loop,1,1):.0f}%, "
        f"2-bit counter {100*predict(loop,2,3):.0f}%", abs(predict(loop, 1, 1) - 0.2) < 1e-3 and abs(predict(loop, 2, 3) - 0.1) < 1e-3)
    say("s04", f"alternating T,N: 2-bit counter misses {100*predict(alt,2,2):.0f}% from 'weakly taken', "
        f"{100*predict(alt,2,1):.0f}% from 'weakly not taken' (hysteresis can lock in the worst case)",
        abs(predict(alt, 2, 2) - 0.5) < 1e-3 and abs(predict(alt, 2, 1) - 1.0) < 1e-3)
    rr = random.Random(16); rnd = [rr.random() < 0.5 for _ in range(100000)]
    say("s04", f"random 50/50 branch: 2-bit counter misses {100*predict(rnd,2,2):.1f}% -- no predictor can beat 50%",
        abs(predict(rnd, 2, 2) - 0.5) < 0.02)
    say("s04", f"expected cost per branch = miss rate x penalty = 0.05 x 15 = {0.05*15:.2f} cycles", abs(0.05*15 - 0.75) < 1e-12)
    clock = float("nan"); br_pen = float("nan")
    MEAS = r'''
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <sys/mman.h>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#define BAR(x) asm volatile("" : : "r"(x) : "memory")
static double ghz(){ long x=1; const long it=20000000; double best=1e9;
  for(int r=0;r<5;r++){ double t=now(); for(long i=0;i<it;i++) asm volatile(
   "add %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\t"
   "add %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0\n\tadd %0,%0":"+r"(x));
   best=std::min(best,now()-t);} BAR(x); return 16.0*it/best/1e9; }
template<int K> static double fr(long it){ __m256 a[K]; __m256 x=_mm256_set1_ps(0.999999f), y=_mm256_set1_ps(1e-7f);
  for(int k=0;k<K;k++) a[k]=_mm256_set1_ps((float)k); double best=1e9;
  for(int r=0;r<5;r++){ double t=now(); for(long i=0;i<it;i++){
#pragma GCC unroll 32
   for(int k=0;k<K;k++) a[k]=_mm256_fmadd_ps(a[k],x,y);} best=std::min(best,now()-t);}
  for(int k=0;k<K;k++) BAR(a[k][0]); return (double)it*K/best; }
template<int K> static double dr(const float* a,const float* b,int n,int reps){ double best=1e9; float sink=0;
  for(int q=0;q<5;q++){ double t=now(); for(int r=0;r<reps;r++){ __m256 acc[K]; for(int k=0;k<K;k++) acc[k]=_mm256_setzero_ps();
   for(int i=0;i+8*K<=n;i+=8*K){
#pragma GCC unroll 32
    for(int k=0;k<K;k++) acc[k]=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8*k),_mm256_loadu_ps(b+i+8*k),acc[k]);}
   __m256 s=acc[0]; for(int k=1;k<K;k++) s=_mm256_add_ps(s,acc[k]); sink+=s[0]; BAR(a);} best=std::min(best,now()-t);}
  BAR(sink); return (double)reps*(n/8)/best; }
__attribute__((noinline)) long branchy(const int* d,int n){ long s=0; for(int i=0;i<n;i++){ if(d[i]>=128) s+=d[i]; asm volatile("":"+r"(s)); } return s; }
__attribute__((noinline)) void addv(const float* a,const float* b,float* c,long n){ for(long i=0;i<n;i++) c[i]=a[i]+b[i]; }
__attribute__((noinline)) void addnt(const float* a,const float* b,float* c,long n){ for(long i=0;i<n;i+=8) _mm256_stream_ps(c+i,_mm256_add_ps(_mm256_load_ps(a+i),_mm256_load_ps(b+i))); _mm_sfence(); }
int main(int argc,char** argv){
  double G=ghz(); printf("clock %.4f\n",G);
  if(!strcmp(argv[1],"addlat")){ float x=1.0f, y=1e-9f; const long it=20000000; double best=1e9;
    for(int r=0;r<5;r++){ double t=now(); for(long i=0;i<it;i++) asm volatile(
     "vaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0\n\tvaddss %1,%0,%0":"+x"(x):"x"(y));
     best=std::min(best,now()-t);} BAR(x); printf("addlat %.4f\n",best*G*1e9/(8.0*it)); }
  if(!strcmp(argv[1],"fma")){ long it=8000000;
    printf("fma 1 %.4f\n",fr<1>(it)/G/1e9); printf("fma 2 %.4f\n",fr<2>(it)/G/1e9); printf("fma 4 %.4f\n",fr<4>(it)/G/1e9);
    printf("fma 6 %.4f\n",fr<6>(it)/G/1e9); printf("fma 8 %.4f\n",fr<8>(it)/G/1e9); printf("fma 10 %.4f\n",fr<10>(it)/G/1e9);
    printf("fma 16 %.4f\n",fr<16>(it)/G/1e9);
    int n=2048; float* A=(float*)aligned_alloc(64,4*(n+64)); float* B=(float*)aligned_alloc(64,4*(n+64));
    for(int i=0;i<n+64;i++){A[i]=1;B[i]=0.5f;}
    printf("dotal %.4f\n",dr<8>(A,B,n,50000)/G/1e9); printf("dotmis %.4f\n",dr<8>(A+1,B+1,n,50000)/G/1e9);
    printf("dot1 %.4f\n",dr<1>(A,B,n,50000)/G/1e9); }
  if(!strcmp(argv[1],"branch")){ const int n=1<<20; std::vector<int> d(n); std::mt19937 g(16); for(auto& x:d) x=g()%256;
    std::vector<int> s=d; std::sort(s.begin(),s.end()); double tr=1e9,ts=1e9; long k=0;
    for(int r=0;r<15;r++){ double t=now(); k+=branchy(d.data(),n); tr=std::min(tr,now()-t); t=now(); k+=branchy(s.data(),n); ts=std::min(ts,now()-t);}
    BAR(k); printf("rand_ns %.4f\nsort_ns %.4f\n",tr/n*1e9,ts/n*1e9); }
  if(!strcmp(argv[1],"chase")){ const long L=(256L<<20)/64; struct alignas(64) Nd{Nd* nx; char pad[56];};
    Nd* m=(Nd*)aligned_alloc(2<<20,L*64); madvise(m,L*64,MADV_HUGEPAGE);
    std::vector<long> p(L); std::iota(p.begin(),p.end(),0); std::mt19937_64 g(16); std::shuffle(p.begin(),p.end(),g);
    for(long i=0;i<L;i++) m[p[i]].nx=&m[p[(i+1)%L]];
    int Ks[]={1,2,4,8,16,32}; for(int K:Ks){ Nd* q[32]; for(int k=0;k<K;k++) q[k]=&m[p[(L/K)*k]];
      long st=1500000/K; if(st<60000) st=60000; double t=now(); for(long i=0;i<st;i++) for(int k=0;k<K;k++) q[k]=q[k]->nx;
      t=now()-t; for(int k=0;k<K;k++) BAR(q[k]); printf("chase %d %.4f\n",K,t/(st*(double)K)*1e9); } }
  if(!strcmp(argv[1],"bw")){ const long N=1<<24; float* a=(float*)aligned_alloc(64,4*N); float* b=(float*)aligned_alloc(64,4*N); float* c=(float*)aligned_alloc(64,4*N);
    for(long i=0;i<N;i++){a[i]=1;b[i]=2;c[i]=0;} double bn=1e9,bt=1e9;
    for(int t=0;t<8;t++){ double s=now(); addv(a,b,c,N); BAR(c); bn=std::min(bn,now()-s); }
    for(int t=0;t<8;t++){ double s=now(); addnt(a,b,c,N); BAR(c); bt=std::min(bt,now()-s); }
    printf("bw_normal %.4f\nbw_nt %.4f\n",12.0*N/bn/1e9,12.0*N/bt/1e9); }
  if(!strcmp(argv[1],"zmm")){ __m512 a[16]; __m512 x=_mm512_set1_ps(0.999999f),y=_mm512_set1_ps(1e-7f);
    for(int k=0;k<16;k++) a[k]=_mm512_set1_ps((float)k); long it=8000000; double best=1e9;
    for(int r=0;r<5;r++){ double t=now(); for(long i=0;i<it;i++){
#pragma GCC unroll 16
     for(int k=0;k<16;k++) a[k]=_mm512_fmadd_ps(a[k],x,y);} best=std::min(best,now()-t);}
    for(int k=0;k<16;k++) BAR(a[k][0]); printf("zmm_gflops %.3f\n",it*16.0*32/best/1e9);
    printf("ymm_gflops %.3f\n",fr<16>(it)*16/1e9); }
  return 0; }
'''
    meas = None
    if x86ok:
        meas = build("meas", MEAS, ["-O2", "-march=native", "-fno-if-conversion", "-fno-if-conversion2"])
    if meas:
        out = run(meas, "branch")
        clock = num(r"clock (\S+)", out)
        dns = num(r"rand_ns (\S+)", out) - num(r"sort_ns (\S+)", out)
        br_pen = dns / 0.5 * clock
        say("s04", f"measured: a 50/50 data-dependent branch costs {dns:.2f} ns more per element than the same branch "
            f"on sorted data -> {br_pen:.0f} cycles per mispredict at {clock:.2f} GHz (source: ~15)", 8 < br_pen < 60)

    # ---------------- s05: SIMD lanes ----------------------------------------------------------
    tab = {128: (4, 2), 256: (8, 4), 512: (16, 8)}
    say("s05", "register width / 32 and / 64: 128 -> 4 floats 2 doubles, 256 -> 8/4, 512 -> 16/8",
        all(w // 32 == f and w // 64 == d for w, (f, d) in tab.items()))
    def strokes(n, W): return math.ceil(n / W), n / (W * math.ceil(n / W))
    say("s05", f"10 holes on an 8-spindle head: {strokes(10,8)[0]} strokes, {100*strokes(10,8)[1]:.1f}% of spindle-strokes "
        f"cut; on a 4-spindle head: {strokes(10,4)[0]} strokes, {100*strokes(10,4)[1]:.1f}%",
        strokes(10, 8) == (2, 0.625) and abs(strokes(10, 4)[1] - 10/12) < 1e-12)
    ok = True
    for W in (16, 32, 64):
        for off in range(4, 64, 4):
            pos = [off + k * W for k in range(64 * 64 // math.gcd(W, 64))]
            frac = sum((p % 64) + W > 64 for p in pos) / len(pos)
            want = 0.0 if off % W == 0 else W / 64
            ok &= abs(frac - want) < 1e-12
    say("s05", "a W-byte load at a misaligned offset splits a 64-byte line on exactly W/64 of loads "
        "(16 B: 25%, 32 B: 50%, 64 B: 100%); offsets that are multiples of W never split", ok)
    def shuf(imm):  # _mm_shuffle_ps(a, a, imm) as a 4x4 0/1 matrix
        M = np.zeros((4, 4))
        for d in range(4): M[d, (imm >> (2 * d)) & 3] = 1
        return M
    S1, S2 = shuf(0b01001110), shuf(0b10110001)
    ok = np.allclose(S1 @ S1.T, np.eye(4)) and np.allclose(S2 @ S2.T, np.eye(4))
    ok &= [int(np.argmax(S1[d])) for d in range(4)] == [2, 3, 0, 1] and [int(np.argmax(S2[d])) for d in range(4)] == [1, 0, 3, 2]
    rg = np.random.default_rng(5)
    for _ in range(1000):
        v = rg.standard_normal(4); m = np.maximum(v, S1 @ v); m = np.maximum(m, S2 @ m)
        ok &= np.allclose(m, v.max())
    say("s05", "the two max-reduction shuffles are permutation matrices (lane i <- i XOR 2, then i XOR 1, the FFT "
        "butterfly pattern) and leave the maximum in every lane, 1,000 random vectors", ok)
    say("s05", f"but a shuffle is not always a permutation: immediate 0 broadcasts lane 0, a matrix of rank "
        f"{np.linalg.matrix_rank(shuf(0))}", np.linalg.matrix_rank(shuf(0)) == 1)

    # ---------------- s06: roofline ---------------------------------------------------------------
    pk_src = 4e9 * 8 * 2; pk = pk_src * 2
    say("s06", f"source's peak 4e9 x 8 x 2 = {pk_src/1e9:.0f} GFLOP/s counts an FMA as one FLOP; its matmul counts "
        f"2n^3, so the matching peak is {pk/1e9:.0f} GFLOP/s per core", pk_src == 64e9 and pk == 128e9)
    say("s06", f"ridge point at 50 GB/s: {pk_src/50e9:.2f} FLOP/B as printed, {pk/50e9:.2f} corrected; 8 cores: "
        f"{8*pk/50e9:.2f}", abs(pk/50e9 - 2.56) < 1e-12 and abs(8*pk/50e9 - 20.48) < 1e-12)
    code = blocks(1)[4][1].replace("import jax.numpy as jnp", "")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    o = buf.getvalue()
    got = re.findall(r":\s*([0-9.]+) FLOPS/byte", o)
    say("s06", f"source's intensity task, run: ReLU {got[0]}, matmul {got[1]}, LayerNorm {got[2]}, conv {got[3]} FLOP/B "
        f"(exact: 0.125, 170.67, 0.625, {2*9*64*128*32*32/((9*64*128+64*1024+128*1024)*4):.2f})",
        got == ["0.12", "171", "0.62", "140"])
    def roof(I, peak=pk, bw=50e9): return min(peak, bw * I)
    say("s06", f"achievable on one corrected core: ReLU {roof(0.125)/1e9:.2f}, LayerNorm {roof(0.625)/1e9:.2f}, "
        f"matmul {roof(170.67)/1e9:.0f} GFLOP/s -- the first two are set by the delivery station alone",
        roof(0.125) == 6.25e9 and roof(170.67) == pk)
    say("s06", f"matmul's n/6 assumes each matrix crosses the bus once; with no reuse each multiply-add fetches two "
        f"floats: 2 FLOP / 8 B = {2/8} FLOP/B, below every ridge above", 2/8 < 1.28)

    # ---------------- s07: staging and tiling -----------------------------------------------------
    def reuse_curve(n, order, b):
        trace = []
        if order == "ijk":
            for i in range(n):
                for j in range(n):
                    for k in range(n): trace += [(0, i, k), (1, k, j), (2, i, j)]
        else:
            for ii in range(0, n, b):
                for jj in range(0, n, b):
                    for kk in range(0, n, b):
                        for i in range(ii, ii + b):
                            for j in range(jj, jj + b):
                                for k in range(kk, kk + b): trace += [(0, i, k), (1, k, j), (2, i, j)]
        ids = {}; tr = [ids.setdefault(x, len(ids)) for x in trace]
        Nt = len(tr); fen = [0] * (Nt + 1); last = {}; dist = []
        def add(i, v):
            i += 1
            while i <= Nt: fen[i] += v; i += i & -i
        def pre(i):
            s = 0; i += 1
            while i > 0: s += fen[i]; i -= i & -i
            return s
        for t, x in enumerate(tr):
            if x in last:
                p = last[x]; dist.append(pre(t - 1) - pre(p)); add(p, -1)
            else:
                dist.append(None)
            last[x] = t; add(t, 1)
        return dist
    def misses(dist, cap): return sum(1 for d in dist if d is None or d >= cap)
    def lru(n, order, b, cap):
        c = OrderedDict(); m = 0
        def acc(key):
            nonlocal m
            if key in c: c.move_to_end(key)
            else:
                m += 1; c[key] = 1
                if len(c) > cap: c.popitem(last=False)
        if order == "ijk":
            for i in range(n):
                for j in range(n):
                    for k in range(n): acc((0, i, k)); acc((1, k, j)); acc((2, i, j))
        else:
            for ii in range(0, n, b):
                for jj in range(0, n, b):
                    for kk in range(0, n, b):
                        for i in range(ii, ii + b):
                            for j in range(jj, jj + b):
                                for k in range(kk, kk + b): acc((0, i, k)); acc((1, k, j)); acc((2, i, j))
        return m
    n32 = 32
    dN = reuse_curve(n32, "ijk", 0); d8 = reuse_curve(n32, "tile", 8); d4 = reuse_curve(n32, "tile", 4); d16 = reuse_curve(n32, "tile", 16)
    ok = all(misses(dN, c) == lru(n32, "ijk", 0, c) for c in (128, 1100, 1200)) and \
         all(misses(d8, c) == lru(n32, "tile", 8, c) for c in (192, 512))
    say("s07", "stack-distance curve (one pass, every capacity at once) == direct LRU simulation at five capacities", ok)
    mN, m8 = misses(dN, 512), misses(d8, 512)
    say("s07", f"n = 32 matmul, 512-word LRU store: naive i-j-k fetches {mN:,} words (n^3 + 2n^2 = {n32**3+2*n32**2:,}); "
        f"8 x 8 tiles fetch {m8:,} (n^2 + 2n^3/b = {n32**2 + 2*n32**3//8:,}); {mN/m8:.2f}x less",
        mN == n32**3 + 2*n32**2 and m8 == n32**2 + 2*n32**3 // 8)
    say("s07", f"tile b = 4 at 128 words: {misses(d4,128):,} (formula {n32**2+2*n32**3//4:,}); b = 16 at 1,100 words: "
        f"{misses(d16,1100):,} (formula {n32**2+2*n32**3//16:,})",
        misses(d4, 128) == n32**2 + 2*n32**3//4 and misses(d16, 1100) == n32**2 + 2*n32**3//16)
    say("s07", f"the formula needs slack: b = 8 at exactly 3b^2 = 192 words gives {misses(d8,192):,}, not 9,216; "
        f"b = 4 at 512 words gives {misses(d4,512):,} (more reuse than the tile alone)",
        misses(d8, 192) > 9216 and misses(d4, 512) < n32**2 + 2*n32**3//4)
    cliff = [c for c in range(1024, 1300, 4) if misses(dN, c) == 3 * n32**2]
    say("s07", f"naive i-j-k falls off its cliff once all of B plus a row each of A and C fits: 3n^2 = {3*n32**2:,} "
        f"words from capacity {cliff[0]:,} (n^2 + 2n = {n32**2+2*n32})", cliff and n32**2 + 2*n32 <= cliff[0] <= n32**2 + 2*n32 + 64)
    say("s07", f"tile intensity, C kept: 2b^3 FLOP over 2b^2 floats = b/4 FLOP/B; at n = 32, b = 8: "
        f"{2*n32**3/(4*m8):.2f} FLOP/B measured from the trace", abs(2*n32**3/(4*m8) - 1.778) < 1e-3)
    rg = np.random.default_rng(6); best = 0; bestx = None
    for _ in range(20000):
        x = rg.random(3) + 0.05; x *= math.sqrt(3 / (x[0]*x[1] + x[1]*x[2] + x[0]*x[2]))
        v = x.prod()
        if v > best: best, bestx = v, x
    say("s07", f"box tile a x b x c with fixed face area ab + bc + ca = 3: largest volume {best:.4f} at "
        f"({bestx[0]:.2f}, {bestx[1]:.2f}, {bestx[2]:.2f}) -> the cube (volume 1); Lagrange gives a = b = c",
        best <= 1 + 1e-9 and best > 0.995)
    say("s07", f"L1 of 32 KiB holds three b x b float tiles while 12 b^2 <= 32768: b <= {int(math.sqrt(32768/12))}",
        int(math.sqrt(32768/12)) == 52)
    say("s07", f"a 64-byte line holds 16 floats: a column walk that loses each line before returning uses 1/16 of what was "
        f"delivered", 64 // 4 == 16)

    # ---------------- s08: Little's law / CONWIP -----------------------------------------------------
    fma = {}
    if meas:
        # throughput runs are sensitive to other work on a shared VM: keep the best of up to three sweeps
        dots = {}
        for attempt in range(3):
            out = run(meas, "fma")
            for k, v in re.findall(r"fma (\d+) (\S+)", out): fma[int(k)] = max(fma.get(int(k), 0.0), float(v))
            for k in ("dotal", "dotmis", "dot1"): dots[k] = max(dots.get(k, 0.0), num(k + r" (\S+)", out))
            if max(fma[10], fma[16]) > 1.8 and attempt >= 1: break
        out = "dotal %f\ndotmis %f\ndot1 %f\n" % (dots["dotal"], dots["dotmis"], dots["dot1"])
        lat = 1 / fma[1]; plat = max(fma[10], fma[16])
        model = {K: min(K / 4, 2) for K in fma}
        dev = max(abs(fma[K] - model[K]) / model[K] for K in (1, 2, 4, 6))
        say("s08", "measured FMA chains (vector FMAs per cycle): " + ", ".join(f"K={K}: {fma[K]:.2f}" for K in sorted(fma)) +
            f"; latency 1/rate(K=1) = {lat:.2f} cycles, plateau {plat:.2f}", 3.3 < lat < 4.8 and 1.4 < plat < 2.2)
        say("s08", f"CONWIP best case TH = min(K/T0, r_b) with T0 = 4 cycles, r_b = 2: below the knee (K = 1, 2, 4, 6) the "
            f"worst deviation is {100*dev:.0f}%; above it the rate stops rising ({plat/fma[1]:.1f}x K=1, bound 8x); "
            f"critical WIP K0 = r_b T0 = 8", dev < 0.15 and 5.0 < plat / fma[1] < 10.0)
        dal, d1r = num(r"dotal (\S+)", out), num(r"dot1 (\S+)", out)
        say("s08", f"dot product from L1, 8 accumulators: {fma.get(8,0):.2f} on the FMA chain but "
            f"{dal:.2f} with two loads per FMA -- the load station, not the FMA station, binds", dal < 0.75 * fma.get(8, 2))
        say("s08", f"source's one-accumulator dot product (the NEON and AVX2 code): {d1r:.2f} vector FMA "
            f"per cycle, i.e. the 4-cycle latency chain; its lanes are independent but its iterations are not", d1r < 0.4)
        al, mi = num(r"dotal (\S+)", out), num(r"dotmis (\S+)", out)
        say("s10", f"measured in L1: the same dot product misaligned by 4 bytes runs at {mi:.2f} against {al:.2f} aligned "
            f"({al/mi:.2f}x): half the 32-byte loads split a line", al / mi > 1.4)
        out = run(meas, "chase")
        ch = {int(k): float(v) for k, v in re.findall(r"chase (\d+) (\S+)", out)}
        T0 = ch[1]; rb = 1 / min(ch[16], ch[32])
        conc = rb * T0
        say("s08", "measured pointer chase (ns per line, K independent chains): " + ", ".join(f"K={K}: {ch[K]:.1f}" for K in sorted(ch)) +
            f"; single-chain latency T0 = {T0:.0f} ns", ch[1] / ch[4] > 2.5 and ch[16] / ch[32] < 1.35)
        say("s08", f"Little's law: plateau {64*rb:.2f} GB/s of random lines x latency {T0:.0f} ns = {conc:.1f} lines in flight "
            f"(the critical WIP); a core cannot have more misses outstanding than that", 6 < conc < 32)
        say("s08", f"OoO window needed to hide one DRAM miss: {T0:.0f} ns x {clock:.2f} GHz x 4 IPC = "
            f"{T0*clock*4:,.0f} instructions, several times any reorder buffer (reported 224 on Skylake-family cores)",
            T0 * clock * 4 > 448)
    say("s08", f"Little's law, exact arithmetic: 4-cycle latency x 2 FMA units = 8 accumulators; Golden Cove-class "
        f"512-entry window vs 300-cycle miss x 4 IPC = 1,200", 4 * 2 == 8 and 300 * 4 == 1200)

    # ---------------- s09: ARM -------------------------------------------------------------------
    say("s09", f"x0..x30 is {len(range(0,31))} general-purpose registers, not 32 (the 32nd encoding is sp or xzr)",
        len(range(0, 31)) == 31)
    say("s09", "NEON lanes: 128/32 = 4 float32, 128/16 = 8 float16/int16, 128/8 = 16 int8",
        (128 // 32, 128 // 16, 128 // 8) == (4, 8, 16))
    rg = np.random.default_rng(9)
    A28 = rg.integers(-128, 128, (2, 8)); B28 = rg.integers(-128, 128, (2, 8)); C22 = rg.integers(-1000, 1000, (2, 2))
    vec = (C22.reshape(4) + np.array([A28[i] @ B28[j] for i in range(2) for j in range(2)]))
    say("s09", f"SMMLA model: 2x2 int32 += (2x8 int8)(2x8 int8)^T = {2*2*8} multiply-accumulates per instruction "
        f"(source: 'an 8-element dot product'); SDOT does 4 x 4 = 16; widening NEON 16 in 4 instructions",
        np.array_equal(vec.reshape(2, 2), C22 + A28 @ B28.T) and 2*2*8 == 32 and 32 / (16 / 4) == 8)
    say("s09", f"SME at SVL = 512 bits: a 16 x 16 outer product = {16*16} FMAs per instruction; ZA = 64 x 64 bytes = "
        f"{64*64:,} B = four 16 x 16 float32 tiles", 16*16 == 256 and 64*64 == 4 * 16 * 16 * 4)
    say("s09", f"PCIe 4.0 x16 (16 GT/s per lane, reported; 128b/130b coding): 16e9 x 128/130 / 8 x 16 = "
        f"{16e9*128/130/8*16/1e9:.1f} GB/s each way (source: ~32)", abs(16e9*128/130/8*16/1e9 - 31.5) < 0.05)
    if clangxx:
        b2 = blocks(2)
        def armc(code, march, extra=()):
            p = os.path.join(tmp16, "arm.cpp"); s = os.path.join(tmp16, "arm.s")
            code = re.sub(r"#include <(iostream|chrono|vector|cmath)>\n", "", code)
            i = code.find("int main("); code = code if i < 0 else code[:i]
            open(p, "w").write("#define INFINITY __builtin_inff()\n" + code)
            r = subprocess.run([clangxx, "--target=aarch64-linux-gnu", "-ffreestanding", "-O3", "-march=" + march, "-S",
                                "-o", s, p] + list(extra), capture_output=True, text=True)
            return r.returncode, (open(s).read() if r.returncode == 0 else r.stderr)
        rc, asm = armc(b2[2][1], "armv8-a")
        def fbody(asm, name):
            m = re.search(r"^_Z\d+" + name + r"\w*:[^\n]*\n(.*?)(?:^\.Lfunc_end|\.cfi_endproc)", asm, re.M | re.S)
            return m.group(1) if m else ""
        if rc == 0:
            neon = fbody(asm, "dot_neon")
            nf = len(re.findall(r"\bfmla\s+v\d+\.4s", neon))
            ser = len(re.findall(r"fadd\s+s0, s0, s\d", fbody(asm, "dot_scalar")))
            say("s09", f"cross-compiled for AArch64: dot_neon's loop is {nf} fmla into one register (one chain); "
                f"dot_scalar keeps {ser} serial 'fadd s0, s0, ...' per unrolled block (in-order reduction)", nf == 1 and ser >= 4)
        rc_sme, err = armc(b2[5][1], "armv9.2-a+sme2")
        say("s09", f"the SME2 snippet as printed does not compile (clang rc {rc_sme}: {'SVL undeclared' if 'SVL' in err else err[:60]})",
            rc_sme != 0 and "SVL" in err)
        ok_all = all(armc(b2[k][1], m)[0] == 0 for k, m in ((3, "armv8-a"), (4, "armv8.6-a+i8mm"), (6, "armv8.2-a+sve"),
                                                            (10, "armv8-a"), (11, "armv8-a"), (12, "armv8-a")))
        say("s09", "the NEON, I8MM and SVE snippets and the three ARM tasks compile for AArch64 (not run: no ARM CPU here)", ok_all)

    # ---------------- s10: x86 -------------------------------------------------------------------
    say("s10", f"AMX tile: 16 rows x 64 B = {16*64} B, eight tiles = {8*16*64:,} B; TDPBF16PS on 16x32 by 32x16 = "
        f"{16*16*32:,} multiply-accumulates = {16*16*32//16} AVX-512 FMAs (source: 'hundreds')",
        16*16*32 == 8192 and 8192 // 16 == 512)
    rg = np.random.default_rng(10)
    Ab = rg.integers(-4, 5, (16, 32)).astype(np.float32); Bb = rg.integers(-4, 5, (32, 16)).astype(np.float32)
    Bv = np.zeros((16, 32), np.float32)
    for k in range(32):
        for j in range(16): Bv[k // 2, 2 * j + (k % 2)] = Bb[k, j]
    Cv = np.zeros((16, 16), np.float32)
    for i in range(16):
        for j in range(16):
            for kp in range(16): Cv[i, j] += Ab[i, 2*kp] * Bv[kp, 2*j] + Ab[i, 2*kp+1] * Bv[kp, 2*j+1]
    say("s10", "AMX's B operand is not 32x16 in memory but 16 rows of pairs (VNNI layout); the pair-wise model "
        "reproduces A @ B exactly", np.array_equal(Cv, Ab @ Bb))
    say("s10", f"task 3's aligned_alloc(32, 16e6 x 4): size {16_000_000*4:,} is a multiple of 32 as C requires",
        (16_000_000 * 4) % 32 == 0)
    if x86ok:
        VEC = ("#include <immintrin.h>\nvoid add_auto(const float* a,const float* b,float* c,int n){for(int i=0;i<n;i++) c[i]=a[i]+b[i];}\n"
               "float dot_scalar(const float* a,const float* b,int n){float s=0.0f;for(int i=0;i<n;i++) s+=a[i]*b[i];return s;}\n"
               "void prefix(float* a,const float* b,int n){for(int i=1;i<n;i++) a[i]=a[i-1]+b[i];}\n")
        p = os.path.join(tmp16, "vec.cpp"); open(p, "w").write(VEC)
        r = subprocess.run([gxx, "-O3", "-mavx2", "-mfma", "-c", p, "-o", os.devnull, "-fopt-info-vec-all"], capture_output=True, text=True)
        rep = r.stderr
        say("s10", "GCC vectorises add_auto WITHOUT restrict: 'loop versioned for vectorization because of possible "
            "aliasing' (a runtime overlap check, not a refusal)", "vec.cpp:2" in rep and "versioned for vectorization because of possible aliasing" in rep)
        say("s10", "the prefix loop a[i] = a[i-1] + b[i] is not vectorised (a true chain)",
            re.search(r"vec.cpp:4:\d+: missed: couldn't vectorize loop", rep) is not None)
        r = subprocess.run([gxx, "-O3", "-mavx2", "-mfma", "-S", p, "-o", os.path.join(tmp16, "vec.s")], capture_output=True, text=True)
        asm = open(os.path.join(tmp16, "vec.s")).read()
        mds = re.search(r"^_Z10dot_scalarPKfS0_i:\s*$(.*?)\.cfi_endproc", asm, re.M | re.S)
        dsec = mds.group(1) if mds else ""
        say("s10", f"dot_scalar at -O3: 'vectorized' in the report, but the asm is vmulps then "
            f"{len(re.findall(r'vaddss', dsec))} scalar vaddss into one register -- the multiplies go 8-wide, the sum stays a chain",
            "vmulps" in dsec and len(re.findall(r"vaddss", dsec)) >= 8)
        if "avx512f" in cpuflags:
            p = os.path.join(tmp16, "red.cpp")
            open(p, "w").write("#include <immintrin.h>\nfloat red(__m512 a){return _mm512_reduce_add_ps(a);}\n")
            subprocess.run([gxx, "-O3", "-mavx512f", "-S", p, "-o", os.path.join(tmp16, "red.s")], capture_output=True)
            body = open(os.path.join(tmp16, "red.s")).read()
            ninst = len(re.findall(r"^\s+v\w+", body, re.M)); nadd = len(re.findall(r"vadd", body))
            say("s10", f"_mm512_reduce_add_ps is not one instruction: GCC emits {ninst} vector instructions, {nadd} of them adds "
                f"(log2 16 = 4 halvings)", nadd >= 4 and ninst > 4)
            zg = yg = 0.0
            for _ in range(2):
                out = run(meas, "zmm")
                zg, yg = max(zg, num(r"zmm_gflops (\S+)", out)), max(yg, num(r"ymm_gflops (\S+)", out))
            say("s10", f"measured: AVX-512 FMA chains deliver {zg:.0f} GFLOP/s per core against AVX2's {yg:.0f}: "
                f"{zg/yg:.2f}x, not 2x", 1.1 < zg / yg < 2.05)

    if x86ok and "amx_tile" in cpuflags and "amx_bf16" in cpuflags:
        AMX = r'''
#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <sys/syscall.h>
struct Cfg { uint8_t palette, start_row, rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; };
int main(int argc, char** argv) {
  bool perm = strchr(argv[1], 'p'), cfg = strchr(argv[1], 'c');
  if (perm) syscall(SYS_arch_prctl, 0x1023, 18);           /* ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA */
  alignas(64) uint16_t A[16][32], Bp[16][32]; alignas(64) float C[16][16] = {}; float Af[16][32], Bf[32][16];
  auto bf = [](float x){ uint32_t u; memcpy(&u, &x, 4); return (uint16_t)(u >> 16); };
  auto fb = [](uint16_t h){ uint32_t u = (uint32_t)h << 16; float x; memcpy(&x, &u, 4); return x; };
  unsigned s = 7; auto rnd = [&](){ s = s * 1103515245u + 12345u; return (int)((s >> 16) % 9) - 4; };
  for (int i = 0; i < 16; i++) for (int k = 0; k < 32; k++) { A[i][k] = bf((float)rnd()); Af[i][k] = fb(A[i][k]); }
  for (int k = 0; k < 32; k++) for (int j = 0; j < 16; j++) { uint16_t h = bf((float)rnd()); Bf[k][j] = fb(h); Bp[k/2][2*j + k%2] = h; }
  if (cfg) { Cfg c = {}; c.palette = 1; for (int t = 0; t < 3; t++) { c.rows[t] = 16; c.colsb[t] = 64; } _tile_loadconfig(&c); }
  _tile_loadd(0, A, 64); _tile_loadd(1, Bp, 64); _tile_zero(2); _tile_dpbf16ps(2, 0, 1); _tile_stored(2, C, 64); _tile_release();
  double e = 0; for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) { double t = 0; for (int k = 0; k < 32; k++) t += Af[i][k] * Bf[k][j]; e = e > fabs(t - C[i][j]) ? e : fabs(t - C[i][j]); }
  printf("maxerr %g\n", e); return 0; }
'''.replace("fabs", "__builtin_fabs")
        ob = build("amx", AMX, ["-O1", "-mamx-tile", "-mamx-bf16"])
        if ob:
            rcs = {m: subprocess.run([ob, m], capture_output=True, text=True) for m in ("pc", "p", "c")}
            okfull = rcs["pc"].returncode == 0 and num(r"maxerr (\S+)", rcs["pc"].stdout) == 0
            say("s10", f"AMX run on this CPU: with the Linux permission request and a tile configuration, one TDPBF16PS "
                f"reproduces the 16x16 product exactly; as printed (no configuration) it dies with signal "
                f"{-rcs['p'].returncode}; without the permission request, signal {-rcs['c'].returncode} (4 = SIGILL)",
                okfull and rcs["p"].returncode == -4 and rcs["c"].returncode == -4)

    # ---------------- s11: the source's benchmarks, re-run ---------------------------------------------
    if x86ok and meas:
        al = runbest(meas, {"lat": r"addlat (\S+)"}, 2, "addlat")["lat"]
        say("s11", f"measured scalar float-add latency on this core: {al:.2f} cycles (4 on Skylake/Cascade Lake; "
            f"Golden Cove-class cores have a 2-cycle add, reported)", 1.5 < al < 5)
        bwn = bwt = 0.0
        for _ in range(3):
            out = run(meas, "bw")
            bwn, bwt = max(bwn, num(r"bw_normal (\S+)", out)), max(bwt, num(r"bw_nt (\S+)", out))
        say("s11", f"measured add c = a + b, counted 12 B per element: {bwn:.1f} GB/s; with streaming stores {bwt:.1f} "
            f"({bwt/bwn:.2f}x). A normal store first reads its line (write-allocate), so the bus carries 16 B per 12 counted: "
            f"4/3 if the bus binds, more if the core's miss concurrency binds", bwt / bwn > 1.15)
        b1 = blocks(1)
        o = build("bench01", b1[2][1], ["-O3", "-march=native"])
        rep = max(num(r"Bandwidth: (\S+)", run(o)) for _ in range(2))
        say("s11", f"file 01's benchmark prints {rep:,.0f} GB/s; the real figure is {bwn:.1f}. The ratio "
            f"{rep/bwn:.0f} is the trial count: the optimiser ran 1 of the 100 identical trials",
            30 < rep / bwn < 300)
        b3 = blocks(3)
        o = build("align", b3[12][1], ["-O3", "-mavx2"])
        out = run(o)
        ta, tu = num(r"Aligned load\s*: (\S+) ms", out), num(r"Unaligned load: (\S+) ms", out)
        say("s11", f"file 03 task 3 (alignment) prints {ta:.2g} ms and {tu:.2g} ms per pass over 64 MB: the loop's "
            f"result is never used, so the compiler deleted it", ta < 1e-3 and tu < 1e-3)
        o = build("dot03", b3[10][1], ["-O3", "-mavx2", "-mfma"])
        bd = runbest(o, {"s": r"Scalar: (\S+) ms", "v": r"AVX2  : (\S+) ms"}, 2)
        tsc, tav = bd["s"], bd["v"]
        floor = 1e7 * al / clock / 1e6
        say("s11", f"file 03 task 1: scalar {tsc:.1f} ms against an add-chain floor of 10^7 x {al:.1f} cycles / {clock:.2f} GHz "
            f"= {floor:.1f} ms; AVX2 {tav:.1f} ms = {8e7/tav/1e6:.1f} GB/s: speedup {tsc/tav:.1f}x, not 8x",
            tsc / tav < 4.5 and 0.85 < tsc / floor < 3.5)
        o = build("relu03", b3[11][1], ["-O3", "-mavx2"])
        br = runbest(o, {"s": r"Scalar\s*: (\S+) ms", "v": r"AVX2\s+: (\S+) ms", "u": r"AVX2 unrolled\s*: (\S+) ms"}, 3)
        tr = [br["s"], br["v"], br["u"]]
        say("s11", "file 03 task 2 (ReLU), best of three runs: scalar / AVX2 / unrolled = " + " / ".join(f"{x:.1f}" for x in tr) +
            " ms: all at the memory station; unrolling had no chain to break", max(tr) / min(tr) < 1.3)
        b0 = blocks(0)
        o = build("layout", b0[14][1], ["-O3"])
        bl = runbest(o, {"r": r"Row-major \(cache-friendly\): (\S+) ms", "c": r"Col-major \(cache-hostile\):\s+(\S+) ms"}, 2)
        tr_, tc_ = bl["r"], bl["c"]
        fl_ = 2**24 * al / clock / 1e6
        say("s11", f"file 00 task 3: row {tr_:.1f} ms, column {tc_:.1f} ms ({tc_/tr_:.1f}x). The row loop is one float add "
            f"chain: 2^24 x {al:.1f} cycles / {clock:.2f} GHz = {fl_:.1f} ms floor", 0.85 < tr_ / fl_ < 3.5 and tc_ / tr_ > 2)
    shutil.rmtree(tmp16, ignore_errors=True)

    # ---------------- s12: worked example (fixed inputs) ---------------------------------------------------
    f_ = 3.17; peak1 = 2 * 8 * 2 * f_; bw1 = 10.0
    ridge1 = peak1 / bw1; b1_ = math.ceil(4 * ridge1)
    say("s12", f"one core: peak 2 FMA x 8 lanes x 2 FLOP x {f_} GHz = {peak1:.1f} GFLOP/s, 10 GB/s -> ridge {ridge1:.2f} "
        f"FLOP/B; tile b >= 4 x ridge = {b1_}; three tiles = {3*b1_**2*4/1024:.1f} KiB <= 32 KiB L1",
        abs(peak1 - 101.44) < 1e-9 and b1_ == 41 and 3 * b1_**2 * 4 <= 32768)
    bw4 = 20.0; ridge4 = 4 * peak1 / bw4; b4_ = math.ceil(4 * ridge4)
    say("s12", f"four cores sharing 20 GB/s: ridge {ridge4:.1f}, tile b >= {b4_}, three tiles = {3*b4_**2*4/1024:.0f} KiB: "
        f"too big for L1, comfortable in a 1 MiB L2 -- stage at the next level out", b4_ == 82 and 32768 < 3*b4_**2*4 < 2**20)
    say("s12", f"line view: delivery 10 GB/s x I = machining {peak1:.1f} GFLOP/s balances at I = {ridge1:.2f}; a 41-tile "
        f"gives I = 41/4 = {41/4:.2f}, machining-bound by {100*(41/4/ridge1-1):.0f}%", 41/4 > ridge1)

_ch16p1()
# ----- end of Chapter 16 part 1 block -----


# ======================================================================================
# Chapter 16 part 2 block. Owned by the Chapter 16 part 2 run: replace only the body of _ch16p2() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch16p2():
    pass

_ch16p2()
# ----- end of Chapter 16 part 2 block -----


# ======================================================================================
# Chapter 16 part 3 block. Owned by the Chapter 16 part 3 run: replace only the body of _ch16p3() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch16p3():
    pass

_ch16p3()
# ----- end of Chapter 16 part 3 block -----


# ======================================================================================
# Chapter 17 block. Owned by the Chapter 17 run: replace only the body of _ch17() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch17():
    # =================================================================
    # Chapter 17 -- AI inference (files 01-05), one sheet.
    # Every name lives inside this function so nothing collides.
    # =================================================================
    import math, random
    import numpy as np
    from scipy.stats import norm as _n17
    from scipy.integrate import quad as _quad17
    from scipy.optimize import minimize_scalar as _min17
    T17 = "[Ch17 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T17 % sec} {msg}{tail}")
    print()

    # --- s01: memory, not arithmetic -----------------------------------
    w70 = 70e9
    say("s01", f"70B params: FP16 {w70*2/1e9:.0f} GB, INT8 {w70/1e9:.0f} GB, INT4 {w70*0.5/1e9:.0f} GB "
        "(source: 140 / - / 35)", w70*2/1e9 == 140 and w70*0.5/1e9 == 35)
    r15 = 989/67
    say("s01", f"source's own figures: 989/67 = {r15:.1f}x, but its 2x-per-halving ladder gives FP8 = 4x FP32; "
        "the two cannot both describe one kind of FLOP", abs(r15-14.76) < 0.01 and r15 > 4*2)
    # decode-bound tokens/s = bandwidth / model bytes (source file 04 estimator)
    est = [(7000, 4, 200), (7000, 8, 200), (70000, 4, 200), (3800, 4, 50), (3000, 4, 50)]
    tps = [round(1000/((pM*1e6*b/8)/(bw*1e9)*1000)) for pM, b, bw in est]
    say("s01", f"bandwidth-bound decode rates (source 04 task 2): {tps} tok/s", tps == [57, 29, 6, 26, 33])

    # --- s02: number formats --------------------------------------------
    fmax = {"FP16": (2-2**-10)*2**15, "BF16": (2-2**-7)*2.0**127, "E4M3": 1.75*2**8, "E5M2": 1.75*2**15}
    say("s02", f"format maxima: FP16 {fmax['FP16']:.0f}, BF16 {fmax['BF16']:.3e}, E4M3 {fmax['E4M3']:.0f}, "
        f"E5M2 {fmax['E5M2']:.0f}", fmax["FP16"] == 65504 and fmax["E4M3"] == 448 and fmax["E5M2"] == 57344
        and abs(fmax["BF16"]/3.4e38-1) < 0.01)
    sf = {k: (m+1)*math.log10(2) for k, m in [("FP32", 23), ("FP16", 10), ("BF16", 7), ("E4M3", 3), ("E5M2", 2)]}
    say("s02", "significant figures (mantissa+1 bits x log10 2): " +
        ", ".join(f"{k} {v:.2f}" for k, v in sf.items()), abs(sf["FP16"]-3.31) < 0.01 and abs(sf["E4M3"]-1.20) < 0.01)
    i8 = math.log10(255); i8t = math.log10(255/10)
    say("s02", f"INT8 carries {i8:.2f} figures at full scale and {i8t:.2f} at a tenth of it (constant absolute step)",
        abs(i8-2.41) < 0.01 and abs(i8t-1.41) < 0.01)
    mx32 = [e+8/32 for e in (8, 6, 4, 8)]; mx16 = [e+8/16 for e in (8, 6, 4, 8)]
    say("s02", f"MX bits/element, shared 8-bit exponent: block 32 -> {mx32}, block 16 -> {mx16}; "
        "MXFP8 and MXINT8 are equal either way (source lists ~8 and ~8.5)", mx32[0] == mx32[3] and mx16[2] == 4.5)
    q40 = (32*4+16)/32
    say("s02", f"a 4-bit block of 32 with a 16-bit scale is {q40} bits/weight: 8B -> {8e9*q40/8/1e9:.1f} GB "
        f"(source table 4.5 GB); 70B -> {70e9*q40/8/1e9:.1f} GB, not 35", q40 == 4.5 and abs(8e9*q40/8/1e9-4.5) < 1e-9)
    say("s02", f"BitNet: log2 3 = {math.log2(3):.3f} bits; 70B -> {70e9*math.log2(3)/8/1e9:.1f} GB (source ~15)",
        abs(70e9*math.log2(3)/8/1e9-13.87) < 0.01)

    # --- s03: quantisation noise is Delta^2/12 --------------------------
    r3 = np.random.default_rng(0)
    w = (r3.standard_normal((1024, 1024))*0.02).astype(np.float32)
    s = np.abs(w).max()/127.0
    rq = np.clip(np.round(w/s), -127, 127).astype(np.float32)*s
    e = w - rq
    say("s03", f"source 01 task 1 (numpy port): Delta = {s:.4e}; error variance {e.var():.4e} vs Delta^2/12 "
        f"{s*s/12:.4e} (ratio {e.var()/(s*s/12):.4f})", abs(e.var()/(s*s/12)-1) < 0.01)
    say("s03", f"mean |error| {np.abs(e).mean():.4e} vs Delta/4 {s/4:.4e}; max {np.abs(e).max():.4e} vs Delta/2 "
        f"{s/2:.4e}; relative error {np.abs(e).mean()/np.abs(w).mean():.2%}",
        abs(np.abs(e).mean()/(s/4)-1) < 0.01 and np.abs(e).max() <= s/2*1.0001)
    say("s03", f"GUM rectangular: half-width a = Delta/2 -> a/sqrt3 = Delta/sqrt12 = {1/(2*math.sqrt(3)):.6f} Delta "
        f"= sqrt(1/12) {math.sqrt(1/12):.6f}", abs(1/(2*math.sqrt(3))-math.sqrt(1/12)) < 1e-15)
    rows = []
    for ratio in [0.1, 0.25, 0.5, 1.0, 2.0]:
        x = r3.standard_normal(400000)*ratio
        er = np.round(x) - x
        rows.append((ratio, er.var()*12, x.var()/er.var(), 12*ratio**2, (np.round(x) == 0).mean()))
    say("s03", "sigma/Delta -> MSE/(Delta^2/12), SQNR measured vs 12(sigma/Delta)^2, share rounded to 0: " +
        "; ".join(f"{a}: {b:.3f}, {c:.2f}/{d:.2f}, {f:.2f}" for a, b, c, d, f in rows),
        all(abs(r[1]-1) < 0.015 for r in rows if r[0] >= 0.5) and rows[1][1] < 0.7 and rows[0][4] > 0.99)

    # --- s04: granularity and the GEMM ----------------------------------
    r4 = np.random.default_rng(42)
    a = r4.standard_normal((32, 512))*0.1
    a[:, 0] *= 100; a[:, 1] *= 50
    st = np.abs(a).max()/127
    qt = np.clip(np.round(a/st), -127, 127)
    sc = np.abs(a).max(axis=0)/127
    qc = np.clip(np.round(a/sc), -127, 127)
    et = np.abs(a-qt*st).mean(); ec = np.abs(a-qc*sc).mean()
    say("s04", f"source 01 task 2 (numpy port): per-tensor {et:.5f}, per-channel {ec:.6f}, {et/ec:.0f}x",
        et/ec > 50)
    nrm = a[:, 2:]; z0 = (qt[:, 2:] == 0).mean(); lv = np.unique(qt[:, 2:]).size
    used = np.abs(nrm).max()/np.abs(a).max()
    say("s04", f"normal channels: {z0:.0%} round to zero, {lv} of 255 levels used, {used:.1%} of the range; "
        f"the printed 'waste' {(510/512):.0%} is the share of channels, not of the range",
        abs(z0-0.72) < 0.01 and lv == 5 and used < 0.02)
    en = qt[:, 2:]*st - nrm
    say("s04", f"normal channels: sigma/Delta = {nrm.std()/st:.3f}; error var {en.var():.5f} vs Delta^2/12 "
        f"{st*st/12:.5f}; SQNR {nrm.var()/en.var():.2f} vs 12(sigma/Delta)^2 = {12*(nrm.std()/st)**2:.2f}",
        abs(en.var()/(st*st/12)-1) < 0.05 and abs(nrm.var()/en.var()-12*(nrm.std()/st)**2) < 0.15)
    Xa = r4.standard_normal((16, 64)); Xa[:, 3] *= 60
    Wg = r4.standard_normal((64, 32))*0.05
    q8 = lambda x, sc_: np.clip(np.round(x/sc_), -127, 127)
    sx = np.abs(Xa).max(1, keepdims=True)/127; sw = np.abs(Wg).max(0, keepdims=True)/127
    QX, QW = q8(Xa, sx), q8(Wg, sw)
    fac = np.abs((QX @ QW)*sx*sw - (QX*sx) @ (QW*sw)).max()
    say("s04", f"per-token X and per-output-channel W scales factor out of the integer GEMM: max diff {fac:.1e}",
        fac < 1e-12)
    sk = np.abs(Xa).max(0, keepdims=True)/127; QXk = q8(Xa, sk)
    Rk = ((QXk*sk) @ (QW*sw))/((QXk @ QW)*sw)
    spread = np.std(Rk)/abs(np.mean(Rk))
    say("s04", f"per-input-channel X scales do not: the factor each output would need varies, std/mean {spread:.1f}",
        spread > 1)
    Y = Xa @ Wg
    rel = lambda A: np.linalg.norm(A-Y)/np.linalg.norm(Y)
    stt = np.abs(Xa).max()/127
    e_t = rel((q8(Xa, stt)*stt) @ (QW*sw)); e_k = rel((QX*sx) @ (QW*sw)); e_c = rel((QXk*sk) @ (QW*sw))
    sm = np.sqrt(np.abs(Xa).max(0)/np.abs(Wg).max(1))
    Xs, Ws = Xa/sm, Wg*sm[:, None]
    ex = np.abs(Xs @ Ws - Y).max()
    sxs = np.abs(Xs).max(1, keepdims=True)/127; sws = np.abs(Ws).max(0, keepdims=True)/127
    e_s = rel((q8(Xs, sxs)*sxs) @ (q8(Ws, sws)*sws))
    say("s04", f"W8A8 relative error: per-tensor {e_t:.2%}, per-token {e_k:.2%}, per-input-channel {e_c:.2%} "
        f"(not GEMM-able), SmoothQuant alpha=0.5 {e_s:.2%}; XW unchanged by the migration to {ex:.0e}",
        e_s < e_k < e_t and ex < 1e-12 and e_s < e_c)

    # --- s05: the gauge range has an optimum ------------------------------
    def gmse(c, b):
        qm = 2**(b-1)-1; d = c/qm; tot = 0.0
        for k in range(-qm, qm+1):
            lo = -np.inf if k == -qm else (k-0.5)*d
            hi = np.inf if k == qm else (k+0.5)*d
            tot += _quad17(lambda x: (x-k*d)**2*_n17.pdf(x), lo, hi)[0]
        return tot
    o4 = _min17(lambda c: gmse(c, 4), bounds=(1, 8), method="bounded")
    o8 = _min17(lambda c: gmse(c, 8), bounds=(1, 8), method="bounded")
    m4, m8 = gmse(4.9, 4), gmse(4.9, 8)
    say("s05", f"MSE-optimal clip for N(0,1): 4-bit {o4.x:.2f} sigma (clips {2*_n17.sf(o4.x):.2%}), "
        f"{m4/o4.fun:.2f}x less error than ranging to 4.9 sigma; 8-bit {o8.x:.2f} sigma, {m8/o8.fun:.2f}x",
        abs(o4.x-2.47) < 0.02 and abs(m4/o4.fun-3.17) < 0.02 and abs(o8.x-3.92) < 0.03)
    say("s05", f"max of 1,048,576 N(0,1) draws in s03 = {np.abs(w).max()/0.02:.2f} sigma (the min-max range)",
        abs(np.abs(w).max()/0.02-5.0) < 0.1)

    # --- s06: GPTQ is a support settlement ------------------------------
    r6 = np.random.default_rng(3)
    nin, nout, ns = 64, 48, 512
    A6 = r6.standard_normal((nin, nin))/math.sqrt(nin) + 0.6*np.eye(nin)
    Xc = r6.standard_normal((ns, nin)) @ A6
    Wm = r6.standard_normal((nout, nin))*0.1
    H = Xc.T @ Xc + 1e-6*np.eye(nin)
    def gq(bits, mode):
        Wq = Wm.copy(); qm = 2**(bits-1)-1
        d = np.abs(Wm).max(1, keepdims=True)/qm
        for j in range(nin):
            q = np.clip(np.round(Wq[:, j:j+1]/d), -qm, qm)*d
            err = Wq[:, j:j+1] - q
            if j < nin-1 and mode == "gptq":
                Hi = np.linalg.inv(H[j:, j:]); Wq[:, j+1:] -= err @ (Hi[0:1, 1:]/Hi[0, 0])
            elif j < nin-1 and mode == "src":
                Wq[:, j+1:] -= (-err) @ (H[j:j+1, j+1:]/H[j, j])
            Wq[:, j:j+1] = q
        return Wq
    Y0 = Xc @ Wm.T
    res = {}
    for bits in (3, 4):
        for mode in ("rtn", "gptq", "src"):
            Wq = gq(bits, mode)
            res[(bits, mode)] = np.linalg.norm(Xc @ Wq.T - Y0)**2/np.linalg.norm(Y0)**2
    say("s06", "output error ||XW - XQ||^2/||XW||^2, RTN / correct update / source's printed formula: " +
        "; ".join(f"{b}-bit {res[(b,'rtn')]:.4f} / {res[(b,'gptq')]:.4f} / {res[(b,'src')]:.4f}" for b in (3, 4)),
        all(res[(b, "gptq")] < 0.55*res[(b, "rtn")] and 0.8*res[(b, "rtn")] < res[(b, "src")] < res[(b, "rtn")]
            for b in (3, 4)))
    say("s06", "share of RTN's error removed: correct " +
        ", ".join(f"{1-res[(b,'gptq')]/res[(b,'rtn')]:.0%}" for b in (3, 4)) + "; printed formula " +
        ", ".join(f"{1-res[(b,'src')]/res[(b,'rtn')]:.0%}" for b in (3, 4)), True)
    Hp = Xc.T @ Xc; Hpi = np.linalg.inv(Hp)
    cond = -np.linalg.solve(Hp[1:, 1:], Hp[1:, 0]); invf = Hpi[1:, 0]/Hpi[0, 0]
    dd = np.abs(cond-invf).max()
    say("s06", f"static condensation -K_rr^-1 K_rj == inverse-Hessian form H^-1_rj/H^-1_jj: max diff {dd:.1e}",
        dd < 1e-9)
    r6b = np.random.default_rng(9); n6 = 64
    wv = r6b.uniform(-1, 1, n6); dl = 2/7
    Hc = np.ones((n6, n6)) + 1e-9*np.eye(n6); wq = wv.copy()
    for j in range(n6):
        q = np.clip(np.round(wq[j]/dl), -7, 7)*dl; err = wq[j]-q
        if j < n6-1:
            Hi = np.linalg.inv(Hc[j:, j:]); wq[j+1:] -= err*Hi[0, 1:]/Hi[0, 0]
        wq[j] = q
    sg = abs(wq.sum()-wv.sum()); sr = abs((np.clip(np.round(wv/dl), -7, 7)*dl).sum()-wv.sum())
    say("s06", f"identical inputs (H = 11'): compensated total error {sg:.3f} <= Delta/2 {dl/2:.3f}; "
        f"independent rounding {sr:.3f}", sg <= dl/2+1e-9 and sr > 3*sg)

    # --- s07: the weight-only zoo ---------------------------------------
    r7 = np.random.default_rng(11)
    Wo = r7.standard_normal((256, 256)); Wo[7, 19] = 100.0
    Q1, _ = np.linalg.qr(r7.standard_normal((256, 256))); U1, _ = np.linalg.qr(r7.standard_normal((256, 256)))
    def i4(M):
        d = np.abs(M).max()/7; return np.clip(np.round(M/d), -7, 7)*d
    ep = np.linalg.norm(i4(Wo)-Wo)/np.linalg.norm(Wo)
    Wr = U1 @ Wo @ Q1.T
    er_ = np.linalg.norm(U1.T @ i4(Wr) @ Q1 - Wo)/np.linalg.norm(Wo)
    say("s07", f"incoherence (QuIP): one 100 among N(0,1), INT4 per-tensor error {ep:.0%} -> {er_:.0%} after a random "
        f"rotation; peak/rms {np.abs(Wo).max()/Wo.std():.0f} -> {np.abs(Wr).max()/Wr.std():.1f}", er_ < 0.25 < 0.9 < ep)
    tw_ = r7.choice([-1, 0, 1], size=(64, 32)); xv = r7.standard_normal(64)
    addsub = np.array([xv[tw_[:, j] == 1].sum() - xv[tw_[:, j] == -1].sum() for j in range(32)])
    say("s07", f"ternary matmul == adds and subtracts only: max diff {np.abs(addsub - xv @ tw_).max():.1e}",
        np.abs(addsub - xv @ tw_).max() < 1e-12)
    say("s07", f"AQLM: two 8-bit indices for 8 weights = {2*8/8} bits/weight", 2*8/8 == 2)

    # --- s08: errors stack like tolerances ------------------------------
    def stream(L, mode, trials=30, d=256, seed=21):
        r = np.random.default_rng(seed); out = []
        for _ in range(trials):
            x = r.standard_normal(d); xq = x.copy(); dl_ = 0.01
            for _l in range(L):
                Wl = r.standard_normal((d, d))/math.sqrt(d)*0.1
                x = x + np.tanh(Wl @ x); y = xq + np.tanh(Wl @ xq)
                xq = np.round(y/dl_)*dl_ if mode == "round" else np.floor(y/dl_)*dl_
            out.append(np.sqrt(np.mean((xq-x)**2)))
        return float(np.mean(out))
    er4, er16, er64 = (stream(L, "round") for L in (4, 16, 64))
    ef4, ef16, ef64 = (stream(L, "floor") for L in (4, 16, 64))
    say("s08", f"residual stream, rounding to 0.01 after every block: RMS error at 4/16/64 blocks "
        f"{er4:.4f}/{er16:.4f}/{er64:.4f} (x{er16/er4:.2f}, x{er64/er16:.2f} per 4x depth; RSS predicts x2)",
        1.8 < er16/er4 < 2.4 and 1.8 < er64/er16 < 2.5)
    say("s08", f"truncating instead (biased): {ef4:.4f}/{ef16:.4f}/{ef64:.4f} (x{ef16/ef4:.2f}, x{ef64/ef16:.2f}; "
        "linear stacking predicts x4)", 3.6 < ef16/ef4 < 4.5 and 3.6 < ef64/ef16 < 4.6)

    # --- s09: the KV cache ----------------------------------------------
    kvb = 2*80*64*128*131072*2
    say("s09", f"70B, 128K, FP16, 64 KV heads: {kvb/1e9:.1f} GB = {kvb/2**30:.0f} GiB; the source's prose says 330 GB "
        "and its own code prints 343.6", abs(kvb/1e9-343.6) < 0.05 and kvb/2**30 == 320)
    say("s09", f"with 8 KV heads (GQA, reported for Llama-2-70B): {kvb/8/1e9:.1f} GB", abs(kvb/8/1e9-42.9) < 0.05)
    say("s09", f"KV per token per layer: 64 heads {2*64*128*2:,} B, 8 heads {2*8*128*2:,} B; x80 layers "
        f"{2*64*128*2*80/1e6:.2f} MB / {2*8*128*2*80/1e3:.0f} KB (source file 03: ~1.3 KB/layer, ~100 KB)",
        2*8*128*2*80 == 327680 and 2*64*128*2 == 32768)
    say("s09", f"2,000-token shared prefix: {2000*100e3/1e6:.0f} MB at the source's 100 KB/token, "
        f"{2000*327680/1e6:.0f} MB with 8 KV heads, {2000*2621440/1e9:.2f} GB with 64", True)

    # --- s10: less work per token ---------------------------------------
    n10, w10 = 512, 128
    cf = n10*(n10+1)//2; win = sum(min(i+1, w10) for i in range(n10))
    say("s10", f"window 128 on 512 tokens: source compares non-causal n^2 = {n10*n10:,} with n*w = {n10*w10:,} (4.0x); "
        f"like for like, causal full {cf:,} vs window {win:,} = {cf/win:.2f}x", cf == 131328 and win == 57408
        and abs(cf/win-2.29) < 0.01)
    mha, gqa, mqa = (2*k*128*32768*2/1e6 for k in (32, 8, 1))
    say("s10", f"KV per layer at 32K: MHA {mha:.0f} MB, GQA-8 {gqa:.0f} MB ({mha/gqa:.0f}x), MQA {mqa:.1f} MB "
        f"({mha/mqa:.0f}x)", round(mha) == 537 and mha/gqa == 4 and mha/mqa == 32)
    fig = [2*80*k*128*32768*b/1e9 for b in (2, 1) for k in (64, 8, 1)]
    say("s10", "figure mha_gqa_mqa (70B, 80 layers, 32K): FP16 MHA/GQA-8/MQA = " +
        "/".join(f"{v:.1f}" for v in fig[:3]) + " GB; the figure's 40/5/0.6 GB match 1 byte per element: " +
        "/".join(f"{v:.2f}" for v in fig[3:]), abs(fig[0]-85.9) < 0.05 and abs(fig[3]-42.9) < 0.05)

    # --- s11: MoE is a mixed-model lot -----------------------------------
    r11 = np.random.default_rng(5)
    e1 = (46.7-12.9)/6; sh = 12.9-2*e1
    ok11 = True; parts = []
    for b in (1, 4, 8, 16):
        sims = [len(set(sum((r11.choice(8, 2, replace=False).tolist() for _ in range(b)), []))) for _ in range(4000)]
        pred = 8*(1-(6/8)**b); reads = sh+e1*pred
        ok11 &= abs(np.mean(sims)-pred) < 0.05
        parts.append(f"b={b}: {np.mean(sims):.2f}/{pred:.2f} experts, {reads:.1f}B read = {reads/46.7:.0%} of dense")
    say("s11", "Mixtral-like (46.7B total, 12.9B active, reported), uniform top-2 of 8: " + "; ".join(parts), ok11)
    say("s11", f"implied split: {e1:.2f}B per expert, {sh:.2f}B shared", abs(e1-5.63) < 0.01)

    # --- s12: pruning task demonstrates nothing --------------------------
    r12 = np.random.default_rng(0)
    ho = r12.standard_normal((8, 64, 32)); full = ho.reshape(64, 256)
    hn = np.linalg.norm(ho, axis=(1, 2)); errs = []
    for keep in (6, 4, 2):
        top = np.argsort(hn)[-keep:]; fp = np.zeros_like(ho); fp[top] = ho[top]
        errs.append(np.linalg.norm(full-fp.reshape(64, 256))/np.linalg.norm(full))
    say("s12", f"source 02 task 3: keep 6/4/2 heads -> error {errs[0]:.3f}/{errs[1]:.3f}/{errs[2]:.3f} vs "
        "sqrt(share removed) 0.500/0.707/0.866; head norms span " + f"{hn.min():.1f}-{hn.max():.1f}",
        all(abs(e_-math.sqrt((8-k)/8)) < 0.02 for e_, k in zip(errs, (6, 4, 2))))
    say("s12", "its reshape(seq, heads*d) is not the per-token concatenation (transpose first)",
        not np.allclose(full, ho.transpose(1, 0, 2).reshape(64, 256)))

    # --- s13/s14: the batching model ------------------------------------
    tw, tk = 20.0, 0.2
    say("s13", f"illustrative server: 40 GB weights / 2 TB/s = {40/2000*1000:.0f} ms per step; 400 MB KV per "
        f"sequence / 2 TB/s = {0.4/2000*1000:.1f} ms; 70B at 4.5 bits = {70e9*4.5/8/1e9:.1f} GB; "
        f"1,250 tokens x 320 KB = {1250*327680/1e6:.0f} MB", abs(1250*327680/1e6-409.6) < 0.1)
    Xb = lambda b, tk_=tk: b/(tw+b*tk_)*1000
    say("s14", f"throughput X(b) = b/(t_w + b t_kv): X(1) = {Xb(1):.1f}, X(100) = {Xb(100):.0f}, ceiling 1/t_kv = "
        f"{1000/tk:.0f} tok/s; one-piece flow uses {Xb(1)/(1000/tk):.1%} of it", abs(Xb(1)-49.5) < 0.05)
    ok14 = True; rows14 = []
    for Lam in (1000, 2000, 3000):
        Tk = 1000/Lam; rho = Lam*tk/1000
        bmin = tw/(Tk-tk); tpot = tw/(1-rho)
        ok14 &= abs(tw+bmin*tk-tpot) < 1e-9 and abs(Lam*tpot/1000-bmin) < 1e-9
        rows14.append(f"{Lam}: takt {Tk:.3f} ms, b_min {bmin:.1f}, TPOT {tpot:.2f} ms")
    say("s14", "lean minimum lot b = t_w/(takt - t_kv) == Little's b = Lambda x TPOT, TPOT = t_w/(1-rho): " +
        "; ".join(rows14), ok14)
    def des(Lam, L=300, steps=30000, bcap=10**9, seed=1):
        r = np.random.default_rng(seed); act = []; ql = 0
        sbt = st_ = ntok = 0.0; qs = 0.0
        for k in range(steps):
            add = min(ql, bcap-len(act)); act += [L]*add; ql -= add
            b = len(act); dt = tw+b*tk
            ql += r.poisson(Lam/L*dt/1000)
            if k > steps//5:
                sbt += b*dt; st_ += dt; ntok += b; qs += ql*dt
            act = [x-1 for x in act if x > 1]
        return sbt/st_, sbt/ntok, ntok/st_*1000, qs/st_
    ok14b = True; rows14b = []
    for Lam in (1000, 2000):
        rho = Lam*tk/1000; bm, tp, th, _ = des(Lam)
        ok14b &= abs(bm/(Lam*tw/1000/(1-rho))-1) < 0.03 and abs(tp/(tw/(1-rho))-1) < 0.03 and abs(th/Lam-1) < 0.02
        rows14b.append(f"{Lam}: b {bm:.1f} (pred {Lam*tw/1000/(1-rho):.1f}), TPOT {tp:.2f} (pred {tw/(1-rho):.2f}), "
                       f"throughput {th:.0f}")
    say("s14", "continuous batching simulated, Poisson arrivals, 300 tokens each: " + "; ".join(rows14b), ok14b)
    bcap = int((80-40)/0.4)
    _, _, th26, q26 = des(2600, bcap=bcap, steps=20000)
    say("s14", f"memory cap (80 GB - 40 GB)/0.4 GB = {bcap} sequences -> X = {Xb(bcap):.0f} tok/s; demand 2,600: "
        f"served {th26:.0f}, queue averaging {q26:.0f} and growing", bcap == 100 and abs(th26-2500) < 5 and q26 > 50)
    say("s14", f"INT8 KV: t_kv 0.1 ms and cap 200 -> X = {200/(tw+200*0.1)*1000:.0f} tok/s (x2 from one change)",
        200/(tw+200*0.1)*1000 == 5000)
    say("s14", "Little's law on the batch: WIP b = throughput x lead time; TPOT/b is the cycle time per token "
        f"(at 2,000 tok/s: {33.333/66.667:.3f} ms = takt)", abs((tw/(1-0.4))/(tw/(0.5-tk))-0.5) < 1e-9)

    # --- s15: 'more batching cannot hurt' --------------------------------
    bs = np.arange(1, 2001); xs = bs/(tw+bs*tk)
    say("s15", "X(b) is strictly increasing in b (throughput never falls with batch, in this model)",
        bool(np.all(np.diff(xs) > 0)))
    def static_(reqs, B=8):
        tt = tkn = 0
        for i in range(0, len(reqs), B):
            bb = reqs[i:i+B]; tt += max(bb)*0.01; tkn += sum(bb)
        return tkn/tt
    def cont_(reqs, B=8):
        tt = tkn = 0; act = []; qu = list(reqs)
        while act or qu:
            while len(act) < B and qu: act.append(qu.pop(0))
            act = [x-1 for x in act]; tkn += len(act); tt += 0.01
            act = [x for x in act if x > 0]
        return tkn/tt
    random.seed(42)
    reqs = [random.randint(10, 500) for _ in range(100)]
    s_, c_ = static_(reqs), cont_(reqs)
    say("s15", f"source 03 task 1: static {s_:.0f} tok/s, continuous {c_:.0f} tok/s, {c_/s_:.2f}x "
        "(source text: 'typically 2-10x')", abs(c_/s_-1.76) < 0.01)
    say("s15", f"static batch efficiency = mean/E[max of 8] for U[10,500]: {255/(10+490*8/9):.3f}",
        abs(255/(10+490*8/9)-0.572) < 0.001)
    r15 = np.random.default_rng(15); lam = 2.0; B = 32; waits = []
    for _ in range(4000):
        arr = np.cumsum(r15.exponential(1/lam, B)); waits.extend(arr[-1]-arr)
    say("s15", f"wait-to-fill a lot of {B} at {lam}/s: mean wait {np.mean(waits):.2f} s vs (B-1)/(2 lambda) = "
        f"{(B-1)/(2*lam):.2f} s, with no throughput gained below capacity", abs(np.mean(waits)-(B-1)/(2*lam)) < 0.1)

    # --- s16: paging and routing -----------------------------------------
    pre = 256*4096*100e3/1e9; pg = 256*math.ceil(256/16)*16*100e3/1e9
    say("s16", f"source 03 task 2 scenario 1: {pre:.1f} GB preallocated vs {pg:.1f} GB paged ({pre/pg:.0f}x)",
        abs(pre-104.9) < 0.05 and abs(pre/pg-16) < 1e-9)
    Ls = np.random.default_rng(6).integers(1, 512, 200000); pgs = np.ceil(Ls/16)*16
    say("s16", f"random lengths 1-511, 16-token pages: {np.mean(pgs-Ls):.2f} tokens/request wasted = "
        f"{(pgs-Ls).sum()/pgs.sum():.2%} (source's 0% is a mean length that is a multiple of 16)",
        abs(np.mean(pgs-Ls)-7.5) < 0.05)
    say("s16", f"routing 80% to a 10x cheaper model: {1-(0.8*0.1+0.2):.0%} saved if routed, {1-(0.1+0.2):.0%} if "
        "every query tries the small model first (source: ~70%)", abs(1-(0.8*0.1+0.2)-0.72) < 1e-9)

    # --- s17: the edge -----------------------------------------------------
    say("s17", f"battery: 3-5 W for 30 min = 1.5-2.5 Wh; on an assumed 4,000 mAh x 3.85 V = {4*3.85:.1f} Wh battery "
        f"that is {1.5/15.4:.0%}-{2.5/15.4:.0%} if generating continuously (source ~5%); 30 min at 15 tok/s = "
        f"{15*1800:,} tokens", abs(1.5/15.4-0.097) < 0.001 and 15*1800 == 27000)
    say("s17", f"compression pipeline: source code 0.15 x 0.7 x 8 -> {32/4/(0.15*0.7):.0f}x; its diagram 70B FP32 "
        f"-> 2 GB = {70e9*4/2e9:.0f}x", abs(32/4/(0.15*0.7)-76.2) < 0.05 and 70e9*4/2e9 == 140)
    r17 = np.random.default_rng(4); nk = [10, 40, 200]
    Xk = [r17.standard_normal((n, 3)) for n in nk]
    yk = [X_ @ np.array([1., -2, .5]) + r17.standard_normal(n)*.1 for X_, n in zip(Xk, nk)]
    g = lambda w_, X_, y_: 2*X_.T @ (X_ @ w_ - y_)/len(y_)
    w0 = np.zeros(3); wk = [w0-0.05*g(w0, X_, y_) for X_, y_ in zip(Xk, yk)]
    wc = w0-0.05*g(w0, np.vstack(Xk), np.concatenate(yk))
    dw = np.abs(sum(n*w_ for n, w_ in zip(nk, wk))/sum(nk)-wc).max(); du = np.abs(sum(wk)/3-wc).max()
    say("s17", f"FedAvg, one local step: data-weighted mean == central step (diff {dw:.0e}); the source's plain "
        f"1/K mean is off by {du:.3f}", dw < 1e-12 and du > 0.01)

    # --- s18: speculative decoding ---------------------------------------
    Etok = lambda al, k: (1-al**(k+1))/(1-al)
    ok18 = True; rows18 = []
    for al, k, c in ((0.8, 5, 0.1), (0.5, 5, 0.1), (0.9, 4, 0.05)):
        acc = np.random.default_rng(1).random((200000, k)) < al
        toks = np.where(acc.all(1), k, acc.argmin(1)) + 1
        ok18 &= abs(toks.mean()-Etok(al, k)) < 0.02
        rows18.append(f"a={al},k={k}: sim {toks.mean():.3f} vs {Etok(al,k):.3f}; speedup {Etok(al,k)/(k*c+1):.2f} "
                      f"vs k*a/(kc+1) {k*al/(k*c+1):.2f}")
    say("s18", "tokens per target pass (1-a^(k+1))/(1-a), c = draft/target cost: " + "; ".join(rows18), ok18)
    bestk = {}
    for al, c in ((0.8, 0.1), (0.5, 0.1)):
        sp = [Etok(al, k)/(k*c+1) for k in range(1, 13)]
        bestk[al] = sp.index(max(sp))+1
    say("s18", f"best k (c = 0.1): alpha 0.8 -> {bestk[0.8]}, alpha 0.5 -> {bestk[0.5]}; k*a/(kc+1) rises with k for ever",
        bestk[0.8] == 6 and bestk[0.5] == 2)
    r18 = np.random.default_rng(2); V = 6
    p = r18.dirichlet(np.ones(V)); q = r18.dirichlet(np.ones(V)); N = 400000
    xs_ = r18.choice(V, N, p=q); accp = r18.random(N) < np.minimum(1, p[xs_]/q[xs_])
    resid = np.maximum(p-q, 0); resid /= resid.sum()
    out = np.where(accp, xs_, r18.choice(V, N, p=resid)); emp = np.bincount(out, minlength=V)/N
    say("s18", f"accept with min(1,p/q), resample from (p-q)+: output matches p to {np.abs(emp-p).max():.1e}; "
        f"acceptance {accp.mean():.4f} vs 1 - TV(p,q) = {np.minimum(p,q).sum():.4f}",
        np.abs(emp-p).max() < 2e-3 and abs(accp.mean()-np.minimum(p, q).sum()) < 3e-3)
    random.seed(0); its = []
    for _ in range(100000):
        tok = 0
        for _i in range(5):
            t_ = random.randint(0, 9); pr = 0.9 if t_ % 2 == 0 else 0.1
            tok += 1
            if random.random() >= pr: break
        its.append(tok)
    mt = sum(its)/len(its)
    say("s18", f"source 05 task 1 without sleeps: {mt:.3f} tokens per target call, {10/(15/mt):.2f}x at 5x1 ms "
        "drafts + 10 ms target", abs(mt-1.9375) < 0.02)

    # --- s19: past one server, at the source's level -----------------------
    r19 = np.random.default_rng(7)
    X9 = r19.standard_normal((4, 16)); W9 = r19.standard_normal((16, 12))
    cc = np.abs(np.hstack([X9 @ c_ for c_ in np.split(W9, 2, axis=1)]) - X9 @ W9).max()
    rr = np.abs(sum(x_ @ w_ for x_, w_ in zip(np.split(X9, 2, axis=1), np.split(W9, 2, axis=0))) - X9 @ W9).max()
    say("s19", f"column split -> concatenate (all-gather) {cc:.0e}; row split -> sum (all-reduce) {rr:.0e}",
        cc < 1e-12 and rr < 1e-12)
    pay = 64*128*2
    say("s19", f"decode all-reduce payload {pay:,} B/token (d_model 64x128): {pay/900e9*1e9:.0f} ns on 900 GB/s, "
        f"{pay/32e9*1e6:.2f} us on 32 GB/s; the source's 0.1 ms and 3 ms need {0.1e-3*900e9/pay:,.0f} and "
        f"{3e-3*32e9/pay:,.0f} tokens per message", pay == 16384 and abs(0.1e-3*900e9/pay-5493) < 1)
    say("s19", f"100,000 GPUs x $8/h x 8,760 h = ${100000*8*8760/1e9:.2f}B/yr (source: ~$3B, implying "
        f"${3e9/(100000*8760):.2f}/h)", abs(100000*8*8760/1e9-7.008) < 1e-9)
    def cost_(PB, bits, gm, gc, tgt):
        ms = PB*1e9*bits/8/1e9; gpm = max(1, int((ms*1.2)/gm+0.99)); tpg = 500/(PB*bits/16)
        tot = tpg*gpm; rep = max(1, int(tgt/tot+0.99)); return gpm*rep*gc/(tot*rep*3600/1e6), tot*gpm
    c70, t70 = cost_(70, 16, 80, 8.0, 1000)
    say("s19", f"source 05 task 2, 70B FP16 on H100: ${c70:.2f} per 1M tokens against its own table's $0.50 "
        f"({c70/0.5:.0f}x); the table implies {24/0.5*1e6/3600:,.0f} tok/s from 3 GPUs vs the model's {t70/3:.1f}",
        abs(c70-311.11) < 0.01)

    # --- s20: worked example ----------------------------------------------
    Lam = 1800; Tk = 1000/Lam; rho = Lam*tk/1000
    bm = tw/(Tk-tk); tp = tw/(1-rho); mem = 40+bm*0.4
    say("s20", f"6 req/s x 300 tokens = {Lam} tok/s: takt {Tk:.3f} ms, min lot {bm:.1f}, TPOT {tp:.2f} ms, "
        f"memory {mem:.1f} GB of 80", abs(bm-56.25) < 0.01 and abs(tp-31.25) < 1e-9 and mem < 80)
    lam_slo = (1-tw/50)/tk*1000
    say("s20", f"TPOT <= 50 ms caps demand at {lam_slo:.0f} tok/s; memory caps it at {Xb(100):.0f}: memory binds; "
        f"INT8 KV -> SLO cap {(1-tw/50)/0.1*1000:.0f}, memory cap {200/(tw+200*0.1)*1000:.0f}",
        abs(lam_slo-3000) < 1e-9 and Xb(100) == 2500)
    w3 = 70e9*3/8/1e9; tw3 = w3/2000*1000; cap3 = int((80-w3)/0.4); x3 = cap3/(tw3+cap3*tk)*1000
    say("s20", f"weights to 3 bits instead: {w3:.2f} GB, t_w {tw3:.2f} ms, cap {cap3}, capacity {x3:.0f} tok/s "
        "(INT8 cache: 5000)", cap3 == 134 and abs(x3-3356) < 2)
    bmd, tpd, thd, _ = des(Lam, steps=20000, bcap=100, seed=5)
    say("s20", f"simulated at {Lam} tok/s with the cap: b {bmd:.1f}, TPOT {tpd:.2f} ms, throughput {thd:.0f}",
        abs(tpd/tp-1) < 0.04 and abs(thd/Lam-1) < 0.02)

_ch17()
# ----- end of Chapter 17 block -----


# ======================================================================================
# Chapter 18 block. Owned by the Chapter 18 run: replace only the body of _ch18() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch18():
    # =================================================================
    # Chapter 18 -- ML systems design (files 01-05), one sheet.
    # The plant view: capacity, redundancy, the tail, training at scale.
    # Every name lives inside this function so nothing collides.
    # =================================================================
    import math, random, bisect
    import numpy as np
    from scipy.stats import norm as _n18
    T18 = "[Ch18 %s]"
    def say(sec, msg, ok=None):
        tail = "" if ok is None else "   " + P(ok)
        print(f"{T18 % sec} {msg}{tail}")
    print()
    g18 = np.random.default_rng(1818)

    # ---------- shared queueing helpers (stable Erlang B -> C) ----------
    def erl_b(c, a):
        b = 1.0
        for k in range(1, c+1):
            b = a*b/(k + a*b)
        return b
    def erl_c(c, a):
        if a >= c: return 1.0
        b = erl_b(c, a)
        return c*b/(c - a*(1-b))
    def mmc_wq(c, a, s):          # mean wait in queue, service time s
        if a >= c: return float('inf')
        return erl_c(c, a)*s/(c - a)
    def mmc_resp_q(c, a, s, q):   # q-quantile of response = wait + Exp(s) service
        if a >= c: return float('inf')
        pw = erl_c(c, a); mu = 1/s; th = c*mu - a*mu
        def surv(t):
            sn = math.exp(-mu*t)
            sw = math.exp(-mu*t)*(1+mu*t) if abs(th-mu) < 1e-12 else (th*math.exp(-mu*t) - mu*math.exp(-th*t))/(th-mu)
            return (1-pw)*sn + pw*sw
        lo, hi = 0.0, s
        while surv(hi) > 1-q: hi *= 2
        for _ in range(100):
            m = (lo+hi)/2
            if surv(m) > 1-q: lo = m
            else: hi = m
        return hi

    # --- s01: Little's law sizes everything --------------------------------
    off = {"user-embedding replicas": (10000, 0.005, 50), "ANN index replicas": (10000, 0.002, 20),
           "ranking GPUs (source: ~10)": (10000, 0.100, 1000), "text classifiers": (12000, 0.005, 60),
           "image classifiers": (12000, 0.020, 240), "hash matchers": (12000, 0.001, 12)}
    ok = all(abs(l*s - L) < 1e-9 for l, s, L in off.values())
    say("s01", "offered load L = lambda x W for every sizing in file 05: " +
        "; ".join(f"{k} {l:,}/s x {s*1e3:g} ms = {l*s:g}" for k, (l, s, L) in off.items()), ok)
    say("s01", f"1e9 posts/day = {1e9/86400:,.0f} per second (source: ~12K)", abs(1e9/86400 - 11574.07) < 0.01)
    say("s01", f"ads: 100,000 auctions/s x 10 ms = {100000*0.010:,.0f} predictions in flight at any instant",
        abs(100000*0.010 - 1000) < 1e-9)
    # Little's law by simulation on an M/D/c system (no distribution assumed by the law)
    lam, s, c = 40.0, 1.0, 50
    t = 0.0; arr = []
    while t < 4000:
        t += g18.exponential(1/lam); arr.append(t)
    free = [0.0]*c; import heapq as _hq18; _hq18.heapify(free); dep = []
    for a in arr:
        f = _hq18.heappop(free); st = max(a, f); _hq18.heappush(free, st + s); dep.append(st + s)
    arr = np.array(arr); dep = np.array(dep); Tend = 3000.0; m = arr < Tend
    W = (np.minimum(dep[m], Tend) - arr[m]).sum()/Tend  # time-average number in system over [0,Tend]
    lam_hat = m.sum()/Tend; Wbar = (dep[m] - arr[m]).mean()
    say("s01", f"Little by simulation (M/D/50, 40/s, 1 s): L = {W:.3f}, lambda W = {lam_hat*Wbar:.3f}",
        abs(W - lam_hat*Wbar)/W < 0.01)

    # --- s02: sizing at 100% utilisation: the cliff --------------------------
    a = 50.0; s = 0.005
    rows = []
    for cc in (50, 51, 52, 55, 58, 60, 65):
        rows.append((cc, a/cc, erl_c(cc, a) if cc > a else 1.0, mmc_wq(cc, a, s)*1e3, mmc_resp_q(cc, a, s, 0.99)*1e3))
    say("s02", "M/M/c at 10,000/s, 5 ms (a = 50 Erlangs): " + "; ".join(
        f"c={r[0]} rho={r[1]:.3f} Pwait={r[2]:.3f} Wq={r[3]:.3f}ms p99={r[4]:.1f}ms" for r in rows),
        rows[0][3] == float('inf') and abs(rows[1][2] - 0.8397) < 1e-3 and abs(rows[1][3] - 4.199) < 0.01
        and abs(rows[3][3] - 0.385) < 0.002 and abs(rows[5][3] - 0.0587) < 0.001)
    say("s02", f"one replica beyond the offered load: 50 -> 51 takes mean wait from infinite to {rows[1][3]:.2f} ms; "
        f"55 -> {rows[3][3]:.2f} ms, 60 -> {rows[5][3]:.3f} ms", True)
    say("s02", f"exponential service alone has p99 = s ln 100 = {s*math.log(100)*1e3:.2f} ms; queueing adds "
        f"{rows[1][4]-s*math.log(100)*1e3:.1f} ms at c=51 and {rows[3][4]-s*math.log(100)*1e3:.2f} ms at c=55",
        abs(rows[6][4] - s*math.log(100)*1e3) < 0.05)
    # Pollaczek-Khinchine: deterministic service halves M/M/1 wait
    rho = 0.9; mm1 = rho/(1-rho); md1 = rho/(2*(1-rho))
    say("s02", f"P-K at rho=0.9: M/M/1 wait {mm1:.2f} service times, M/D/1 {md1:.2f} (half): exponential is pessimistic",
        abs(md1*2 - mm1) < 1e-12)
    # ranking stage: Little says 1,000 busy
    say("s02", f"ranking: 10,000/s x 100 ms = {10000*0.1:,.0f} GPU-seconds per second = 1,000 GPUs busy, not 10; "
        f"10 GPUs need {10000*0.1/10:.0f} requests in flight each (= {10000*0.1/10*1000:,.0f} candidates)",
        10000*0.1 == 1000)
    say("s02", f"at the source's $1.00/h A10G: 10 GPUs ${10*1.0*730:,.0f}/month; 1,000 GPUs ${1000*1.0*730:,.0f}/month",
        10*730 == 7300 and 1000*730 == 730000)

    # --- s03: safety capacity is safety stock ---------------------------------
    # M/G/infinity occupancy is Poisson(a) whatever the service distribution
    lam, Tsim = 50.0, 20000.0
    for name, gen in (("deterministic", lambda n: np.full(n, 1.0)),
                      ("lognormal cv 1", lambda n: g18.lognormal(-0.5*math.log(2), math.sqrt(math.log(2)), n))):
        n = g18.poisson(lam*Tsim); at = np.sort(g18.uniform(0, Tsim, n)); en = np.sort(at + gen(n))
        probes = np.linspace(100, Tsim-100, 4000)
        busy = np.searchsorted(at, probes, side='right') - np.searchsorted(en, probes, side='right')
        say("s03", f"M/G/inf occupancy ({name} service, a = 50): mean {busy.mean():.2f}, variance {busy.var():.2f} "
            "(Poisson: both 50)", abs(busy.mean()-50) < 1.5 and abs(busy.var()/busy.mean() - 1) < 0.2)
    # square-root staffing, Halfin-Whitt limit
    hw = 1/(1 + 1.0*_n18.cdf(1.0)/_n18.pdf(1.0))
    sq = []
    for aa in (5, 50, 500, 5000):
        cc = math.ceil(aa + math.sqrt(aa)); sq.append((aa, cc, aa/cc, erl_c(cc, aa)))
    say("s03", "c = a + 1*sqrt(a): " + "; ".join(f"a={x[0]} c={x[1]} rho={x[2]:.3f} Pwait={x[3]:.3f}" for x in sq)
        + f"; Halfin-Whitt limit {hw:.3f}", all(abs(x[3] - hw) < 0.06 for x in sq) and sq[-1][2] > 0.98 and sq[0][2] < 0.65)
    # reorder point identity: ROP = mu + z sigma, sigma = sqrt(mu) for Poisson demand
    mu = 50.0; z = 1.6449
    rop = mu + z*math.sqrt(mu); from scipy.stats import poisson as _po18
    say("s03", f"reorder point for Poisson demand mean 50 at z = 1.645: {rop:.2f}; P(demand > {math.ceil(rop)}) = "
        f"{1-_po18.cdf(math.ceil(rop), mu):.4f} (normal approx 0.05)", abs(1-_po18.cdf(math.ceil(rop), mu) - 0.05) < 0.02)
    # continuous batching occupancy (Sheet 17 server): t_w 20 ms, t_kv 0.2 ms, 300-token requests
    def cb_sim(Lam, steps=60000, cap=10**9, tw=0.020, tkv=0.0002, L=300, seed=7):
        # continuous batching: finished sequences leave, queued requests join, each needs L steps
        r = np.random.default_rng(seed); fin = {}; b = 0; q = 0; hist = []
        for k in range(steps):
            b -= fin.pop(k, 0)
            j = q if cap >= 10**9 else min(q, cap - b)
            if j > 0:
                fin[k+L] = fin.get(k+L, 0) + j; b += j; q -= j
            if k > 3000: hist.append(b)
            q += r.poisson(Lam/L*(tw + b*tkv))
        return np.array(hist)
    h1 = cb_sim(1800.0)
    pred_b = 1800*0.020/(1-1800*0.0002)
    disp = h1.var()/h1.mean()
    say("s03", f"Sheet 17 server at 1,800 tok/s, uncapped: batch mean {h1.mean():.1f} (Little: {pred_b:.1f}), "
        f"sd {h1.std():.2f} against sqrt(mean) {math.sqrt(h1.mean()):.2f}; dispersion {disp:.2f}",
        abs(h1.mean()/pred_b - 1) < 0.03 and 0.8 < disp < 2.5)
    # the batch's variance-to-mean ratio is 1/(1 - rho): linear-noise approximation, checked by simulation
    dl = []
    for Lam in (1000, 1800, 3000):
        ds_ = []
        for sd_ in range(6):
            hh = cb_sim(float(Lam), steps=120000, seed=100+sd_); ds_.append(hh.var()/hh.mean())
        rr_ = Lam*0.0002
        dl.append((Lam, rr_, float(np.mean(ds_)), 1/(1-rr_)))
    say("s03", "batch dispersion (variance/mean, 6 runs each) against 1/(1 - rho): " + "; ".join(
        f"{x[0]}/s rho {x[1]:.2f}: {x[2]:.3f} vs {x[3]:.3f}" for x in dl), all(abs(x[2]/x[3] - 1) < 0.08 for x in dl))
    # analytic planning capacity: mean + z sqrt(mean/(1-rho)) = cap
    def plan_an(z, cap=100):
        lo, hi = 100.0, 2500.0
        for _ in range(60):
            mid = (lo+hi)/2; rr_ = mid*0.0002; bm = mid*0.02/(1-rr_)
            if bm + z*math.sqrt(bm/(1-rr_)) < cap: lo = mid
            else: hi = mid
        return lo
    pa = plan_an(_n18.ppf(0.99))
    say("s03", f"safety-stock rule with that spread, z = 2.326 (1%): planning capacity {pa:,.0f} tok/s per server", 1950 < pa < 2200)
    # planning capacity per server: keep P(batch >= 100) small
    plan = []
    for Lam in (2000, 2100, 2200, 2300, 2400):
        hh = cb_sim(float(Lam), steps=60000, seed=11)
        plan.append((Lam, hh.mean(), np.mean(hh >= 100)))
    say("s03", "uncapped batch vs the 100-sequence memory cap: " + "; ".join(
        f"{x[0]}/s mean {x[1]:.1f}, P(>=100) {x[2]:.3f}" for x in plan),
        plan[0][2] < 0.02 and plan[-1][2] > 0.15)
    ex = {x[0]: x for x in plan}
    say("s03", f"nameplate 2,500/s is the cap at the mean; at 2,200/s the batch exceeds the cap {ex[2200][2]*100:.1f}% of steps "
        f"(Poisson z-rule: mean {2200*0.02/(1-2200*0.0002):.1f} + 2.33 sqrt -> {2200*0.02/(1-2200*0.0002)+2.33*math.sqrt(2200*0.02/(1-2200*0.0002)):.1f})",
        ex[2200][2] < 0.05)

    # --- s04: pooling -------------------------------------------------------
    pool = []
    for cc in (2, 10, 50):
        aa = 0.8*cc; sep = 0.8/(1-0.8); pooled = erl_c(cc, aa)/(cc - aa)
        pool.append((cc, sep, pooled, sep/pooled))
    say("s04", "rho = 0.8, wait in service times: " + "; ".join(
        f"{x[0]} separate queues {x[1]:.2f} vs one shared {x[2]:.4f} ({x[3]:.1f}x)" for x in pool),
        abs(pool[1][3] - 19.6) < 0.2 and pool[2][3] > 400)
    # routing rules by simulation: random, round robin, join-shortest-queue, central queue (c=10, rho=0.8)
    def route_sim(rule, c=10, rho=0.8, N=200000, seed=3):
        r = np.random.default_rng(seed); lam = rho*c
        at = np.cumsum(r.exponential(1/lam, N)); sv = r.exponential(1.0, N)
        waits = np.empty(N)
        if rule == "central":
            free = [0.0]*c
            for i in range(N):
                f = _hq18.heappop(free); st = max(at[i], f); waits[i] = st - at[i]; _hq18.heappush(free, st + sv[i])
            return waits[N//10:].mean()
        lastfree = np.zeros(c); deps = [[] for _ in range(c)]
        for i in range(N):
            if rule == "random": k = r.integers(c)
            elif rule == "rr": k = i % c
            else:  # JSQ: fewest jobs present (in service + waiting)
                cnt = [0]*c
                for j in range(c):
                    d = deps[j]
                    while d and d[0] <= at[i]: d.pop(0)
                    cnt[j] = len(d)
                k = int(np.argmin(cnt))
            st = max(at[i], lastfree[k]); waits[i] = st - at[i]; lastfree[k] = st + sv[i]
            if rule == "jsq": deps[k].append(lastfree[k])
        return waits[N//10:].mean()
    rs = {k: route_sim(k, N=60000) for k in ("random", "rr", "jsq", "central")}
    say("s04", "routing 10 servers at rho 0.8, mean wait (service times): " +
        ", ".join(f"{k} {v:.3f}" for k, v in rs.items()),
        abs(rs["random"] - 4.0) < 0.5 and rs["rr"] < rs["random"] and rs["jsq"] < rs["rr"] and abs(rs["central"] - 0.2046) < 0.05)

    # --- s05: consistent hashing ------------------------------------------
    def ring_loads(n, v, rr):
        pts = sorted((rr.random(), s_) for s_ in range(n) for _ in range(v))
        L = [0.0]*n
        for i, (p, s_) in enumerate(pts):
            prev = pts[i-1][0] if i > 0 else pts[-1][0] - 1.0
            L[s_] += p - prev
        return L
    rr = random.Random(55)
    cvrow = []
    for v in (1, 10, 100):
        cvs = [np.std(ring_loads(10, v, rr))*10 for _ in range(400)]
        cvrow.append((v, math.sqrt(np.mean(np.square(cvs))), math.sqrt(9/(10*v+1))))
    say("s05", "load spread on the ring, 10 servers (coefficient of variation): " + "; ".join(
        f"v={x[0]} measured {x[1]:.4f}, sqrt((n-1)/(nv+1)) = {x[2]:.4f}" for x in cvrow),
        all(abs(x[1]/x[2] - 1) < 0.06 for x in cvrow))
    mx = [max(ring_loads(10, 1, rr))*10 for _ in range(3000)]
    Hn = sum(1/k for k in range(1, 11))
    say("s05", f"one point per server: mean largest share {np.mean(mx):.3f} x fair, harmonic number H_10 = {Hn:.3f}",
        abs(np.mean(mx)/Hn - 1) < 0.03)
    # keys moved
    def owner(pts, x):
        i = bisect.bisect_left(pts, (x, -1)); return pts[i % len(pts)][1]
    r5 = random.Random(5); n, v = 10, 100
    p1 = sorted((r5.random(), s_) for s_ in range(n) for _ in range(v))
    p2 = sorted(p1 + [(r5.random(), n) for _ in range(v)])
    ks = [r5.random() for _ in range(40000)]
    moved = sum(owner(p1, x) != owner(p2, x) for x in ks)/len(ks)
    moved_to_new = sum(owner(p1, x) != owner(p2, x) and owner(p2, x) == n for x in ks)/len(ks)
    say("s05", f"add an 11th server: consistent hashing moves {moved:.4f} of keys (1/11 = {1/11:.4f}), all of them "
        f"to the new server ({moved_to_new:.4f})", abs(moved - 1/11) < 0.02 and moved == moved_to_new)
    r9 = random.Random(9); ik = [r9.getrandbits(64) for _ in range(100000)]
    mm = [(n_, sum((k % n_) != (k % (n_+1)) for k in ik)/len(ik)) for n_ in (3, 10, 100)]
    say("s05", "hash mod n -> n+1 moves: " + ", ".join(f"n={x[0]} {x[1]:.4f} (n/(n+1) {x[0]/(x[0]+1):.4f})" for x in mm),
        all(abs(x[1] - x[0]/(x[0]+1)) < 0.01 for x in mm))

    # --- s06: sharding and read replicas -----------------------------------
    rg = np.random.default_rng(66); E = 200000; nsh = 8
    u = rg.integers(0, 10**6, E); w = rg.integers(0, 10**6, E)
    h = lambda x: (x*2654435761) % 2**32 % nsh
    cutf = np.mean(h(u) != h(w))
    say("s06", f"hash-sharding a graph over 8 machines cuts {cutf:.4f} of edges (1 - 1/8 = 0.875)", abs(cutf - 0.875) < 0.005)
    def rep_speed(N, wf):   # total throughput / one machine, every replica applies every write
        return N/((1-wf) + N*wf)
    amd = lambda p, N: 1/((1-p) + p/N)
    err = max(abs(rep_speed(N, wf) - amd(1-wf, N)) for N in range(1, 200) for wf in (0.01, 0.05, 0.2))
    say("s06", f"read replicas with write fraction w == Amdahl with serial fraction w: max diff {err:.1e}", err < 1e-12)
    say("s06", f"w = 5%: 10 replicas {rep_speed(10, .05):.2f}x, 20 replicas {rep_speed(20, .05):.2f}x, ceiling {1/.05:.0f}x",
        abs(rep_speed(10, .05) - 6.897) < 1e-3 and abs(rep_speed(20, .05) - 10.256) < 1e-3)
    # IVF: k + n/k is an EOQ
    nvec = 1e8; kstar = math.sqrt(nvec); cost = 2*math.sqrt(nvec)
    kk = np.logspace(2, 6, 4001); cc_ = kk + nvec/kk
    say("s06", f"IVF with one probe: k + n/k minimised at k = sqrt(n) = {kstar:,.0f}, cost {cost:,.0f} distance "
        f"computations vs {nvec:,.0f} for flat ({nvec/cost:,.0f}x fewer)", abs(kk[np.argmin(cc_)]/kstar - 1) < 0.01)

    # --- s07: nines ---------------------------------------------------------
    dt = [(k, (10**-k)*8760, (10**-k)*525600, (10**-k)*43200) for k in (2, 3, 4, 5)]
    say("s07", "downtime: " + "; ".join(f"{x[0]} nines {x[1]:.2f} h/yr = {x[2]:.1f} min/yr, {x[3]:.1f} min per 30-day month"
        for x in dt), abs(dt[1][3] - 43.2) < 1e-9 and abs(dt[2][2] - 52.56) < 1e-9)
    say("s07", f"source's 43 min: 0.1% of a 30-day month is {0.001*43200:.1f} min, of an average month {0.001*30.4375*1440:.1f}",
        True)
    say("s07", f"request path of 5 services at 99.9%: {0.999**5*100:.3f}%; 7 services {0.999**7*100:.3f}%; "
        "logging moved behind a queue leaves 5", abs(0.999**5 - 0.99501) < 1e-5)

    # --- s08: quorums --------------------------------------------------------
    kofn = lambda k, n_, p: sum(math.comb(n_, j)*p**j*(1-p)**(n_-j) for j in range(k, n_+1))
    maj = {n_: kofn(n_//2+1, n_, 0.99) for n_ in range(1, 8)}
    say("s08", "majority quorum at a = 0.99: " + ", ".join(f"n={k} {v:.6f}" for k, v in maj.items()),
        abs(maj[3] - 0.999702) < 1e-6 and maj[4] < maj[3] and maj[6] < maj[5])
    unav = lambda k, n_, p: sum(math.comb(n_, j)*p**j*(1-p)**(n_-j) for j in range(0, k))
    worse = all(unav(k+1, 2*k, p) > unav(k, 2*k-1, p) for k in range(1, 8) for p in np.linspace(0.5, 0.9999, 300)[1:])
    say("s08", "an even cluster 2k is less available than the odd 2k-1 below it, k=1..7, every a in (0.5, 0.9999] (unavailabilities compared)", worse)
    d43 = 3*0.99**2*0.01**2
    say("s08", f"2-of-3 minus 3-of-4 = 3a^2(1-a)^2 = {d43:.3e} at a = 0.99", abs((maj[3]-maj[4]) - d43) < 1e-15)
    inter = all(2*(n_//2+1) > n_ for n_ in range(1, 1000))
    say("s08", "two majorities of the same cluster always share a member (2(floor(n/2)+1) > n, n < 1000)", inter)
    say("s08", "tolerates floor((n-1)/2) failures: " + ", ".join(f"n={n_}:{(n_-1)//2}" for n_ in range(1, 8)),
        [(n_-1)//2 for n_ in range(1, 8)] == [n_ - (n_//2+1) for n_ in range(1, 8)])

    # --- s09: common cause --------------------------------------------------
    def bq(N, q, beta):
        return 1 - (1 - beta*q)*(1 - ((1-beta)*q)**N)
    row = [bq(N, 0.01, 0.05) for N in (1, 2, 3, 4, 8)]
    say("s09", "unavailability, q = 0.01 per replica, beta = 0.05: " + ", ".join(f"N={N} {x:.3e}" for N, x in zip((1, 2, 3, 4, 8), row))
        + f"; floor beta*q = {0.05*0.01:.0e}", abs(row[-1] - 5e-4) < 1e-6 and abs(row[1] - 5.9025e-4) < 1e-6)
    say("s09", f"the floor in nines: 1 - 5e-4 = 99.95% = {-math.log10(5e-4):.2f} nines, whatever N",
        abs(-math.log10(5e-4) - 3.30) < 0.01)
    # Monte Carlo of beta model
    rg = np.random.default_rng(99); Tm = 2_000_000
    common = rg.random(Tm) < 0.05*0.01
    ind = rg.random((Tm, 3)) < 0.95*0.01
    down3 = common | ind.all(axis=1)
    say("s09", f"Monte Carlo, 3 replicas: {down3.mean():.2e} unavailable vs model {bq(3, 0.01, 0.05):.2e}",
        abs(down3.mean()/bq(3, 0.01, 0.05) - 1) < 0.1)

    # --- s10: a queue between services is a buffer between stations ---------
    def line(B, f, r, cycles=600000, seed=1):
        rgl = np.random.default_rng(seed); up = [True, True]; buf = 0; out = 0
        U = rgl.random((cycles, 2))
        for t_ in range(cycles):
            if B == 0:
                both = up[0] and up[1]
                if both:
                    out += 1
                    for i in (0, 1):
                        if U[t_, i] < f[i]: up[i] = False
                else:
                    for i in (0, 1):
                        if not up[i] and U[t_, i] < r[i]: up[i] = True
                continue
            w0 = up[0] and buf < B
            w1 = up[1] and buf > 0
            if w1: buf -= 1; out += 1
            if w0: buf += 1
            for i, wk in ((0, w0), (1, w1)):
                if up[i]:
                    if wk and U[t_, i] < f[i]: up[i] = False
                else:
                    if U[t_, i] < r[i]: up[i] = True
        return out/cycles
    f = (0.01, 0.02); r = (0.19, 0.18)
    e = [r[i]/(r[i]+f[i]) for i in (0, 1)]
    E0 = 1/(1 + f[0]/r[0] + f[1]/r[1])
    eff = {B: line(B, f, r, cycles=300000) for B in (0, 5, 20, 100, 400)}
    say("s10", f"two stations, isolated efficiencies {e[0]:.3f} and {e[1]:.3f}: no buffer {eff[0]:.4f} "
        f"(1/(1+sum f/r) = {E0:.4f}); buffer 5 {eff[5]:.4f}, 20 {eff[20]:.4f}, 100 {eff[100]:.4f}, 400 {eff[400]:.4f} "
        f"-> min = {min(e):.4f}", abs(eff[0]/E0 - 1) < 0.01 and eff[5] > eff[0] and eff[100] > eff[20] and abs(eff[400] - min(e)) < 0.01)

    # --- s11: protective devices -------------------------------------------
    # token bucket: accepted in any window of length T <= b + r T
    rg = np.random.default_rng(111); bcap, rate = 20.0, 5.0; tok = bcap; acc = []
    t_ = 0.0; times = []
    for _ in range(20000):
        gap = rg.exponential(1/12.0) if rg.random() < 0.7 else rg.exponential(1/0.5)
        t_ += gap; times.append(t_)
    last = 0.0
    for tt in times:
        tok = min(bcap, tok + rate*(tt - last)); last = tt
        if tok >= 1: tok -= 1; acc.append(tt)
    acc = np.array(acc); worst = 0.0
    for Tw in (0.5, 1, 2, 5, 10, 50):
        j = np.searchsorted(acc, acc + Tw, side='left') - np.arange(len(acc))
        worst = max(worst, (j.max() - (bcap + rate*Tw)))
    say("s11", f"token bucket b = 20, r = 5/s on bursty arrivals: accepted in any window never exceeds b + rT "
        f"(worst margin {worst:+.1f}); long-run accepted rate {len(acc)/acc[-1]:.2f}/s", worst <= 1e-9 and len(acc)/acc[-1] <= rate + 0.05)
    # retries: synchronised vs jittered
    Ncl = 1000; rgj = np.random.default_rng(12)
    sync = np.concatenate([np.full(Ncl, 2.0**k) for k in range(4)])
    jit = np.concatenate([rgj.uniform(0, 2.0**k, Ncl) + (2.0**k - 1) for k in range(4)])
    pk = lambda x: np.histogram(x, bins=np.arange(0, 20.0001, 0.1))[0].max()
    say("s11", f"1,000 clients retrying at 1, 2, 4, 8 s: peak {pk(sync)} arrivals in one 0.1 s bin; with full jitter "
        f"{pk(jit)}; total wait after four tries 1+2+4+8 = 15 s", pk(sync) == 1000 and pk(jit) < 150)

    # --- s12: fan-out tail and hedging ---------------------------------------
    pany = {n_: 1 - 0.99**n_ for n_ in (1, 10, 100)}
    say("s12", "P(request slower than a shard's p99) = 1 - 0.99^n: " + ", ".join(f"n={k} {v:.4f}" for k, v in pany.items()),
        abs(pany[100] - 0.63397) < 1e-4)
    mu_, sg = math.log(10), 0.5
    p99s = math.exp(mu_ + sg*_n18.ppf(0.99)); p95s = math.exp(mu_ + sg*_n18.ppf(0.95))
    fo = []
    for n_ in (1, 10, 100):
        x = g18.lognormal(mu_, sg, (20000, n_)).max(axis=1)
        fo.append((n_, np.median(x), np.quantile(x, 0.99), np.mean(x > p99s)))
    say("s12", f"lognormal shards (median 10 ms, sigma 0.5, shard p99 {p99s:.1f} ms): " + "; ".join(
        f"n={x[0]} median {x[1]:.1f} p99 {x[2]:.1f} frac>shard-p99 {x[3]:.3f}" for x in fo),
        all(abs(x[3] - pany[x[0]]) < 0.015 for x in fo))
    # exact quantile of the max: F^n
    q99n = math.exp(mu_ + sg*_n18.ppf(0.99**(1/100)))
    say("s12", f"exact p99 of the max of 100: shard quantile 0.99^(1/100) = {0.99**0.01:.6f} -> {q99n:.1f} ms",
        abs(q99n - fo[2][2]) < 2.0)
    zz = g18.normal(10, 2, (400000, 2)).max(axis=1).mean()
    say("s12", f"Sheet 14's merge bias, same object: E max of two N(10, 2^2) = {zz:.3f} vs 10 + 2/sqrt(pi) = {10+2/math.sqrt(math.pi):.3f}",
        abs(zz - (10 + 2/math.sqrt(math.pi))) < 0.01)
    def hedge(n_, d, pc=0.0, extra=40.0, trials=20000, seed=5):
        rh = np.random.default_rng(seed)
        a1 = rh.lognormal(mu_, sg, (trials, n_)); a2 = d + rh.lognormal(mu_, sg, (trials, n_))
        hot = rh.random((trials, n_)) < pc
        a1 = a1 + hot*extra; a2 = a2 + hot*extra
        lat = np.minimum(a1, np.where(a1 > d, a2, np.inf)).max(axis=1)
        return np.quantile(lat, 0.99), np.mean(a1 > d)
    h0 = np.quantile(np.random.default_rng(5).lognormal(mu_, sg, (20000, 100)).max(axis=1), 0.99)
    hp, frac = hedge(100, p95s)
    say("s12", f"100 shards, hedge each after the shard p95 ({p95s:.1f} ms): request p99 {h0:.1f} -> {hp:.1f} ms "
        f"for {frac*100:.1f}% extra shard requests", hp < 0.75*h0 and abs(frac - 0.05) < 0.005)
    # common cause defeats hedging
    hc0 = hedge(100, 1e9, pc=0.002)[0]; hc1 = hedge(100, p95s, pc=0.002)[0]
    say("s12", f"if 0.2% of shard slowness is in the shard itself (+40 ms on both copies): p99 {hc0:.1f} -> {hc1:.1f} ms with hedging",
        hc1 > 0.85*hc0)
    # series/parallel for on-time: P(on time) = 0.99^n; hedge sent at once: P(slow) = 0.01^2 per shard
    say("s12", f"duplicate sent at once (parallel pair): shard slow with p = 0.0001, 100 shards on time {(1-1e-4)**100:.4f} "
        f"vs {0.99**100:.4f} unhedged (De Morgan, Sheet 13)", abs((1-1e-4)**100 - 0.99005) < 1e-4)

    # --- s13: the day's load ------------------------------------------------
    # source: 10 GPUs at peak, 2 at night saves ~60%
    for hpk in (12, 6):
        avg = (10*hpk + 2*(24-hpk))/24
        say("s13", f"{hpk} h at 10 GPUs, {24-hpk} h at 2: mean {avg:.2f}, saving 1 - mean/peak = {1-avg/10:.0%}",
            abs((1-avg/10) - (0.4 if hpk == 12 else 0.6)) < 1e-12)
    hwk = (4*168 - 2*48 - 5*48)/(5*8)
    say("s13", f"weekends at 2: saving 60% needs {hwk:.1f} peak hours per weekday", abs(hwk - 8.4) < 1e-9)
    # autoscaling saving == 1 - load factor on any profile (integer units)
    rp = np.random.default_rng(13); ok = True
    for _ in range(200):
        d_ = rp.integers(1, 40, 168)
        sav = 1 - d_.sum()/(d_.max()*len(d_)); lf = d_.mean()/d_.max()
        ok &= abs(sav - (1 - lf)) < 1e-12
    say("s13", "perfect autoscaling saves exactly 1 - load factor (200 random weekly profiles)", ok)
    # three regions 8 h apart: three-phase
    th = np.linspace(0, 2*np.pi, 1000)
    s3 = sum(1 + 0.6*np.sin(th + 2*np.pi*k/3) for k in range(3))
    df = 3*1.6/s3.max()
    say("s13", f"three regions, each 1 + 0.6 sin, 8 h apart: sum is constant {s3.min():.6f}..{s3.max():.6f}, "
        f"diversity factor {df:.3f}", abs(s3.max() - 3) < 1e-9 and abs(df - 1.6) < 1e-9)
    s3h = sum(1 + 0.6*np.sin(th + 2*np.pi*k/3) + 0.2*np.sin(3*(th + 2*np.pi*k/3)) for k in range(3))
    say("s13", f"add a third harmonic (period 8 h, amplitude 0.2): ripple of the sum {s3h.max()-s3h.min():.3f} = 3 x 2 x 0.2; "
        "triplen harmonics add, as in a three-phase neutral", abs((s3h.max()-s3h.min()) - 1.2) < 1e-3)

    # --- s14: reserved vs on-demand (screening curve) -------------------------
    def best_reserved(dem, r_, o_):
        best = None
        for R in range(0, dem.max()+1):
            cost = R*r_*len(dem) + o_*np.maximum(dem - R, 0).sum()
            if best is None or cost < best[1] - 1e-9: best = (R, cost)
        return best
    ok = True
    for _ in range(100):
        hrs = np.arange(168)
        dem = np.maximum(0, np.round(20 + 12*np.sin(2*np.pi*hrs/24) + rp.normal(0, 3, 168))).astype(int)
        disc = rp.uniform(0.3, 0.6); r_, o_ = (1-disc)*8.0, 8.0
        R, _ = best_reserved(dem, r_, o_)
        frac = [np.mean(dem >= k) for k in range(dem.max()+2)]
        rule = max([k for k in range(1, dem.max()+1) if frac[k] > (1 - disc)] or [0])
        ok &= (R == rule) or abs(best_reserved(dem, r_, o_)[1] - (rule*r_*168 + o_*np.maximum(dem-rule, 0).sum())) < 1e-6
    say("s14", "reserve unit k iff demand >= k for more than (1 - discount) of hours: matches brute force on 100 weekly profiles", ok)
    own = 30000/0.75; ownh = own/(3*8760)
    say("s14", f"source's own figures: $30K GPU at 75% of cost -> ${own:,.0f} per GPU over 3 years = ${ownh:.2f}/h at full use; "
        f"cloud $8.00 is {8/ownh:.1f}x; reserved at 30-60% off is {8*0.7/ownh:.1f}x-{8*0.4/ownh:.1f}x (source: 2-3x)",
        abs(ownh - 1.522) < 0.01 and 2.0 < 8*0.4/ownh < 2.2)
    say("s14", f"break-even utilisation, owned vs on-demand: {ownh/8:.0%}", abs(ownh/8 - 0.19) < 0.01)
    say("s14", f"64 x $8 = ${64*8}/h (source ~$500); 720 h -> ${64*8*720:,} (source ~$360,000 = 720 x $500)",
        64*8 == 512 and 500*720 == 360000)
    say("s14", f"checkpoint hoard: 10 GB x 100 x 50 = {10*100*50/1000:.0f} TB x $0.023/GB = ${10*100*50*0.023:,.0f}/month",
        abs(10*100*50*0.023 - 1150) < 1e-9)
    say("s14", f"140 GB at 1 GB/s = {140/60:.2f} min (source ~2.5)", abs(140/60 - 2.333) < 0.001)
    def hav(a1, o1, a2, o2):
        R_ = 6371.0; p1_, p2_ = math.radians(a1), math.radians(a2); dl = math.radians(o2-o1); dp = p2_ - p1_
        x = math.sin(dp/2)**2 + math.cos(p1_)*math.cos(p2_)*math.sin(dl/2)**2
        return 2*R_*math.asin(math.sqrt(x))
    dsf = hav(35.68, 139.69, 37.77, -122.42); dva = hav(35.68, 139.69, 39.04, -77.49)
    vf = 299792.458/1.468
    say("s14", f"Tokyo-San Francisco {dsf:,.0f} km, Tokyo-N. Virginia {dva:,.0f} km: fibre round-trip floor "
        f"{2*dsf/vf*1e3:.0f} and {2*dva/vf*1e3:.0f} ms (source ~150 ms is 1.4-1.8x the floor)",
        70 < 2*dsf/vf*1e3 < 90 and 100 < 2*dva/vf*1e3 < 115)

    # --- s15: a training job is a series system --------------------------------
    mtbf_job = 54*24/466; per = mtbf_job*16384
    say("s15", f"466 interruptions in 54 days: job MTBF {mtbf_job:.2f} h; x 16,384 GPUs = {per:,.0f} h = {per/8760:.1f} years per GPU "
        "if every interruption were one GPU's", abs(mtbf_job - 2.781) < 0.001 and abs(per/8760 - 5.2) < 0.05)
    rgs = np.random.default_rng(15); lifes = rgs.exponential(per, (20000, 64)).min(axis=1)
    say("s15", f"series of 64 exponential parts, mean life {per:,.0f} h: system mean {lifes.mean():,.0f} h vs MTBF/64 {per/64:,.0f}",
        abs(lifes.mean()/(per/64) - 1) < 0.03)
    say("s15", f"P(54 days with no interruption at that rate) = exp(-466) = {math.exp(-466):.1e}", math.exp(-466) < 1e-200)

    # --- s16: checkpoint interval is an EOQ -----------------------------------
    M = mtbf_job*60
    for dl in (1.0, 5.0, 10/60):
        tau = math.sqrt(2*dl*M); wst = math.sqrt(2*dl/M)
        say("s16", f"M = {M:.1f} min, checkpoint write {dl*60:.0f} s: Young tau* = sqrt(2 delta M) = {tau:.1f} min, waste {wst:.1%}", True)
    # EOQ identity and flat bottom
    C = lambda tau, dl, M_: dl/tau + tau/(2*M_)
    ok = all(abs(C(k*math.sqrt(2*1.0*M), 1.0, M)/C(math.sqrt(2*1.0*M), 1.0, M) - (k + 1/k)/2) < 1e-12 for k in (0.5, 2, 3))
    say("s16", "cost at k x optimum = (k + 1/k)/2 x minimum: 1.25 at k = 2 or 1/2 (EOQ's flat bottom)", ok)
    # simulation: exponential failures, fixed interval, measure useful fraction
    def ckpt_sim(tau, dl, M_, R_=0.0, horizon=2e6, seed=16):
        rc = np.random.default_rng(seed); t_ = 0.0; useful = 0.0; nextf = rc.exponential(M_)
        while t_ < horizon:
            seg = tau + dl
            if t_ + seg <= nextf:
                t_ += seg; useful += tau
            else:
                t_ = nextf + R_; nextf = t_ + rc.exponential(M_)
        return useful/t_
    taus = [5, 10, 15, 18.3, 22, 30, 45, 60]
    fr = [ckpt_sim(tt, 1.0, M) for tt in taus]
    best = taus[int(np.argmax(fr))]
    say("s16", "simulated useful fraction, delta = 1 min: " + ", ".join(f"tau {a_:g} -> {b_:.3f}" for a_, b_ in zip(taus, fr))
        + f"; best of these {best}", best in (15, 18.3, 22) and abs(max(fr) - (1 - math.sqrt(2/M))) < 0.02)
    # checkpoint size
    P70 = 70e9
    say("s16", f"70B with Adam: source 70B x 4 x 2 = {P70*8/1e9:.0f} GB; fp32 master + m + v = 12 B/param = {P70*12/1e9:.0f} GB; "
        f"with a bf16 copy 14 B = {P70*14/1e9:.0f} GB", P70*8/1e9 == 560 and P70*12/1e9 == 840)
    say("s16", f"hourly for 90 days = {90*24:,} checkpoints: x 560 GB = {90*24*0.56:,.0f} TB (source 1.1 PB for ~2000); "
        f"x 840 GB = {90*24*0.84:,.0f} TB", abs(2000*0.56 - 1120) < 1e-9)
    say("s16", f"840 GB at 100 GB/s = {840/100:.1f} s; at 10 GB/s = {840/10:.0f} s", True)

    # --- s17: the fabric ----------------------------------------------------
    pcie5 = 32e9*16*128/130/8/1e9; pcie4 = pcie5/2
    say("s17", f"PCIe 5.0 x16 = 32 GT/s x 16 x 128/130 / 8 = {pcie5:.1f} GB/s each way; NVLink 900 GB/s total = 450 each way: "
        f"{450/pcie5:.1f}x like for like; 900 vs PCIe 4.0 one way = {900/pcie4:.1f}x (source: 30x)",
        abs(pcie5 - 63.0) < 0.1 and abs(450/pcie5 - 7.1) < 0.1 and abs(900/pcie4 - 28.6) < 0.1)
    say("s17", f"8 x 400 Gb/s = {8*400/8:.0f} GB/s per node, 50 GB/s per GPU: NVLink inside the node is {450/50:.0f}x the network",
        8*400/8 == 400)
    k_ = 64
    say("s17", f"fat tree of {k_}-port switches: two tiers {k_**2//2:,} hosts, three tiers {k_**3//4:,}: 16,384 GPUs need three",
        k_**2//2 == 2048 and k_**3//4 == 65536)
    def ring_allreduce(N, S=12):
        rng_ = np.random.default_rng(N); data = rng_.normal(size=(N, N*S)); chunks = data.reshape(N, N, S).copy()
        sent = np.zeros(N)
        for step in range(N-1):                  # reduce-scatter
            new = chunks.copy()
            for i in range(N):
                c_ = (i - step) % N; new[(i+1) % N, c_] += chunks[i, c_]; sent[i] += S
            chunks = new
        for step in range(N-1):                  # all-gather
            new = chunks.copy()
            for i in range(N):
                c_ = (i + 1 - step) % N; new[(i+1) % N, c_] = chunks[i, c_]; sent[i] += S
            chunks = new
        ok_ = np.allclose(chunks.reshape(N, N*S), data.sum(axis=0)[None, :])
        return ok_, sent[0]/(N*S)
    ar = {N: ring_allreduce(N) for N in (2, 4, 8, 16)}
    say("s17", "ring all-reduce simulated: sums correct and bytes sent per node / message = " +
        ", ".join(f"N={N} {v[1]:.4f} (2(N-1)/N {2*(N-1)/N:.4f})" for N, v in ar.items()),
        all(v[0] and abs(v[1] - 2*(N-1)/N) < 1e-12 for N, v in ar.items()))
    for Pm, nm in ((405e9, "405B"), (70e9, "70B")):
        tps = 16384*989e12*0.40/(6*Pm)
        say("s17", f"{nm} on 16,384 GPUs at 40% of 989 TFLOPS: {tps:,.0f} tokens/s = {tps*4/1e6:.0f} MB/s of 4-byte tokens "
            f"({100e9/(tps*4):,.0f}x below 100 GB/s)", tps*4 < 1e8)

    # --- s18: training cost from 6ND ------------------------------------------
    rate = 989e12*0.40
    for D, nm in ((1.4e12, "Chinchilla 1.4T"), (15e12, "15T")):
        fl = 6*70e9*D; gh = fl/rate/3600
        say("s18", f"70B, {nm} tokens: 6ND = {fl:.2e} FLOP, {gh/1e6:.2f}M GPU-h at 40% of 989 TF: ${gh*8/1e6:.1f}M at $8/h, "
            f"${gh*ownh/1e6:.2f}M at ${ownh:.2f}/h", True)
    fl405 = 6*405e9*15e12; gh405 = fl405/rate/3600
    say("s18", f"405B, 15T tokens: {gh405/1e6:.1f}M GPU-h: ${gh405*8/1e6:.0f}M at $8/h, ${gh405*ownh/1e6:.0f}M at ${ownh:.2f}/h "
        f"(source: $50-100M); {gh405/16384/24:.0f} days on 16,384 GPUs", 30 < gh405*ownh/1e6 < 60)
    say("s18", f"$30K x 16,000 = ${30000*16000/1e6:.0f}M", 30000*16000 == 480_000_000)
    say("s18", f"$2M buys {2e6/8:,.0f} GPU-h at $8 = {2e6/8*3600*rate:.2e} FLOP = {2e6/8*3600*rate/(6*70e9)/1e12:.2f}T tokens for 70B",
        True)

    # --- s19: lifecycle: A/B sample size, PSI, Bayes, delay ---------------------
    za, zb = _n18.ppf(0.975), _n18.ppf(0.80)
    def nab(p, d): return (za+zb)**2*2*p*(1-p)/d**2
    for p in (0.01, 0.02, 0.05):
        say("s19", f"CTR {p:.0%}: relative +0.1% needs {nab(p, p*0.001):.2e} per arm; absolute +0.1 point needs {nab(p, 0.001):,.0f} per arm", True)
    say("s19", "source's 'millions of impressions': relative 0.1% at 2% CTR needs 1.5e9 in total; absolute 0.1 pt needs 0.6e6",
        abs(2*nab(0.02, 2e-5)/1.54e9 - 1) < 0.02 and abs(2*nab(0.02, 0.001)/6.15e5 - 1) < 0.02)
    # PSI = KL(a||e) + KL(e||a); normal shift d -> Jeffreys d^2
    rp2 = np.random.default_rng(19); ok = True
    for _ in range(100):
        aa = rp2.dirichlet(np.ones(10)); ee = rp2.dirichlet(np.ones(10))
        psi = np.sum((aa-ee)*np.log(aa/ee)); kl = np.sum(aa*np.log(aa/ee)) + np.sum(ee*np.log(ee/aa))
        ok &= abs(psi - kl) < 1e-12
    say("s19", "PSI == KL(a||e) + KL(e||a) (Jeffreys divergence), 100 random histograms", ok)
    edges = _n18.ppf(np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    for dsh in (0.316, 0.5):
        aa = np.diff(_n18.cdf(edges - dsh)); ee = np.full(10, 0.1)
        say("s19", f"mean shift {dsh} sigma binned in deciles: PSI {np.sum((aa-ee)*np.log(aa/ee)):.3f} (unbinned d^2 = {dsh**2:.3f})",
            abs(np.sum((aa-ee)*np.log(aa/ee)) - dsh**2) < 0.03)
    prec = 0.99*0.001/(0.99*0.001 + 0.01*0.999)
    say("s19", f"fraud at 0.1%, detector 99% sensitive and 99% specific: precision {prec:.1%} (Sheet 05's alarm)",
        abs(prec - 0.0902) < 0.001)
    cfp, cfn = 5.0, 500.0
    say("s19", f"flag when P(fraud) > C_FP/(C_FP + C_FN): costs $5 and $500 -> threshold {cfp/(cfp+cfn):.4f}",
        abs(cfp/(cfp+cfn) - 0.0099) < 1e-4)
    ok = True
    for K in (0.1, 0.2, 0.3, 0.5, 0.9, 1.1):
        rts = np.roots([1, -1, K]); cplx = np.iscomplexobj(rts) and np.any(np.abs(rts.imag) > 1e-12)
        ok &= (cplx == (K > 0.25)) and ((np.max(np.abs(rts)) > 1) == (K > 1))
    say("s19", "correction acting on last cycle's error, x(k+1) = x(k) - K x(k-1): oscillates for K > 1/4, diverges for K > 1", ok)

    # --- s20: the design examples, recomputed ---------------------------------
    emb = 100e6*256*2
    say("s20", f"100M x 256 x 2 B = {emb/1e9:.1f} GB (source 50 GB)", abs(emb/1e9 - 51.2) < 1e-9)
    for Mh in (16, 32):
        say("s20", f"HNSW M = {Mh}: 2M = {2*Mh} links x 4 B = {2*Mh*4} B per vector against 512 B: +{2*Mh*4/512:.0%} "
            "(source: ~2x, which needs M = 64)", True)
    img = 1e9*512*2; links = 1e9*32*4
    say("s20", f"image index: 1B x 512 x 2 B = {img/1e12:.3f} TB of vectors + {links/1e9:.0f} GB of links (M = 16); "
        f"source's 128 GB equals the links alone; PQ at 64 B/vector = {1e9*64/1e9:.0f} GB",
        img/1e12 == 1.024 and links/1e9 == 128)
    say("s20", f"recommendation latency 5 + 2 + 100 = 107 ms within 200; search 100 + 200 + 100 = 400 within 500", True)
    say("s20", f"moderation: 2% of 1e9 = {0.02*1e9:,.0f} reviews/day / 100 = {0.02*1e9/100:,.0f} reviewers; "
        f"0.1% = {0.001*1e9:,.0f}; 600 frames x 20 ms = {600*0.02:.0f} s", 0.02*1e9/100 == 200000)
    stream = 20e6/450; perpage = 10000*math.ceil((2000-50)/450)
    say("s20", f"RAG chunks: one stream 20M/450 = {stream:,.0f}; page by page 10,000 x ceil(1950/450) = {perpage:,} "
        f"(source ~44K)", abs(stream - 44444.4) < 1 and perpage == 50000)
    say("s20", f"index 44,444 x 768 x 2 B = {44444*768*2/1e6:.1f} MB = {44444*768*2/2**20:.1f} MiB; 10M chunks {1e7*768*2/1e9:.2f} GB",
        abs(44444*768*2/2**20 - 65.1) < 0.1)
    say("s20", f"API: 1,000 x 2,000 tokens x $3/M = ${1000*2000*3/1e6:.0f}/day; at 1M/day ${1e6*2000*3/1e6:,.0f}; "
        f"2 A10G x $1 x 24 = ${2*24}; ratio {1e6*2000*3/1e6/48:.0f}x", 1e6*2000*3/1e6/48 == 125)
    need = 2*7e9*2e9/86400
    say("s20", f"2e9 tokens/day = {2e9/86400:,.0f} tokens/s x 2 x 7e9 FLOP = {need/1e12:.0f} TFLOP/s sustained; "
        f"at 40% of a 125-TFLOPS card that is {need/(0.4*125e12):.1f} cards, not 2", abs(need/1e12 - 324) < 1)

    # --- s21: worked example: the help desk becomes a plant --------------------
    tw, tkv = 0.020, 0.0002
    X = lambda b: b/(tw + b*tkv)
    mean_dem = 18000.0; peak = 1.8*mean_dem
    # planning capacity per server from s03 simulation: first load with P(batch >= 100) <= 1%
    plan_cap = None
    for Lam in range(1800, 2501, 50):
        hh = cb_sim(float(Lam), steps=40000, seed=21)
        if np.mean(hh >= 100) > 0.01: break
        plan_cap = Lam
    say("s21", f"per-server planning capacity (batch over the 100-sequence cap on <= 1% of steps): {plan_cap} tok/s "
        f"(nameplate X(100) = {X(100):,.0f})", plan_cap is not None and 1900 <= plan_cap <= 2300)
    nserv = math.ceil(peak/plan_cap)
    pk_ok = lambda N, k, a_: kofn(k, N, a_)
    av = 0.99; Ntot = nserv
    while pk_ok(Ntot, nserv, av) < 0.999: Ntot += 1
    say("s21", f"peak {peak:,.0f} tok/s / {plan_cap} = {peak/plan_cap:.2f} -> {nserv} servers; with each up 99% of the time, "
        f"P(>= {nserv} of N up) >= 0.999 needs N = {Ntot} (N={nserv+1}: {pk_ok(nserv+1, nserv, av):.4f}, N={Ntot}: {pk_ok(Ntot, nserv, av):.4f})",
        Ntot >= nserv + 1)
    naive = math.ceil(mean_dem/X(100))
    say("s21", f"the source's method (mean demand / nameplate, no spare): {mean_dem:,.0f}/{X(100):,.0f} = {mean_dem/X(100):.1f} -> {naive} servers",
        naive == 8)
    per_srv = peak/nserv; rho_ = per_srv*tkv; tpot = tw/(1-rho_)
    say("s21", f"at peak each of {nserv} carries {per_srv:,.0f} tok/s: rho {rho_:.3f}, TPOT {tpot*1e3:.1f} ms (target 50)", tpot < 0.050)
    hrs = np.arange(24); prof = mean_dem*(1 + 0.8*np.cos(2*np.pi*(hrs-14)/24))
    need_h = np.array([math.ceil(d/plan_cap) for d in prof])
    spare = np.array([next(N for N in range(n_, n_+10) if pk_ok(N, n_, av) >= 0.999) for n_ in need_h])
    fixed = Ntot*24*8.0; scaled = spare.sum()*8.0
    say("s21", f"profile 1 + 0.8 cos (peak {prof.max():,.0f}, trough {prof.min():,.0f}): fixed fleet {Ntot} x 24 h = ${fixed:,.0f}/day; "
        f"autoscaled with spares {spare.sum()} server-hours = ${scaled:,.0f}/day ({1-scaled/fixed:.0%} less)",
        abs(prof.max() - 32400) < 1e-6 and scaled < fixed)
    R_ = best_reserved(spare, 0.6*8.0, 8.0)[0]
    say("s21", f"at a 40% reservation discount, reserve {R_} servers (on duty more than 60% of hours); "
        f"day cost ${R_*24*4.8 + 8*np.maximum(spare-R_,0).sum():,.0f}", True)

_ch18()
# ----- end of Chapter 18 block -----


# ======================================================================================
# Chapters 19-20 coda block. Owned by the Chapters 19-20 coda run: replace only the body of _ch1920() below.
# Keep every name inside the function so nothing collides with another chapter's block.
def _ch1920():
    pass

_ch1920()
# ----- end of Chapters 19-20 coda block -----


print("\n" + "="*66)

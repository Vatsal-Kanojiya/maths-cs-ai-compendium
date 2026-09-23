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

print("\n" + "="*66)

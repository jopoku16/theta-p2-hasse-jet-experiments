"""Branch-free re-run of the band law.

The earlier whole-band checks computed om_p(theta^jmax f) from a CLOSED-FORM
classical cycle with the ORDINARY branch hard-coded.  Here om_p(theta^jmax f)
is computed DIRECTLY by an echelon filtration test mod p, so no ordinarity
assumption enters anywhere.  Runs on ordinary and non-ordinary (p,k) alike.

Reports, per cell: observed delta mod p^2, D from Theorem 1.1, conic flag.
"""
import numpy as np, sys, collections

def build(p,k,NMAX,DWIN=4):
    M,P1=p*p,p*(p-1)
    imax=NMAX*p+p-1
    WMAX=k+2*imax+DWIN*P1+24
    N=WMAX//12+40
    def conv(a,b): return (np.convolve(a,b)[:N+1])%M
    def sig(kk):
        s=np.zeros(N+1,dtype=np.int64)
        for d in range(1,N+1):
            dk=pow(d,kk,M); s[d::d]=(s[d::d]+dk)%M
        return s
    one=np.zeros(N+1,dtype=np.int64); one[0]=1
    def Eis(kk,c):
        a=(c*sig(kk-1))%M; a[0]=1; return a
    E4,E6=Eis(4,240),Eis(6,(-504)%M)
    e=one.copy()
    for n in range(1,N+1):
        ne=e.copy(); ne[n:]=(ne[n:]-e[:N+1-n])%M; e=ne
    p24=one.copy()
    for _ in range(24): p24=conv(p24,e)
    Dl=np.zeros(N+1,dtype=np.int64); Dl[1:]=p24[:N]
    rec={8:[],12:[],16:[4],18:[6],20:[4,4],22:[4,6],26:[4,4,6]}
    f=Dl.copy() if k!=8 else Eis(8,480)
    for t in rec[k]: f=conv(f,E4 if t==4 else E6)
    ordy=(int(f[p])%p)!=0
    jm,am=WMAX//12+2,WMAX//4+2
    Dp=[one.copy()]
    for j in range(1,jm+1): Dp.append(conv(Dp[-1],Dl))
    A4=[one.copy()]
    for a in range(1,am+1): A4.append(conv(A4[-1],E4))
    bc={}
    def belt(w,j):
        key=(w,j)
        if key in bc: return bc[key]
        r=w-12*j; v=None
        if r>=0 and r!=2:
            b=0 if r%4==0 else 1; a=(r-6*b)//4
            if a>=0:
                v=A4[a] if b==0 else conv(A4[a],E6)
                v=conv(Dp[j],v)
        bc[key]=v; return v
    def reduce_Mw(g,w,mod):
        """Return the canonical remainder after echelon reduction by M_w."""
        if w<0 or w%2: return None
        h=g.copy()%mod
        for j in range(0,w//12+1):
            el=belt(w,j)
            if el is None: continue
            c=int(h[j])%mod
            if c: h=(h-c*el)%mod
        return h
    def in_Mw(g,w,mod):
        """echelon forward substitution: j-th basis elt has q-valuation j"""
        h=reduce_Mw(g,w,mod)
        return h is not None and not np.any(h%mod)
    def theta(g,i):
        return np.array([(pow(int(t),i,M)*int(g[t]))%M for t in range(N+1)],dtype=np.int64)
    def om_p(g,wcl):
        """direct mod-p weight filtration of g; wcl is any weight in its class.
        Scans only w = wcl mod (p-1), which is where the filtration must lie
        since E_{p-1} = 1 mod p.  No ordinarity assumption."""
        gg=g%p
        w0=wcl%(p-1)
        while w0<0: w0+=(p-1)
        if w0%2: w0+=(p-1)
        for w in range(w0,WMAX+1,(p-1)):
            if in_Mw(gg,w,p): return w
        return None
    return dict(M=M,P1=P1,N=N,f=f,ordy=ordy,in_Mw=in_Mw,
                reduce_Mw=reduce_Mw,theta=theta,om_p=om_p,p=p,k=k)

def run(p,k,NMAX,label="",DWIN=4,DMAX=6):
    E=build(p,k,NMAX,DWIN)
    M,P1,f,ordy=E["M"],E["P1"],E["f"],E["ordy"]
    in_Mw,theta,om_p=E["in_Mw"],E["theta"],E["om_p"]
    VERBOSE=(p>=53)
    rows=[]; CSV=[]
    omcache={}
    for n in range(1,NMAX+1):
        for ip in range(p-k+2,p):
            i=n*p+ip
            g=theta(f,i)
            if not np.any(g%p): continue
            obs=None
            for d in range(-2,DMAX):
                w=k+2*i+d*P1
                if w>=0 and in_Mw(g,w,M): obs=d; break
            jmax=(n-1)*p+ip
            if jmax not in omcache:
                omcache[jmax]=om_p(theta(f,jmax),k+2*jmax)
            omcl=omcache[jmax]
            wA=omcl+p*(p+1)
            Dv=0
            while k+2*i+Dv*P1 < wA: Dv+=1
            while Dv>-6 and k+2*i+(Dv-1)*P1 >= wA: Dv-=1
            conic=((i*i+(k-1)*i-n*(n+1))%p==0)
            omti=om_p(g,k+2*i)
            rows.append((n,ip,obs,Dv,conic))
            CSV.append((p,k,n,ip,obs,Dv,1 if conic else 0,1 if ordy else 0,
                        omcl,wA,omti,k+2*i,jmax))
            if VERBOSE:
                print("   n=%d ip=%d obs=%s D=%d conic=%s ok=%s"%(n,ip,obs,Dv,conic,(obs is not None and obs<=Dv)));sys.stdout.flush()
        sys.stdout.write("."); sys.stdout.flush()
    print()
    import csv,os
    fn="cells_%d_%d.csv"%(p,k)
    with open(fn,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["p","k","n","ip","delta","D","conic","ordy",
                                      "om_jmax","wA","om_theta_i","k2i","jmax"])
        w.writerows(CSV)
    print("wrote",fn,len(CSV),"cells")
    tag="ORDINARY" if ordy else "NON-ORDINARY"
    print("=== p=%d k=%d  %s %s : %d cells ==="%(p,k,tag,label,len(rows)))
    # (a) upper bound Theorem 1.2
    ub=[r for r in rows if r[2]>r[3]]
    print("  Thm 1.2 upper bound delta<=D : %d violations"%len(ub))
    if ub: print("    e.g.",ub[:6])
    # (b) Proposition D2 : D==2 => delta = 2 - [conic]
    sub=[r for r in rows if r[3]==2]
    bad2=[r for r in sub if r[2]!=2-(1 if r[4] else 0)]
    print("  Prop 6.7 (D=2 cells): %d cells, %d violations"%(len(sub),len(bad2)))
    if bad2: print("    ",bad2[:8])
    # (c) full conjectural law
    badL=[r for r in rows if r[2]!=r[3]-(1 if (r[4] and r[3]==2) else 0)]
    print("  Candidate law over all cells: %d/%d hold, %d exceptions"%(len(rows)-len(badL),len(rows),len(badL)))
    if badL:
        for r in badL: print("      (n,i')=(%d,%d) obs=%s D=%d conic=%s"%(r[0],r[1],r[2],r[3],r[4]))
    sys.stdout.flush()
    return rows,ordy

if __name__=="__main__":
    import sys
    args=sys.argv[1:]
    p,k,n=int(args[0]),int(args[1]),int(args[2])
    dw=int(args[3]) if len(args)>3 else 4
    dm=int(args[4]) if len(args)>4 else 6
    run(p,k,n,DWIN=dw,DMAX=dm)

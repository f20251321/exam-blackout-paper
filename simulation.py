import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('figures_psych', exist_ok=True)

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'lines.linewidth': 1.4,
})

P = dict(alpha=0.8,beta=0.4,gamma=0.6,mu=0.2,E=0.0,
         rho=1.8,kw=1.0,delta=0.25,sigma=1.2,tau=1.8,nu=0.35)

def unpack(p):
    return (p['alpha'],p['beta'],p['gamma'],p['mu'],p['E'],
            p['rho'],p['kw'],p['delta'],p['sigma'],p['tau'],p['nu'])

def system(t, y, T_val, p):
    S,R,L = y
    a,b,g,m,e,r,k,d,s,ta,n = unpack(p)
    S=max(S,0); R=np.clip(R,0,1); L=np.clip(L,0,1)
    W=1/(1+k*S)
    return [a*T_val-b*S-g*R-m*e, r*W*(1-L)*(1-R)-d*R, s*S*(1-L)-ta*R*L-n*L]

def Tval(t, te=3.0): return t/te if t<te else 0.0
t_eval = np.linspace(0, 5.5, 2000)
TE = 3.0

W  = 3.4   # figure width inches
H  = 1.9   # figure height inches
MARGINS = dict(left=0.17, right=0.97, top=0.88, bottom=0.22)

def single(fname, title=''):
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(**MARGINS)
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold')
    return fig, ax

def save(fname):
    plt.savefig(f'figures_psych/{fname}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  {fname}')

# Shared: compute three student trajectories
p_pos = dict(P); p_pos['E'] = 0.3
SCENS = [
    ([0.15,0.80,0.05], P,     'steelblue', 'Calm (low stress)'),
    ([0.80,0.40,0.35], P,     'tomato',    'Anxious (E=0)'),
    ([0.80,0.40,0.35], p_pos, 'seagreen',  r'Anxious, $E\!=\!+0.3$'),
]
SOLS = [solve_ivp(lambda t,y,p=p: system(t,y,Tval(t),p),
        [0,5.5],ic,t_eval=t_eval,method='RK45',max_step=0.005)
        for ic,p,_,_ in SCENS]

# FIG ts_S, ts_R, ts_L -- three-scenario timeseries, one variable each
print('Three-scenario timeseries:')
for vi, vname, ylim, fname in [
    (0, r'Stress $S(t)$',    (0,3.2), 'ts_S'),
    (1, r'Recall $R(t)$',    (0,1.05),'ts_R'),
    (2, r'Rumin. $L(t)$',    (0,1.05),'ts_L'),
]:
    fig, ax = single(fname)
    for (ic,p,col,lab), sol in zip(SCENS, SOLS):
        ax.plot(sol.t, sol.y[vi], color=col, label=lab)
    ax.axvline(TE, color='k', ls='--', lw=0.9, label='Exam ends')
    ax.axvspan(0, TE, alpha=0.07, color='gold')
    ax.set_ylabel(vname); ax.set_xlabel(r'Time $t$')
    ax.set_ylim(ylim); ax.grid(True, alpha=0.2, lw=0.5)
    if vi == 0:
        ax.legend(loc='upper right', ncol=1, fontsize=6.5, framealpha=0.8)
    save(fname)

# Shared: compute blackout trajectories
def sys_pert(t,y):
    T_val=Tval(t); shock=0.9*np.exp(-15*(t-1.5)**2) if t>1 else 0
    S,R,L=y; a,b,g,m,e,r,k,d,s,ta,n=unpack(P)
    S=max(S,0); R=np.clip(R,0,1); L=np.clip(L,0,1); W2=1/(1+k*S)
    return [a*T_val-b*S-g*R-m*e+shock, r*W2*(1-L)*(1-R)-d*R, s*S*(1-L)-ta*R*L-n*L]

ic0=[0.3,0.70,0.10]
sol_ns=solve_ivp(lambda t,y: system(t,y,Tval(t),P),[0,5.5],ic0,
                 t_eval=t_eval,method='RK45',max_step=0.005)
sol_sh=solve_ivp(sys_pert,[0,5.5],ic0,t_eval=t_eval,method='RK45',max_step=0.005)

# FIG bk_S, bk_R, bk_L -- blackout mechanism, one variable each
print('Blackout mechanism:')
annots = {
    1: [(1.5, 'orange', 'Spike',    1.5,  0.55, 0.7),
        (2.2, 'tomato', 'Collapse', 2.5,  0.5,  1),
        (3.6, 'navy',   'Recovery', 4.0,  0.35, 1)],
    2: [(1.85,'tomato', 'Floods',   2.2,  0.45, 1)],
}
ylims_bk = {0:(0,3.0), 1:(0,1.05), 2:(0,1.05)}
vnames_bk = {0:r'Stress $S(t)$', 1:r'Recall $R(t)$', 2:r'Rumin. $L(t)$'}
fnames_bk = {0:'bk_S', 1:'bk_R', 2:'bk_L'}
for vi in [0,1,2]:
    fig, ax = single(fnames_bk[vi])
    ax.plot(sol_ns.t, sol_ns.y[vi], 'steelblue', label='No perturbation')
    ax.plot(sol_sh.t, sol_sh.y[vi], 'tomato',    label='Stress spike $t=1.5$')
    ax.axvline(TE,  color='k',      ls='--', lw=0.9, label='Exam ends')
    ax.axvline(1.5, color='orange', ls=':',  lw=0.9)
    ax.axvspan(0, TE, alpha=0.06, color='gold')
    ax.set_ylabel(vnames_bk[vi]); ax.set_xlabel(r'Time $t$')
    ax.set_ylim(ylims_bk[vi]); ax.grid(True, alpha=0.2, lw=0.5)
    if vi == 0:
        ax.legend(loc='upper right', fontsize=6.5, framealpha=0.8)
    for (tx, col, lab, xtxt, ytxt, yi) in annots.get(vi, []):
        idx=np.argmin(abs(sol_sh.t-tx))
        ax.annotate(lab,xy=(tx,sol_sh.y[vi,idx]),xytext=(xtxt,ytxt),
            arrowprops=dict(arrowstyle='->',lw=0.7,color=col),
            fontsize=6.5,color=col)
    save(fnames_bk[vi])

# FIG prior_neg, prior_neu, prior_pos -- one panel per E value
print('Prior performance E:')
E_cases = [
    (-0.5, 'tomato',    r'$E=-0.5$ (negative history)',  'prior_neg'),
    ( 0.0, 'goldenrod', r'$E=0$ (neutral history)',       'prior_neu'),
    (+0.5, 'steelblue', r'$E=+0.5$ (positive history)',   'prior_pos'),
]
for E_val, col, title, fname in E_cases:
    p = dict(P); p['E'] = E_val
    sol = solve_ivp(lambda t,y,p=p: system(t,y,Tval(t),p),
                    [0,5.5],[0.5,0.55,0.20],t_eval=t_eval,
                    method='RK45',max_step=0.005)
    fig, ax = single(fname, title)
    ax.plot(sol.t, sol.y[0], 'firebrick', lw=1.1, label=r'Stress $S$', alpha=0.8)
    ax.plot(sol.t, sol.y[1], col,         lw=1.4, label=r'Recall $R$')
    ax.plot(sol.t, sol.y[2], 'purple',    lw=1.1, ls='--', label=r'Rumin. $L$', alpha=0.8)
    ax.axvline(TE, color='k', ls='--', lw=0.8)
    ax.axvspan(0, TE, alpha=0.07, color='gold')
    ax.set_ylim(-0.05, 2.5); ax.set_xlabel(r'Time $t$')
    ax.set_ylabel('Value'); ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(loc='upper right', ncol=3, fontsize=6.5, framealpha=0.8)
    save(fname)

# FIG yd -- Yerkes-Dodson single panel (unchanged)
print('Yerkes-Dodson:')
T_vals = np.linspace(0, 1.2, 60)
R_ss = []
for T in T_vals:
    s = solve_ivp(lambda t,y: system(t,y,T,P),[0,60],[0.3,0.5,0.2],
                  method='RK45',max_step=0.05,t_eval=np.linspace(40,60,300))
    R_ss.append(np.mean(s.y[1,-30:]))
T_ref = np.linspace(0,1.2,200)
YD    = np.clip(T_ref*np.exp(-2.5*T_ref)*4.5, 0, 1)

fig, ax = single('yd')
ax.plot(T_ref, YD, 'k--', lw=1, alpha=0.5, label='Yerkes-Dodson (reference)')
ax.plot(T_vals, np.clip(R_ss,0,1), 'steelblue', lw=1.8, label=r'Model $R^*(T)$')
ax.set_xlabel(r'Time Pressure $T$'); ax.set_ylabel(r'Steady-state Recall $R^*$')
ax.set_title('Yerkes-Dodson Compliance', fontsize=8, fontweight='bold')
ax.legend(); ax.set_xlim(0,1.2); ax.set_ylim(0,1.05)
ax.grid(True, alpha=0.2, lw=0.5)
save('yd')

# FIG phase -- Phase portrait (unchanged)
print('Phase portrait:')
alpha,beta,gamma,mu,E,rho,kw,delta,sigma,tau,nu = unpack(P)
T_pp = 0.65
R_r = np.linspace(0.01,0.99,50); L_r = np.linspace(0.01,0.99,50)
RG,LG = np.meshgrid(R_r, L_r)
S_a = np.clip((alpha*T_pp-gamma*RG-mu*E)/beta,0.01,10)
W_a = 1/(1+kw*S_a)
dR = rho*W_a*(1-LG)*(1-RG)-delta*RG
dL = sigma*S_a*(1-LG)-tau*RG*LG-nu*LG

fig, ax = single('phase')
ax.streamplot(R_r,L_r,dR,dL,color=np.sqrt(dR**2+dL**2),
              cmap='Blues',density=1.4,linewidth=0.6,arrowsize=0.8)
for S_nc in [0.5,1.0,1.5]:
    W_nc=1/(1+kw*S_nc)
    Rn=np.linspace(0.01,0.95,200)
    dn=rho*W_nc*(1-Rn)
    Lrn=np.where(dn>1e-10,1-delta*Rn/dn,np.nan)
    mask=(Lrn>=0)&(Lrn<=1)
    ax.plot(Rn[mask],Lrn[mask],'r-',lw=0.8,alpha=0.6)
    ax.plot(Rn,sigma*S_nc/(sigma*S_nc+tau*Rn+nu),'g--',lw=0.8,alpha=0.6)
ax.plot([],[],'r-',lw=1.2,label=r'$\dot{R}=0$')
ax.plot([],[],'g--',lw=1.2,label=r'$\dot{L}=0$')
ax.fill_between([0.55,0.99],[0,0],[0.35,0.35],alpha=0.12,color='steelblue')
ax.fill_between([0,0.35],[0.65,0.65],[1,1],alpha=0.12,color='tomato')
ax.text(0.77,0.14,'Functional',ha='center',fontsize=6.5,color='steelblue',fontweight='bold')
ax.text(0.17,0.82,'Blackout', ha='center',fontsize=6.5,color='tomato',fontweight='bold')
for R0,L0,col in [(0.85,0.05,'steelblue'),(0.10,0.90,'tomato'),(0.50,0.45,'purple')]:
    S0=max((alpha*T_pp-gamma*R0-mu*E)/beta,0.1)
    s=solve_ivp(lambda t,y: system(t,y,T_pp,P),[0,15],[S0,R0,L0],
                t_eval=np.linspace(0,15,600),method='RK45',max_step=0.02)
    ax.plot(s.y[1],s.y[2],color=col,lw=1.2,alpha=0.85)
    ax.plot(s.y[1,0],s.y[2,0],'o',color=col,ms=4,zorder=5)
    ax.plot(s.y[1,-1],s.y[2,-1],'s',color=col,ms=4,zorder=5)
ax.set_xlabel(r'Recall $R$'); ax.set_ylabel(r'Rumination $L$')
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title(r'Phase Portrait $(R,L)$, $T=0.65$',fontsize=8,fontweight='bold')
ax.legend(fontsize=7,loc='center right'); ax.grid(True,alpha=0.2,lw=0.5)
save('phase')

# FIG thresh_curve, thresh_bars -- cascade threshold, two separate panels
print('Cascade threshold:')
A_base = P['delta']*P['sigma']/(P['beta']+P['nu'])

def get_FP(T_val):
    s=solve_ivp(lambda t,y: system(t,y,T_val,P),[0,200],[0.2,0.8,0.05],
                method='RK45',max_step=0.2,t_eval=np.linspace(180,200,100))
    return max(s.y[0,-1],0),s.y[1,-1],s.y[2,-1]

DS_plot = np.linspace(0,3.0,35)
T_cases_c=[(0.4,'T=0.4','steelblue'),(0.7,'T=0.7','goldenrod'),(0.9,'T=0.9','tomato')]

# Panel 1: R_min vs Delta_S
fig, ax = single('thresh_curve')
for T_v,lab,col in T_cases_c:
    SF,RF,LF=get_FP(T_v)
    Rmins=[]
    for DS in DS_plot:
        s=solve_ivp(lambda t,y: system(t,y,T_v,P),[0,20],[SF+DS,RF,LF],
                    method='RK45',max_step=0.01,t_eval=np.linspace(0,20,1000))
        Rmins.append(s.y[1].min())
    ax.plot(DS_plot,Rmins,color=col,label=f'Sim. ({lab})')
    ax.plot(DS_plot,np.clip(RF*(1-A_base*DS_plot),0,1),color=col,lw=0.9,ls='--',alpha=0.7)
ax.axhline(0.5,color='gray',ls=':',lw=0.7,alpha=0.6)
ax.set_xlabel(r'Perturbation $\Delta S$'); ax.set_ylabel(r'Min.\ recall $R_{\min}$')
ax.set_title(r'Blackout depth vs.\ $\Delta S$'+'\n'+r'(solid: sim.; dashed: $R_F(1\!-\!A\Delta S)$)',
             fontsize=8,fontweight='bold')
ax.legend(fontsize=7); ax.set_xlim(0,3); ax.set_ylim(0,1.02)
ax.grid(True,alpha=0.2,lw=0.5)
save('thresh_curve')

# Panel 2: intervention bars
fig, ax = single('thresh_bars')
int_data=[
    ('Baseline\n'+r'$A=\delta\sigma/(\beta\!+\!\nu)$', A_base,                        'steelblue'),
    ('Breathing\n'+r'($\beta\!=\!0.6$)',                P['delta']*P['sigma']/(0.6+P['nu']),  'seagreen'),
    ('Worry red.\n'+r'($\nu\!=\!0.55$)',                P['delta']*P['sigma']/(P['beta']+0.55),'purple'),
    ('Both\n'+r'($\beta\!=\!0.6,\nu\!=\!0.55$)',        P['delta']*P['sigma']/(0.6+0.55),     'goldenrod'),
]
labs_i=[d[0] for d in int_data]
A_i=[d[1] for d in int_data]
cols_i=[d[2] for d in int_data]
bars=ax.barh(range(len(labs_i)),A_i,color=cols_i,alpha=0.85,height=0.55)
ax.axvline(A_base,color='steelblue',ls='--',lw=0.9,alpha=0.5)
ax.set_xlabel(r'Cascade sensitivity $A$',fontsize=8)
ax.set_yticks(range(len(labs_i))); ax.set_yticklabels(labs_i,fontsize=7)
ax.set_title('Interventions reduce $A$\n(less recall lost per perturbation)',
             fontsize=8,fontweight='bold')
ax.grid(True,alpha=0.2,axis='x',lw=0.5)
for bar,val in zip(bars,A_i):
    ax.text(val+0.003,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=7)
save('thresh_bars')

print()
print('Done. Files in figures_psych/:')
all_expected = ['ts_S','ts_R','ts_L','bk_S','bk_R','bk_L',
                'prior_neg','prior_neu','prior_pos',
                'yd','phase','thresh_curve','thresh_bars']
for f in all_expected:
    ok = os.path.exists(f'figures_psych/{f}.png')
    print(f'  {"OK" if ok else "MISSING"}: {f}.png')
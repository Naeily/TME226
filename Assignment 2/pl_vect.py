#%%
import scipy.io as sio
import numpy as np
import csv
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})
import pandas as pd
import math
import matplotlib.ticker as mticker


rho = 998.29#[kg/m^3]
mu = 0.001003# [Nms/kg]
nu = mu/rho# [Pas]

#--------------------------------------------------------------------------
ni=200 #Do not change it.
nj=200 #Do not change it.
#--------------------------------------------------------------------------
tt=np.arange(nj+1)
tt[1]=int(0)
for j  in range (2,nj+1):
    tt[j]=tt[j-1]+j

ss=np.zeros((ni+1,nj+1),dtype=int)
ss[:,1]= tt
for j  in range (1,nj+1):
   for i  in range (2,ni+1):
        ss[j,i]=ss[j,i-1]+i-1+(j-1)

pp=np.zeros((ni),dtype=int)
for i  in range (1,ni):
    pp[i] = int((i)**2)

subtx=np.ones((ni+1,nj+1),dtype=int)
count=0
for n in range (nj,0,-1):
    count = nj-n+1
    subtx[1:count+1,n]=0
    subtx[count+1:,n]=pp[1:ni-count+1]

gridIndex=ss-subtx
#--------------------------------------------------------------------------
nmax=ni*nj
u=np.zeros(nmax)
v=np.zeros(nmax)
p=np.zeros(nmax)
te=np.zeros(nmax)
diss=np.zeros(nmax)
vist=np.zeros(nmax)
x=np.zeros(nmax)
y=np.zeros(nmax)
#with open('./output_standard-keps-low-re.csv') as csv_file:
with open('./FieldParams_xh2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    n = 0  
    for row in csv_reader:
        if n == 0:
            print(f'Column names are {", ".join(row)}')
            n += 1
        else:
#           if n < 10:  # print the 10 first lines
                #print(f'\t{row[0]}: U={row[1]}, V={row[2]},  P={row[3]},  k={row[4]},  eps={row[5]}, vist={row[6]}, x={row[7]}, y={row[8]}')

            u[n-1]=row[1]
            v[n-1]=row[2]
            p[n-1]=row[3]
            te[n-1]=row[4]
            diss[n-1]=row[5]
            vist[n-1]=row[6]
            x[n-1]=row[7]
            y[n-1]=row[8]
            n += 1
        #if n == 10:
           #break

print(f'Processed {n} lines.')


x1_2d=np.zeros((ni,nj))
x2_2d=np.zeros((ni,nj))
v1_2d=np.zeros((ni,nj))
p_2d=np.zeros((ni,nj))
v2_2d=np.zeros((ni,nj))
te_2d=np.zeros((ni,nj))
vist_2d=np.zeros((ni,nj))
diss_2d=np.zeros((ni,nj))

for j  in range (1,nj+1):
   for i  in range (1,ni+1):
       n=gridIndex[i,j]
       x1_2d[i-1,j-1]=x[n]
       x2_2d[i-1,j-1]=y[n]
       p_2d[i-1,j-1]=p[n]
       v1_2d[i-1,j-1]=u[n]
       v2_2d[i-1,j-1]=v[n]
       te_2d[i-1,j-1]=te[n]
       vist_2d[i-1,j-1]=vist[n]
       diss_2d[i-1,j-1]=diss[n]

v1_2d_org=v1_2d
x2_2d_org=x2_2d

x1_2d=np.flipud(np.transpose(x1_2d))
x2_2d=np.flipud(np.transpose(x2_2d))
v1_2d=np.flipud(np.transpose(v1_2d))
v2_2d=np.flipud(np.transpose(v2_2d))
p_2d=np.flipud(np.transpose(p_2d))
te_2d=np.flipud(np.transpose(te_2d))
vist_2d=np.flipud(np.transpose(vist_2d))
diss_2d=np.flipud(np.transpose(diss_2d))


# The STAR-CCM data do no include the boundaries. 
# below we add wall at the bottom (low x_2, south) and top (high x_2, north)
deltaYBottom=(x2_2d[:,1]-x2_2d[:,0])/2
deltaYTop=(x2_2d[:,-1]-x2_2d[:,-1-1])/2


# duplicate first column  (south boundary)
x1_2d=np.insert(x1_2d,0,x1_2d[:,0],axis=1)
x2_2d=np.insert(x2_2d,0,x2_2d[:,0],axis=1)
v1_2d=np.insert(v1_2d,0,v1_2d[:,0],axis=1)
v2_2d=np.insert(v2_2d,0,v2_2d[:,0],axis=1)
te_2d=np.insert(te_2d,0,te_2d[:,0],axis=1)
diss_2d=np.insert(diss_2d,0,diss_2d[:,0],axis=1)
vist_2d=np.insert(vist_2d,0,vist_2d[:,0],axis=1)
p_2d=np.insert(p_2d,0,p_2d[:,0],axis=1)
diss_2d=np.insert(diss_2d,0,diss_2d[:,0],axis=1)
zero_col=np.zeros(ni)

# set south boundary to zero
v1_2d[:,0]=0.
v2_2d[:,0]=0.
te_2d[:,0]=0.
diss_2d[:,0]=0.
vist_2d[:,0]=0.

x2_2d[:,0]=x2_2d[:,1]-deltaYBottom;

# duplicate last column and put it at the end (north boundary)
x1_2d=np.insert(x1_2d,-1,x1_2d[:,-1],axis=1)
x2_2d=np.insert(x2_2d,-1,x2_2d[:,-1],axis=1)
v1_2d=np.insert(v1_2d,-1,v1_2d[:,-1],axis=1)
v2_2d=np.insert(v2_2d,-1,v2_2d[:,-1],axis=1)
te_2d=np.insert(te_2d,-1,te_2d[:,-1],axis=1)
diss_2d=np.insert(diss_2d,-1,diss_2d[:,-1],axis=1)
vist_2d=np.insert(vist_2d,-1,vist_2d[:,-1],axis=1)
p_2d=np.insert(p_2d,-1,p_2d[:,-1],axis=1)
diss_2d=np.insert(diss_2d,-1,diss_2d[:,-1],axis=1)

# append a column with zeros and put it at the end (north boundary)

nj=nj+2
#--------------------------------------------------------------------------
#*************** DO NOT CHANGE ANY PART OF THE ABOVE LINES. ***************
#--------------------------------------------------------------------------
#
hmax=0.050 # Maximum hill height.
H=3.035*hmax # Cahnnel height.
L=9*hmax # Space between two hills summit. 

#**** LOADING MEASUREMENT DATA AT DIFFERENT X_1 (STREAMWISE) LOCATIONS. ****
#--------------------------------------------------------------------------
xh1=np.genfromtxt("xh1.xy", comments="%")
y_1=xh1[:,0] # x_2 coordinates, wall-normal direction.
v1_Exp_1=xh1[:,1] # mean velocity in the streamwise direction (x_1) along wall-normal direction (x_2). 
v2_Exp_1=xh1[:,2] # mean velocity in the streamwise direction (x_1) along wall-normal direction (x_2). 
uu_Exp_1=xh1[:,3] # Normal Reynolds stress (Re_xx) along wall-normal direction (x_2).  
vv_Exp_1=xh1[:,4] # Normal Reynolds stress (Re_yy) along wall-normal direction (x_2).
uv_Exp_1=xh1[:,5] # Shear Reynolds stress (Re_xy) along wall-normal direction (x_2).
# The locations for the measurement data are: x/h=0.05, 0.5, 1, 2, 3, 4, 5, 6, 7 and 8.
# You should find appropriate "i" corresponds to measurement x locations.
#For example, "xh005.xy", "xh05.xy" and "xh1.xy" are the measurment data at x/h=0.05,x/h=0.5 and x/h=1, repectively.



#%% AH 3.1

v_b = np.array([np.trapz(v1_2d[i,:],x2_2d[i,:])/(x2_2d[i,-1]-x2_2d[i,0]) for i in range(ni)])
P_b_ccm = np.array([np.trapz(p_2d[i,:],x2_2d[i,:])/(x2_2d[i,-1]-x2_2d[i,0]) for i in range(ni)])
P_b_bernoulli = np.zeros(ni)
P_b_bernoulli[0] = P_b_ccm[0]
for i in range(1,ni):
    P_b_bernoulli[i] = P_b_bernoulli[i-1] + rho*(v_b[i-1]**2 - v_b[i]**2)/2


delta_dyn_pressure = rho*(v_b[-1]**2-v_b[0]**2)/2
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,0],P_b_bernoulli, label = 'Bernoulli')
plt.plot(x1_2d[:,0],P_b_ccm, label = 'CCM+')
#plt.plot(x1_2d[:,0],tau_theoretical, label = 'theoretical_lower')
plt.title('Bulk preassure for CCM+ and Bernoulli')
plt.legend()
#plt.axis([0,1.5,0,0.01]) # set x & y axis
plt.xlabel('$x_1$') 
plt.ylabel('$P_b[Pa]$')
plt.grid()
plt.show()

#%% AH3.2
Shear_top = pd.read_csv('Shear_top.csv')
Shear_bottom = pd.read_csv('Shear_bottom.csv')
C_f = np.zeros((ni,2)) #x1 coords, [top = 1,bot = 0]
C_f[:,1] = Shear_top['Wall Shear Stress: Magnitude (Pa)'].values[::-1]/(0.5*rho*v_b**2)
C_f[:,0] = Shear_bottom['Wall Shear Stress: Magnitude (Pa)'].values[::-1]/(0.5*rho*v_b**2)

plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,0],C_f[:,0], label = 'bottom wall')
plt.plot(x1_2d[:,0],C_f[:,1], label = 'top wall')
#plt.plot(x1_2d[:,0],tau_theoretical, label = 'theoretical_lower')
plt.title('Skin friction at the walls')
plt.legend()
#plt.axis([0,1.5,0,0.01]) # set x & y axis
plt.xlabel('$x_1$') 
plt.ylabel('$C_f$')
plt.grid()
plt.show()

#%% AH3.3
dv1dx1_2d= np.zeros((ni,nj))
dv1dx2_2d= np.zeros((ni,nj))
dv2dx1_2d= np.zeros((ni,nj))
dv2dx2_2d= np.zeros((ni,nj))

# note that the routine 'dphidx_dy' wants x_1 and x_2 at faces (of size (ni-1)x(nj-1))
# Here we cheat a bit and give the coordinate at cell center (but using correct size (ni-1)x(nj-1))
dv1dx1_2d,dv1dx2_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],v1_2d)
dv2dx1_2d,dv2dx2_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],v2_2d)

w3_2d = dv2dx1_2d-dv1dx2_2d

fig1,ax1 = plt.subplots()
climits = [-14000,-70]
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.contourf(x1_2d,x2_2d,w3_2d,levels = np.linspace(climits[0],climits[1],100))
plt.xlabel("$x_1$[m]")
plt.ylabel("$x_2$[m]")
plt.title("$\omega_3$ for the flow",fontsize = 16,pad = 20)
#plt.axis([0,0.1,0,0.01]) # zoom-in on the first 0.1m from the inlet
plt.colorbar(ticks = np.linspace(climits[0],climits[1],5))
#plt.xticks(ticks = [0.05,0.25,0.45,0.63])
plt.show()
#%%
fig1,ax1 = plt.subplots()
#climits = [-50,50]
plt.subplots_adjust(left=0.25,bottom=0.10)
cax = plt.contourf(x1_2d[:40,:90],x2_2d[:40,:90],np.log10(abs(w3_2d[:40,:90])),levels = 100)
plt.xlabel("$x_1$[m]")
plt.ylabel("$x_2$[m]")
plt.title("$log_{10}(|\omega_3|)$ for the flow",fontsize = 16,pad = 20)
#plt.axis([0,0.1,0,0.01]) # zoom-in on the first 0.1m from the inlet
#cbar = fig1.colorbar(caxticks = [1,2,3,4], format = mticker.FixedFormatter(['$10^{-1}$','$10^{-2}$','$10^{-3}$','$10^{-4}$']))
plt.colorbar()
#plt.xticks(ticks = [0.05,0.25,0.45,0.63])
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.contourf(x1_2d,x2_2d,vist_2d/mu,20)
plt.xlabel("$x_1$[m]")
plt.ylabel("$x_2$[m]")
plt.title("$\mu_t / \mu$ for the flow",fontsize = 16,pad = 20)
#plt.axis([0,0.1,0,0.01]) # zoom-in on the first 0.1m from the inlet
plt.colorbar()#plt.colorbar(ticks = np.linspace(climits[0],climits[1],5))
#plt.xticks(ticks = [0.0,0.2,0.4])
plt.show()
max_pos = np.where(vist_2d == np.max(vist_2d))
print(f'the maximum turbulent viscosity is {np.max(vist_2d)}$[Ns/m^2]$ at x_1={x1_2d[max_pos[0][0],max_pos[1][0]]}[m], y={x2_2d[max_pos[0][0],max_pos[1][0]]}[m].')



#%% AH3.4
"""
Field_params = pd.read_csv('Fieldparams_xh2.csv')
mu_t = Field_params['Turbulent Viscosity (Pa-s)']
X = Field_params['X (m)']
Y = Field_params['Y (m)']
plt.tricontourf(X,Y,mu_t/mu,20)
"""

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.contourf(x1_2d,x2_2d,vist_2d/mu,20)
plt.xlabel("$x_1$[m]")
plt.ylabel("$x_2$[m]")
plt.title("$\mu_t / \mu$ for the flow",fontsize = 16,pad = 20)
#plt.axis([0,0.1,0,0.01]) # zoom-in on the first 0.1m from the inlet
plt.colorbar()#plt.colorbar(ticks = np.linspace(climits[0],climits[1],5))
#plt.xticks(ticks = [0.0,0.2,0.4])
plt.show()
max_pos = np.where(vist_2d == np.max(vist_2d))
print(f'the maximum turbulent viscosity is {np.max(vist_2d)}$[Ns/m^2]$ at x_1={x1_2d[max_pos[0][0],max_pos[1][0]]}[m], y={x2_2d[max_pos[0][0],max_pos[1][0]]}[m].')

def x2_plus(x2,u_t,nu): #Kanske fixa så att vi använda endast x2[1:,:] och använder x2[0,:] som vägg höjd?
    return (x2-x2[0])*u_t/nu
u_t_bot = Shear_bottom['Ustar (m/s)'].values[::-1]
#x1_index = [70,100,130] #Somewhat deliberately only choose points where bot is at heigth 0
x1_index = [5,100,190]
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
for i in x1_index:
    plt.plot(vist_2d[i,:]/mu,x2_2d[i,:]-x2_2d[i,0],label=f'$x_1$={round(x1_2d[i,0],3)}[m]')
plt.grid()
plt.xlabel('$\mu_t/\mu$')
plt.ylabel('$x_2$')
plt.title('$\mu_t/\mu$ vs $x_2$')
plt.legend(prop = {'size':12})
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
for i in x1_index:
    plt.plot(vist_2d[i,:]/mu,x2_plus(x2_2d[i,:],u_t_bot[i],nu),label=f'$x_1$={round(x1_2d[i,0],4)}[m]')
plt.grid()
plt.xlabel('$\mu_t/\mu$')
plt.ylabel('$x_2^+$')
plt.title('$\mu_t/\mu$ vs $x_2^+$')
plt.legend(prop = {'size':12})
plt.show()

#%% AH3.5 Kanske plotta dessa some logaritmer
x1_index = [3,100]
dv1dx21_2d,dv1dx22_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],dv1dx2_2d*nu)
dv1dx11_2d,dv1dx12_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],dv1dx1_2d*nu)
dv2dx12_2d,dv2dx11_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],dv2dx1_2d*nu)

nudv1dx21_2d,nudv1dx22_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],vist_2d*dv1dx2_2d/rho)
nudv1dx11_2d,nudv1dx12_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],vist_2d*dv1dx1_2d/rho)
nudv2dx12_2d,nudv2dx11_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],vist_2d*dv2dx1_2d/rho)

viscdisp_2d = dv2dx12_2d+dv1dx22_2d+dv1dx11_2d*2
turbdisp_2d = nudv2dx12_2d+nudv1dx22_2d+nudv1dx11_2d*2

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
for i in x1_index:
    plt.plot(viscdisp_2d[i,:],x2_2d[i,:],label=f'viscous')
    plt.plot(turbdisp_2d[i,:],x2_2d[i,:],label=f'turbulent')
    plt.grid()
    plt.xlabel('Diffusion')
    plt.ylabel('$x_2$')
    plt.title(f'Diffusion for $x_1$={round(x1_2d[i,0],4)}[m]')
    plt.legend(prop = {'size':12})
    plt.show()

    plt.plot(viscdisp_2d[i,20:-20],x2_2d[i,20:-20],label=f'viscous')
    plt.plot(turbdisp_2d[i,20:-20],x2_2d[i,20:-20],label=f'turbulent')
    plt.grid()
    plt.xlabel('Diffusion')
    plt.ylabel('$x_2$')
    plt.title(f'Diffusion for $x_1$={round(x1_2d[i,0],4)}[m], without edges.')
    plt.legend(prop = {'size':12})
    plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
for i in x1_index:
    plt.plot(viscdisp_2d[i,:],x2_plus(x2_2d[i,:],u_t_bot[i],nu),label=f'viscous')
    plt.plot(turbdisp_2d[i,:],x2_plus(x2_2d[i,:],u_t_bot[i],nu),label=f'turbulent')
    plt.grid()
    plt.xlabel('Diffusion')
    plt.ylabel('$x_2^+$')
    plt.title(f'Diffusion for $x_1$={round(x1_2d[i,0],4)}[m]')
    plt.legend(prop = {'size':12})
    plt.show()


#%% AH3.6

#mu_t = Field_params['Turbulent Viscosity (Pa-s)']

P_k = vist_2d/rho * (2*dv1dx1_2d**2 + dv1dx2_2d**2 + 2*dv2dx2_2d**2 + dv2dx1_2d**2 + 2*dv1dx2_2d*dv2dx1_2d)

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
climits = [0,0.05]
plt.contourf(x1_2d,x2_2d,P_k,levels = np.linspace(climits[0],climits[1],100))
#plt.grid()
plt.colorbar(ticks = np.linspace(climits[0],climits[1],5))
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'the produktion term $P^k$',pad=15)
#plt.legend(prop = {'size':12})
plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.contourf(x1_2d,x2_2d,te_2d,100)
#plt.grid()
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'the turbulent kinetic energy k',pad=15)
#plt.legend(prop = {'size':12})
plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.contourf(x1_2d,x2_2d,vist_2d,100)
#plt.grid()
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'turbulent viscosity $\mu_t$',pad=15)
#plt.legend(prop = {'size':12})
plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
plt.show()

#%%AH3.7
#diss_2d = epsilon
eps_top = 2*nu*te_2d[:,-1]/(abs(x2_2d[:,-1]-0.15175)**2)
eps_bot = 2*nu*te_2d[:,1]/(abs(x2_2d[:,1]-x2_2d[:,0])**2)

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.plot(x1_2d[:,-1],diss_2d[:,-1],label='Simulated',)
plt.plot(x1_2d[:,-1],eps_top,label='Calculated',linestyle = '--')
plt.grid()
plt.legend()
plt.xlabel('$x_1$')
plt.ylabel('$\epsilon[m^2/s^3]$')
plt.title('$\epsilon$ at the top wall',pad=15)
#plt.legend(prop = {'size':12})
plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.10)
plt.plot(x1_2d[:,1],diss_2d[:,1],label='Simulated')
plt.plot(x1_2d[:,1],eps_bot,label='Calculated')
plt.grid()
plt.legend()
plt.xlabel('$x_1$')
plt.ylabel('$\epsilon[m^2/s^3]$')
plt.title('$\epsilon$ at the bottom wall',pad=15)
#plt.legend(prop = {'size':12})
plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
plt.show()

#%%AH3.9
"""
xh1=np.genfromtxt("xh1.xy", comments="%")
y_1=xh1[:,0] # x_2 coordinates, wall-normal direction.
v1_Exp_1=xh1[:,1]
"""
hmax = 0.05
hmult_list = [0.005,0.05,1,2,3,4,5,6,8]
hmult_str_list = ['005','05','1','2','3','4','5','6','8']
for string,hmult in zip(hmult_str_list,hmult_list):
    x1_current = hmult*hmax
    min_index = np.argmin(abs(x1_2d-x1_current)) #% 0.05 = x/h = 1
    min_index_2d = np.unravel_index(min_index, x1_2d.shape)
    real_min_index = min_index_2d[0]
    xh=np.genfromtxt("xh" + string + ".xy", comments="%")
    y=xh[:,0] # x_2 coordinates, wall-normal direction.
    v1_Exp=xh[:,1]

    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.10)
    plt.plot(v1_2d[real_min_index,:],x2_2d[real_min_index,:],label = 'Simulated')
    plt.plot(v1_Exp,y,label = 'Experimental') #%% v1 och y1 till gerneral loop
    plt.grid()
    plt.legend(prop = {'size':12})
    plt.xlabel('$v_1[m/s]$')
    plt.ylabel('$x_2$[m]')
    plt.title(f'$v_1$ for $x_1$={np.round(x1_current,5)}',pad=15)
    #plt.legend(prop = {'size':12})
    #plt.xticks(ticks = [0.00,0.11,0.22,0.33,0.44])
    plt.show()
#%%
#diss = epsilon

#%% plots
#################################### plot v_1 vs. x_2 at x_1=hmax
fig1,ax1 = plt.subplots()
xx=hmax
i1 = (np.abs(xx-x1_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(v1_2d[i1,:],x2_2d[i1,:],'b-')
plt.plot(v1_Exp_1,y_1,'bo')
plt.xlabel("$V_1$")
plt.ylabel("$x_2$")
plt.title("Velocity")
plt.axis([-0.1,0.6,0.0225,H+0.01])
# Create inset of width 30% and height 40% of the parent axes' bounding box
# at the lower left corner (loc=3)
# upper left corner (loc=2)
# use borderpad=1, i.e.
# 22 points padding (as 22pt is the default fontsize) to the parent axes
axins1 = inset_axes(ax1, width="40%", height="30%", loc=2, borderpad=2)
plt.plot(v1_2d[i1,:],x2_2d[i1,:],'b-')
plt.plot(v1_Exp_1,y_1,'bo')
plt.axis([-0.1,0.01,0.0225,0.04])
# reduce fotnsize 
axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)
# Turn ticklabels of insets off
#axins1.tick_params(labelleft=False, labelbottom=False)
# put numbers on the right y axis
axins1.yaxis.set_label_position("right")
axins1.yaxis.tick_right()

plt.savefig('Vel_python.eps')

#################################### contour p
fig2 = plt.figure("Figure 2")
plt.clf() #clear the figure
plt.contourf(x1_2d,x2_2d,p_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("contour pressure plot")
plt.colorbar()
plt.savefig('p_contour.eps')




# compute velociy gradients
dv1dx1_2d= np.zeros((ni,nj))
dv1dx2_2d= np.zeros((ni,nj))
dv2dx1_2d= np.zeros((ni,nj))
dv2dx2_2d= np.zeros((ni,nj))

# note that the routine 'dphidx_dy' wants x_1 and x_2 at faces (of size (ni-1)x(nj-1))
# Here we cheat a bit and give the coordinate at cell center (but using correct size (ni-1)x(nj-1))
dv1dx1_2d,dv1dx2_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],v1_2d)
dv2dx1_2d,dv2dx2_2d = dphidx_dy(x1_2d[0:-1,0:-1],x2_2d[0:-1,0:-1],v2_2d)


#################################### vector plot
fig3 = plt.figure("Figure 1")
plt.clf() #clear the figure
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(x1_2d[::k,::k],x2_2d[::k,::k],v1_2d[::k,::k],v2_2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("vector plot")
plt.savefig('vect_python.eps')
#

# %%

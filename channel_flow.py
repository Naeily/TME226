#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

plt.interactive(True)
# Channel flow
data=np.genfromtxt("channel_flow_data.dat", comments="%")

ni=199  # number of grid nodes in x_1 direction
nj=28  # number of grid nodes in x_2 direction
x1=data[:,0] #don't use this array
x2=data[:,1] #don't use this array
v1=data[:,2] #don't use this array
v2=data[:,3] #don't use this array
p=data[:,4]  #don't use this array

# transform the arrays from 1D fields x(n) to 2D fields x(i,j)
# the first index 'i', correponds to the x-direction
# the second index 'j', correponds to the y-direction

x1_2d=np.reshape(x1,(nj,ni)) #this is x_1 (streamwise coordinate)
x2_2d=np.reshape(x2,(nj,ni)) #this is x_2 (wall-normal coordinate)
v1_2d=np.reshape(v1,(nj,ni)) #this is v_1 (streamwise velocity component)
v2_2d=np.reshape(v2,(nj,ni)) #this is v_1 (wall-normal velocity component)
p_2d=np.reshape(p,(nj,ni))   #this is p   (pressure) #ask if this is wrong with nj, ni in template

x1_2d=np.transpose(x1_2d)
x2_2d=np.transpose(x2_2d)
v1_2d=np.transpose(v1_2d)
v2_2d=np.transpose(v2_2d)
p_2d=np.transpose(p_2d)
#%% C1
# index 13 and 14 describe the middle in Y direction, symmetry aroudn "index 13.5"
nu = 1.56*10**-5
V_in = 0.9
h = 0.01
L = 0.6385
x1e_t = 0.016*V_in*4*h**2/nu
dv1dx1_center = np.gradient(v1_2d[:,13],x1_2d[:,13])
index_001 = np.argmax(dv1dx1_center < 0.01)
index_99 = np.argmax(v1_2d[:,13] > 0.99*v1_2d[:,13].max())
x1e_001 = x1_2d[index_001,0]
x1e_99 = x1_2d[index_99,0]
# plotta dessa två mot teoretisk: v1_2d[index_001,:], v1_2d[index_99,:]
# kolla på v2_2d[index_001,x2=h/4] 

#duxk you

#%% C2
mu = 1.81 * 10**-5
dpdx1 = np.gradient(p_2d[:,0],x1_2d[:,0]) 
tau_wall_lower = mu * (v1_2d[:,1]-v1_2d[:,0])/(x2_2d[0,1]-x2_2d[0,0])
tau_wall_upper = mu * -(v1_2d[:,-1]-v1_2d[:,-2])/(x2_2d[0,-1]-x2_2d[0,-2])
tau_theoretical = h/2 * -dpdx1 

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,0],tau_wall_lower, label = 'lower')
plt.plot(x1_2d[:,0],tau_wall_upper, label = 'upper')
#plt.plot(x1_2d[:,0],tau_theoretical, label = 'theoretical_lower')
plt.title('Shear stress at the lower wall')
plt.legend()
#plt.axis([0,1.5,0,0.01]) # set x & y axis
plt.xlabel('$x_1$') 
plt.ylabel('$tau_{wall}$') 
#plt.text(0.04,0.004,'$x_1=0.008$ and $0.34$') # show this text at (0.04,0.004)
#plt.savefig('velprof.eps')



#%% C3
dv1dx1_wall = np.gradient(v1_2d[:,2],x1_2d[:,2])
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,13],v1_2d[:,13], label = 'v1_{center}')
plt.plot(x1_2d[:,2],v1_2d[:,2], label = 'v1_{edge}')

#plt.plot(x1_2d[:,13],dv1dx1_center, label = 'dv1dx1_{center}')
#plt.plot(x1_2d[:,2],dv1dx1_wall, label = 'dv1dx1_{edge}')
plt.legend()
plt.show()

def xi(x1_index):
      return np.trapz(v1_2d[x1_index,:],x2_2d[0,:])

plt.plot(x1_2d[:,0], [xi(i) for i in range(ni)])


#%% C4
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,13],v2_2d[:,13], label = 'v2_{center}')
plt.plot(x1_2d[:,2],v2_2d[:,2], label = 'v2_{edge}')
plt.legend()


#%% C5
dv2dx1_2d = np.zeros((ni,nj))
dv1dx2_2d = np.zeros((ni,nj))
for i in range(nj):
   dv2dx1_2d[:,i] = np.gradient(v2_2d[:,i],x1_2d[:,i]) #list comprehension
for i in range(ni):
   dv1dx2_2d[i,:] = np.gradient(v1_2d[i,:],x2_2d[i,:]) #list comprehension
w_3_2d = dv2dx1_2d - dv1dx2_2d
plt.imshow(w_3_2d.T, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
#%%
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(w_3_2d[index_99,:],x2_2d[index_99,:])
plt.title('$\omega_3$ for the fully developed flow')
plt.xlabel('$\omega_3$') 
plt.ylabel('$x_2$')
#plt.savefig('C5_w_3_fully_developed.eps')
plt.show()
#%%
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(w_3_2d[0,:],x2_2d[0,:])
plt.title('$\omega_3$ for the inflow')
plt.xlabel('$\omega_3$') 
plt.ylabel('$x_2$')
#plt.savefig('C5_w_3_inflow.eps')
plt.show()


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(w_3_2d[int(index_99/2),:],x2_2d[int(index_99/2),:])
plt.title('$\omega_3$ for the developing flow')
plt.xlabel('$\omega_3$') 
plt.ylabel('$x_2$')
#plt.savefig('C5_w_3_developing.eps')
plt.show()


#%% C6 #Ask if we are supposed to plot at fully developed flow?
S_12_2d = 1/2*(dv1dx2_2d + dv2dx1_2d)
Omega_12_2d = 1/2*(dv1dx2_2d - dv2dx1_2d)
plt.imshow(S_12_2d.T, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
plt.imshow(Omega_12_2d.T, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(Omega_12_2d[index_99,:],x2_2d[index_99,:])
plt.title('$\Omega_{12}$ for the fully developed flow')
plt.xlabel('$\Omega_{12}$') 
plt.ylabel('$x_2$')
#plt.savefig('C6_Omega12.eps')
plt.show()


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(S_12_2d[index_99,:],x2_2d[index_99,:])
plt.title('$S_{12}$ for the fully developed flow')
plt.xlabel('$S_{12}$') 
plt.ylabel('$x_2$')
#plt.savefig('C6_S12.eps')
plt.show()

#%% C7
c_p = 1006 #J/kg 20c 1bar
rho = 1.204 #kg/m^3 20c 1atm ########################Maybe a bit wrong assuming aprox atm preassure lmaooooooooooo
T_in = 20 # same as T_b_in
dv2dx2_2d = np.zeros((ni,nj))
dv1dx1_2d = np.zeros((ni,nj))
for i in range(nj):
   dv1dx1_2d[:,i] = np.gradient(v1_2d[:,i],x1_2d[:,i]) #list comprehension
for i in range(ni):
   dv2dx2_2d[i,:] = np.gradient(v2_2d[i,:],x2_2d[i,:]) #list comprehension


# Kankse inte ha tau som np.zeros för 11 och 22
tau_11_2d = 2*mu*dv1dx1_2d - 1/3*mu*(dv1dx1_2d+dv2dx2_2d) #np.zeros((ni,nj))
tau_12_2d = mu*(dv1dx2_2d + dv2dx1_2d)
tau_22_2d = 2*mu*dv2dx2_2d - 1/3*mu*(dv1dx1_2d+dv2dx2_2d) #np.zeros((ni,nj))
#skriv om phi_2d med hjälp av tau
phi_2d = ( tau_11_2d*dv1dx1_2d + tau_12_2d*dv1dx2_2d + tau_12_2d*dv2dx1_2d + tau_22_2d*dv2dx2_2d )
#phi_2d = mu*(np.square(dv1dx2_2d) + np.square(dv2dx1_2d) + 2*(np.square(dv1dx1_2d) + np.square(dv2dx2_2d) + dv1dx2_2d*dv2dx1_2d))

plt.imshow(phi_2d.T, cmap='hot', interpolation='nearest')
plt.title('Phi')
plt.colorbar()
plt.show()

#%% fråga om denna och fixa denna
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.contourf(x1_2d,x2_2d,phi_2d,10000)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.clim(0.1,10.)
plt.title("contour phi plot")
plt.axis([0,0.1,0,0.011]) # zoom-in on the first 0.1m from the inlet
plt.colorbar()
plt.show()
#plt.savefig('v1_grad.eps')
#%%
#plot of difference in phi between inlet and outlet
plt.plot(phi_2d[-1,:]-phi_2d[0,:], range(nj))
#we can see that there is slightly less overall dissipation which makes sense since it has lost energy travelign from entrence to exit
phi_integral = np.trapz(np.array([np.trapz(phi_2d[:,i],x1_2d[:,i]) for i in range(nj)]),x2_2d[0,:])
Delta_T_b = phi_integral/(c_p*rho*xi(ni-1))
# do the same derivation, but just assume xi(-1) = xi(0) xdedddd

#%% C8
tau = np.zeros((2,2,ni,nj))
n = np.zeros((2,2,ni,nj))
lambda_ = np.zeros((2,ni,nj))
for i in range(ni):
   for j in range(nj):
      tau[:,:,i,j] = np.array(( [tau_11_2d[i,j],tau_12_2d[i,j]] , [tau_12_2d[i,j],tau_22_2d[i,j]] ))
      lambda_[:,i,j], n[:,:,i,j] = np.linalg.eig(tau[:,:,i,j])

plt.plot(x2_2d[index_99,:],lambda_[0,index_99,:], label = 'first eigenvalue')
plt.plot(x2_2d[index_99,:],lambda_[1,index_99,:], label = 'second eigenvalue')
plt.legend()
plt.show() # actually the eigenvalue for a corresponding eigenvector is lienar similar to tau 12 since eigenvectors are the same for all x_2 but magnitude differs
# it is just that linalg.eig mixes the eigenvectors around from time to time hence the jumps in the plot
for i in [0,1]:
   for j in [0,1]:
      plt.plot(x2_2d[index_99,:],tau[i,j,index_99,:],label = f'tau_{str(i+1) + str(j+1)}')
      plt.legend()
      plt.show()


#%% C9
plt.quiver(x1_2d[::5,:],x2_2d[::5,:],n[0,0,::5,:],n[1,0,::5,:],scale = 70)
plt.title('Normal vector')
plt.show()
#t = np.zeros((2,ni,nj))
t_1 = n[:,0,:,:]*lambda_[0,:,:] #+ n[:,1,:,:]*lambda_[1,:,:]
plt.quiver(x1_2d[::5,:],x2_2d[::5,:],t_1[0,::5,:],t_1[1,::5,:])
plt.title('Traction_1')
plt.show()

t_2 = n[:,1,:,:]*lambda_[1,:,:]
plt.quiver(x1_2d[::5,:],x2_2d[::5,:],t_2[0,::5,:],t_2[1,::5,:])
plt.title('Traction_2')
plt.show()




#%%
#************
# velocity profile plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=169 # plot the velocity profile for i=169
plt.plot(v1_2d[i,:],x2_2d[i,:],'b-')
i=4 # plot the velocity profile for i=4
plt.plot(v1_2d[i,:],x2_2d[i,:],'r--')  #red dashed line
plt.title('Velocity profile')
plt.axis([0,1.5,0,0.01]) # set x & y axis
plt.xlabel('$V_1$') 
plt.ylabel('$x_2$') 
plt.text(0.04,0.004,'$x_1=0.008$ and $0.34$') # show this text at (0.04,0.004)
plt.show()
#plt.savefig('velprof.eps')

################################ contour plot of v1
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.contourf(x1_2d,x2_2d,v1_2d, 50)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.clim(0.1,1.)
plt.title("contour v_1 plot")
plt.axis([0,0.1,0,0.011]) # zoom-in on the first 0.1m from the inlet
plt.show()
#plt.savefig('v1_contour.eps')

################################ compute the velocity gradient dv_1/dx_2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
dv1_dx2=np.zeros((ni,nj))
for i in range(0,ni-1):
   for j in range(0,nj-1):
      dx2=x2_2d[i,j+1]-x2_2d[i,j-1]
      dv1_dx2[i,j]=(v1_2d[i,j+1]-v1_2d[i,j-1])/dx2

# fix the derivative at the walls
for i in range(0,ni-1):
# lower wall
   dx2=x2_2d[i,1]-x2_2d[i,0]
   dv1_dx2[i,0]=(v1_2d[i,1]-v1_2d[i,0])/dx2
# upper wall
   dx2=x2_2d[i,nj-1]-x2_2d[i,nj-2]
   dv1_dx2[i,nj-1]=(v1_2d[i,nj-1]-v1_2d[i,nj-2])/dx2

# you can also use the built-in command
x1_1d=x1_2d[:,1] # make 1d array
x2_1d=x2_2d[1,:] # make 1d array
dv1_dx1_built_in, dv1_dx2_bulit_in=np.gradient(v1_2d,x1_1d,x2_1d)


i=169 # plot the velocity gradient for i=169
plt.plot(dv1_dx2[i,:],x2_2d[i,:])
plt.axis([-550,550,0,0.01]) # set x & y axis
plt.title('Velocity gradient')
plt.xlabel('$\partial v_1/\partial x_2$')
plt.ylabel('$x_2$') 
plt.text(-380,0.004,'$x_1=0.52$')
plt.show()
#plt.savefig('v1_grad.eps')


# %%

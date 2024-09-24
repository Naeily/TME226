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
   dx2 = np.zeros((28))
   dx2[1:27] = (x2_2d[0,2:]-x2_2d[0,0:-2])/2
   dx2[0] = (x2_2d[0,1]-x2_2d[0,0])/2
   dx2[27] = (x2_2d[0,-1]-x2_2d[0,-2])/2
   return sum(v1_2d[x1_index,:]*dx2)
plt.plot(x1_2d[:,0], [xi(i) for i in range(len(x1_2d[:,0]))])





#%% C4
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x1_2d[:,13],v2_2d[:,13], label = 'v2_{center}')
plt.plot(x1_2d[:,2],v2_2d[:,2], label = 'v2_{edge}')
plt.legend()


#%% C5
dv2dx1_2d = np.gradient(v2_2d,x1_2d) #list comprehension
dv1dx2_2d = np.gradient(v1_2d,x2_2d) #list comprehension
w_3_2d = dv2dx1_2d - dv1dx2_2d
plt.imshow(w_3_2d, cmap='hot', interpolation='nearest')
plt.colorbar()
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
plt.savefig('velprof.eps')

################################ contour plot of v1
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.contourf(x1_2d,x2_2d,v1_2d, 50)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.clim(0.1,1.)
plt.title("contour v_1 plot")
plt.axis([0,0.1,0,0.011]) # zoom-in on the first 0.1m from the inlet
plt.savefig('v1_contour.eps')

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
plt.savefig('v1_grad.eps')


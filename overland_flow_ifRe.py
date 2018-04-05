#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Overland Flow (laminar flow)

Created on Wed Mar 28 12:31:27 2018

@author: nadine
"""

#%% import modules
import numpy as np
import matplotlib.pyplot as plt

#%% NUMERICAL SOLUTION
# set variables

#### time ####
days = .2
tmax = days * 24. * 60 * 60 # one day in seconds
dt = .2 # seconds
time = np.arange(0,tmax+dt,dt)

#### space ####
slope = np.radians(2)
g = 9.81 # m/s
grav = g*np.sin(slope)
xmax = 1000. #meter 
dx = 10.
channel_width = 1.
x = np.arange(0,xmax+dx,dx)
zmin = 0.
zmax = 100.
z = zmax - x*slope
z0 = (1./30.) * .05

#zb = zbmax - (slope*x)
#zb = zb+(zbmax2*(np.exp(-x/xstar)))

#plt.plot(x,z)
#plt.title('hillslope')


#### water ####
rho = 1000. # kg/m^3
mu = 0.0012083 # N*s/m2
karman = 0.408 # von Karman's constant
#Type of Storm Rate
# from: https://scool.larc.nasa.gov/lesson_plans/RainfallRatesWaterVolume.pdf
light = .003/(60*60)  # 2 - 4 mm/hr. 3 mm/hr = .003m/hr = .003/60*60 m/s
moderate = .009/(60*60) # 5 - 9 mm/hr --> m/s
heavy = .02/(60*60) #10 - 40 mm/hr --> m/s
violent = .06/(60*60) # >50 mm/hr --> m/s

hours = .5
storm_length = np.int(hours*60*60/dt) # three hours in seconds
storm_rate1 = light
storm_rate2 = moderate
storm_rate3 = heavy
storm_rate4 = light 

precip = np.zeros(len(time))
precip[0:storm_length] = light
precip[storm_length:storm_length*2] = moderate
precip[storm_length*2:storm_length*3] = heavy
precip[storm_length*3:storm_length*4] = moderate
precip[storm_length*4:storm_length*5] = light

#Type of Infiltration Rate
#from: https://stormwater.pca.state.mn.us/index.php?title=Design_infiltration_rates
none = 0.
reallyslow = .0005/(60*60)
slow = .0015/(60*60)  # 0.15 cm/hr --> .0015 m/hr = .0015/60*60 m/s
medium = .005/(60*60) # 0.5 cm/hr --> m/s
fast = .01/(60*60) # .76 - 1.14 cm/hr --> m/s
extreme = .04/(60*60) # 2.03 - 4.14 cm/hr --> m/s

infiltration = np.zeros(len(time))
infiltration[:] = slow

#Type of Evaporation Rate
evap = np.zeros(len(time))
evap[:] = none

#### plot initial conditions ###
# precip
plt.plot(time/(60*60), precip*100*60*60,'b',label='precipitation')
plt.plot(time/(60*60), evap*100*60*60,'g--',lw=1.,label='evaporation')
plt.plot(time/(60*60), infiltration*100*60*60,'r:',lw=.75,label='infiltration')
plt.title('Water Regime')
plt.xlabel('time (hrs)')  
plt.ylabel('rate (cm/hr)')
plt.legend()
plt.show()

Ubase_heavy = (heavy*xmax) / (np.cbrt(3*mu*heavy*xmax)/(rho*g*np.sin(slope)))
Ubase_moderate = (moderate*xmax) / (np.cbrt(3*mu*moderate*xmax)/(rho*g*np.sin(slope)))
Ubase_light = (light*xmax) / (np.cbrt(3*mu*light*xmax)/(rho*g*np.sin(slope)))

dt_heavy = dx/Ubase_heavy
dt_moderate = dx/Ubase_moderate
dt_light = dx/Ubase_light

print('dt_heavy = '+str(dt_heavy))
print('dt_moderate = '+str(dt_moderate))
print('dt_light = '+str(dt_light))
print('dt is: '+ str(dt))

#%% RUN MODEL WITH LAMINAR / TURBULENT FLOW
# initialize arrays
q = np.zeros(shape=(len(x),len(time)))
h = np.zeros(shape=(len(x),len(time)))
#U = np.zeros(len(x))
U = np.zeros(shape=(len(x),len(time)))
dqdx = np.zeros(len(x)-1)
dhdt = np.zeros(len(x)-1)
Re = np.zeros(shape=(len(x),len(time)))

h[:,0] = (precip[0] - infiltration[0] - evap[0]) * x # give some water depth at first time step
q[0,:] = 0. # set first discharge cell (spatially) to always be zero


for i in range(len(time)-1):
    #hedge[0:-1] = h[0:-1] or avg of two cells
    #hedge[-1] = h[-1]
    U[:,i] = ((rho*g*np.sin(slope))/mu) * ((h[:,i]**2.)/3.) # calculate water velocity for this time and all space
    #U[:,i] = ((np.sqrt(g*h[:,i]*np.sin(slope))) / karman) * (np.log(z[:]/z0)-1)
#    if np.any(np.isnan(U[:,i])) or np.any(np.isinf(U[:,i])):
#        print('warning! U is nan or inf at timestep '+ str(i))
#        break
    q[:,i] = U[:,i] * h[:,i] #* channel_width
    Re[:,i] = (q[:,i]*rho)/mu
    if np.any(Re[:,i] > 2000):
        U[:,i] = ((np.sqrt(g*h[:,i]*np.sin(slope))) / karman) * (np.log(z[:]/z0)-1)
        q[:,i] = U[:,i] * h[:,i] #* channel_width
    dqdx = np.diff(q[:,i]) / dx
    dhdt = (precip[i] - infiltration[i] - evap[i]) - dqdx
    h[1:,i+1] = h[1:,i] + (dhdt * dt)
    h[1:,i+1] = np.maximum(0,h[1:,i+1])

# plot at i = 
i = np.int(60*60*1.5)

# plot hydrograph 
#plt.fill_between(time/(60*60),q[-1,:],0) # plot time in hours and discharge at bottom hill through all time
plt.plot(time/(60*60),q[-1,:]) # plot time in hours and discharge at bottom hill through all time
plt.plot(i/(60.*60.),0,'r.')
plt.xlabel('time (hrs)')
plt.ylabel('discharge (m^3/s)')
plt.title('hydrograph')
plt.show()

# plot Reynolds number
plt.plot(time/(60*60),Re[-1,:]) # plot time in hours and discharge at bottom hill through all time
plt.plot(i/(60.*60.),0,'r.')
plt.xlabel('time (hrs)')
plt.ylabel('Re')
plt.title('Reynolds Number at Bottom')
plt.show()

# plot spatial distribution at time step i
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,sharex=True)
plt.tight_layout()
ax4.plot(x,h[:,i],'royalblue')
ax4.set_title('Water Depth',fontsize=10)
ax4.set_ylabel('m')
ax4.set_xlabel('distance (m)')
ax2.plot(x,q[:,i],'limegreen')
ax2.set_title('Specific Discharge',fontsize=10)
ax2.set_ylabel('m^2/s')
ax3.plot(x,U[:,i],'orange')
ax3.set_title('Velocity',fontsize=10)
ax3.set_ylabel('m/s')
ax1.plot(x,z[:],'brown')
ax1.set_title('Hillslope',fontsize=10)
ax1.set_ylabel('m')
plt.show()


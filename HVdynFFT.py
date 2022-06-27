#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy import fftpack
# %matplotlib widget


# In[29]:


dt = int(input('dt=')) # in ps


# In[30]:


ratioAW=6.75


# In[31]:


def fftdata(N, deltaT, data):
    # deltaT in ps, f in THz
    f_s = 1/deltaT
    delta_f = 1/(N*deltaT)
    F = fft(data)
    # f = fftpack.fftfreq(N, deltaT)
    f = np.fft.fftfreq(N, d=deltaT)
    mask = np.where(f >= 0)
    return f[mask], F[mask]


# ## CMD FFT HV

# In[32]:


# PEG2water = np.loadtxt('water2PEG_allParts')[:,2]
# HwaterBulk = np.loadtxt('down_HorizontalHB_Bulk.dat')[:,1]-ratioAW
HwaterBIL = np.loadtxt('down_HorizontalHB.dat')[:,1]-ratioAW
# PEG2water = np.zeros(len(Hwater))
HVratio = HwaterBIL/(1)-ratioAW


# In[33]:


# fCMDBulk, FCMDBulk = fftdata(1800, 50, HwaterBulk)
fCMDBIL, FCMDBIL = fftdata(HwaterBIL.size, dt, HwaterBIL)
# fVCMD, FVCMD = fftdata(1800, 50, PEG2water)
# fHV, FHV = fftdata(1800, 50, HVratio)


# In[35]:


def plotRatio(r):
    bwidth = 1.5
    s = 20
    fig, ax = plt.subplots()
    # fig.set_size_inches(6.3,5.7)
    label_list = [str(int(i*0.05)) for i in np.arange(0, 1801, 200)]
    x = range(1,len(r)+1)
    ratio = ax.bar(x, r, align='center', alpha=1, width=0.4, color='blue', ecolor='grey', capsize=s)
    ax.set_ylabel(r"H/V ratio (CMD)", size=s)
    ax.set_xlabel('t (ns)', size=s)
    ax.set_xticks(x)
    # start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, 1801, 200))
    ax.tick_params(which='major', direction='out', length=6, labelsize=s)
    ax.set_xticklabels(label_list, size=s)
    ax.set_ylim(-7, 0)
    ax = plt.gca()  # get frame
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth)
    plt.tight_layout()
#     plt.savefig('CMD-HBulk-PEG90ns.pdf', dpi = 1000)


# ## Plot the HV FFT

# In[36]:


s, bwidth = 20, 1.5

fig, ax = plt.subplots()
# ax.plot(fDFT, abs(FDFT)/40, label = 'DFT-MD', linewidth=2)
# ax.plot(fCMDBulk, abs(FCMDBulk)/1800, label = 'H-Bulk')
# ax.plot(fVCMD, abs(FVCMD)/1800, label = 'V-PEGwater', color='blue')
ax.plot(fCMDBIL, abs(FCMDBIL)/HwaterBIL.size, label = 'H-BIL', color='orange')
# ax.plot(fHV, abs(FHV)/1800, label = 'HV ratio', color = 'green')

# ax.set_ylabel(r"H/V ratio (DFT-MD)", size=s)
ax.set_xlabel('frequency (THz)', size=s)
# ax.set_xticks(x)
# ax.xaxis.set_ticks(np.arange(0, 41, 10))
ax.tick_params(which='major', direction='out', length=6, labelsize=s)
plt.legend(prop={'size': 10})
# ax.set_xticklabels(label_list, size=s)
# ax.set_xlim(0, 0.010)
# ax.set_ylim(0, 0.4)
ax = plt.gca()  # get frame
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['left'].set_linewidth(bwidth)
ax.spines['top'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
plt.tight_layout()
plt.savefig('FFT-HV.pdf', dpi = 200)


# In[ ]:





# In[ ]:





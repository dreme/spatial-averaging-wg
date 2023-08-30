# -*- coding: utf-8 -*-
"""
Created on 29/8/2023
@author: Vitas Anderson
This module contains some useful functions for RF calculations

CHANGELOG
+ Copied functions from RFcalcUC_v11
"""
__version_info__ = (0, 0)
__version__ = '.'.join(map(str, __version_info__))
__author__ = 'Vitas Anderson'

# Import modules
import pandas as pd
import numpy as np
import math

# Create dictionary of Mayavi color coordinates
COLORS = {'blue': (65/255., 105/255., 225/255.),
          'darkblue': (0,0,0.55),
          'green': (0, 1, 0),
          'darkgreen': (0, 0.55, 0),
          'red': (1, 0, 0),
          'darkred': (0.55,0,0),
          'coral2': (238/255,106/255,80/255),
          'yellow': (1, 1, 0),
          'gold': (1,215/255,0),
          'black': (0, 0, 0),
          'orange': (1, 140/255, 0),
          'magenta': (1, 0, 1),
          'white': (1,1,1),
          'pink': (1,0.75,0.8),
          'brown': (0.35,0.24,0.12),
          'olive': (0.29,0.33,0.13),
          'lightgrey': (0.8706, 0.8706, 0.8706),
          'lightgrey2':(0.34,0,34,0,34),
          'darkgrey': (0.1,0.1,0.1)
          }

def SARlimit(setting='pub'):
    """This function returns the WBA SAR limit in W/kg
    usage: Slimit(setting)
    setting = lower tier (public/uncontrolled) or upper tier (occupational/controlled) setting
    """
    setdic = dict(pub='pub',
                  occ='occ',
                  unc='pub',
                  con='occ')
    setting = setdic[setting[:3].lower()]  # just use the first 3 letters to identify the setting type
    assert setting in setdic.keys(), f"setting ({setting}) must start with one of {setdic.keys()}"
    
    limit = 0.08 if setting == 'pub' else 0.4
    return limit

def Slimit(freq, setting='pub', standard='RPS S-1 WB'):
    """This function returns the compliance standard limit for S in W/m²
    usage: Slimit(f, setting, standard)
    INPUTS:
      setting = lower tier (public/uncontrolled) or upper tier (occupational/controlled)
         freq = exposure frequency in MHz (10 <= f <= 300,000)
     standard = applicable compliance standard"""

    setdic = dict(pub='pub',
                  occ='occ',
                  unc='pub',
                  con='occ')
    setting = setdic[setting[:3].lower()]  # just use the first 3 letters to identify the setting type

    assert setting in setdic.keys(), f"setting ({setting}) must start with one of {setdic.keys()}"
    standards = ['RPS3','FCC','ICNIRP 2020 WB','ICNIRP 2020 local',
                 'RPS S-1 WB', 'RPS S-1 local']
    assert standard in standards, f"standard ({standard}) must be in {standards}"
    assert type(freq) in (int, float), f"freq ({freq}) is not an integer or float"
    assert 10 <= freq <= 300_000, f"freq ({freq}) is not between 10 and 300,000 MHz"

    if standard == 'RPS3':
        if freq > 2000:
            S = 10
        elif freq > 400:
            S = freq / 200.
        elif freq >= 10:
            S = 2
        if setting == 'occ': S = S * 5
    elif standard == 'FCC':
        if freq > 1500:
            S = 10
        elif freq > 300:
            S = freq / 150.
        elif freq >= 30:
            S = 2
        if setting == 'occ': S = S * 5
    elif standard in ['ICNIRP 2020 WB', 'RPS S-1 WB']:
        if freq > 2000:
            S = 10
        elif freq > 400:
            S = freq / 200.
        elif freq >= 10:
            S = 2
        if setting == 'occ': S = S * 5
    elif standard in ['ICNIRP 2020 local', 'RPS S-1 local']:
        if freq > 6000:
            S = 55 / (freq/1000)**0.177
        elif freq > 2000:
            S = 40
        elif freq > 400:
            S = 0.058 * freq**0.86
        elif freq >= 30:
            S = 10
        if setting == 'occ': S = S * 5

    return S

def SAR_IEC_front(d,body,fMHz,P,N,DdBi,L,Φdeg,Θdeg):
    '''This function calculates upper bound estimates of local 1g, 10g and whole body average
       SAR induced by exposure in front of a directional (vertical or cross polarized) or
       omni antenna for frequencies 300 to 5000 MHz in accordance with the SAR formulas
       in section B.4.2.2 in the IEC 62232 (2018) standard.
       The driven elements of the antenna must lie on the same vertical axis so it is generally
       only suitable for MBS omni and panel antennas
       
       Inputs:
          d = closest distance between the outermost point of the antenna and 
              a box enclosing the body
       body = body type for SAR calculations ("adult" or "child")
       fMHz = frequency of the exposure (MHz)
          P = radiated power of the antenna in (W)
          N = Number of driven elements
       DdBi = peak directivity of the antenna (dBi)
          L = overall height of the antenna (m)
       Φdeg = horizontal half-power beamwidth (degrees) of the antenna
       Θdeg = vertical half-power beam width (degrees) of the antenna
       
       Outputs:
         SARwb, SAR10g, SAR1g (W/kg)       
    '''
    
    # Constants
    π = math.pi
    nan = math.nan  # nan = not a number
    λ = 300 / fMHz
    
    # check inputs
    assert 300 <= fMHz <= 5000, f'fMHz ({fMHz} MHz) must lie within range 300 to 5000 MHz'
    assert body in ['adult','child'], f'body ({body}) must be either "adult" or "child"'
    assert 0.1 < L < 10, f'value for L ({L} m) does not look right'
    assert 1 <= N <= 15, f'value for N ({N}) does not look right'
    assert 10 < Φdeg <= 360, f'Φdeg ({Φdeg}°) should lie within range 10 to 360 degrees'
    assert 4 < Θdeg <= 180, f'Θdeg ({Φdeg}°) should lie within range 4 to 180 degrees'
        
    # Convert DdBi to linear gain
    D = 10**(DdBi/10) # linear

    # convert Φdeg and Θdeg to radians
    Φ = Φdeg * π / 180
    Θ = Θdeg * π / 180
    
    # calculate Hbeam
    Hbeam = 2 * d * math.tan(Θ/2)
    
    # Set A and B
    A, B = (0.089, 1.54) if body.lower() == 'adult' else (0.06, 0.96)
    
    # Calculate Rwb10g, Rwb1g
    Rwb10g, Rwb1g = (1.5,0.6) if fMHz <= 2500 else (1,0.3)
        
    # calculate Heff
    if L >= B:
        Heff = B
    else:
        if Hbeam < L and Hbeam < B:
            Heff = L
        elif L <= Hbeam < B:
            Heff = Hbeam
        elif Hbeam >= B:
            Heff = B

    # Calculate C(f)
    dmm = d * 1000  # convert d from m to mm
    if fMHz <= 900:
        if dmm < 200:
            C = nan
        elif dmm <= 400:
            C = (3.5 + (fMHz-300)/600 ) * (1 + 0.8*dmm/400)
        else:
            C = 6.3 + (fMHz-300)/600 * 1.8
    else:
        if dmm <= 400:
            C = 4.5 * (1 + 0.8*dmm/400)
        else:
            C = 8.1
    C = C * 1E-4
    
    # Calculate whole body average SAR
    if d > λ/(2*π):
        SARwb  = C*Heff*P / (A*B*Φ*L*d) * (1 + ((4*π*d) / (Φ*D*L))**2)**-0.5
    else:
        SARwb = nan
        
    # Calculate 10g, 1g localised SAR
    if d >= 0.2:
        SAR10g = 25 * SARwb * B / (Heff * Rwb10g)
        SAR1g  = 20 * SARwb * B / (Heff * Rwb1g)
    else:
        SAR10g, SAR1g = nan, nan
        
    return SARwb, SAR10g, SAR1g

def cint(x):
    '''Returns the nearest integer of x'''
    return int(round(x))

def find_idx(arr, v):
    '''find index of nearest value to v in a sorted numpy array
    usage: find_idx(arr, v)
    arr is a sorted numpy array which is being searched for value v'''
    idx = np.searchsorted(arr, v, side="left")
    if idx > 0 and (idx == len(arr) or math.fabs(v - arr[idx - 1]) < math.fabs(v - arr[idx])):
        return idx - 1
    else:
        return idx

def round_to_1(x):
    '''round x down to one significant digit'''
    return round(x, -int(math.floor(math.log10(x))))

def getgrid(df):
    '''Create a dataframe of a grid's xyz characteristics:
       min, max, n, delta'''
    
    def n(arr):
        '''Return number of unique elements in array'''
        return arr.unique().size

    def max_delta(arr):
        '''Return dx, dy or dz increment between x, y or z points'''
        return np.diff(np.sort(np.unique(arr))).max()
   
    grid = df[['x','y','z']].agg(['min','max',n,max_delta]).T
    grid['n'] = grid.n.astype(int)
    return grid

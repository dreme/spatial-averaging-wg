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

def dB(x):
    '''Convert x to dB(x)'''
    return 10. * np.log10(x)

def use_best_dtype(df, verbose=True):
    '''This function converts the dtype of each column in the dataframe
       to the most memory efficient type
       INPUTS:
         df = pandas dataframe
    '''
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def sf(m, S, trial, antboxsize, setting='pub', data='Smax', offset=0,
       errtol=0, spatavgL=None, power=None, standard='RPS S-1 WB'):
    '''creates filter masks for S data columns
    usage: .sf(m, setting, data, offset, errtol, power)
          m = string to indicate defined filter mask or boolean expression for filter
              all -> all points
              outant -> points outside offsetted antenna box
              ant -> points inside and on offsetted antenna box (= ~outant)
              spatavg -> valid points for spatial average window or SAR body length
              spatavg_outant -> valid spatial averaging points outside offsetted antenna box
              cb -> points near compliance boundary
              icb -> points inside compliance boundary
          S = dataframe containing xyz and S values
       fMHz = exposure frequency in MHz
 antboxsize = size of a box encompassing the antenna 
    setting = the upper or lower tier of the limit (e.g. 'pub', 'uncontrolled', 'occupational')
       data = S dataset in S (e.g. Sixus, SE, SH, Smax)
     offset = dimensional offset in metres for enlarging the antenna box or 
              the additional z direction offset from the antenna box for the nobody volume
     errtol = error tolerance for selecting points near compliance boundary
   spatavgL = length (m) of the spatial averaging window 
      power = adjusted power of antenna (the limit value is scaled by self.power/power)'''

    # Set internal variables
    if spatavgL == None: spatavgL = 1.6
    Slim = Slimit(fMHz, setting, standard)
    Slim_adjusted = Slim * self.power / power

    # functions for filter masks
    def fnAll():
        # All points
        mask = np.repeat(True,len(S))
        return mask

    def fnOutant(offset):
        # Points outside of the offsetted antenna box
        x0, x1, y0, y1, z0, z1 = self.antboxsize(offset)
        mask = (S.x < x0) | (S.x > x1) | \
               (S.y < y0) | (S.y > y1) | \
               (S.z < z0) | (S.z > z1)
        return mask

    def fnNearField(offset, spatavgL):
        # Valid points inside the offsetted antenna box
        mask1 = fnOutant(offset)
        mask2 = fnValid(0.001,spatavgL)
        mask = ~mask1 & mask2
        return mask

    def fnValid(offset, spatavgL):
        # Valid points for the antenna offset and spatial averaging length
        x0, x1, y0, y1, z0, z1 = self.antboxsize(0)
        zmin, zmax, dz = self.grid['z']
        zoffset = max(offset, spatavgL/2)
        mask1 = (S.x > x0-offset)  & (S.x < x1+offset) & \
                (S.y > y0-offset)  & (S.y < y1+offset) & \
                (S.z > z0-zoffset) & (S.z < z1+zoffset)
        mask2 = (S.z >= zmin+spatavgL/2) & (S.z <= zmax-spatavgL/2)
        mask = ~mask1 & mask2
        return mask

    def fnSpatavg(spatavgL):
        x0, x1, y0, y1, z0, z1 = self.antboxsize(0)
        zmin, zmax, dz = self.grid['z']
        mask = ((S.x < x0) | (S.x > x1) | (S.y < y0) | (S.y > y1)) & \
               (S.z >= zmin+spatavgL/2) & (S.z <= zmax-spatavgL/2)
        return mask

    def fnSpatavgOutant(offset,spatavgL):
        mask1 = fnOutant(offset)
        mask2 = fnSpatavg(spatavgL)
        mask = mask1 & mask2
        return mask

    def fnCb(Slim, data, offset, errtol):
        mask1 = fnOutant(offset) 
        mask2 = np.isclose(S[data], Slim, rtol=errtol)
        mask = mask1 & mask2
        return mask

    def fnIcb(Slim, data, offset):
        mask1 = fnOutant(offset) 
        mask2 = S[data] >= Slim
        mask = mask1 & mask2
        return mask

    def fnMeval(m, offset):
        meval = m
        for c in S.columns:
            if c in meval:
                meval = meval.replace(c, 'S.' + c)
                print(f'replacing {c} with S.{c} --> {meval}')
        meval = meval.replace('SmaS.x', 'S.Smax')
        meval = meval.replace('S.S.', 'S.')
        meval = meval.replace('SiS.xus', 'S.Sixus')
        print(f'meval: {meval}')
        mask1 = fnOutant(offset)
        mask2 = eval(meval)
        mask = mask1 & mask2
        return mask            

    # Select filter
    if m == 'all':
        mask = fnAll()
        name = 'all points'
    elif m == 'outant':
        mask = fnOutant(offset)
        if offset == 0:
            name = 'points outside antenna box'
        else:
            name = f'points further than {offset}m from antenna box'
    elif m == 'near field':
        mask = fnNearField(offset,spatavgL)
        name = f'Valid near field points within {offset}m of antenna box'
    elif m == 'valid':
        mask = fnValid(offset,spatavgL)
        name = f'valid points for {offset}m antenna offset and {spatavgL}m spatial average window'
    elif m == 'spatavg':
        mask = fnSpatavg(spatavgL)
        name = f'valid points for {spatavgL}m spatial average window'
    elif m == 'spatavg_outant':
        mask = fnSpatavgOutant(offset, spatavgL)
        name = f'valid points for {spatavgL}m spatial average window outside {offset}m offsetted antenna box'
    elif m == 'cb':
        mask = fnCb(Slim_adjusted, data, offset, errtol)
        name = f'{offset}m antenna offset points within {errtol * 100:g}% of {standard} {setting} limit ({Slim} W/m²) for {data} data and {power}W radiated power'
    elif m == 'icb':
        mask = fnIcb(Slim_adjusted, data, offset)
        name = f'{offset}m antenna offset points inside {standard} {setting} compliance boundary ({Slim} W/m²) for {data} data and {power}W radiated power'
    else:
        mask = fnMeval(m, offset)
        name = f'points outside antenna box (offset = {offset}) where {m}'

    return Filter(mask, name, m, offset, spatavgL, errtol, power)

def contourplot(df,col,fMHz,levels=None,dB=False,R=False):
    '''Generate contour plot of df[col] in xz plane
       Depict the omni antenna in the plot at x=0, z=0
       INPUTS:
       df = data dataframe containing x,y,z,Smax,Ssa,SAR data
       col = data column name in df, e.g. "R1.6m-5        
       fMHz = frequency of the exposure in MHz (used for scaling antenna)
       levels = contour levels
       dB = switch for plotting R: True -> plot dB(R), False -> plot R
       R = switch for indicating compliance ration data
    '''
    
    # make mgrids for x, z, S, SAR
    X = make_mgrid(df,'x')
    Z = make_mgrid(df,'z')
    C = make_mgrid(df,col)
    if dB == True: 
        C = 10. * np.log10(C)    
    
    # Create plot title
    if R == True:
        title = f"compliance ratio for {col[2:]}"
        if dB == True:
            title = "dB " + title
    else:
        title = col if dB == False else f"dB({col})"
    title = title + f" at {fMHz} MHz"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8,8))

    # Contour plots for R
    levels = np.linspace(0,10,11)
    CS1 = ax.contourf(X, Z, C, levels=levels)

    # Label axes and plot and display grid
    fig.suptitle(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(left=0)
    ax.grid(ls='--')

    # Draw omni antenna
    wl = 300 / fMHz           # wavelength
    dipole_len = wl / 2       # dipole length
    dipole_sep = 0.75 * wl    # separation between dipole centres
    zdcs = [dipole_sep * i for i in [-1.5,-0.5,0.5,1.5]]  # z for dipole centres
    
    for zdc in zdcs:
        zlow =  zdc - dipole_len / 2
        zhigh = zdc + dipole_len / 2
        ax.plot([0,0],[zlow,zhigh],'b-',lw=4)
        ax.plot(0,zdc,'ro')
        

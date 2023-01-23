# -*- coding: utf-8 -*-
"""
Created on Thurs 6/8/2021
@author: Vitas Anderson
+ Added sf function to create filtered collections of assessment points
+ Refactored for python 3
+ Comment out _plot_default since it contains chaco elements which are very hard to install now
+ Remove specific reference to RF-Map
+ Changed base data of RFc object to just xyz coordinates and general model S data. 
Values for freq, power, grid & antennabox are entered as separate inputs when creating an RFc object.
+ Added importS function to import just S data from FAC
+ Added functionality to calculate spatial averages
+ Got rid of storing the mgrids for the S data in mdata in order to reduce memory load. Use the make_mgrid function instead
+ Amended ExclusionZone to accept from 1 to 4 S data sets
+ Added ExclusionZoneP function to display exclusion zones in Plotly
+ Made sure that only immutable defaults (tuples) are used in ExclusionZone and ExclusionZoneP
+ Added functionality to include a man figure in ExclusionZone function
+ Placed ExclusionZone image title at the bottom of the figure to avoid clipping as the image height is varied
+ Changed color of antena box to dark gray
+ Moved mlabox function to the package level, rather than within the RFc class
+ Added some extra mlab colors (pink, brown, olive) to the RFc class
+ Deleted the plotly exclusion zone function as it is way too slow compared to mayavi
+ Added 'spatialpeak' function to calculate spatial peak values from rolling maximum over vertical lines
+ Added functionality to 'ExclusionZone' for y=ycut cutplane of field points
+ Added capability to show SAR exclusion zones
+ Added ycut and zcut features for ExclusionZone
+ Added check that contour level lies within range of S for ExclusionZone function
+ Replace mlabbox with panelAntenna function which draws the IEC 62232 panel antenna
+ Add the COLORS parameter to define all mayavi colors
+ Replaced 'RPS3' with 'RPS S-1 WB' as the default standard
+ Add a filter option for valid spatial averaging points
+ Add assertion tests for RFc parameters
+ Tidy up sf filter code and add spat_avg_outant option
+ Placed mayavi calls inside try/except blocks
+ Added function for calculating WBA SAR using IEC 62232 formula
"""
__version_info__ = (0, 9)
__version__ = '.'.join(map(str, __version_info__))
__author__ = 'Vitas Anderson'

# Import modules
import pandas as pd
import numpy as np
import math
import numbers
from collections import namedtuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from mayavi import mlab
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate

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

def get_trial_data(trialsheet):
    '''Read the metadata spreadsheet for the RFmap/IXUS/RV models'''
    trials = pd.read_excel(trialsheet, skiprows=[0, 14, 15, 16], index_col=0)
    trials.drop('unit', axis=1, inplace=True)
    return trials


def extract2Ddata(xm, ym, zm, sm, d, L):
    '''extracts 2D data plane at d=L from 3D xm,ym,zm,sm mgrid arrays
    for creating contour plots
    xm, ym, zm, are the mgrid forms of the xyz coordinate columns
    sm is the mgrid form of the S data
    d is the axis dimension, x, y or z
    L is the distance along the axis'''

    # Extract the 2D data plane for d = L
    if d == 'x':
        i = find_idx(xm[:, 0, 0], L)  # find index where x = n
        X = ym[i, :, :]
        Y = zm[i, :, :]
        S = sm[i, :, :]
        axislabel = ['y', 'z']
    elif d == 'y':
        i = find_idx(ym[0, :, 0], L)  # find index where y = n
        X = xm[:, i, :]
        Y = zm[:, i, :]
        S = sm[:, i, :]
        axislabel = ['x', 'z']
    else:
        i = find_idx(zm[0, 0, :], L)  # find index where z = n
        X = xm[:, :, i]
        Y = ym[:, :, i]
        S = sm[:, :, i]
        axislabel = ['x', 'y']

    return X, Y, S, axislabel

def mlabbox(x, y, z):
    '''Function which returns a 3D box using mlab.mesh
    x,y,z are 2-el lists for the physical extents of the box'''
    
    yellow = (1,1,0)
    darkgrey = (0.1,0.1,0.1)
    darkred = (0.55,0,0)
    darkblue = (0,0,0.55)
    
    # create 2D mgrids for x,y,z and f (which is 0,0)
    [xa, ya] = np.mgrid[x[0]:x[1]:2j, y[0]:y[1]:2j]
    [xb, zb] = np.mgrid[x[0]:x[1]:2j, z[0]:z[1]:2j]
    [yc, zc] = np.mgrid[y[0]:y[1]:2j, z[0]:z[1]:2j]
    [f, f] = np.mgrid[0:0:2j, 0:0:2j]
    print(f'{f=}, {xa=}, {ya=}, {xb=}, {zb=}, {yc=}, {zc=}')

    # Draw the six surfaces of the box
    surfaces = [[xa, ya, f + z[0], xa, ya, f + z[1]],
                [xb, f + y[0], zb, xb, f + y[1], zb],
                [f + x[0], yc, zc, f + x[1], yc, zc]]

    for s in surfaces:
        mlab.mesh(s[0], s[1], s[2], opacity=1, color=darkblue)
        mlab.mesh(s[3], s[4], s[5], opacity=1, color=darkblue)
    return

def panelAntenna(antcolor, x0=-0.04, y0=0, z0=0):
    '''Function which returns a 3D box depiction of the IEC 62232
       RBS panel antenna encompassing the reflector and antenna elements
       Dipole centres are shown as red dots
       The antenna phase centre is shown as a green dot
       antcolor = color of the antenna
       x0,y0,z0 = location of the centre of the reflector'''
    
    # Antenna box lengths
    xlength = 0.04
    ylength = 0.3
    zlength = 2.25
    
    # Antenna dipole centres
    dz = 0.25
    xd = [x0 + 0.04] * 9
    yd = [y0] * 9
    zd = [z0 + n*dz for n in [-4,-3,-2,-1,0,1,2,3,4]]
    
    # antenna box axis dimensions
    x = [x0, x0 + xlength]
    y = [y0-ylength/2, y0+ylength/2]
    z = [z0-zlength/2, z0+zlength/2]    
    
    # create 2D mgrids for x,y,z and f (which is 0,0)
    [xa, ya] = np.mgrid[x[0]:x[1]:2j, y[0]:y[1]:2j]
    [xb, zb] = np.mgrid[x[0]:x[1]:2j, z[0]:z[1]:2j]
    [yc, zc] = np.mgrid[y[0]:y[1]:2j, z[0]:z[1]:2j]
    [f, f] = np.mgrid[0:0:2j, 0:0:2j]

    # Draw the six surfaces of the box
    surfaces = [[xa, ya, f + z[0], xa, ya, f + z[1]],
                [xb, f + y[0], zb, xb, f + y[1], zb],
                [f + x[0], yc, zc, f + x[1], yc, zc]]

    for s in surfaces:
        mlab.mesh(s[0], s[1], s[2], opacity=1, color=COLORS[antcolor])
        mlab.mesh(s[3], s[4], s[5], opacity=1, color=COLORS[antcolor])
    
    # Draw the dipole centres
    mlab.points3d(xd,yd,zd,scale_factor=0.03,color=COLORS['red'])
    
    # Draw the phase centre
    mlab.points3d(x0,y0,z0,scale_factor=0.03,color=COLORS['green'])
    
    return

def vyagi(antcolor, x0=0, y0=0, z0=0):
    '''Draw vertical Yagi antenna
       color = color of antenna
       x0,y0,z0 = coords of the centre of the rear reflector element'''

    # Yagi dimensions in m
    radius = 0.006    # radius of dipole wires
    l_element = 0.14   # element length
    l_reflector = 0.2  # reflector length
    l_dipole = 0.18    # length of folded dipole
    l_beam = 0.49      # length of central beam
    w_dipole = 0.02    # width of folded dipole
    loc_dipole = 0.10145   # location of dipole along beam
    loc_elements = (0.16347, 0.27248, 0.37467, 0.47685)  # location of elements along beam
    
    # build yagi in Mayavi
    col = COLORS[antcolor]
    
    # beam
    mlab.plot3d([x0,x0+l_beam],[0]*2, [0]*2,
                color=col, tube_radius=radius)
    # reflector
    mlab.plot3d([x0]*2,[0]*2, [-l_reflector/2,l_reflector/2],
                color=col, tube_radius=radius)
    # elements
    for loc in loc_elements:
        mlab.plot3d([x0+loc]*2, [0]*2, [-l_element/2,l_element/2],
                    color=col, tube_radius=radius)
    # dipole
    mlab.plot3d([x0+loc_dipole]*2, [0]*2,
                [-(l_dipole-w_dipole)/2,(l_dipole-w_dipole)/2],
                color=col, tube_radius=radius)        
    mlab.plot3d([x0+loc_dipole]*2, [-w_dipole]*2,
                [-(l_dipole-w_dipole)/2,(l_dipole-w_dipole)/2],
                color=col, tube_radius=radius)
    mlab.plot3d([x0+loc_dipole]*3,
                [0,-w_dipole/2,-w_dipole],
                [(l_dipole-w_dipole)/2,l_dipole/2,(l_dipole-w_dipole)/2],
                color=col, tube_radius=radius)        
    mlab.plot3d([x0+loc_dipole]*3,
                [0,-w_dipole/2,-w_dipole],
                [-(l_dipole-w_dipole)/2,-l_dipole/2,-(l_dipole-w_dipole)/2],
                color=col, tube_radius=radius)        
    mlab.points3d(x0+loc_dipole,-w_dipole,0,color=(1,0,0),scale_factor=0.01)
    return

def hyagi(antcolor, x0=0, y0=0, z0=0):
    '''Draw horizontal Yagi antenna
       color = color of antenna
       x0,y0,z0 = coords of the centre of the rear reflector element'''

    # Yagi dimensions in m
    radius = 0.006    # radius of dipole wires
    # radius = 0.003    # radius of dipole wires
    l_element = 0.14   # element length
    l_reflector = 0.2  # reflector length
    l_dipole = 0.18    # length of folded dipole
    l_beam = 0.49      # length of central beam
    w_dipole = 0.02    # width of folded dipole
    loc_dipole = 0.10145   # location of dipole along beam
    loc_elements = (0.16347, 0.27248, 0.37467, 0.47685)  # location of elements along beam
    
    # build yagi in Mayavi
    col = COLORS[antcolor]
    
    # beam
    mlab.plot3d([x0,x0+l_beam],[0]*2, [0]*2,
                color=col, tube_radius=radius)
    # reflector
    mlab.plot3d([x0]*2,[-l_reflector/2,l_reflector/2],[0]*2,
                color=col, tube_radius=radius)
    # elements
    for loc in loc_elements:
        mlab.plot3d([x0+loc]*2, [-l_element/2,l_element/2], [0]*2,
                    color=col, tube_radius=radius)
    # dipole
    mlab.plot3d([x0+loc_dipole]*2,
                [-(l_dipole-w_dipole)/2,(l_dipole-w_dipole)/2],
                [0]*2,
                color=col, tube_radius=radius)        
    mlab.plot3d([x0+loc_dipole]*2, 
                [-(l_dipole-w_dipole)/2,(l_dipole-w_dipole)/2],
                [-w_dipole]*2,
                color=col, tube_radius=radius)
    mlab.plot3d([x0+loc_dipole]*3,
                [(l_dipole-w_dipole)/2,l_dipole/2,(l_dipole-w_dipole)/2],
                [0,-w_dipole/2,-w_dipole],
                color=col, tube_radius=radius)        
    mlab.plot3d([x0+loc_dipole]*3,
                [-(l_dipole-w_dipole)/2,-l_dipole/2,-(l_dipole-w_dipole)/2],
                [0,-w_dipole/2,-w_dipole],
                color=col, tube_radius=radius)        
    mlab.points3d(x0+loc_dipole,0,-w_dipole,color=(1,0,0),scale_factor=0.01)
    return

def show_grid_points(df, fields=['SARwb'],axv=(True,False,False),hman=None,antenna=hyagi,
                   bgcolor='lightgrey',fgcolor='black',antcolor='blue',ycut=False,title=None):
    '''Show S and SAR grid points
    usage: .showgrids(S, SAR)
        df = dataframe
    fields = list of fields to display grid points for
       avx = X,Y,Z axis visibility flags (True/False,True/False,True/False)
      hman = height of body model behind antenna in m
             If hman = None, then man model is not displayed
   bgcolor = background color
   fgcolor = foreground color
  antcolor = color of the MBS panel antenna       
      ycut = True/False toggle to set y cut plane (i.e. remove points for y < 0)
    '''
    from mayavi import mlab
    from collections.abc import Iterable
    
    # Make sure that fields is iterable
    if not isinstance(fields, Iterable): fields = [fields]

    # check that fields is a valid selection
    possible_fields = df[3:].columns.to_list()
    for field in fields:
        assert field in possible_fields, f'field ({field}) must be one of {possible_fields}'
    
    # create the Mayavi figure
    fig = mlab.figure(1, size=(1200,900), 
                      bgcolor=COLORS[bgcolor],
                      fgcolor=COLORS[fgcolor])
    mlab.clf()

    # draw each of the field points
    for field in fields:

        # Get field grid point data
        dfd = df[['x','y','z',field]].dropna()
        
        # Check for y cut plane
        if ycut:
            dfd = dfd[dfd.y >= 0]

        # draw the field grid points
        if 'SAR' in field:
            pointcolor = COLORS['coral2']
            opacity = 1
            scale_factor = 0.05
        else:
            pointcolor = COLORS['blue']
            opacity = 0.3
            scale_factor = 0.02
        mlab.points3d(dfd.x.values,dfd.y.values,dfd.z.values,
                      scale_factor=scale_factor,color=pointcolor,
                      opacity=opacity)

    # draw the human figure        
    if hman != None:
        # draw man figure behind the antenna
        assert type(hman) in (float, int), f'type of hman ({type(hman)}) must be int or float'
        assert 0.1 <= hman <= 3, f'hman ({hman}) must be within range 0.1 to 3m'
        mlabman(h=hman, xc=-2, yc=0, zc=0)

    # Get the extents
    g = df[['x','y','z']].agg(['min','max']).T
    extents = g.values.flatten().tolist()        

    # Set title
    if title == None:
        title = 'grid points for: ' + ', '.join(fields)

    # draw the axes
    ax = mlab.axes(x_axis_visibility=axv[0], y_axis_visibility=axv[1],
                   z_axis_visibility=axv[2], line_width=1,
                   extent=extents,color=(0,0,0))
    ax.label_text_property.color = (0,0,0)
    ax.title_text_property.color = (0,0,0)
    ax.axes.label_format = '%g'
    ax.axes.font_factor = 1

    # Draw the panel antenna
    antenna(antcolor)
    
    # Draw the scene
    mlab.title(title, height=0.85, size=0.15, color=COLORS[fgcolor])
    fig.scene.parallel_projection = True
    mlab.show()
    
def cylinder(xc,yc,z1,z2,rx,ry,color,n=30,cap1=True,cap2=True):
    '''Draw a capped eliptical cylinder oriented in z direction with Mayavi mlab
       xc = x coord of cylinder centre
       yc = y coord of cylinder centre
       z1 = lower z level of cylinder
       z2 = upper z level of cylinder
       rx = x-axis elliptical radius of cylinder
       ry = y-axis elliptical raduius of cylinder
       color = color of cylinder (normalised RGB coordinates)
       n = number of cylinder facets
       cap1 = whether to include lower cylinder cap (True, False)
       cap2 = whether to include upper cylinder cap (True, False)
       '''
    
    pi = np.pi
    
    # Create cylinder mesh points
    theta,zcyl = np.mgrid[-pi:pi:n*1j, z1:z2:2j]
    xcyl = rx * np.cos(theta) + xc
    ycyl = ry * np.sin(theta) + yc
    
    # Create disk mesh points
    r, theta = np.mgrid[0:1:2j, -pi:pi:n*1j]
    xdisk = xc + rx*r*np.cos(theta)
    ydisk = yc + ry*r*np.sin(theta)
    zdisk = np.ones(xdisk.shape)
    
    # Assemble the cylinder and disk meshes  
    mlab.mesh(xcyl, ycyl, zcyl, color=color)
    if cap1:
        mlab.mesh(xdisk, ydisk, zdisk * z1,color=color)
    if cap2:
        mlab.mesh(xdisk, ydisk, zdisk * z2,color=color)
    return
        
def mlabman(h, xc, yc, zc):
    '''Function which generates a simple man figure with mlab
       oriented along the z axis
       h = height of the man
       xc, yc, zc = xyz coords of the centre of the man'''

    scale = h / 12  # scaled size of each body segment
    n = 20
    blue = (0,0,1)
    pink = (1,0.75,0.8)
    red = (1,0,0)
    brown = (0.35,0.24,0.12)
    olive = (0.29,0.33,0.13)
    yellow = (1,1,0)

    # spherical head segment     
    mlab.points3d(xc, yc, zc + 5*scale, scale_factor=scale*2,
                  color=pink, mode='sphere', resolution=n)
    
    # eyes
    mlab.points3d(xc + 0.3*scale, yc - 0.9*scale, zc + 5.2*scale, scale_factor=scale*0.25,
                  color=blue, mode='sphere', resolution=n)
    mlab.points3d(xc - 0.3*scale, yc - 0.9*scale, zc + 5.2*scale, scale_factor=scale*0.25,
                  color=blue, mode='sphere', resolution=n)
     
    # centre point
    mlab.points3d(xc, yc - scale, zc, scale_factor=scale*0.25,
                  color=yellow, mode='sphere', resolution=n)
    # neck
    z1 = zc + 3.5*scale
    z2 = zc + 4.5*scale
    rx, ry = 0.5*scale, 0.5*scale
    cylinder(xc,yc,z1,z2,rx,ry,pink,n,False,False)
    
    # body
    z1 = zc - scale
    z2 = zc + 3.5*scale
    rx, ry = 1.5*scale, scale
    cylinder(xc,yc,z1,z2,rx,ry,olive,n)
    
    # legs
    z1 = zc - 6*scale
    z2 = zc - scale
    rx, ry = 0.7*scale, 0.7*scale
    cylinder(xc-0.8*scale,yc,z1,z2,rx,ry,brown,n,True,False)
    cylinder(xc+0.8*scale,yc,z1,z2,rx,ry,brown,n,True,False)

class Filter():
    '''This is just an object package for a filter mask and its name'''
    def __init__(self, mask, name, m, offset, spatavgL, errtol, power):
        self.mask = mask
        self.name = name
        self.m = m
        self.offset = offset
        self.spatavgL = spatavgL
        self.errtol = errtol
        self.power = power
        self.n = mask.sum()
        
    def __repr__(self):
        s = '*Filter object*\n'
        s += f'              name: {self.name}\n'
        s += f'No. of mask points: {self.n:,d}\n'
        s += f'                 m: {self.m}\n'
        s += f'            offset: {self.offset}\n'
        s += f'          spatavgL: {self.spatavgL}\n'
        s += f'            errtol: {self.errtol}\n'
        s += f'             power: {self.power}\n'
        return s

class RFc:
    '''Creates an object which facilitates quantification and viewing of RF calculation
    uncertainty error in S field outputs from FACs compared to FEKO'''

    def __init__(self, freq, power, grid, antbox, 
                 errtol=0.15, offset=0, spatavgL=1.6):
        '''
            freq = RF frequency of the model data in MHz
           power = nominal power of the source for the simulations
            grid = specification of the 3D point grid
          antbox = specification of the panel antenna box enclosure
          errtol = default error tolerance in S for picking points near the compliance limit
          offset = distance (m) away from the surface of the antenna box when selecting points outside it
        spatavgL = default length (m) of the spatial averaging window
        '''
        
        # Assertion tests
        assert isinstance(freq,numbers.Number), f'freq ({freq}) must be a number'
        assert isinstance(power,numbers.Number), f'power ({power}) must be a number'
        assert isinstance(errtol,numbers.Number), f'errtol ({errtol}) must be a number'
        assert isinstance(offset,numbers.Number), f'offset ({offset}) must be a number'
        assert isinstance(spatavgL,numbers.Number), f'spatavgL ({spatavgL}) must be a number'
        assert freq >= 100, f'freq ({freq}) must be >= 100 MHz'
        assert power > 0, f'power ({power}) must be > 0 W'
        assert errtol >= 0, f'errtol ({errtol}) must be >= 0'
        assert offset >= 0, f'offset ({offset}) must be >= 0 m'
        assert 0.1 <= spatavgL <= 3, f'spatavgL ({spatavgL}) must be in range 0.1 to 3 m'
        
        # Assign self properties
        self.freq = freq
        self.power = power
        self.grid = grid
        self.xb, self.yb, self.zb = antbox
        self.errtol = errtol
        self.offset = offset
        self.spatavgL = spatavgL
        self.datatitles = {}

        # Read the grid parameters
        xstart, xend, dx = grid['x']
        ystart, yend, dy = grid['y']
        zstart, zend, dz = grid['z']

        # Determine number of grid points along each x y z axis
        self.nx = int(np.rint((xend - xstart) / dx)) + 1
        self.ny = int(np.rint((yend - ystart) / dy)) + 1
        self.nz = int(np.rint((zend - zstart) / dz)) + 1

        # Create S dataframe, starting with columns for the xyz coordinates
        grid_x, grid_y, grid_z = np.mgrid[xstart:xend:self.nx * 1j,
                                          ystart:yend:self.ny * 1j,
                                          zstart:zend:self.nz * 1j]
        self.S = pd.DataFrame({'x': grid_x.flatten(),
                               'y': grid_y.flatten(),
                               'z': grid_z.flatten()}).round(4)

        # Add columns for r and phi coords to S dataframe
        self.S['r'] = np.sqrt(self.S.x**2 + self.S.y**2)
        self.S['phi'] = np.degrees(np.arctan2(self.S.y, self.S.x))

        # Create x, y, z mgrid arrays for iso-surface and point plots
        self.xm = self.make_mgrid('x')
        self.ym = self.make_mgrid('y')
        self.zm = self.make_mgrid('z')

    def make_mgrid(self, data):
        '''Creates an mgrid from a column in self.S'''
        Sm = self.S[data].values.reshape(self.nx, self.ny, self.nz)
        return Sm

    def importS(self, Sfile):
        '''Load dataframe of S column(s) from HDF5 data file
        Note that the S columns are assumed to correspond to points
        on the xyz coordinate grid and have been sorted in xyz order'''
        print(f'{Sfile=}')
        with pd.HDFStore(Sfile) as store:
            Sdata = store['mydata']
            Snames = store.get_storer('mydata').attrs.metadata

        # Add the S data to the self.S dataframe
        for col in Sdata:
            self.S[col] = Sdata[col]

        # Add the S data names to self.datatitles
        self.datatitles.update(Snames)

    def addS(self, Sdata, name, title):
        '''Add a data column to the self.S dataframe'''
        self.S[name] = Sdata
        self.datatitles[name] = title

    def antboxsize(self, offset):
        '''Calculates antenna box size which may be enlarged 
        by the specified distance offset in each of the x,y,z directions'''
        xb, yb, zb = self.xb, self.yb, self.zb
        x0, x1 = xb[0] - offset, xb[1] + offset
        y0, y1 = yb[0] - offset, yb[1] + offset
        z0, z1 = zb[0] - offset, zb[1] + offset
        
        return x0, x1, y0, y1, z0, z1

    def sf(self, m, setting='pub', data='Smax', offset=None,
           errtol=None, spatavgL=None, power=None, standard='RPS S-1 WB'):
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
        setting = the upper or lower tier of the limit (e.g. 'pub', 'uncontrolled', 'occupational')
           data = S dataset in S (e.g. Sixus, SE, SH, Smax)
         offset = dimensional offset in metres for enlarging the antenna box or 
                  the additional z direction offset from the antenna box for the nobody volume
         errtol = error tolerance for selecting points near compliance boundary
       spatavgL = length (m) of the spatial averaging window 
          power = adjusted power of antenna (the limit value is scaled by self.power/power)'''

        # Set internal variables
        S = self.S
        if offset == None: offset = self.offset
        if errtol == None: errtol = self.errtol
        if spatavgL == None: spatavgL = self.spatavgL
        if power == None: power = self.power
        Slim = Slimit(self.freq, setting, standard)
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

    def printfilters(self):
        '''Print the number of xyz points in the standard filters'''
        print('antenna box offset = {} m'.format(self.offset))
        print('error tolerance for point selection = {}%'.format(self.errtol * 100.))
        print('\n** Number of points for standard filter masks **')
        for f in ['outant', 'ant', 'all']:
            Sf = self.sf(f, offset=self.offset, errtol=self.errtol)  # create filter mask
            nmask = sum(Sf.mask)  # count number of points in filter mask
            print(f'{f:>7s}:{nmask:9,g} ({Sf.name})')
        for f, setting in [('cb', 'public'), ('cb', 'occ'), ('icb', 'public'), ('icb', 'occ')]:
            Sf = self.sf(f, setting=setting, offset=self.offset, errtol=self.errtol)  # create filter mask
            nmask = sum(Sf.mask)  # count number of points in filter mask
            print(f'{f:>7s}:{nmask:9,g} ({Sf.name})')

    def ci(self, data, f=None, CI=95):
        '''Generates a dataframe of coverage interval (CI) values
        data is the data column in the self.S dataframe that is being analysed
        f is the filter for selecting points within the 3D grid
        CI is the span of a probabilistically symmetric confidence interval (default value is 95)'''

        if isinstance(data,str):
            data = [data]
        datatitles = self.datatitles
        for dat in data:
            assert dat in list(datatitles.keys()), f'data {dat} must be in {list(datatitles.keys())}'

        if f == None:
            f = self.sf('outant')
        Sh = self.S.loc[f.mask, data]

        CIlow = (100 - CI) / 2.
        CIhigh = 100. - (100 - CI) / 2.

        cidf = pd.DataFrame()
        cidf['n'] = Sh.apply(len)
        cidf[str(CIlow) + ' ptile'] = Sh.apply(np.nanpercentile, args=(CIlow,))
        cidf['median'] = Sh.median()
        cidf['mean'] = Sh.mean()
        cidf[str(CIhigh) + ' ptile'] = Sh.apply(np.nanpercentile, args=(CIhigh,))

        print('{}% CI for {}'.format(CI, f.name))
        
        return cidf

    def confunc(self, arr, Si, convector):
        """Returns numpy array for convolution transformation of S values
           Is used to do vertical spatial averaging
                  Si = The column of S to be spatially averaged, e.g. Smax, Sfac, etc
           convector = the convolution vector, e.g. np.ones(21)/21 for 2 meter averaging 
                       with 0.1m intervals over 21 points
        """
        return np.convolve(arr[Si].values, convector, 'same')

    def spatavg(self, name, Si, convector, title=None):
        """Creates a column in the S dataframe for spatially averaged field
           values in the z (i.e. vertical) orientation
        name = the name of the new column, e.g. 'Sfac_avg'
          Si = The column of S to be spatially averaged, e.g. Smax, Sfac, etc
   convector = the convolution vector for the spatial averaging, e.g. np.ones(21)/21 for 2 meter averaging
        """
        print(f'creating {name} spatial average of {Si}')
        Savg = self.S[['x', 'y', Si]].groupby(['x', 'y']).apply(self.confunc, Si, convector, )
        Savg = np.concatenate(Savg.to_list())
        if title == None:
            title = f'spatially averaged {Si} S levels'
        self.addS(Savg, name, title)

    def rolling_max(self,arr,Si,z,n):
        '''This function calculates the rolling maximum of the Si column.
           It extrapolates values at the start and end of the Si column 
           to accommodate the size of the rolling window.
           It assumes that the z values are ascending
       Si = the S column for which the rolling maximum is obtained
        z = corresponding z array for the S array
        n = size of the rolling window (must be an odd integer)'''

        # Create extrapolation function for S values
        S = arr[Si].values
        f = interpolate.interp1d(z, S, kind='quadratic',fill_value="extrapolate")

        # Create additional z values for extrapolating S at start and end of array
        dz = z[1] - z[0]
        m = int((n-1)/2)
        z_start = z[0:m] - dz*m
        z_end   = z[-m:] + dz*m

        # Expand S array with start and end extrapolated values
        S_start = f(z_start)
        S_end   = f(z_end)
        
        S = np.insert(S, 0, S_start)
        S = np.append(S, S_end)

        # Calculate rolling maximum for S from extrapolated S array
        Smax = np.max(sliding_window_view(S, window_shape = n), axis = 1)

        return Smax

    def peakspatial(self, name, Si, n, title=None):
        """Creates a column in the S dataframe for peak spatial field
           values in the z (i.e. vertical) orientation
        name = the name of the new column, e.g. 'Sfac_avg'
          Si = The column of S to be spatially averaged, e.g. Smax, Sfac, etc
           n = the size in field points of the rolling window for the peak spatial field
               e.g. 21 for a 2m window when the poinst are spaced in 0.1m increments
        """
        print(f'creating {name} peak spatial values of {Si}')
        gridz = self.grid['z']
        z = np.linspace(gridz[0],gridz[1],self.nz)
        Sps = self.S[['x', 'y', Si]].groupby(['x', 'y']).apply(self.rolling_max, Si, z, n,)
        Sps = np.concatenate(Sps.to_list())
        if title == None:
            title = f'peak spatial {Si} S levels'
        self.addS(Sps, name, title)

    def mcp(self, data, d='y', L=0, nc=5):
        '''Creates 2D matplotlib contour plot for axis plane
        usage: .mcp(data, d, L, nc)
        d = 'x','y' or 'z' for x=L, y=L or z=L planes in mdata[data]
        nc = the number of contour levels'''

        # check input data
        d = d.lower()
        assert d in ['x', 'y', 'z'], "d ({}) must be 'x','y' or 'z'".format(d)
        assert data in list(self.datatitles.keys()), "data {} must be in {}".format(data, list(self.datatitle.keys()))

        # Extract the 2D data and contour levels for the contour plot
        Sm = self.make_mgrid(data)
        X, Y, S, axislabel = extract2Ddata(self.xm, self.ym, self.zm, Sm, d, L)

        # Create the contour levels
        S5, S95 = np.nanpercentile(S, 1), np.nanpercentile(S, 99)
        span = S95 - S5
        dS = span / float(nc - 1)
        dS = round_to_1(span / float(nc - 1))
        start = (S5 * 1.E12) // (dS * 1.E12) * dS  # make start a multiple of dS
        V = [start + k * dS for k in range(-1, int(nc) + 1)]

        # Set the default tick directions for the contour plot
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.direction'] = 'out'

        # Create the contour plot
        figratio = (X.max() - X.min()) / (Y.max() - Y.min())
        fig, ax = plt.subplots(figsize=(figratio * 6, 6))
        CS = ax.contour(X, Y, S, V)
        plt.clabel(CS, inline=1, fontsize=10, fmt='%g')  # , fmt='%1.1f')
        title = '{} in the {} = {}m plane'.format(self.datatitles[data], d, L)
        fig.suptitle(title, fontsize=14)
        plt.xlabel(axislabel[0] + ' (m)')
        plt.ylabel(axislabel[1] + ' (m)')
        # plt.show()
        return ax

    def hist(self, data, f=None, CI=95, bins=50, xrange=None):
        '''creates histogram of values in filtered S dataset
        usage: RFc.hist(filter, data, CI)
        data = dataset in S
           f = filter mask applied to data (default=.sf('outant'))
        bins = No. of histogram bins (default=50)
          CI = range of CI, e.g. 95 or 90 (default=95)
      xrange = x axis range (default is automatic)'''

        datatitles = self.datatitles
        assert data in list(datatitles.keys()), 'data {} must be in {}'.format(data, list(datatitles.keys()))
        datatitle = datatitles[data]

        if f == None:
            f = self.sf('outant')
        if 'dB' in datatitle:
            unit = 'dB'
        elif 'SAR' in datatitle:
            unit = 'W/kg'
        else:
            unit = r'W/m$^2$'
        Sh = self.S.loc[f.mask, data]

        fig, ax = plt.subplots(figsize=(12, 6))
        Sh.hist(bins=bins, ax=ax, range=xrange)

        fig.subplots_adjust(top=0.85)
        xlabel = datatitle
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('counts', fontsize=16)
        CIlow = np.nanpercentile(Sh, (100 - CI) / 2.)
        CIhigh = np.nanpercentile(Sh, 100 - (100 - CI) / 2.)
        titlestr = '{}% CI is {:.2f} to {:.2f} {} for {}\n[ {:,d} points ]'
        title = titlestr.format(CI, CIlow, CIhigh, unit, f.name, len(Sh))
        ax.set_title(title, fontsize=16, color='blue')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    def __str__(self):
        s = 'Object parameters:\n'
        s += f'  {self.freq} MHz, {self.power} W\n'
        s += f'  errtol = {self.errtol}\n'
        s += f'  offset = {self.offset} m\n'
        s += f'  default length of spatial averagering window = {self.spatavgL} m\n'
        s += f'  {len(self.S):,d} grid points\n'
        s += f'  nx = {self.nx}, ny = {self.ny}, nz = {self.nz}'
        return s

    def __repr__(self):
        return str(self)

    def setcontour(self, setting, plotpower, standard='RPS S-1 WB'):
        '''Set the contour level for plot, scaled for plot power'''
        limit = Slimit(self.freq, setting, standard)
        C = self.power / plotpower * limit
        infoString = 'power = {}, plotpower = {}, setting = {}, limit = {}, contour level = {0.3f}'
        print(infoString.format(self.power, plotpower, setting, limit, C))
        
        return C, limit

    def ExclusionZone(self, data, power, bg='lightgrey', fg='black',
                      color=['green','blue','orange','magenta'],
                      alpha=[0.5]*8, setting=['public']*8, standard=['RPS-S1']*8, 
                      title='', axv=[True,False,False], ycut=None, zcut=None,
                      hman=None, xyzman=[-1.5,0,0], 
                      antenna=panelAntenna, antdz=None, 
                      gridpoints=False, gridpoint_filter=None, gridpoint_size=0.02, gridpoint_opacity=0.3,
                      figsize=(1200,900)):
        '''
        Draw Mayavi figures of exclusion zones for datasets in S
             data = list of S data sets, e.g.['Smax','SE','SH','SARwb']
            power = list of scaled power levels for data, e.g. [200,200,100]
               bg = background color, selected from COLORS dictionary, e.g. 'white'
               fg = foreground color, selected from COLORS dictionary, e.g. 'black'
            color = list of colours for exclusion zones for data, e.g. ['red','blue','crimson']
            alpha = opacity of the exclusion zone [0 to 1]
          setting = list of settings for data, e.g. ['pub','occ']
         standard = EME exposure standard for limit value ['RPS S-1 WB','FCC']
            title = displayed title for the plot
              axv = list or tuple indication visibility of x,y,z axes, e.g. (False,False,True) for just z axis visible
             ycut = y value to set cutplane where all points have y > ycut [ystart to yend]
             zcut = z value to set cutplane where all points have z > zcut [zstart to zend]
             hman = height of man figure
           xyzman = [x,y,z] coords of centre of man
       gridpoints = toggle to display gridpoint. Defau;ti is False to NOT display gridpoints
 gridpoint_filter = Mask for gridpoints. Default is valid points for 1.6m spatial averaging
   gridpoint_size = size of gridpoints. Default is 0.02
gridpoint_opacity = opacity of the gripoints [0 to 1]
          antenna = function for display of antenna
            antdz = displacement of second antenna in vertical direction. Enter None to not display
       gridpoints = toggle for displaying gridpoints
          figsize = tuple of width and height f figure in pixels, e.g. (1200,900)
        '''

        # make sure that data, power, color, setting, standard are lists or tuples
        # and have at least as many elements as in data
        def makeIterable(arg, argname):
            if isinstance(arg, (str,float,int)):  # make sure that single string inputs are contained in a list
                arg = [arg]
            if argname != 'data':
                errmsg = f'{argname} must have at least as many elements as in data {data}'
                assert len(arg) >= len(data), errmsg 
            return arg

        data = makeIterable(data,'data')
        power = makeIterable(power,'power')
        color = makeIterable(color,'color')
        alpha = makeIterable(alpha,'alpha')
        setting = makeIterable(setting,'setting')
        standard = makeIterable(standard,'standard')

        # assertion tests
        Scols = [col for col in self.S.columns if col not in ('x','y','z','r','phi')]  # i.e. all columns except the x y z r phi coordinates
        for d in data:
            assert d in Scols, f"data ({d}) must be one of {Scols}"
        for c in color:
            assert c in COLORS.keys(), f"color ({c}) must be in {list(COLORS.keys())}"
        for a in alpha:
            assert (a >= 0) and (a <= 1), f"alpha ({a}) must be in [0,1] range."
        for p in power:
            assert p >= 0, f"power ({p}) must be >= 0"
        if hman != None:
            assert 0.5 <= hman <= 3, f"hman ({hman}) must be between 0.5 and 3"
        assert bg in COLORS.keys(), f"bg {bg} must be one of {COLORS.keys()}"
        assert fg in COLORS.keys(), f"fg {fg} must be one of {COLORS.keys()}"
        
            
        # Background and foreground colors
        bgc = COLORS[bg]
        fgc = COLORS[fg]

        # Calculate the S and SAR limits
        limits, limtexts = [], []
        for dat, s, std in zip(data, setting, standard):
            if 'SAR' in dat:
                limit = SARlimit(s)
                limtext = f'{limit} W/kg'
                limits.append(limit)
                limtexts.append(limtext)
            else:
                limit = Slimit(self.freq, s, std)
                limtext = f'{limit:0.1f} W/m²'
                limits.append(limit)
                limtexts.append(limtext)
    
        # Calculate the contour level of each exclusion zone
        contour = [self.power / p * lim for p, lim in zip(power, limits)]
        
        # Calculate X, Y, Z mgrid
        X = self.xm
        Y = self.ym
        Z = self.zm
        
        if ycut is not None:
            ystart, yend, dy = self.grid['y']
            assert ystart < ycut < yend, f"ycut ({ycut} must lie within the grid's y value range [{ystart}, {yend}])"
            nc = int((ycut - ystart) / dy)
            X = X[:,nc:,:]
            Y = Y[:,nc:,:]
            Z = Z[:,nc:,:]
            
        if zcut is not None:
            zstart, zend, dz = self.grid['z']
            assert zstart < zcut < zend, f"zcut ({zcut} must lie within the grid's z value range [{zstart}, {zend}])"
            nc = int((zcut - zstart) / dz)
            X = X[:,:,nc:]
            Y = Y[:,:,nc:]
            Z = Z[:,:,nc:]

        # Calculate the x,y,z extents of the plot for all exclusion zones
        ScbAll = pd.DataFrame(columns=['x','y','z'])
        for dat, con in zip(data,contour):
            print(f'{dat=}, {con=}')
            mask = (self.S[dat] >= con) & (self.S[dat] < 1.1*con)
            Scb = self.S.loc[mask,['x','y','z']]
            # ScbAll = ScbAll.append(Scb)
            ScbAll = pd.concat([ScbAll, Scb])
        extent = ScbAll.apply([min,max]).T.values.flatten().round(1).tolist()
        print(extent)

        # create the Mayavi figure
        try:
            fig = mlab.figure(1, size=figsize, bgcolor=bgc, fgcolor=fgc)
            mlab.clf()
        except:
            raise Exception("Could not create mlab figure")

        # draw the iso-surfaces
        titles = [] if title == '' else [title + '\n']
        for dat,limtext,p,col,a,s,std,con in zip(data,limtexts,power,color,alpha,setting,standard,contour):
            print(f'power={self.power}, plotpower={p}, setting={s}, limit={limtext}, contour level={con:0.3f}')
            t = f'{col.upper()}: {std} {s} exclusion zone ({limtext}) for {p} W {self.datatitles[dat]}'
            titles.append(t)
            S = self.make_mgrid(dat)
            if ycut is not None:
                S = S[:,nc:,:]
            if zcut is not None:
                S = S[:,:,nc:]
            S = np.nan_to_num(S, nan=0.0)  # replace nans in S with zeros
            if S.min() < con < S.max():    # check that con lies within range of values in S field
                try:
                    src = mlab.pipeline.scalar_field(X, Y, Z, S, name=dat)
                    mlab.pipeline.iso_surface(src, contours=[con, ], opacity=a,
                                              color=COLORS[col])
                except:
                    raise Exception(f"Could not draw iso-surface for {dat}")

        # display gridpoints
        if gridpoints == True:
            pointcolor = COLORS['blue']
            if gridpoint_filter == None:
                # create mask for valid 1.6m spatial averaging points
                gridpoint_filter = self.sf('spatavg', offset=0.001, spatavgL=1.6)
            Sp = self.S[gridpoint_filter.mask]
            if ycut != None:
                Sp = Sp[Sp.y >= ycut]
            extent = Sp[['x','y','z']].apply([min,max]).T.values.flatten().round(1).tolist()
            mlab.points3d(Sp.x.values,Sp.y.values,Sp.z.values,
                          scale_factor=gridpoint_size,color=pointcolor,
                          opacity=gridpoint_opacity)

        # draw the axes
        ax = mlab.axes(x_axis_visibility=axv[0], y_axis_visibility=axv[1],
                  z_axis_visibility=axv[2], line_width=1, extent=extent)            
        ax.label_text_property.color = fgc
        ax.title_text_property.color = fgc
        ax.axes.label_format = '%g'
        ax.axes.font_factor = 1

        # print plot title
        height, size = 0.08, 0.08
        mlab.title('\n'.join(titles), height=height, size=size)

        # draw the antenna
        antenna('blue')
        
        # draw second elevated antenna
        if antdz != None:
            antenna('blue',z0=antdz)
        
        # draw the man figure
        if hman != None:
            xc, yc, zc = xyzman
            mlabman(hman, xc, yc, zc)
            
        # Set up the mayavi view
        mlab.view(90, -90)  # xz plane (azimuth=90°, elevation=90°)
        fig.scene.parallel_projection = True
        mlab.show()
        return

    def msp(self, data='errR', f=None, mp=2, V=[-4, 4], nlabels=9, ncolors=8,
            scale=0.075, antenna=None, ctitle=None, standard='RPS S-1 WB'):
        '''Show data in 3D Mayavi scatter plot
        usage: .msp(data, f, mp, V, nlabels, ncolors, ctitle)\n
           data = S dataset column in the self.S dataframe
        sfilter = filter applied to select points in S
              V = list of Vmin and Vmax for colorbar values
             mp = fraction of points to show (e.g mp = 3 -> show 1/3rd of points)
        nlabels = no. of colorbar labels
        ncolors = no. of colorbar colors
         ctitle = title of colorbar
        Note: Seems to crap out if too many points'''

        # Get point data
        if f == None:
            f = self.sf('outant')
            
        S = self.S[f.mask]
        if 'SARwbi' in data:
            SARmask = S.SARwbi.isna()
            S = S[~SARmask]
        x, y, z = S.x.values, S.y.values, S.z.values
        s = S[data].values

        # create the Mayavi figure
        mlab.figure(1, size=(1200, 900), bgcolor=(0.5, 0.5, 0.5))
        mlab.clf()
        
        # Display the 3D scatter plot points, colorbar and title
        mlab.points3d(x, y, z, s, mask_points=mp, colormap='RdYlGn',
                           vmin=V[0], vmax=V[1], scale_mode='none', scale_factor=scale)
        mlab.colorbar(orientation='horizontal', nb_labels=nlabels, nb_colors=ncolors,
                      title=ctitle, label_fmt='%g')
        title = '{} \nfor {}'.format(self.datatitles[data], f.name)
        mlab.title(title, height=0.85, size=0.1)
        
        # Display the antenna
        if antenna != None: antenna('darkblue')
            
        mlab.show()

    def showgrids(self, S=True, SAR=True, hman=None, antenna=hyagi, axv=(True,False,False) ):
        '''Show S and SAR grid points in 3D Mayavi scatter plot
        usage: .showgrids(S, SAR)
          S = flag to toggle S grid visibility (True/False)
        SAR = flag to toggle SAR grid visibility (True/False)
        hman = height of body model behind antenna in m
               If hman = None, then man model is not displayed
     antenna = function for dsiplaying the antenna
        avx = X,Y,Z axis visibility flags (True/False,True/False,True/False)
        '''

        # create the Mayavi figure
        from mayavi import mlab
        fig = mlab.figure(1, size=(1200,900), 
                          bgcolor=COLORS['lightgrey'],
                          fgcolor=sCOLORS['black'])
        mlab.clf()

        def make_mgrid(df):
            df = df.sort_values(['x','y','z'])
            nx = len(df.x.unique())
            ny = len(df.y.unique())
            nz = len(df.z.unique())
            X = df.x.values.reshape(nx, ny, nz)
            Y = df.y.values.reshape(nx, ny, nz)
            Z = df.z.values.reshape(nx, ny, nz)
            return X, Y, Z, nx, ny, nz
        
        title = ''

        if SAR:
            # Get SAR grid point data
            SARdf = self.S[['x','y','z','SARwb']].dropna()
            X1, Y1, Z1, nx1, ny1, nz1 = make_mgrid(SARdf)
            title += f'SAR grid: {nx1} x {ny1} x {nz1} = {nx1*ny1*nz1} points\n'
            
            # Get the extents
            g = SARdf[['x','y','z']].agg(['min','max']).T
            extents = g.values.flatten().tolist()        
                       
            # draw the SAR grid popints
            mlab.points3d(X1,Y1,Z1,scale_factor=0.1,color=(0,0,1),opacity=1)   # SAR grid
            
        if S:         
            # Get S grid point data
            X2, Y2, Z2, nx2, ny2, nz2 = make_mgrid(self.S)
            title += f'S grid: {nx2} x {ny2} x {nz2} = {nx2*ny2*nz2:,} points\n'
            
            # Get the extents
            g = self.S[['x','y','z']].agg(['min','max']).T
            extents = g.values.flatten().tolist()        
                       
            # draw the S grid points
            mlab.points3d(X2,Y2,Z2,scale_factor=0.04,color=(1,1,0),opacity=0.1)  # S grid
            
        if hman != None:
            # draw man figure behind the antenna
            assert type(hman) in (float, int), f'type of hman ({type(hman)}) must be int or float'
            assert 0.1 <= hman <= 3, f'hman ({hman}) must be within range 0.1 to 3m'
            mlabman(h=hman, xc=-1, yc=0, zc=0)

        # Add the antenna box
        antenna('darkblue')
        
        # draw the axes
        ax = mlab.axes(x_axis_visibility=axv[0], y_axis_visibility=axv[1],
                       z_axis_visibility=axv[2], line_width=1,
                       extent=extents,color=(0,0,0))
        ax.label_text_property.color = (0,0,0)
        ax.title_text_property.color = (0,0,0)
        ax.axes.label_format = '%g'
        ax.axes.font_factor = 1
        
        # Draw the scene
        mlab.title(title, height=0.85, size=0.15, color=(0,0,0))
        fig.scene.parallel_projection = True
        mlab.show()

    def AnimatedExclusionZone(self, data, power, bg='white', fg='black',
                              color=('green','blue','orange','crimson'),
                              alpha=(0.5)*8, setting=('public')*8, standard=('RPS-S1')*8, 
                              title='', axv=(True,False,False), ycut=None, zcut=None, 
                              daz=1, elevation =90, distance=None, showtitle=True,
                              hman=None, xyzman=[-1.5,0,0], antenna=hyagi, figsize=(1200,900)):
        '''
        Draw Mayavi figures of exclusion zones for datasets in S
             data = list of S data sets, e.g.['Smax','SE','SH','SARwb']
            power = list of scaled power levels for data, e.g. [200,200,100]
            color = list of colours for exclusion zones for data, e.g. ['red','blue','crimson']
            alpha = opacity of the exclusion zone [0 to 1]
          setting = list of settings for data, e.g. ['pub','occ']
         standard = EME exposure standard for limit value ['RPS S-1 WB','FCC']
            title = displayed title for the plot
              axv = list or tuple indication visibility of x,y,z axes, e.g. (False,False,True) for just z axis visible
             ycut = y value to set cutplane where all points have y > ycut [ystart to yend]
             zcut = z value to set cutplane where all points have z > zcut [zstart to zend]
              daz = azimuthal increment for animation
        elevation = elevation angle of view
         distance = distance of camera from scene
             hman = height of man figure
           xyzman = [x,y,z] coords of centre of man
          antenna = function for dispay of antenna
          figsize = tuple of width and height f figure in pixels, e.g. (1200,900)
        '''

        # make sure that data, power, color, setting, standard are lists or tuples
        # and have at least as many elements as in data
        def makeIterable(arg, argname):
            if isinstance(arg, (str,float,int)):  # make sure that single string inputs are contained in a list
                arg = [arg]
            if argname != 'data':
                errmsg = f'{argname} must have at least as many elements as in data {data}'
                assert len(arg) >= len(data), errmsg 
            return arg

        data = makeIterable(data,'data')
        power = makeIterable(power,'power')
        color = makeIterable(color,'color')
        alpha = makeIterable(alpha,'alpha')
        setting = makeIterable(setting,'setting')
        standard = makeIterable(standard,'standard')

        # assertion tests
        Scols = self.S.columns[5:].tolist()  # i.e. all columns except the x y z r phi coordinates
        for d in data:
            assert d in Scols, f"data ({d}) must be one of {Scols}"
        for c in color:
            assert c in COLORS.keys(), f"color ({c}) must be in {list(COLORS.keys())}"
        for a in alpha:
            assert (a >= 0) and (a <= 1), f"alpha ({a}) must be in [0,1] range."
        for p in power:
            assert p >= 0, f"power ({p}) must be >= 0"
        if hman != None:
            assert 0.5 <= hman <= 3, f"hman ({hman}) must be between 0.5 and 3"
        assert bg in COLORS.keys(), f"bg {bg} must be one of {COLORS.keys()}"
        assert fg in COLORS.keys(), f"fg {fg} must be one of {COLORS.keys()}"
        
            
        # Background and foreground colors
        bgc = COLORS[bg]
        fgc = COLORS[fg]

        # Calculate the S and SAR limits
        limits, limtexts = [], []
        for dat, s, std in zip(data, setting, standard):
            if 'SAR' in dat:
                limit = SARlimit(s)
                limtext = f'{limit} W/kg'
                limits.append(limit)
                limtexts.append(limtext)
            else:
                limit = Slimit(self.freq, s, std)
                limtext = f'{limit:0.1f} W/m²'
                limits.append(limit)
                limtexts.append(limtext)
    
        # Calculate the contour level of each exclusion zone
        contour = [self.power / p * lim for p, lim in zip(power, limits)]
        
        # Calculate X, Y, Z mgrid
        X = self.xm
        Y = self.ym
        Z = self.zm
        
        if ycut is not None:
            ystart, yend, dy = self.grid['y']
            assert ystart < ycut < yend, f"ycut ({ycut} must lie within the grid's y value range [{ystart}, {yend}])"
            nc = int((ycut - ystart) / dy)
            X = X[:,nc:,:]
            Y = Y[:,nc:,:]
            Z = Z[:,nc:,:]
            
        if zcut is not None:
            zstart, zend, dz = self.grid['z']
            assert zstart < zcut < zend, f"zcut ({zcut} must lie within the grid's z value range [{zstart}, {zend}])"
            nc = int((zcut - zstart) / dz)
            X = X[:,:,nc:]
            Y = Y[:,:,nc:]
            Z = Z[:,:,nc:]

        # Calculate the x,y,z extents of the plot for all exclusion zones
        ScbAll = pd.DataFrame(columns=['x','y','z'])
        for dat, con in zip(data,contour):
            print(f'{dat=}, {con=}')
            mask = (self.S[dat] >= con) & (self.S[dat] < 1.1*con)
            Scb = self.S.loc[mask,['x','y','z']]
            # ScbAll = ScbAll.append(Scb)
            ScbAll = pd.concat([ScbAll, Scb])
        extent = ScbAll.apply([min,max]).T.values.flatten().round(1).tolist()

        # mayavi animation function
        @mlab.animate(delay = 50)
        def updateAnimation():
            az = 0.0
            while True:
                mlab.view(azimuth=az, elevation=elevation,distance=distance)
                az += 1
                yield
        
        # create the Mayavi figure
        fig = mlab.figure(1, size=figsize, bgcolor=bgc, fgcolor=fgc)
        mlab.clf()

        # draw the iso-surfaces
        titles = [] if title == '' else [title + '\n']
        for dat,limtext,p,col,a,s,std,con in zip(data,limtexts,power,color,alpha,setting,standard,contour):
            print(f'power={self.power}, plotpower={p}, setting={s}, limit={limtext}, contour level={con:0.3f}')
            t = f'{col.upper()}: {std} {s} exclusion zone ({limtext}) for {p} W {self.datatitles[dat]}'
            titles.append(t)
            S = self.make_mgrid(dat)
            if ycut is not None:
                S = S[:,nc:,:]
            if zcut is not None:
                S = S[:,:,nc:]
            S = np.nan_to_num(S, nan=0.0)  # replace nans in S with zeros
            if S.min() < con < S.max():    # check that con lies within range of values in S field
                src = mlab.pipeline.scalar_field(X, Y, Z, S, name=dat)
                mlab.pipeline.iso_surface(src, contours=[con, ], opacity=a,
                                          color=COLORS[col])

        # draw the axes
        ax = mlab.axes(x_axis_visibility=axv[0], y_axis_visibility=axv[1],
                  z_axis_visibility=axv[2], line_width=1, extent=extent)
        ax.label_text_property.color = fgc
        ax.title_text_property.color = fgc
        ax.axes.label_format = '%g'
        ax.axes.font_factor = 1

        # print plot title
        if showtitle == True:
            height, size = 0.08, 0.08
            mlab.title('\n'.join(titles), height=height, size=size)

        # draw the antenna
        antenna('blue')
        
        # draw the man figure
        if hman != None:
            xc, yc, zc = xyzman
            mlabman(hman, xc, yc, zc)

        # Set prallel projection
        fig.scene.parallel_projection = True
        
        # Start animation
        updateAnimation()        
        mlab.show()
        
        return


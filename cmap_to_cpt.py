# -*- coding: utf-8 -*-
"""
Function to convert a pyplot cmap to a gmt-compatible cpt file.

Created on Thu Jan 18 2018
Last edit on Fri Sep 3 2021

@author: ImportanceOfBeingErnest (https://stackoverflow.com/users/4124317/importanceofbeingernest)
Adapted from: Stackoverflow (https://stackoverflow.com/questions/48322741/is-there-a-way-to-save-a-custom-matplotlib-colorbar-to-use-elsewhere)
Adapted by: Jesse Arens
"""

import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')

def export_cmap_to_cpt(cmap, vmin=0,vmax=1, N=255, filename="test.cpt",**kwargs):
    """
    Export a pyplot cmap to a cpt file.

    Parameters
    ----------
    cmap : Colormap object
        Pyplot color map to convert.
    vmin : int, optional
        Start point of cpt. The default is 0.
    vmax : int, optional
        End point of cpt. The default is 1.
    N : int, optional
        Colorspace size. The default is 255, corresponding with RGB colorspace.
    filename : str, optional
        Name of the cpt file to save. The default is "test.cpt".
    **kwargs : dictionary
        Extra keyword arguments, to define extra parameters for the cpt file.

    Returns
    -------
    None.

    """
    # create string for upper, lower colors
    b = np.array(kwargs.get("B", cmap(0.)))
    f = np.array(kwargs.get("F", cmap(1.)))
    na = np.array(kwargs.get("N", (0,0,0))).astype(float)
    ext = (np.c_[b[:3],f[:3],na[:3]].T*255).astype(int)
    extstr = "B {:3d} {:3d} {:3d}\nF {:3d} {:3d} {:3d}\nN {:3d} {:3d} {:3d}"
    ex = extstr.format(*list(ext.flatten()))
    #create colormap
    cols = (cmap(np.linspace(0.,1.,N))[:,:3]*255).astype(int)
    vals = np.linspace(vmin,vmax,N)
    arr = np.c_[vals[:-1],cols[:-1],vals[1:],cols[1:]]
    # save to file
    fmt = "%e %3d %3d %3d %e %3d %3d %3d"
    np.savetxt(filename, arr, fmt=fmt,
               header="# COLOR_MODEL = RGB",
               footer = ex, comments="")

# test case: create cpt file from RdYlBu colormap
cmap = plt.get_cmap("Greys",255)
# you may create your colormap differently, as in the question

export_cmap_to_cpt(cmap, vmin=0,vmax=1,N=20,filename='Greys.cpt')
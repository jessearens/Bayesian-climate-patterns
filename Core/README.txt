This is the core folder of the Bayesian climate patterns code. It contains the following files:
MLP.py : implementation of Artificial neural network as a class. 
TrainMLP.py : script used to train and test an Artificial neural network.
BNN.py : implementation of Bayesian neural network as a class.
TrainBNN.py : script used to train and test a Bayesian neural network.
supportfuncs.py : several supporting functions used for preparing data or plotting results.
regridding.sh : Climate data operators (CDO) shell script used to merge, regrid and compute yearly mean maps from CMIP5 output models.
BrBG.cpt / BuRd.cpt / Greys.cpt : Color Palette Tables, used in coloring pygmt maps.
cmap_to_cpt.py : code to convert a matplotlib pyplot color map into a CPT file.

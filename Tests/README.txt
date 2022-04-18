Contains supplementary tests used to compute optimal hyperparameters and tests used to check robustness of models.
The following files are present:
BNN_testing.py : BNN class implementation with extra settings for different hyperparameters.
Prior_BNN_testing.py : Script used to test performance of different priors for BNNs.
KL_BNN_testing,py : Script used to test performance of different KL weighting values for BNNs.
ComputeStatistics.py : Script used to compute correlation and variance versus mean weight ratios.
Iterative_statistics.py : Script used to compute several iterations of the standard ANNs and BNNs, to check for robust results and statements.
TrainMLP_extratests.py : Script used to run several different settings of the ANN, i.e. networks for data with global mean removed.
TrainBNN_extratests.py : Script used to run several different settings of the BNN, such as different subsetting between train and test data, or different architectures.
TrainLinReg.py : Script used to create a linear regression model.

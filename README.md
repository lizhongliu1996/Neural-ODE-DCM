# Neural-ODE-DCM
this is the repository that contain the code of my dissertation, the goal is to accurate estimated brain connectivity network matrix from fMRI timeseries data with ground unknown.

This project focuses on applying Graph Neural Networks (GNNs) for time series analysis, with a specific focus on functional Magnetic Resonance Imaging (fMRI) data. The main objectives of this project are:

GNN for time series, with applications to fMRI There are 3 aims towards analysis fMRI data.

Aim 1: Neural ODE for Estimating Brain Connectivity Networks during rs-fMRI Sessions
Configuration: In this step, we generate synthetic data to test the effectiveness of the algorithm.
Validation: The algorithm is tested against Patel's tau on simulation data with ground truth (GT) known.
Application: The algorithm is applied to NCANDA rs-fMRI data to estimate brain connectivity networks.

Aim 2: Neural ODE for Estimating Brain Connectivity Networks during t-fMRI Sessions with External Stimulus Incorporation
Simulation: Synthetic data is generated to test the algorithm's performance with external stimulus incorporation.
Application: The algorithm is applied to ABCD t-fMRI data and compared against Patel's tau.
Evaluation: Binary classification performance is evaluated on brain connectivity matrices derived from the previous step.

Aim 3: GNN Fusion for Leveraging Brain Connectivity for Symptom Classification
Supervised Learning: We propose Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) with the fusion of subject-level phenotypic data.
Self-supervised Learning: The approach involves node-level pretraining (AttrMasking), graph-level pretraining (supervised), and finetuning.

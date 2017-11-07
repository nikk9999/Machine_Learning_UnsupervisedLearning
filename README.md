# Machine_Learning_UnsupervisedLearning
Following algorithms are implemented and analysed
1. K-Means
2. Expectation Maximization
3. Principal Component Analysis (PCA)
4. Independent Component Analysis (ICA)
5. Random Projections (RP)
6. Feature Selection 

Clustering is performed by isolating the labels and it is observed to organize data in an unsupervised manner and is a useful pre-processing technique. Dimensionality reduction assists in removing non-useful features to improve clustering and learner accuracy. Usage of unsupervised learning for improving the performance of supervised learning algorithms like ANN is demonstrated.

KMeans, EM, Data reduction are performed using Python code with Scikit-learn library.

Datasets can be found here.
https://drive.google.com/open?id=0BxnjqxEVUcSTbE95WmtxQ1RyTjA

This takes care of output of reduced data and data with clusters as added features which is given as input to multilayer perceptron in weka.

The following parameters are used for each ANN in weka.

	Learning	Momentum	Hidden
Original	0.4	0.7	8,2,14
PCA	0.4	0.7	8,2,14
ICA	0.4	0.2	8
RP	0.4	0.7	8,2,14
FS	0.4	0.7	8,2,14
Kmeans			
PCA	0.4	0.7	8,2,14
ICA	0.4	0.7	8,2,14
RP	0.4	0.7	8,2,14
FS	0.1	0.2	14,3,2
EM			
PCA	0.4	0.7	8,2
ICA	0.5	0.6	8, 14
RP	0.4	0.7	8,2,14
FS	0.4	0.2	8


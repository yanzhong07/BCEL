# BCEL
Biclustering with the exclusive Lasso penalty

Implementation of the paper:

Zhong, Yan, and Jianhua Z. Huang. "Biclustering via structured regularized matrix decomposition." Statistics and Computing 32.3 (2022): 1-15.

(https://link.springer.com/article/10.1007/s11222-022-10095-1)

Abstract：
Biclustering is a machine learning problem that deals with simultaneously clustering of rows and columns of a data matrix. Complex structures of the data matrix such as overlapping biclusters have challenged existing methods. This paper develops a new biclustering method called BCEL based on structured regularized matrix decomposition. The biclustering problem is formulated as a penalized least-squares problem that approximates the data matrix X by a multiplicative matrix decomposition UV^T with sparse columns in both U and V. The squared l_{1,2}-norm penalty, also called the exclusive Lasso penalty, is applied to both U and V to assist identification of rows and columns included in the biclusters. The penalized least-squares problem is solved by a novel computational algorithm that combines alternating minimization and the proximal gradient method. A subsampling based procedure called stability selection is developed to select the tuning parameters and determine the bicluster membership. BCEL is shown to be competitive to existing methods in a simulation study and an application to a real-world single-cell RNA sequencing dataset.



Main functions:

exclusivelasso_project(): proximal operator of exclusive lasso.

exclusivelasso(): regression with exclusive lasso penalty.

bcel(): biclustering with exclusive lasso penalty, written with Rcpp.

bcel_stable(): stable method to select models.

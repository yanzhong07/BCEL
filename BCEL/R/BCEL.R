library(RSpectra)
library(Matrix)
library(Rcpp)
library(RcppArmadillo)
library(gtools)
library(MASS)


#sourceCpp("/Volumes/work/dropbox/Dropbox/Dr. Huang/project/V2_biclustering/20171026_buildpackage/20180802_newcode/BCEL/src/Exclusivelasso.cpp")

## Function 1: exclusive lasso project
#' @param x projected vector. 
#' @param lambda penalty parameter.
exclusivelasso_project = function(z, lambda){
  return(exclusive_lasso_project(z, lambda))
}

## Function 2
## Exclusive lasso with sparse x
#' @param y the response vector, a length n vector.
#' @param x saved as sparse matrix, n*p design matrix. The dense matrix will be transformed to "sparseMatrix" type first.
#' @param group groups of features, a length p vector.
#' @param lambda penalty parameter.
#' @param iter_max max times of iterations.
#' @param tol tolerance of convergence.
exclusivelasso = function(y, x, group, lambda, iter_max, tol){
  X = as(x, "sparseMatrix")
  model = exclusive_lasso(y, X, group, lambda, iter_max = 100, tol = 10^(-6))
  return(list(beta = model))
}

## Function 3:
## main function, exclusive lasso in biclustering.
#' @param X n*p matrix to decompose, should saved as unsparsed matrix.
#' @param r number of subgroups to detect.
#' @param lambda_u,lambda_v penalty parameter for rows and columns.
#' @param iter_max max times of iterations.
#' @param tol tolerance of convergence.
bcel = function(X, r, lambda_u, lambda_v, iter_max, tol){
  stopifnot(r >0)
  stopifnot(lambda_u >0 ,lambda_v >0)
  stopifnot(iter_max > 0, tol > 0)
  
  X1 = as(X, "matrix")
  model = BiElasso(X1, r, lambda_u, lambda_v, iter_max, tol)
  names(model) = c("U","V","times of iterations")
  return(model)
}

# BiElasso_r(X, r = 3, 0.1, 0.1, iter_max = 100, tol = 0.000001)

######
## Function 4
## Stable method to select parameter in BCEL
#' @param X n*p matrix to decompose, should saved as unsparsed matrix.
#' @param r number of subgroups to detect.
#' @param q_u,q_v tolerance error of negative true in selected rows/cols.
#' @param B1 max tolerance of percenrage of negative ture.
#' @param select_upper_u,select_upper_v max percentage of variables selected.
#' @param pi_thr_l,pi_thr_u interval of pi, the cutpoint of effective variables.
#' @param B1,B2 number of bootstraps at the first and second times.
#' @param size_per percentage of samples selected each bootstrap.
#' @param speed the change speed of lambda.
#' @param iter_max max times of iterations.
#' @param tol tolerance of convergence.
#' @param sparse if the input data are actually sparse data, 0 no, 1 yes.

bcel_stable = function(X, r, q_u = 0.2,q_v = 0.2, q_upper = 0.5,
                             select_upper_u = 0.45, select_upper_v = 0.45,
                             B1 = 10, B2 = 200,
                             pi_thr_l = 0.65, pi_thr_u = 0.7,
                             size_per = 0.5, speed = 0.3,
                             iter_max = 100, tol = 0.000001, sparse = 0){
  stopifnot(q_u <1, q_u >0, q_v <1, q_v >0, q_u <q_upper, q_v <q_upper)
  stopifnot(q_u <1, q_u >0, q_v <1, q_v >0, q_upper <1, q_upper >0, q_u <q_upper, q_v <q_upper)
  stopifnot(r>0, B1 >0, B2 > 0)
  stopifnot(size_per <1, size_per >0, size_per <1, size_per >0)
  stopifnot( sparse %in% c(0,1), iter_max > 0, tol > 0)
  
  
  stopifnot(r*(dim(X)[1]+dim(X)[2]) < dim(X)[1]*dim(X)[2]*size_per)
  X1 =  as(X, "matrix")
  model = BiElasso_stable(X, r, q_u, q_v, q_upper, select_upper_u, select_upper_v,
              B1, B2, pi_thr_l, pi_thr_u, size_per, speed, iter_max, tol, sparse)
  names(model) = c("U","V","converge", "lambda_u", "lambda_v")
  return(model)
  
}

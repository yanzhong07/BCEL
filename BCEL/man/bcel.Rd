\name{bcel}
\alias{bcel}
\alias{bcel_stable}
\alias{bcel_chooser}
\title{Bi-clustering with Exclusive Lasso}
\description{
Main function of BCEL and its stable version. Method to choose r by CV.
}

\usage{
bcel(X, r, lambda_u, lambda_v, iter_max, tol)
bcel_stable(X, r, q_u = 0.2, q_v = 0.2, q_upper = 0.5, select_upper_u = 0.45, select_upper_v = 0.45,
               B1 = 10, B2 = 200, pi_thr_l = 0.65, pi_thr_u = 0.7, size_per = 0.5, speed = 0.3, iter_max = 100, tol = 0.000001, sparse = 0)
                                      

\arguments{
  \item{X}{a data matrix.}
  \item{r}{number of maximun lattent groups.}
  \item{lambda_u, lambda_v}{penlaty parameter on U and V.}
  \item{iter_max}{max times of iteration.}
  \item{tol}{tolerance for iteration.}
  \item{q_u, q_v}{percentage of false positive in row and column clustering result.}
  \item{q_upper}{protential max percentage of false positive}
  \item{select_upper_u, select_upper_v}{The upper percentage of parameters selected.}
  \item{pi_thr_l, pi_thr_u}{range of accepting rate}
  \item{size_per}{size of subsample}
  \item{speed}{speed of searching lambda.}
  \item{B1,B2}{the number of subsamples. B1 is used in lambda selection. B2 is used in final stable variable selection}
  \item{sparse}{input matrix X is sparseMatrix or Matrix.}

}
\value{
  \code{U} estimated U matrix.
  \code{V} estimated V matrix.
  \code{lambda_u} selected lambda_u.
  \code{lambda_v} selected lambda_v.
  \code{r} selected r.

}
\details{

}
\references{
}
\examples{

}
\author{Yan Zhong}


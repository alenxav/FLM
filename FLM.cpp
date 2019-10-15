#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
SEXP FLM(NumericVector y, NumericMatrix X){
  // Convergence settings
  int maxit = 300; double tol = 10e-8;
  // Initial settings and starting values
  int p=X.ncol(), n=X.nrow(), numit=0;
  double b0,b1,eM,Ve,cnv=1,mu=mean(y),Lmb2=0;
  NumericVector e=y-mu,Vb(p),b(p),fit(n);
  // Cross-products and shape parameter
  NumericVector xx(p),sx(p),bc(p);
  for(int k=0; k<p; k++){
    xx[k]=sum(X(_,k)*X(_,k));
    if(xx[k]==0) xx[k]=0.1;
    Lmb2=Lmb2+var(X(_,k));}
  NumericVector iTau2=p+Lmb2;
  // Looping across parameters until convergence
  while(numit<maxit){   
    // Updating markers effects
    bc=b+0; for(int j=0; j<p; j++){ b0=b[j];
      b1=(sum(X(_,j)*e)+xx[j]*b0)/(iTau2(j)+xx(j));
      b[j]=b1; e=e-X(_,j)*(b1-b0);}
    // Updating intercept
    eM=mean(e); mu=mu+eM; e=e-eM;
    // Updating variance components
    Ve=sum(e*y)/(n-1); 
    Vb=b*b+Ve/(xx+iTau2);
    iTau2=sqrt(Lmb2*Ve/Vb);
    // Check parameters convergence
    ++numit; cnv=sum(abs(bc-b));
    if(cnv<tol){break;}}
  // Fit model
  for(int k=0; k<n; k++){fit[k]=mu+sum(X(k,_)*b);}
  // Return output
  return List::create(Named("mu")=mu,
                      Named("b")=b, Named("fit")=fit,
                      Named("T2")=1/iTau2, Named("Ve")=Ve);}
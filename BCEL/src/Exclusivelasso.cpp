// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#define ARMA_DONT_PRINT_ERRORS
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>




using namespace Rcpp;
using namespace arma;
arma::vec sfunction(const arma::vec&x,  const double& lambda){
  vec zhong1 = abs(x) - lambda;
  return(sign(x) % zhong1 % (zhong1>=0));
}


// [[Rcpp::export]]
arma::vec exclusive_lasso_project(const arma::vec& x, const double& lambda) {
    int n = x.n_elem;
    vec x_abs = abs(x);
    uvec a = sort_index(x_abs, "descent");
    vec x_order = x_abs(a);
    vec seq = linspace<vec>(1,n,n);
    vec x_cummu = cumsum(x_order) / (1/lambda + seq);
    vec z_zhong = x_order - x_cummu;
    int n_nozero = max(seq.elem( find(z_zhong >0)));
    vec z = sfunction(x_order,x_cummu(n_nozero-1));
    vec z_final = z;
    z_final(a) = z;
    z_final = z_final % sign(x);
    return(z_final);

}

// [[Rcpp::export]]
arma::vec exclusive_lasso(const arma::vec& y, const arma::sp_mat& x, const arma::vec& group, const double& lambda, const int& iter_max, const double& tol) {
    sp_mat xtx = x.t() * x;
    vec xty = x.t() * y;
    int begin_index = 0;
    vec eigval;
    while(true){
        begin_index += 1;
        try{
            eigval = eigs_sym(xtx, fmin(begin_index+1,6));
            break;
        }catch(const std::runtime_error &e){
            if(begin_index > 10){
                //arma_stop_runtime_error("eigs_sym(): decomposition failed");
                arma_stop_runtime_error("Bad sample, resample.");
            }
            continue;
        }
    }

    int L = ceil(eigval[0]);
    double lambda2 = lambda/L;
    int p = xtx.n_cols;
    vec beta0 = vec(p, fill::randn);
    
    vec group_index = unique(group);
    int group_m = group_index.n_elem;
    
    vec betat = beta0, betat_1 = beta0;
    double diff = 1;
    int count = 0;
    vec z(p);
    

    while(count < iter_max && diff > tol){
        count++;
        z = betat_1 - (xtx * betat_1 - xty)/L;
        
        // This part can be solve in parallel.
        vec zg;
        vec betag_star1;
        for(int i = 0; i < group_m ;i++){
            uvec idx = find(group == group_index[i]); // Substitute == with >, <, >=, <=, !=
            zg = z.elem(idx);
            betag_star1 = exclusive_lasso_project(zg, lambda2);
            betat.elem(idx) = betag_star1;
        }
        diff = sqrt(sum(pow((betat - betat_1), 2)));
        betat_1 = betat;
    }

    return(betat);
}


// [[Rcpp::export]]
List BiElasso(const arma::mat& X, int r, const double& lambda_u, const double& lambda_v, const int& iter_max, const double& tol){
    
    int p = X.n_cols;
    int n = X.n_rows;
    mat Ut = arma::mat(n, r, fill::randn);
    mat Vt = arma::mat(p, r, fill::randn);
    mat Ut_1 = arma::mat(n, r, fill::randn);
    mat Vt_1 = arma::mat(p, r, fill::randn);
    
    int count = 0;
    double diff = 10;
    
    uvec seq1 = linspace<uvec>(1,n,n);
    uvec seq2 = linspace<uvec>(0,(p-1),p);
    uvec seq3 = linspace<uvec>(1,r,r);
    uvec i_index_u = repmat(seq1,r*p,1) + reshape(repmat(seq2 * n, 1,n*r).t(), r*p*n,1) - 1;
    uvec j_index_u = reshape(repmat(seq3,p,n).t(), r*p*n,1) + reshape(repmat(seq2 * r, 1,n*r).t(), r*p*n,1) - 1;
    umat ij_u = join_cols(i_index_u.t(), j_index_u.t());
    
    uvec i_index_v = repmat(seq2+1,r*n,1) + reshape(repmat((seq1-1) * p, 1,p*r).t(), r*p*n,1) - 1;
    uvec j_index_v = reshape(repmat(seq3,n,p).t(), r*p*n,1) + reshape(repmat((seq1-1) * r, 1, p*r).t(), r*p*n,1) - 1;
    umat ij_v = join_cols(i_index_v.t(), j_index_v.t());
    
    
    // init point
    vec seq3v = linspace<vec>(1,(r-1),r);
    
    vec p_element_u = zeros<vec>(r*p*n);
    sp_mat U_design = sp_mat(ij_u, p_element_u);
    vec V_model = zeros<vec>(r*p);
    
    vec p_element_v = zeros<vec>(r*p*n);
    sp_mat V_design = sp_mat(ij_v, p_element_v);
    vec U_model = zeros<vec>(r*n);
    double normlize_constant1, normlize_constant2;
    
    printf("\n iteration");
    while(count < iter_max && diff > tol){
        printf("%d,",count);
        count = count + 1;
        // fix U, iterate V
        p_element_u = repmat(reshape(Ut_1, n*r, 1), p, 1);
        U_design = sp_mat(ij_u, p_element_u);
        V_model = exclusive_lasso(reshape(X, n*p, 1), U_design, repmat(seq3v,p,1), lambda_v, iter_max, tol);
        Vt = reshape(V_model, r, p).t();

        // fix U, iterate V
        p_element_v = repmat(reshape(Vt, p*r, 1), n, 1);
        V_design = sp_mat(ij_v, p_element_v);
        U_model = exclusive_lasso(reshape(X.t(), n*p, 1), V_design, repmat(seq3v,n,1), lambda_u, iter_max, tol);
        Ut = reshape(U_model, r, n).t();
        
        normlize_constant1 = accu(square(Ut));
        normlize_constant2 = accu(square(Vt));
        
        diff = accu(square(Vt-Vt_1) / normlize_constant2)+ accu(square(Ut - Ut_1) / normlize_constant1);
        Ut_1 = Ut;
        Vt_1 = Vt;
    }
    
    List list1;
    list1.push_back(Ut);
    list1.push_back(Vt);
    list1.push_back(count);

    return(list1);
}

uvec select_subset(uvec x1, uvec x2, int np){
    int m2 = x2.n_elem;
    uvec new_index1 = linspace<uvec>(1,m2,m2);
    uvec new_index2 = zeros<uvec>(np);
    new_index2(x2) = new_index1;
    uvec new_index3 = new_index2(x1);
    return(new_index3);
}

uvec Omega_tran(uvec x, int n, int p){
    uvec col_num = floor(x/n);
    uvec row_num = x - col_num * n;
    return(row_num * p + col_num);
}






// for B1 sample
// [[Rcpp::export]]
List BiElasso_bootstrap(const arma::mat& X, arma::vec& Omega, int r, const double& lambda_u, const double& lambda_v, const int& iter_max, const double& tol){
    
    int p = X.n_cols;
    int n = X.n_rows;
    mat Ut = arma::mat(n, r, fill::randn);
    mat Vt = arma::mat(p, r, fill::randn);
    mat Ut_1 = arma::mat(n, r, fill::randn);
    mat Vt_1 = arma::mat(p, r, fill::randn);
    uvec Omega_index = conv_to<uvec>::from(Omega);
    uvec Omega_index_t = Omega_tran(Omega_index, n, p);

    int count = 0;
    double diff = 10;
    uvec seq1 = linspace<uvec>(1,n,n);
    uvec seq2 = linspace<uvec>(0,(p-1),p);
    uvec seq3 = linspace<uvec>(1,r,r);
    uvec i_index_u = repmat(seq1,r*p,1) + reshape(repmat(seq2 * n, 1,n*r).t(), r*p*n,1) - 1;
    uvec j_index_u = reshape(repmat(seq3,p,n).t(), r*p*n,1) + reshape(repmat(seq2 * r, 1,n*r).t(), r*p*n,1) - 1;
    
    uvec new_index_u = select_subset(i_index_u, Omega_index, n*p);

    uvec select_index_u = find(conv_to<vec>::from(new_index_u) > 0);
    umat ij_u = join_cols(new_index_u(select_index_u).t() - 1, j_index_u(select_index_u).t());
    uvec i_index_v = repmat(seq2+1,r*n,1) + reshape(repmat((seq1-1) * p, 1,p*r).t(), r*p*n,1) - 1;
    uvec j_index_v = reshape(repmat(seq3,n,p).t(), r*p*n,1) + reshape(repmat((seq1-1) * r, 1, p*r).t(), r*p*n,1) - 1;
    uvec new_index_v = select_subset(i_index_v, Omega_index_t, n*p);
    uvec select_index_v = find(conv_to<vec>::from(new_index_v) > 0);
    umat ij_v = join_cols(new_index_v(select_index_v).t() - 1, j_index_v(select_index_v).t());


    
    // init point
    vec seq3v = linspace<vec>(1,(r-1),r);
    sp_mat U_design, V_design;
    vec p_element_u = zeros<vec>(r*p*n);
    vec V_model = zeros<vec>(r*p);
    
    vec p_element_v = zeros<vec>(r*p*n);
    vec U_model = zeros<vec>(r*n);
    double normlize_constant1, normlize_constant2;
    vec y_zhong = zeros<vec>(p*n);
    
    while(count < iter_max && diff > tol){
        count = count + 1;
        // fix U, iterate V
        p_element_u = repmat(reshape(Ut_1, n*r, 1), p, 1);
        U_design = sp_mat(ij_u, p_element_u(select_index_u));
        y_zhong = reshape(X, n*p, 1);
        V_model = exclusive_lasso(y_zhong(Omega_index), U_design, repmat(seq3v,p,1), lambda_v, iter_max, tol);
        Vt = reshape(V_model, r, p).t();
        
        // fix U, iterate V
        p_element_v = repmat(reshape(Vt, p*r, 1), n, 1);
        V_design = sp_mat(ij_v, p_element_v(select_index_v));
        y_zhong = reshape(X.t(), n*p, 1);
        U_model = exclusive_lasso(y_zhong(Omega_index_t), V_design, repmat(seq3v,n,1), lambda_u, iter_max, tol);
        Ut = reshape(U_model, r, n).t();
        normlize_constant1 = accu(square(Ut));
        normlize_constant2 = accu(square(Vt));
        
        diff = accu(square(Vt-Vt_1) / normlize_constant2)+ accu(square(Ut - Ut_1) / normlize_constant1);
        Ut_1 = Ut;
        Vt_1 = Vt;

    }
    

    
    List list1;
    list1.push_back(Ut);
    list1.push_back(Vt);
    list1.push_back(count);
    
    return(list1);
}


// for B2 sample, only one step
List BiElasso_bootstrap_one(const arma::mat& X, arma::vec& Omega, int r, arma::mat Ut, arma::mat Vt, const double& lambda_u, const double& lambda_v, const int& iter_max, const double& tol){
    
    int p = X.n_cols;
    int n = X.n_rows;
    mat Ut_1 = Ut;
    mat Vt_1 = Vt;
    uvec Omega_index = conv_to<uvec>::from(Omega);
    uvec Omega_index_t = Omega_tran(Omega_index, n, p);

    uvec seq1 = linspace<uvec>(1,n,n);
    uvec seq2 = linspace<uvec>(0,(p-1),p);
    uvec seq3 = linspace<uvec>(1,r,r);
    uvec i_index_u = repmat(seq1,r*p,1) + reshape(repmat(seq2 * n, 1,n*r).t(), r*p*n,1) - 1;
    uvec j_index_u = reshape(repmat(seq3,p,n).t(), r*p*n,1) + reshape(repmat(seq2 * r, 1,n*r).t(), r*p*n,1) - 1;
    uvec new_index_u = select_subset(i_index_u, Omega_index, n*p);
    uvec select_index_u = find(conv_to<vec>::from(new_index_u) > 0);
    
    umat ij_u = join_cols(new_index_u(select_index_u).t() - 1, j_index_u(select_index_u).t());
    
    uvec i_index_v = repmat(seq2+1,r*n,1) + reshape(repmat((seq1-1) * p, 1,p*r).t(), r*p*n,1) - 1;
    uvec j_index_v = reshape(repmat(seq3,n,p).t(), r*p*n,1) + reshape(repmat((seq1-1) * r, 1, p*r).t(), r*p*n,1) - 1;
    uvec new_index_v = select_subset(i_index_v, Omega_index_t, n*p);
    uvec select_index_v = find(conv_to<vec>::from(new_index_v) > 0);
    umat ij_v = join_cols(new_index_v(select_index_v).t() - 1, j_index_v(select_index_v).t());
    
    
    // init point
    vec seq3v = linspace<vec>(1,(r-1),r);
    sp_mat U_design, V_design;
    vec p_element_u = zeros<vec>(r*p*n);
    vec V_model = zeros<vec>(r*p);
    
    vec p_element_v = zeros<vec>(r*p*n);
    vec U_model = zeros<vec>(r*n);
    vec y_zhong = zeros<vec>(p*n);
    

    // fix U, iterate V
    p_element_u = repmat(reshape(Ut_1, n*r, 1), p, 1);
    U_design = sp_mat(ij_u, p_element_u(select_index_u));
    y_zhong = reshape(X, n*p, 1);
    V_model = exclusive_lasso(y_zhong(Omega_index), U_design, repmat(seq3v,p,1), lambda_v, iter_max, tol);
    Vt = reshape(V_model, r, p).t();
    
    // fix U, iterate V
    p_element_v = repmat(reshape(Vt, p*r, 1), n, 1);
    V_design = sp_mat(ij_v, p_element_v(select_index_v));
    y_zhong = reshape(X.t(), n*p, 1);
    U_model = exclusive_lasso(y_zhong(Omega_index_t), V_design, repmat(seq3v,n,1), lambda_u, iter_max, tol);
    Ut = reshape(U_model, r, n).t();
    
    List list1;
    list1.push_back(Ut);
    list1.push_back(Vt);
    
    return(list1);
}

/*
// at least, each row/column should be sampled once.
arma::vec good_sample(const arma::vec& x, int n, int p, int sample_size, int r){
    vec sample_index = zeros<vec>(sample_size);
    while(true){
        sample_index = Rcpp::RcppArmadillo::sample(x, sample_size, false);
        vec list1 = unique(floor((sample_index-1) / n));
        vec list2 = unique(sample_index-1 - floor((sample_index-1) / n) * n);
        if( list1.n_elem < p || list2.n_elem  < n){
            continue;
        }
        break;
    }
    return(sample_index);
}
*/

// [[Rcpp::export]]
arma::vec good_sample(const arma::vec& x, int n, int p, int sample_size, int r){

    vec sample_index = zeros<vec>(sample_size);
    while(true){
        sample_index = Rcpp::RcppArmadillo::sample(x, sample_size, false);
        
        uvec list1 = conv_to<uvec>::from(floor((sample_index-1) / n));
        uvec list2 = conv_to<uvec>::from(sample_index-1 - floor((sample_index-1) / n) * n);
        vec count1 = zeros<vec>(p);
        vec count2 = zeros<vec>(n);
        
        count1(list1) +=1;
        count2(list2) +=1;
        if(accu(count1 <= r) >0 || accu(count2 <= r) >0){
            continue;
        }
        break;
    }
    return(sample_index);
}



// [[Rcpp::export]]
List BiElasso_stable(const arma::mat& X, int r,
                     const double& q_u = 0.2, const double& q_v = 0.2, const double& q_upper = 0.5,
                     const double& select_upper_u = 0.45, const double& select_upper_v = 0.45,
                     const double& B1 = 10, const double& B2 = 200,
                     const double& pi_thr_l = 0.65, const double& pi_thr_u = 0.7,
                     const double& size_per = 0.5, const double& speed = 0.3,
                     const int& iter_max = 100, const double& tol = 0.000001, int sparse = 0){
    
    int p = X.n_cols;
    int n = X.n_rows;
    sp_mat XtX;
    if(sparse ==1){
        sp_mat Xs = sp_mat(X);
        XtX = Xs.t() * Xs;
    }else{
        XtX = sp_mat(X.t() * X);
    }
    vec eigval = eigs_sym(XtX, 1);
    double lambda_max = log(sqrt(eigval[0]));
    double lambda_min = -15.0;
    int lambda_num = 100;
    vec lambda_u_list = exp(linspace<vec>(lambda_min,lambda_max,lambda_num));
    vec lambda_v_list = exp(linspace<vec>(lambda_min,lambda_max,lambda_num));
    
    int v_index_begin = 0;
    int u_index_begin = 0;
    int v_index_end = lambda_num - 1;
    int u_index_end = lambda_num - 1;
    int u_index_select, v_index_select;
    double lambda_u, lambda_v;
    
    vec sample_list = linspace<vec>(0, n*p-1, n*p);
    int sample_size = floor(n*p*size_per);
    vec sOmega_B1 = zeros<vec>(sample_size);
    vec sOmega_B2 = zeros<vec>(sample_size);

    vec ut_nozero_times = zeros<vec>(B1);
    vec vt_nozero_times = zeros<vec>(B1);
    
    vec ut_beta_times = zeros<vec>(n*r);
    vec vt_beta_times = zeros<vec>(p*r);
    vec Ut_vector = zeros<vec>(n*r);
    vec Vt_vector = zeros<vec>(p*r);

    
    float qt_u, qt_v, pi_need_u, pi_need_v;
    int B1_2 = B1 -1;
    bool stop = true;
    for(int no_use = 1; no_use <= 2 * lambda_num; no_use++){
        // case one, end and begin are too close to change
        if(v_index_end - v_index_begin <= 2 || u_index_end - u_index_begin <= 2){
            if(v_index_end - v_index_begin <= 2){
                printf("column not stable, increase q_v or decrease select_upper_v");
            }
            if(u_index_end - u_index_begin <= 2){
                printf("row not stable, increase q_u or decrease select_upper_u");
            }
            stop = false;
            break;
        }
        
        // update new select lambda
        u_index_select = floor(u_index_begin + (u_index_end - u_index_begin) * speed);
        v_index_select = floor(v_index_begin + (v_index_end - v_index_begin) * speed);
        lambda_u = lambda_u_list[u_index_select];
        lambda_v = lambda_v_list[v_index_select];
        
        printf("\n choice %d： lambda_u = %f, lambda_v = %f\n", no_use, lambda_u, lambda_v);

        
        // subsample
        int count_wrong = 0;
        for(int i = 0; i < B1; i++){
            printf("=");
            
            while(true){
                try{
                    printf(">");
                    sOmega_B1 = good_sample(sample_list, n, p, sample_size, r);
                    List model_B1 = BiElasso_bootstrap(X, sOmega_B1, r, lambda_u, lambda_v, iter_max, tol);
                    mat zhong2 = model_B1[0];
                    ut_nozero_times[i] = accu(zhong2 != 0);
                    mat zhong3 = model_B1[1];
                    vt_nozero_times[i] = accu(zhong3 != 0);
                    break;
                }

                catch(const std::exception &e1){
                    cerr << e1.what() << endl;
                    count_wrong += 1;
                    printf("bad%d", count_wrong);
                    if(count_wrong > B1 * 1.5){ // sample at most B1*2.5
                        //arma_stop_runtime_error("Too much bad samples in B1.");
                        B1_2 = i-1;
                        break;
                    }
                    continue;
                    
                }
            }
            
            if(B1_2 == i-1){
                break;
            }
            
        }
        
        
        if(B1_2 == -1){ // lambda is too large
            if( lambda_u > lambda_v){
                u_index_end = u_index_select;
            }
            if( lambda_v >= lambda_u){
                v_index_end = v_index_select;
            }
            continue;
        }
        
        // calculate qt_u and qt_v
        qt_u = median(ut_nozero_times(span(0,B1_2)));
        pi_need_u = (qt_u * qt_u/(r*n*q_u*r*n)+1)/2;
        qt_v = median(vt_nozero_times(span(0,B1_2)));
        pi_need_v = (qt_v * qt_v/(r*p*q_v*r*p)+1)/2;
        
        // Three cases to update and break
        
        if((qt_u/r/n) > select_upper_u || (qt_v/r/p) > select_upper_v){
            if((qt_u/r/n) > select_upper_u){
                u_index_begin = u_index_select;
            }
            if((qt_v/r/p) > select_upper_v){
                v_index_begin = v_index_select;
            }
            //printf("A");
            continue;
        }
        
        if(pi_need_u > pi_thr_u || pi_need_v > pi_thr_u){
            if(pi_need_u > pi_thr_u){
                double q_zhong = q_u;
                while(q_zhong <= q_upper){
                    q_zhong = q_zhong + 0.01;
                    pi_need_u = (qt_u * qt_u/(r*n*q_zhong*r*n)+1)/2;
                    if(pi_need_u <= pi_thr_u) break;
                    }
                if(q_zhong > q_upper){
                    u_index_begin = u_index_select;
                    //printf("B");
                    continue;
                }
            }
            if(pi_need_v > pi_thr_u){
                double q_zhong = q_v;
                while(q_zhong <= q_upper){
                    q_zhong = q_zhong + 0.01;
                    pi_need_v = (qt_v * qt_v/(r*p*q_zhong*r*p)+1)/2;
                    if(pi_need_v <= pi_thr_u) break;
                    }
                if(q_zhong > q_upper){
                    v_index_begin = v_index_select;
                    //printf("B");
                    continue;
                }
            }
        }
        
        if(pi_need_u < pi_thr_l | pi_need_v < pi_thr_l){
            if(pi_need_u < pi_thr_l){
                u_index_end = u_index_select;
            }
            if(pi_need_v < pi_thr_l){
                v_index_end = v_index_select;
            }
            //printf("C");
            continue;
        }
        
        if(pi_need_u > pi_thr_l && pi_need_v > pi_thr_l){
            break;
        }
    }
    
    double pi_v = pi_need_v;
    double pi_u = pi_need_u;

    List model_final = BiElasso(X, r, lambda_u, lambda_v, iter_max*3, tol);
    mat Ut = model_final[0];
    mat Vt = model_final[1];

    
    
    int count_wrong2;
    int B2_2 = B2;
    for(int i = 1; i<= B2; i++){
        
        while(true){
            try{
                sOmega_B2 = good_sample(sample_list, n, p, sample_size, r);
                List model_B2 = BiElasso_bootstrap_one(X, sOmega_B2, r, Ut, Vt, lambda_u, lambda_v, iter_max, tol);
                mat zhong4 = model_B2[0];
                ut_beta_times = ut_beta_times + reshape(zhong4 != 0, r*n, 1);
                mat zhong5 = model_B2[1];
                vt_beta_times = vt_beta_times + reshape(zhong5 != 0, r*p, 1);
                break;
            }
            catch(const std::runtime_error &e2){
                count_wrong2 += 1;
                printf(",%d", count_wrong2);
                if(count_wrong2 > B2){
                    //arma_stop_runtime_error("Too much bad subsamples in B2.");
                    B2_2 = i;
                    break;
                }
                continue;
            }
            catch(const std::exception &e3){
                cerr << e3.what() << endl;
                count_wrong2 += 1;
                printf(",%d", count_wrong2);
                if(count_wrong2 > B2){
                    //arma_stop_runtime_error("Too much bad subsamples in B2.");
                    B2_2 = i;
                    break;
                }
                continue;
            }
        }

    }
     
    
    uvec zero_idx_u = find((ut_beta_times / 1.0 /B2_2) <= pi_u);
    Ut_vector = reshape(Ut, n*r ,1);
    Ut_vector.elem(zero_idx_u).zeros();
    Ut = reshape(Ut_vector,n,r);
    
    uvec zero_idx_v = find((vt_beta_times / 1.0 /B2_2) <= pi_v);
    Vt_vector = reshape(Vt, p*r ,1);
    Vt_vector.elem(zero_idx_v).zeros();
    Vt = reshape(Vt_vector,p,r);
    
     
    List list1;
    list1.push_back(Ut);
    list1.push_back(Vt);
    list1.push_back(stop);
    list1.push_back(lambda_u);
    list1.push_back(lambda_v);
    return(list1);
}



// [[Rcpp::export]]
List BiElasso_cv(const arma::mat& X, arma::vec r_list,
                 const double& q_u = 0.2, const double& q_v = 0.2, const double& q_upper = 0.5,
                 const double& select_upper_u = 0.45, const double& select_upper_v = 0.45,
                 const double& B1 = 10, const double& B2 = 200, const int& cv_k = 5,
                 const double& pi_thr_l = 0.65, const double& pi_thr_u = 0.7,
                 const double& size_per = 0.5, const double& speed = 0.3,
                 const int& iter_max = 100, const double& tol = 0.000001, int sparse = 0){
    //cv_k is new parameter.
    vec pre_error = zeros<vec>(r_list.n_elem);
    double lambda_u, lambda_v;
    int p = X.n_cols;
    int n = X.n_rows;
    vec all_index = linspace<vec>(0,p*n-1,p*n);
    vec sample_cv, sOmega;
    int m = floor(n * p / cv_k);
    mat X_pre;
    vec pre_diff;
    List model_lambda;
    List Ut_list, Vt_list, lambda_u_list, lambda_v_list;
    
    for(int i = 0; i< r_list.n_elem; i++){
        //select lambda for each r first
        model_lambda = BiElasso_stable(X, r_list[i], q_u, q_v, q_upper, select_upper_u, select_upper_v,
                        B1, B2, pi_thr_l, pi_thr_u, size_per, speed, iter_max, tol, sparse);
        printf("1");
        Ut_list.push_back(model_lambda[0]);
        Vt_list.push_back(model_lambda[1]);
        lambda_u_list.push_back(model_lambda[3]);
        lambda_v_list.push_back(model_lambda[4]);

        lambda_u = model_lambda[3];
        lambda_v = model_lambda[4];

        while(true){
            try{
                sample_cv = Rcpp::RcppArmadillo::sample(all_index, n*p, false);
                pre_error[i] = 0;

            
                for(int j = 0; j < cv_k; j++){
                    uvec subset_index;
                    uvec test_index;

                    if(j ==0){
                        subset_index = linspace<uvec>(0, (cv_k-1)*m-1, (cv_k-1)*m);
                        test_index = linspace<uvec>((cv_k-1)*m, n * p - 1, n * p - (cv_k-1)*m);
                    }
                    if(j == (cv_k - 1)){
                        subset_index = linspace<uvec>(m, n * p - 1, n * p - m);
                        test_index = linspace<uvec>(0, m- 1, m);

                    }
                    if(j !=0 && j != (cv_k - 1)){
                        subset_index = zeros<uvec>(n*p - m);
                        subset_index(span(0, m*j - 1)) = linspace<uvec>(0, j*m-1, j*m);
                        subset_index(span(m*j, n*p - m - 1)) = linspace<uvec>((j+1)*m, n * p - 1, n * p - (j+1)*m);
                        test_index = linspace<uvec>(m*j, (j+1)*m-1, m);

                    }
                    
                    sOmega = sample_cv(subset_index);
                    List model_pre = BiElasso_bootstrap(X, sOmega, r_list[i], lambda_u, lambda_v, iter_max, tol);
                    mat Ut = model_pre[0];
                    mat Vt = model_pre[1];
                    X_pre = Ut * Vt.t();
                    pre_diff = reshape(X_pre - X, n*p, 1);
                    pre_error[i] += sum(abs(pre_diff(test_index)));
                    
                }
                
                break;
            }
            catch(...){
                continue;
            }
        }
    }
    
    uvec r_index = sort_index(pre_error, "ascent");
    int r_select = r_list[r_index[0]];
    
    List list1;
    list1.push_back(r_select);
    list1.push_back(pre_error);
    list1.push_back(Ut_list[r_index[0]]);
    list1.push_back(Vt_list[r_index[0]]);
    list1.push_back(lambda_u_list[r_index[0]]);
    list1.push_back(lambda_v_list[r_index[0]]);



    return(list1);
}

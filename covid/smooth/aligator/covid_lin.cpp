//#include<iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#include<cmath>
#include<vector>
#include<utility>

using namespace std;

namespace py = pybind11;

double prev_pred = 0;
int final_idx;

struct expert{
  double prediction;
  double loss;
  //int count;
  double weight;

  int start_idx;
  int end_idx;


  expert(){
    prediction = 0;
    loss = 0;
    //count = 0;
    weight = 0;

    start_idx = 0;
    end_idx = 0;
    
  }
  
};


int init_experts(std::vector<std::vector<expert> >& pool, int n){
  int count = 0;
  for(int k = 0; k <= floor(log2(n)); k++){

    int stop = ((n+1) >> k) - 1;
    
    //int stop = floor((n+1)/pow(2,k) -1);

    if(stop < 1)
      break;

    std::vector<expert> elist;
    for (int i=1; i<= stop; i++){
      expert e;

      e.start_idx = i* (1 << k) - 1; // 0-based index
      e.end_idx = (i+1)*(1 << k) - 2 < final_idx ? (i+1)*(1 << k) - 2 : final_idx;
      
      elist.push_back(e);
      count++;
    }
    pool.push_back(elist);
  }
  return count;
}

std::pair<double, double> perform_ols(int start, int end, int idx, std::vector<double>& y){
  // start - end + 1 must be greater than 2
  

  //calculation of slope
  int n = end - start + 1;
  n = n -1; // beacuse of leave one out

  double sum_xy = 0;
  double sum_x = 0;
  double sum_y = 0;
  double sum_x2 = 0;
  
  for(int i = start; i<= end; i++){
    if(i == idx) continue;

    sum_xy = sum_xy + (i+1)*y[i];
    sum_x = sum_x + (i+1);
    sum_y = sum_y + y[i];
    sum_x2 = sum_x2 + ((i+1)*(i+1));
  }

  double m = ((n*sum_xy) - (sum_x * sum_y))/((n*sum_x2)-(sum_x * sum_x));
  double c = (sum_y - (m*sum_x))/n;

  return std::make_pair(m,c);
  
}

void get_awake_set(std::vector<int>& index, int t, int n){
  //std::vector<int> index;
  
  for(int k=0; k<= floor(log2(t)); k++){
      int i = (t >> k);
      if(((i+1) << k) - 1 > n)
	index.push_back(-1);
      else
	index.push_back(i);
    }

  //return index;
}

double get_forecast(std::vector<int>& awake_set,
		    std::vector<std::vector<expert> >& pool,
		    double& normalizer, int pool_size, int idx,
		    std::vector<double> y){
  double output = 0;
  normalizer = 0;
  int i;
  double prediction;
  
  for(int k=0; k<awake_set.size(); k++){
    if(awake_set[k] == -1) continue;

    i = awake_set[k] - 1;

    if(pool[k][i].weight == 0){
      pool[k][i].weight = 1.0/pool_size;
      // added to reduce jittery output for isotonic case
      prediction = prev_pred;
    }

    if(pool[k][i].end_idx - pool[k][i].start_idx +1 <= 2)
      prediction = prev_pred;
    else{
      std::pair<double,double> ols = perform_ols(pool[k][i].start_idx, pool[k][i].end_idx,
				  idx, y);
      prediction = (ols.first * (idx+1) ) + ols.second;
    }
      
    pool[k][i].prediction = prediction;
    output = output + (pool[k][i].weight * prediction);
    normalizer = normalizer + pool[k][i].weight;
  }
  return output/normalizer;
}

void compute_losses(std::vector<int>& awake_set,
		    std::vector<std::vector<expert> >& pool,
		    std::vector<double>& losses, double y,
		    double B, int n, double sigma, double delta){
  int i;
  //double norm = 2*(B + sigma*sqrt(log(2*n/delta)))*
  //(B + sigma*sqrt(log(2*n/delta)));

  // using sigma as a proxy for step size
  double norm = 2*sigma*B*B;
  
  for(int k=0; k<awake_set.size(); k++){
    if(awake_set[k] == -1){
      losses.push_back(-1);
    }
    else{
      i = awake_set[k] - 1;
      double loss= (y-pool[k][i].prediction)*(y-pool[k][i].prediction)/norm;
      
      losses.push_back(loss);
    }
  }
}


void update_weights(std::vector<int>& awake_set,
		    std::vector<std::vector<expert> >& pool,
		    std::vector<double>& losses,
				    double normalizer){
  double norm = 0;
  int i;

  // compute new normalizer
  for(int k=0; k < awake_set.size(); k++){
    if(awake_set[k] == -1) continue;

    i = awake_set[k] - 1;

    //assert(losses[k] != -1);

    norm = norm + pool[k][i].weight * exp(-losses[k]);
    
  }

  // update weights
  for(int k=0; k < awake_set.size(); k++){
    if(awake_set[k] == -1) continue;

    i = awake_set[k] - 1;

    pool[k][i].weight = pool[k][i].weight * exp(-losses[k]) *
      normalizer/norm;

    
  }
  
}

std::vector<double> make_durational_forecast(int n,std::vector<std::vector<expert> >& pool,
					     std::vector<double>& y, int duration){
  std::vector<int> awake_set;
  get_awake_set(awake_set,n,n);

  double m = 0;
  double c = 0;
  double normalizer = 0;
  int i;

  for(int k=0; k<awake_set.size(); k++){
    if(awake_set[k] == -1) continue;

    i = awake_set[k] - 1;


    if(pool[k][i].end_idx - pool[k][i].start_idx +1 < 2)
      c = c + pool[k][i].weight * y[n-1];
    else{
      std::pair<double,double> ols = perform_ols(pool[k][i].start_idx, pool[k][i].end_idx,
				  -1, y);
      m = m + (pool[k][i].weight * ols.first);
      c = c + (pool[k][i].weight * ols.second);
    }
      
    normalizer = normalizer + pool[k][i].weight;
  }

  m = m/normalizer;
  c = c/normalizer;

  std::vector<double> estimates;

  for(int j = n+1; j <= n+duration; j++){
    double pred = m*j + c; 
    estimates.push_back(pred);
  }
  
  return estimates;
  
}

std::vector<double> run_aligator(int n, std::vector<double> y,
				 std::vector<int> index,
				 double sigma,
				 double B, double delta,
				 int duration ){
  prev_pred = 0;
  final_idx = n-1;
  std::vector<double> estimates(n);
  std::vector<std::vector<expert> > pool;

  int pool_size = init_experts(pool,n);


  for(int t=0; t<n; t++){
    double normalizer = 0;
    std::vector<int> awake_set;

    int idx = index[t];
    double y_curr = y[idx];
    
    //get_awake_set(awake_set,t+1,n);
    get_awake_set(awake_set,idx+1,n);

    double output = get_forecast(awake_set, pool, normalizer, pool_size, idx, y);
    //estimates.push_back(output);
    estimates[idx] = output;

    std::vector<double> losses;
    //compute_losses(awake_set, pool, losses, y[t], B, n, sigma, delta);
    compute_losses(awake_set, pool, losses, y_curr, B, n, sigma, delta);
    
    //update_weights_and_predictions(awake_set, pool, losses, normalizer, y[t]);
    update_weights(awake_set, pool, losses, normalizer);

    //prev_pred = output;
    prev_pred = y_curr;
    
  }
  if(duration == -1)
    return estimates;

  // forecast the upcoming duration
  return make_durational_forecast(n,pool,y,duration);
  
  
}
		    

PYBIND11_MODULE(covid_lin, m) {
    m.doc() = "pybind11 covid plugin"; // optional module docstring

    m.def("run_aligator", [](int n, std::vector<double> y, std::vector<int> index, \
			     double sigma, double B, double delta, int duration) -> py::array {
	    auto v = run_aligator(n,y,index,sigma,B,delta,duration);
	return py::array(v.size(), v.data());
	  },py::arg("n"), py::arg("index"), py::arg("y"), py::arg("sigma"),	\
	  py::arg("B"), py::arg("delta"), py::arg("duration"));
}

/*

c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` covid_lin.cpp -o covid_lin`python3-config --extension-suffix`



 */

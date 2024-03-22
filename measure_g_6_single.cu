#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "../timer.cuh"
#include <math.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <sys/stat.h>
#include "../MT.h"
#include <complex.h>
using namespace std;

//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 128; //Num of the cuda threads.
const int  NP = 1e4; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const int  NN = 100; //nearest neighbor maximum numbers
const double dt_initial = 0.01;
const int timeeq = 1000;
const double zeta = 1.0;
const double zeta_zero = 1.;
const double RCHK = 4.0;
const double rcut = 1.0;

//Initiallization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}

//Gaussian random number's generation
__global__ void genrand_kernel(float *result, curandState *state){  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  result[i_global] = curand_normal(&state[i_global]);
}

__global__ void p_bound(double *x_dev, double *y_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}

int calc_com(double *x_corr, double *y_corr, double *corr_x, double *corr_y){
  *corr_x = 0.;
  *corr_y = 0.; 
  for (int i=0; i<NP; i++){
    *corr_x += x_corr[i];
    *corr_y += y_corr[i];
  }
  //printf("%f  %f\n",*corr_x, *corr_y);

  return 0;
}

double calc_MSD(double *MSD_host){
  double msd = 0.;

  for (int i=0; i<NP; i++){
    msd += MSD_host[i];
  }
  return msd;
}

double calc_ISF(double *ISF_host){
  double isf = 0.;

  for (int i=0; i<NP; i++){
    isf += ISF_host[i];
  }
  return isf;
}

double calc_K(double *vx, double*vy){
  double K = 0.;
  for (int i=0; i<NP; i++){
    K += (vx[i]*vx[i]+vy[i]*vy[i])*0.5/(double) NP;
  }
  return K;
}

void save_position(double *x, double*y, double *xi, double *yi, double corr_x, double corr_y){
  for (int i=0; i<NP; i++){
    xi[i] = x[i] - corr_x;
    //if(i%1000==0){printf("%.4f %.4f %.4f\n",*corr_x, x0[i], x[i]);}
    yi[i] = y[i] - corr_y;
  }
  //return 0;
}

__global__ void com_correction(double *x_dev, double *y_dev, double *x_corr_dev, double *y_corr_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  static double x0[NP], y0[NP];
  static bool IsFirst=true;

  if(i_global<NP){
    if(IsFirst){
      x0[i_global] = x_dev[i_global];
      y0[i_global] = y_dev[i_global];
      IsFirst = false;
    }
    double dx, dy;
    dx = x_dev[i_global] - x0[i_global];
    dy = y_dev[i_global] - y0[i_global];

    dx -= LB*floor(dx/LB+0.5);
    dy -= LB*floor(dy/LB+0.5);

    x_corr_dev[i_global] += dx/NP;
    y_corr_dev[i_global] += dy/NP;
    //if(i_global%1000 == 0){printf("%d %.5f	%.5f\n", i_global, x0[i_global], x_dev[i_global]);}
    x0[i_global] = x_dev[i_global];
    y0[i_global] = y_dev[i_global];
  }
}

__global__ void ini_gate_kernel(int *gate_dev,int c)
{
  gate_dev[0]=c;
}

__global__ void disp_gate_kernel(double LB,double *vx_dev,double *vy_dev,double *dx_dev,double *dy_dev,int *gate_dev, double dt)
{
  double r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
 
 if(i_global<NP){
    dx_dev[i_global]+=vx_dev[i_global]*dt; //particles dispacement update
    dy_dev[i_global]+=vy_dev[i_global]*dt;
    r2 = dx_dev[i_global]*dx_dev[i_global]+dy_dev[i_global]*dy_dev[i_global];
    // printf("dr2=%d\n",dr2);
    if(r2> 0.25*(RCHK-rcut)*(RCHK-rcut)){ //threshold for update. rcut = potential cut distance.
      gate_dev[0]=1; //update's gate open!! many particle rewrite okay!
      //printf("i=%d,dr=%f\n",i_global,sqrt(r2));
    }
  }
}

__global__ void disp_gate_kernel_cABP(double LB,double *vx_dev,double *vy_dev,double *dx_dev,double *dy_dev,int *gate_dev, int select_potential, double dt)
{
  double r2, cut;

  if (select_potential ==0){cut = 3.0;}
  else{cut = 1.0;}  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  
 if(i_global<NP){
    dx_dev[i_global]+=vx_dev[i_global]*dt; //particles dispacement update
    dy_dev[i_global]+=vy_dev[i_global]*dt;
    r2 = dx_dev[i_global]*dx_dev[i_global]+dy_dev[i_global]*dy_dev[i_global];
    // printf("dr2=%d\n",dr2);
    if(r2> 0.25*(RCHK-cut)*(RCHK-cut)){ //threshold for update. rcut = potential cut distance.
      //printf("i_global = %d\n", i_global);
      gate_dev[0]=1; //update's gate open!! many particle rewrite okay!
      //printf("i=%d,dr=%f\n",i_global,sqrt(r2));
    }
  }
}


__global__ void update(double LB,double *x_dev,double *y_dev,double *dx_dev,double *dy_dev,int *list_dev,int *gate_dev)
{
  double dx,dy,r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    list_dev[NN*i_global]=0;     // single array in GPU, = list[i][0] 
    for (int j=0; j<NP; j++)
      if(j != i_global){  //ignore self particle
        dx = x_dev[i_global] - x_dev[j];
        dy = y_dev[i_global] - y_dev[j];
        dx -= LB*round(dx/LB); //boudary condition
        dy -= LB*round(dy/LB);	  
        r2 = dx*dx + dy*dy;
        
        if(r2 < RCHK*RCHK){
          list_dev[NN*i_global]++; //list[i][0] ++;
          list_dev[NN*i_global+list_dev[NN*i_global]]=j; //list[i][list[i][0]] = j;
      }
    //  printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.; //reset dx
    dy_dev[i_global]=0.; //
    
    }
  }
}

__global__ void calc_force_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB,int *list_dev){
  double dx,dy,dr,dU,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  //a_i  = a_dev[i_global];

  if(i_global<NP){
    fx_dev[i_global] = 0.0;
    fy_dev[i_global] = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){ //list[i][0]
      dx=x_dev[list_dev[NN*i_global+j]]-x_dev[i_global]; //x[list[i][j]-x[i]
      dy=y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);	
      dr = sqrt(dx*dx+dy*dy);
      a_ij=0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);  //0.5*(a[i]+a[i][j])
      if(dr < a_ij){ //cut off
	      dU = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r for harmonic potential.
         fx_dev[i_global] += dU*dx/dr; //only deal for i_global, don't care the for "j"
         fy_dev[i_global] += dU*dy/dr;
      }     
    }
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}

__global__ void calc_force_kernel_HP(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB,int *list_dev, double *pot){
  double dx,dy,dr,dU,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  //a_i  = a_dev[i_global];

  if(i_global<NP){
    fx_dev[i_global] = 0.0;
    fy_dev[i_global] = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){ //list[i][0]
      dx=x_dev[list_dev[NN*i_global+j]]-x_dev[i_global]; //x[list[i][j]-x[i]
      dy=y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);	
      dr = sqrt(dx*dx+dy*dy);
      a_ij=0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);  //0.5*(a[i]+a[i][j])
      if(dr < a_ij){ //cut off
	      dU = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r for harmonic potential.
         fx_dev[i_global] += dU*dx/dr; //only deal for i_global, don't care the for "j"
         fy_dev[i_global] += dU*dy/dr;
         pot[i_global] = 0.25*(1-dr/a_ij)*(1-dr/a_ij);
         pot[i_global] /= (double) NP;
      }     
    }
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}

__global__ void calc_force_BHHP_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB,int *list_dev, double *pot){
  double dx,dy,dU,a_ij,r2, w2,w4,w12,cut;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  //a_i  = a_dev[i_global];
  cut = 3.0;
  if(i_global<NP){
    fx_dev[i_global] = 0.0;
    fy_dev[i_global] = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){ //list[i][0]
      dx= x_dev[list_dev[NN*i_global+j]] - x_dev[i_global]; //x[list[i][j]-x[i]
      dy= y_dev[list_dev[NN*i_global+j]] - y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);	
      //dr = sqrt(dx*dx+dy*dy);
      a_ij=0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);  //0.5*(a[i]+a[i][j])
      r2 = dx * dx + dy * dy;
      w2 = a_ij * a_ij / r2;
      w4 = w2*w2;
      w12 = w4*w4*w4;
      if(r2 < cut*cut){ //cut off
	      dU = (-12.0)*w12/r2; //derivertive of U wrt r for harmonic potential.
         fx_dev[i_global] += dU*dx; //only deal for i_global, don't care the for "j"
         fy_dev[i_global] += dU*dy;
         pot[i_global] = w12*0.5;
      }     
    }
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}



__global__ void calculate_rdf(double *x, double *y, double LB, double delta_r,
                   double *r, int ri, double *histogram) {
    int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
    int j;
    if(i_global<NP){
        for (j = 0 ; j < NP; j++) {
            double dx = x[i_global] - x[j];
            double dy = y[i_global] - y[j];
            dx -= LB*floor(dx/LB+0.5);
            dy -= LB*floor(dy/LB+0.5);
            double distance = sqrt(dx * dx + dy * dy);
            int bin_index = (int)(distance / delta_r);
            if (bin_index < ri) {
                histogram[i_global*ri + bin_index] += 1.;
            }
        }
    }
}

__global__ void psi_6_measure(double *x_dev, double *y_dev, int *list_dev, double LB, double *psi_real_dev, double *psi_imag_dev, double *psi_dev, double cutoff){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    double dx, dy, dr, theta_ij;
    int count=0;
    double psi_real_sum = 0.0;
    double psi_imag_sum = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){
      dx= x_dev[list_dev[NN*i_global+j]] - x_dev[i_global]; //x[list[i][j]-x[i]
      dy= y_dev[list_dev[NN*i_global+j]] - y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);

      dr = (dx*dx + dy*dy);
      
      if (dr < cutoff*cutoff){
        count +=1;
        theta_ij = atan2(dy,dx);
        psi_real_sum += cos(6*theta_ij);
        psi_imag_sum += sin(6*theta_ij);
      }
    }
    if( count > 0 ){
    psi_real_dev[i_global] = psi_real_sum/count;
    psi_imag_dev[i_global] = psi_imag_sum/count;
    }
    psi_dev[i_global] = sqrt(psi_real_dev[i_global]*psi_real_dev[i_global] + psi_imag_dev[i_global]*psi_imag_dev[i_global]);
  }
}

__global__ void calculate_g_6(double *x, double *y, double LB, double delta_r,
                   double *r, int ri, double *histogram_real, double *histogram_imag, double *psi_real_dev, double *psi_imag_dev) {
    int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
    int j;
    if(i_global<NP){
        for (j = 0 ; j < NP; j++) {
            double dx = x[j] - x[i_global];
            double dy = y[j] - y[i_global];
            dx -= LB*floor(dx/LB+0.5);
            dy -= LB*floor(dy/LB+0.5);
            double distance = sqrt(dx * dx + dy * dy);
            int bin_index = (int)(distance / delta_r);
            if (bin_index < ri) {
                histogram_real[i_global*ri + bin_index] += psi_real_dev[i_global]*psi_real_dev[j] + psi_imag_dev[i_global]*psi_imag_dev[j];
                histogram_imag[i_global*ri + bin_index] += psi_imag_dev[i_global]*psi_real_dev[j] - psi_imag_dev[j]*psi_real_dev[i_global];
            }
        }
    }
}

__global__ void init_histogram(double *histogram_real, double *histogram_imag, int ri){
    int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
    int j;
    if(i_global<NP){
        for (j = 0 ; j < NP; j++) {
                histogram_real[i_global*ri + j] = 0.;
                histogram_imag[i_global*ri + j] = 0.;
        }
    }
}

__global__ void reduce_g_6(int ri, double *r, double *g_6_dev, double *histogram_real, double *histogram_imag, double delta_r, int rdf_count, double rho)
{    // Calculate RDF
    int i_global = threadIdx.x + blockIdx.x*blockDim.x;
    int k;
    if(i_global<ri){
        r[i_global] = delta_r * (i_global + 0.5);  // Midpoint of the bin
        for (k=0;k<NP;k++){
        g_6_dev[i_global] += histogram_real[i_global+k*ri]/(2*M_PI*r[i_global]*delta_r*rho*NP)/(double)rdf_count;
        }
	}    
}

__global__ void reduce_rdf(int ri, double *r, double *rdf_dev, double *histogram, double delta_r, int rdf_count, double rho)
{    // Calculate RDF
    int i_global = threadIdx.x + blockIdx.x*blockDim.x;
    int k;
    if(i_global<ri){
        r[i_global] = delta_r * (i_global + 0.5);  // Midpoint of the bin
        for (k=0;k<NP;k++){
        rdf_dev[i_global] += histogram[i_global+k*ri]/(2*M_PI*r[i_global]*delta_r*rho*NP)/(double) rdf_count;
        }
	}    
}

__global__ void calculate_structure_factor(double *x_dev, double *y_dev, double LB, double *q_dev, double *Sq_dev, int si){
	int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
	int j;
	double dq = 2.0 * M_PI / LB;
    double cos_sum = 0., sin_sum=0.;
    if(i_global<si){
        q_dev[i_global] = (double) i_global * dq;
        for (j = 0; j<NP; j++){
            double arg = q_dev[i_global] * x_dev[j], arg2 = q_dev[i_global] * y_dev[j];
            cos_sum += cos(arg) + cos(arg2);
            sin_sum += sin(arg) + sin(arg2);
        }
        Sq_dev[i_global] += (cos_sum*cos_sum+sin_sum*sin_sum)/(double)(NP);
    }
}

__global__ void Jq_measure(double *x_dev, double *y_dev, double *vx_dev, double *vy_dev, double LB, double *q_dev, 
                           double *Jx_real_dev, double*Jx_imag_dev,double *Jy_real_dev, double*Jy_imag_dev, int si){
	int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
	int j;
	double dq = 2.0 * M_PI / LB;
  double Jx_real_sum = 0., Jx_imag_sum=0., Jy_real_sum = 0., Jy_imag_sum=0.;
  if(i_global<si){
      q_dev[i_global] = (double) i_global * dq;
      for (j = 0; j<NP; j++){
          double arg = q_dev[i_global] * x_dev[j];
          Jx_real_sum += vx_dev[j] * (cos(-arg));
          Jx_imag_sum += vx_dev[j] * (sin(-arg));
          Jy_real_sum += vy_dev[j] * (cos(-arg));
          Jy_imag_sum += vy_dev[j] * (sin(-arg));
      }
      Jx_real_dev[i_global] = (Jx_real_sum);
      Jx_imag_dev[i_global] = (Jx_imag_sum);
      Jy_real_dev[i_global] = (Jy_real_sum);
      Jy_imag_dev[i_global] = (Jy_imag_sum);
  }
}

__global__ void Jq_divide(double LB, double *q_dev, double *Jx_real_dev, double*Jx_imag_dev, double *Jy_real_dev, double*Jy_imag_dev, 
                           double *Jp_real_dev, double *Jp_imag_dev, double *Jt_real_dev, double *Jt_imag_dev, int si){
	int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
  if(i_global<si){
      Jp_real_dev[i_global] = Jx_real_dev[i_global];
      Jp_imag_dev[i_global] = Jx_imag_dev[i_global];

      Jt_real_dev[i_global] = Jy_real_dev[i_global];
      Jt_imag_dev[i_global] = Jy_imag_dev[i_global];
  }
}

__global__ void velocity_correlation(double LB, double *Jp_real_dev, double *Jp_imag_dev, 
                           double *Jt_real_dev, double *Jt_imag_dev,double *omega_p, double *omega_t, int si){
	int i_global = threadIdx.x + blockIdx.x*blockDim.x; 
  if(i_global<si){
      omega_p[i_global] += (Jp_real_dev[i_global] * Jp_real_dev[i_global] + Jp_imag_dev[i_global] * Jp_imag_dev[i_global]);
      omega_t[i_global] += (Jt_real_dev[i_global] * Jt_real_dev[i_global] + Jt_imag_dev[i_global] * Jt_imag_dev[i_global]);
      //omega_t[i_global] += ;
  }
}

__global__ void reduce_omega(int si, double *q_dev, double *omega_p_dev, double *omega_t_dev, int rdf_count)
{    // Calculate RDF
    int i_global = threadIdx.x + blockIdx.x*blockDim.x;
    if(i_global<si){
        omega_p_dev[i_global] /= (double)(NP*rdf_count);  // Midpoint of the bin
        omega_t_dev[i_global] /= (double)(NP*rdf_count);
	}    
}

__global__ void copy_kernel(double *x0_dev, double *y0_dev, double *x_dev, double *y_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
  x0_dev[i_global]=x_dev[i_global];
  y0_dev[i_global]=y_dev[i_global];
  //printf("%f,%f\n",x_dev[i_global],x0_dev[i_global]);
  } 
}

//For MSD measure, save initial positions
__global__ void copy_kernel2(double *xi_dev, double *yi_dev, double *x_dev, double *y_dev, double *corr_x_dev, double *corr_y_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    xi_dev[i_global] = x_dev[i_global] - *corr_x_dev;
    yi_dev[i_global] = y_dev[i_global] - *corr_y_dev;
    //if(i_global%1000==0){printf("%d, %f, %f\n",i_global, *corr_x_dev,*corr_y_dev);}
  } 
}

// homogeneous system -> binary you have to change
__global__ void init_array(double *x_dev, double c){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c;
}

__global__ void init_binary(double *x_dev, double c, double c2){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global < NP){ 
    if(i_global%2==0){x_dev[i_global] =c;}
    else {x_dev[i_global] =c2;}
  }
}

//position randomly allocate
__global__ void init_array_rand(double *x_dev, double c,curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c*curand_uniform(&state[i_global]); 
}


__global__ void MSD_ISF_device(double *x_dev, double *y_dev, double *xi_dev, double *yi_dev, double *corr_x_dev, double *corr_y_dev, double *MSD_dev, double *ISF_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double dx, dy;
  double q = 2. * M_PI / 1.0;

  if (i_global<NP){
     dx = x_dev[i_global] - xi_dev[i_global] - *corr_x_dev;
     dy = y_dev[i_global] - yi_dev[i_global] - *corr_y_dev;
    //if(i_global%1000==0){printf("%d  %.3f %.3f\n", i_global, dx, dy);}
     dx -= LB*floor(dx/LB+0.5); //boudary condition
     dy -= LB*floor(dy/LB+0.5);	  
     
     MSD_dev[i_global] = (dx*dx + dy*dy)/(double)NP;
     ISF_dev[i_global] = (cos(- q * dx) + cos(- q * dy)) / (double)NP / 2.0;
     //if(i_global%1000==0){printf("%d	%.4f\n",i_global, *corr_x_dev);}
  }
}

__global__ void ISF_device(double *x_dev, double *y_dev, double *xi_dev, double *yi_dev, double *corr_x_dev, double *corr_y_dev, double *ISF_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double dx, dy;
  double q = 2. * M_PI / 1.0;
  if (i_global<NP){
     dx = x_dev[i_global] - xi_dev[i_global] - *corr_x_dev; //including COM displacement 
     dy = y_dev[i_global] - yi_dev[i_global] - *corr_y_dev;

     dx -= LB*floor(dx/LB+0.5); //boudary condition
     dy -= LB*floor(dy/LB+0.5);	  

     ISF_dev[i_global] = 0.5 * (cos(-q * dx) + cos(-q * dy)) / (double)NP;
     //if(i_global%1000==0){printf("%d	%.4f\n",i_global, *corr_x_dev);}
  }
}

void g_6_Measure(int ri, double *r, double *g_6_host, int si, double *q, double *Sq_host, double *omega_p_host, double *omega_t_host, int rdf_count, double omega, double tau_p, double rho, int select_potential){
  char filename[128], filename2[128], filename3[128];
  mkdir("g_6_data",0755);
  mkdir("omega_data",0755);
  mkdir("Sq_data",0755);
  if(select_potential==0){
    sprintf(filename,"omega_data/LJ_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename2,"g_6_data/LJ_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename3,"Sq_data/WCA_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
  if(select_potential == 1){
    sprintf(filename,"omega_data/WCA_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename2,"g_6_data/WCA_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename3,"Sq_data/WCA_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
 if(select_potential == 2){
    sprintf(filename,"omega_data/HP_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename2,"g_6_data/HP_Pe%.4f_omega%.4f_phi%.1f.dat",tau_p,omega,rho);
    sprintf(filename3,"Sq_data/WCA_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }

  FILE *fp,*fp2,*fp3;
  
  fp = fopen(filename, "w+");
  for(int i=1;i<si;i++){
    fprintf(fp, "%.4f\t%.9f\t%.9f\n", q[i], 0.5*omega_p_host[i], 0.5*omega_t_host[i]);
  }
  fclose(fp);

  fp2 = fopen(filename2, "w+");
  for(int i=1;i<ri;i++){
    fprintf(fp2, "%.4f\t%.9f\n", r[i], g_6_host[i]);
  }
  fclose(fp2);

    fp3 = fopen(filename3, "w+");
  for(int i=1;i<si;i++){
    fprintf(fp3, "%.4f\t%.4f\n", q[i], Sq_host[i]/(double)rdf_count);
  }
  fclose(fp3);

}

void output_Measure(double *measure_time, double *MSD, double *ISF, double *count, int time_count, int eq_count, int ri, double *r, double *rdf_host, int si, double *q_host, double *Sq_host, int rdf_count, double omega, double tau_p, double rho, int select_potential){
  char filename[128], filename2[128], filename3[128];
  mkdir("data",0755);

  if(select_potential==0){
    sprintf(filename,"data/LJ_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/LJ_Pe%.4f_omega%.4f_phi%.4f",tau_p,omega,rho);
    sprintf(filename3,"data/LJ_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
  if(select_potential == 1){
    sprintf(filename,"data/WCA_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/WCA_Pe%.4f_omega%.4f_phi%.4f.dat",tau_p,omega,rho);
    sprintf(filename3,"data/WCA_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
 if(select_potential == 2){
    sprintf(filename,"data/HP_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/HP_Pe%.4f_omega%.4f_phi%.4f.dat",tau_p,omega,rho);
    sprintf(filename3,"data/HP_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }

  FILE *fp,*fp2, *fp3;
  fp = fopen(filename, "w+");
  for(int i=1;i<time_count;i++){
    fprintf(fp, "%.4f\t%.4f\t%.4f\n", measure_time[i]-measure_time[0], MSD[i]/(count[i]-eq_count), ISF[i]/(count[i]-eq_count));
  }
  fclose(fp);
  
  fp2 = fopen(filename2, "w+");
  for(int i=1;i<ri;i++){
    fprintf(fp2, "%.4f\t%.4f\n", r[i], rdf_host[i]);
  }
  fclose(fp2);

  fp3 = fopen(filename3, "w+");
  for(int i=1;i<si;i++){
    fprintf(fp3, "%.4f\t%.4f\n", q_host[i], Sq_host[i]/(double)rdf_count);
  }
  fclose(fp3);
}

int output(double *x,double *y, double *theta, double *vx, double *vy, double *r1, double t, double omega, double tau_p, double rho ,int select_potential){
  int i;
  static int count = 0;
  char filename[128], foldername[128];
  if (select_potential==0){
    sprintf(foldername, "position_LJ"); 
    {mkdir(foldername, 0755);}
    sprintf(foldername, "position_LJ/N%d_Pe%.1f_omega%.3f_phi%.2f", NP, tau_p, omega, rho);
  }
  if (select_potential==1){
    sprintf(foldername, "position_WCA");
    {mkdir(foldername, 0755);}
    sprintf(foldername, "position_WCA/N%d_Pe%.1f_omega%.3f_phi%.2f", NP, tau_p, omega, rho);
  }
  if (select_potential==2){
    sprintf(foldername, "position_WCA");
    {mkdir(foldername, 0755);}
    sprintf(foldername, "position_HP/N%d_Pe%.1f_omega%.3f_phi%.2f", NP, tau_p, omega, rho);
  }

  {mkdir(foldername, 0755);}
  sprintf(filename,"%s/count_%d.dat",foldername, count);
  ofstream file;
  file.open(filename);
  for(i=0;i<NP;i++)
    file << x[i] << " " << y[i]<< " " << theta[i] << " " << t << endl;
  file.close();
  count++;
  
  return 0;
}


int main(int argc, char** argv){
  double *x,*xi, *x_dev, *xi_dev, *x0_dev,*vx,*vx_dev, *y, *y_dev, *yi, *yi_dev, *y0_dev, *vy, *a, *dx_dev,*dy_dev,*vy_dev,*a_dev,*fx_dev,*fy_dev;
  double *theta, *theta_dev; 
  double *x_corr_dev, *y_corr_dev, *x_corr, *y_corr, corr_x=0., corr_y=0., *corr_x_dev, *corr_y_dev;
  int *list_dev,*gate_dev, time_count, init_count;
  double *MSD_dev, *MSD_host, *ISF_dev,*ISF_host;
  double sampling_time, time_stamp=0.;
  int ri=5000, si = 2000;
  double delta_r = 0.01;
  double *histogram, *rdf_dev, *rdf_host, *r_dev, *r_host, *histogram_real, *histogram_imag, *g_6_dev, *g_6_host;
  double *psi, *psi_dev, *psi_real_dev, *psi_imag_dev;
  double *Sq_dev, *Sq_host, *q_dev, *q_host;
  double *Jx_real_dev, *Jx_imag_dev, *Jy_real_dev, *Jy_imag_dev;
  double *Jp_real_dev, *Jp_imag_dev;
  double *Jt_real_dev, *Jt_imag_dev;
  double *omega_p_dev, *omega_t_dev;
  double *omega_p_host, *omega_t_host;
  double Sq_MPI[si], rdf_MPI[ri];
  double pot, *pot_dev;
  double sampling_time_max =1e4;
  curandState *state; //Cuda state for random numbers
  double omega, tau_p, rho;
  int select_potential; //0 is LJ, 1 is WCA, 2 is Harmonic
  int timemax, eq_count;
  double dt, position_start;

  char* char_tau_p = argv[1];
  sscanf(char_tau_p, "%lf", &tau_p);
  
  char* char_omega = argv[2];
  sscanf(char_omega, "%lf", &omega);

  char* char_rho = argv[3];
  sscanf(char_rho, "%lf", &rho); 
	
  char* char_pot = argv[4];
  sscanf(char_pot, "%d", &select_potential);

   // start simulation
  x  = (double*)malloc(NB*NT*sizeof(double)); // memory allocattion on CPU
  xi  = (double*)malloc(NB*NT*sizeof(double)); // memory allocattion on CPU
  y  = (double*)malloc(NB*NT*sizeof(double));
  yi  = (double*)malloc(NB*NT*sizeof(double));
  vx = (double*)malloc(NB*NT*sizeof(double));
  vy = (double*)malloc(NB*NT*sizeof(double));
  a  = (double*)malloc(NB*NT*sizeof(double));
  x_corr  = (double*)malloc(NB*NT*sizeof(double));
  y_corr  = (double*)malloc(NB*NT*sizeof(double));
  MSD_host  = (double*)malloc(NB*NT*sizeof(double));
  ISF_host  = (double*)malloc(NB*NT*sizeof(double));
  Sq_host  = (double*)malloc(NB*NT*sizeof(double));
  omega_p_host  = (double*)malloc(NB*NT*sizeof(double));
  omega_t_host  = (double*)malloc(NB*NT*sizeof(double));
  q_host  = (double*)malloc(NB*NT*sizeof(double));
  theta = (double*)malloc(NB*NT*sizeof(double));
  psi = (double*)malloc(NB*NT*sizeof(double));
  rdf_host  = (double*)malloc(NB*NT*sizeof(double));
  g_6_host  = (double*)malloc(NB*NT*sizeof(double));
  r_host  = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev,  NB * NT * sizeof(double)); // CudaMalloc should be executed once in the host. 
  cudaMalloc((void**)&y_dev,  NB * NT * sizeof(double));
  cudaMalloc((void**)&theta_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&xi_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&yi_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&x0_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&y0_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dx_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&pot_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dy_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&rdf_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&x_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&y_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&corr_x_dev, sizeof(double));
  cudaMalloc((void**)&corr_y_dev, sizeof(double));
  cudaMalloc((void**)&MSD_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&ISF_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&Sq_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jx_real_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jy_real_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jx_imag_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jy_imag_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jp_real_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jp_imag_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jt_real_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&Jt_imag_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&omega_p_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&omega_t_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&q_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&gate_dev, sizeof(int)); // for update 
  cudaMalloc((void**)&list_dev,  NB * NT * NN* sizeof(int)); 
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState)); 
  cudaMalloc((void**)&psi_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&psi_real_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&psi_imag_dev, NB * NT * sizeof(double));    
  cudaMalloc((void**)&g_6_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&histogram, ri * NP * sizeof(double));
  cudaMalloc((void**)&histogram_real, ri * NP * sizeof(double));
  cudaMalloc((void**)&histogram_imag, ri * NP * sizeof(double));
  cudaMalloc((void**)&r_dev,  NB * NT * sizeof(curandState)); 

  char filename[128];
  double dummy1, dummy2, dummy3;
  int rdf_count;

  
 // double omega_list[9] = {0.001, 0.010, 0.020, 0.040, 0.100, 0.300, 1.000, 3.000, 5.000};
  //double cutoff_list[9] = {1.525, 1.525, 1.525, 1.505, 1.505, 1.565, 1.575, 1.905, 2.065};
  double cutoff = 2.065;
  double final_psi6;

  FILE *data;
  mkdir("psi_6_data", 0755);
  char filename2[128]; 
  sprintf(filename2, "psi_6_data/Pe%.1f_omega%.4f_phi%.1f.dat", tau_p, omega, rho);
  data = fopen(filename2, "w+");

  //for (int index = 0 ; index < 9 ; index++)
  {
    init_array<<<NB,NT>>>(g_6_dev,0.);
    init_histogram<<<NB,NT>>>(histogram_imag, histogram_real, ri);
    init_array<<<NB,NT>>>(psi_imag_dev,0.);
    init_array<<<NB,NT>>>(psi_real_dev,0.);
    init_array<<<NB,NT>>>(Jp_real_dev,0.);
    init_array<<<NB,NT>>>(Jp_imag_dev,0.);
    init_array<<<NB,NT>>>(Jt_real_dev,0.);
    init_array<<<NB,NT>>>(Jt_imag_dev,0.);
    init_array<<<NB,NT>>>(omega_p_dev,0.);
    init_array<<<NB,NT>>>(omega_t_dev,0.);
    init_array<<<NB,NT>>>(Sq_dev,0.);

    //omega = omega_list[index];
    //cutoff = cutoff_list[index];
    rdf_count = 0;
    final_psi6 = 0.0;
    for (int count_time = 500; count_time < 999; count_time++){
      if (select_potential==0){sprintf(filename, "position_LJ/N%d_Pe%.1f_omega%.3f_phi%.2f/count_%d.dat", NP, tau_p, omega,rho,count_time);}
      if (select_potential==1){sprintf(filename, "WCA2/N%d_Pe%.1f_omega%.3f_phi%.2f/count_%d.dat", NP, tau_p, omega,rho,count_time);}
      if (select_potential==2){sprintf(filename, "position_HP/N%d_Pe%.1f_omega%.3f_phi%.2f/count_%d.dat", NP, tau_p, omega,rho,count_time);}

      printf("filename = %s\n", filename);
      
      FILE *file;
      file = fopen(filename, "r");
      
      if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return 1;
      }

      for (int j = 0; j < NP; j++) {
        fscanf(file, "%lf  %lf %lf %lf %lf %lf ", &x[j], &y[j], &theta[j], &vx[j], &vy[j], &dummy3);
      }

      fclose(file);
      double LB = sqrt(M_PI*(1.0*1.0)*(double)NP/(4.* rho));  //system size  
      
      cudaMemcpy(x_dev, x, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(y_dev, y,  NB * NT* sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(vx_dev, vx, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(vy_dev, vy,  NB * NT* sizeof(double),cudaMemcpyHostToDevice);

      update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev);
      psi_6_measure<<<NB,NT>>>(x_dev, y_dev, list_dev, LB, psi_real_dev, psi_imag_dev, psi_dev, cutoff);
      //double psi_6 = thrust::reduce(thrust::device_pointer_cast(psi_dev), thrust::device_pointer_cast(psi_dev + NB * NT), 0.0, thrust::plus<double>());
      double psi_6_real = thrust::reduce(thrust::device_pointer_cast(psi_real_dev), thrust::device_pointer_cast(psi_real_dev + NB * NT), 0.0, thrust::plus<double>());
      double psi_6_imag = thrust::reduce(thrust::device_pointer_cast(psi_imag_dev), thrust::device_pointer_cast(psi_imag_dev + NB * NT), 0.0, thrust::plus<double>());
      double temp_psi = sqrt(psi_6_real*psi_6_real + psi_6_imag * psi_6_imag)/(double)NP;
      final_psi6 += temp_psi;

      //qx direction
      Jq_measure<<<NB,NT>>>(x_dev, y_dev, vx_dev, vy_dev,LB,q_dev, Jx_real_dev, Jx_imag_dev, Jy_real_dev, Jy_imag_dev, si);
      Jq_divide<<<NB,NT>>>(LB, q_dev, Jx_real_dev, Jx_imag_dev, Jy_real_dev, Jy_imag_dev, 
                           Jp_real_dev, Jp_imag_dev, Jt_real_dev, Jt_imag_dev, si);
      velocity_correlation<<<NB,NT>>>(LB, Jp_real_dev, Jp_imag_dev,Jt_real_dev, Jt_imag_dev, omega_p_dev, omega_t_dev, si);
      
      //qy direction
      Jq_measure<<<NB,NT>>>(y_dev, x_dev, vy_dev, vx_dev,LB,q_dev, Jx_real_dev, Jx_imag_dev, Jy_real_dev, Jy_imag_dev, si);
      Jq_divide<<<NB,NT>>>(LB, q_dev, Jx_real_dev, Jx_imag_dev, Jy_real_dev, Jy_imag_dev, 
                           Jp_real_dev, Jp_imag_dev, Jt_real_dev, Jt_imag_dev, si);
      velocity_correlation<<<NB,NT>>>(LB, Jp_real_dev, Jp_imag_dev,Jt_real_dev, Jt_imag_dev, omega_p_dev, omega_t_dev, si);
      printf("psi_6 = %f\n", temp_psi);
      calculate_g_6<<<NB,NT>>>(x_dev, y_dev, LB, delta_r, r_dev, ri, histogram_real, histogram_imag, psi_real_dev,psi_imag_dev);
      calculate_structure_factor<<<NB,NT>>>(x_dev, y_dev, LB, q_dev, Sq_dev, si);
      rdf_count += 1; //it is the number of measuring
    }
  
  reduce_g_6<<<NB,NT>>>(ri, r_dev, g_6_dev, histogram_real, histogram_imag, delta_r, rdf_count, rho);
  reduce_omega<<<NB,NT>>>(si, q_dev, omega_p_dev, omega_t_dev, rdf_count);
  cudaMemcpy(g_6_host, g_6_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(r_host, r_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(omega_p_host, omega_p_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(omega_t_host, omega_t_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(Sq_host, Sq_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(q_host, q_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  g_6_Measure(ri,r_host,g_6_host, si, q_host, Sq_host, omega_p_host, omega_t_host, rdf_count, omega, tau_p, rho, select_potential);
  fprintf(data, "%.4f\t%.4f\n", tau_p, final_psi6/(double)rdf_count);
  }

  fclose(data);
  cudaFree(g_6_dev);
  cudaFree(histogram);
  cudaFree(histogram_real);
  cudaFree(histogram_imag);
  cudaFree(r_dev);
  cudaFree(psi_real_dev);
  cudaFree(psi_imag_dev);
  cudaFree(psi_dev);
  cudaFree(Jp_real_dev);
  cudaFree(Jp_imag_dev);
  cudaFree(Jt_real_dev);
  cudaFree(Jt_imag_dev);
  cudaFree(omega_p_dev);
  cudaFree(omega_t_dev);
  cudaFree(Jx_real_dev);
  cudaFree(Jx_imag_dev);
  cudaFree(Jy_real_dev);
  cudaFree(Jy_imag_dev);
  free(omega_p_host);
  free(omega_t_host);
  free(g_6_host);
  free(psi);
  cudaFree(x_dev);
  cudaFree(xi_dev);
  cudaFree(vx_dev);
  cudaFree(yi_dev);
  cudaFree(vy_dev);
  cudaFree(dx_dev);
  cudaFree(dy_dev);
  cudaFree(x_corr_dev);
  cudaFree(y_corr_dev);
  cudaFree(corr_x_dev);
  cudaFree(corr_y_dev);
  cudaFree(MSD_dev);
  cudaFree(ISF_dev);
  cudaFree(rdf_dev);
  cudaFree(q_dev);
  cudaFree(Sq_dev);
  cudaFree(gate_dev);
  cudaFree(state);
  cudaFree(a_dev);
  cudaFree(theta_dev);
  cudaFree(pot_dev);
  free(x); 
  free(xi); 
  free(vx); 
  free(y); 
  free(yi); 
  free(vy); 
  free(x_corr); 
  free(y_corr); 
  free(MSD_host); 
  free(ISF_host);
  free(rdf_host);
  free(r_host);
  free(Sq_host);
  free(q_host);
  free(theta);
  free(a);
  return 0;
}

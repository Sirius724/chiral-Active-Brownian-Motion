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
#include <mpi.h>  //MPI header file
#include "../MT.h"
using namespace std;

//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 128; //Num of the cuda threads.
const int  NP = 1e4; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const int  NN = 200; //nearest neighbor maximum numbers
const double dt_initial = 0.01;
//const int timemax = 1e5;
const int timeeq = 1000;
//Langevin parameters
const double zeta = 1.0;
const double zeta_zero = 1.;
const double temp = 0.65;
const double epsilon = 1./24.;
//const double rho = 0.4;
const double RCHK = 4.0;
const double rcut = 1.0;
//const double tau_p = 200.; 
//const double omega = 0.1;
const double v0 = 1.0;
//const int position_interval = 0.1 / dt ; 
//const int time_interval = 10./dt;

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

//Gaussian random number's generation
__global__ void langevin_kernel(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    vx_dev[i_global] += -zeta_zero*vx_dev[i_global]*dt_initial+ fx_dev[i_global]*dt_initial + noise_intensity*curand_normal(&state[i_global]);
    vy_dev[i_global] += -zeta_zero*vy_dev[i_global]*dt_initial+ fy_dev[i_global]*dt_initial + noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += vx_dev[i_global]*dt_initial;
    y_dev[i_global] += vy_dev[i_global]*dt_initial;

    x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}

__global__ void langevin_kernel_cABP(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev, double *theta_dev, double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB, double omega, double dt){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    //if(i_global==0){printf("omega = %.3f\n", omega);}
	vx_dev[i_global] = v0*cos(theta_dev[i_global]) + 1./zeta*fx_dev[i_global];
    vy_dev[i_global] = v0*sin(theta_dev[i_global]) + 1./zeta*fy_dev[i_global];
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    x_dev[i_global] -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global] -= LB*floor(y_dev[i_global]/LB);
    theta_dev[i_global] += noise_intensity*curand_normal(&state[i_global]) + omega*dt;
	theta_dev[i_global] = fmod(theta_dev[i_global]+ 2 * M_PI, 2 * M_PI);
 		//printf("%f\n",x_dev[i_global]);
  }
}

__global__ void calc_force_WCA_kernel(double* x_dev, double* y_dev, double* fx_dev, double* fy_dev, double* a_dev, double LB, int* list_dev, double *pot) {
    double dx, dy, dU, a_ij, r2, w2, w6, cut;
    int i_global = threadIdx.x + blockIdx.x * blockDim.x;

    if (i_global < NP) {
        fx_dev[i_global] = 0.0;
        fy_dev[i_global] = 0.0;

        for (int j = 1; j <= list_dev[NN * i_global]; j++) {
            dx = x_dev[list_dev[NN * i_global + j]] - x_dev[i_global];
            dy = y_dev[list_dev[NN * i_global + j]] - y_dev[i_global];

            dx -= LB * floor(dx / LB + 0.5);
            dy -= LB * floor(dy / LB + 0.5);

            a_ij = 0.5 * (a_dev[i_global] + a_dev[list_dev[NN * i_global + j]]);
            cut = pow(2.0, 1.0 / 6.0) * a_ij; // Cutoff distance for WCA potential
            //r = sqrt(dx * dx + dy * dy);
            r2 = dx * dx + dy * dy;

            // Apply WCA potential force
            if (r2 < cut * cut ) {
                w2 = a_ij * a_ij / r2;
                w6 = w2 * w2 * w2;
                dU = -24.* epsilon *  w6 * (2.* w6 - 1.0) / r2 ;
                fx_dev[i_global] += dU * dx;
                fy_dev[i_global] += dU * dy;
                pot[i_global] = 2.* epsilon * (w6 * w6 - w6 + 0.25);
                pot[i_global] /= (double) NP;
            }
        }
    }
}

__global__ void calc_force_LJ_kernel(double* x_dev, double* y_dev, double* fx_dev, double* fy_dev, double* a_dev, double LB, int* list_dev, double *pot) {
    double dx, dy, dU, a_ij, r2, w2, w6, cut;
    int i_global = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i_global < NP) {
        fx_dev[i_global] = 0.0;
        fy_dev[i_global] = 0.0;

        for (int j = 1; j <= list_dev[NN * i_global]; j++) {
            dx = x_dev[list_dev[NN * i_global + j]] - x_dev[i_global];
            dy = y_dev[list_dev[NN * i_global + j]] - y_dev[i_global];

            dx -= LB * floor(dx / LB + 0.5);
            dy -= LB * floor(dy / LB + 0.5);

            a_ij = 0.5 * (a_dev[i_global] + a_dev[list_dev[NN * i_global + j]]);
            cut = 3.0 * a_ij;
            r2 = dx * dx + dy * dy;
            //r = sqrt(r2);

            if (r2 < cut * cut) {
                // Avoid division by zero
                if (r2 > 0) {
                    w2 = a_ij * a_ij / r2;
                    w6 = w2 * w2 * w2;
                    dU = - 24. * w6 * (2.* w6 - 1.0) / r2;
                    fx_dev[i_global] += dU * dx;
                    fy_dev[i_global] += dU * dy;
                    pot[i_global] = 2. * (w6 * w6 - w6);
                    pot[i_global] /= (double) NP;
                }
            }
        }
    }
}

/*
__global__ void langevin_kernel_nobound(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    vx_dev[i_global] += -zeta*vx_dev[i_global]*dt+ fx_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    vy_dev[i_global] += -zeta*vy_dev[i_global]*dt+ fy_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    //x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    //y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}
*/

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
  if(gate_dev[0] == 1 && i_global<NP){
    list_dev[NN*i_global]=0;     // single array in GPU, = list[i][0] 
    for (int j=0; j<NP; j++)
      if(j != i_global){  //ignore self particle
        dx = x_dev[i_global] - x_dev[j];
        dy = y_dev[i_global] - y_dev[j];
        dx -= LB*floor(dx/LB+0.5); //boudary condition
        dy -= LB*floor(dy/LB+0.5);	  
        r2 = dx*dx + dy*dy;
        
        if(r2 < RCHK*RCHK){
          list_dev[NN*i_global]++; //list[i][0] ++;
          list_dev[NN*i_global+list_dev[NN*i_global]]=j; //list[i][list[i][0]] = j;
        }
      }
    //  printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.; //reset dx
    dy_dev[i_global]=0.; //
    if(i_global==0) //only first thread change to the gate value
      gate_dev[0] = 0; // for only single thread.
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

__global__ void reduce_rdf(int ri, double *r, double *rdf_dev, double *histogram, double delta_r, int rdf_count, double rho)
{    // Calculate RDF
    int i_global = threadIdx.x + blockIdx.x*blockDim.x;
    int k;
    if(i_global<ri){
        r[i_global] = delta_r * (i_global + 0.5);  // Midpoint of the bin
        for (k=0;k<NP;k++){
        rdf_dev[i_global] += histogram[i_global+k*ri]/(2*M_PI*r[i_global]*delta_r*rho*NP*(double)rdf_count);
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

void output_Measure(double *measure_time, double *MSD, double *ISF, double *count, int time_count, int eq_count, int ri, double *r, double *rdf_host, int si, double *q_host, double *Sq_host, int rdf_count, double omega, double tau_p, double rho, int select_potential){
  char filename[128], filename2[128], filename3[128];
  mkdir("data",0755);

  if(select_potential==0){
    sprintf(filename,"data/LJ_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/LJ_rdf_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename3,"data/LJ_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
  if(select_potential == 1){
    sprintf(filename,"data/WCA_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/WCA_rdf_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename3,"data/WCA_Sq_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
 if(select_potential == 2){
    sprintf(filename,"data/HP_MSD_ISF_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
    sprintf(filename2,"data/HP_rdf_MPI_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
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
    file << x[i] << " " << y[i]<< " " << theta[i] << " " << vx[i] << " " << vy[i] << " " << t << endl;
  file.close();
  count++;
  
  return 0;
}

int pot_output(double *pot){



  return 0;
}

/*
int output_t(double *x,double *y, double t, double time_stamp, double x_corr,double y_corr, double K)
{
  int i;
  char filename[64];
  char filename2[64];
  FILE *fp;
  FILE *fp2;
  char foldername[64];
  sprintf(foldername, "position");
  mkdir(foldername, 0755);
  sprintf(filename,"%s/time_coord_%.3f.dat", foldername, temp);
  fp=fopen(filename,"a+");

  for(i=0;i<NP;i++)
    fprintf(fp,"%f\t%f\t%f\n",t-time_stamp, x[i]-x_corr, y[i]-y_corr);
  fclose(fp);

  sprintf(filename2,"%s/time_energy_%.3f.dat", foldername, temp);
  fp2=fopen(filename2,"a+");
  fprintf(fp2,"%f\t%f\t%f\t%f\n", t-time_stamp, x_corr, y_corr, K);
  fclose(fp2);

  return 0;
}
*/

/*
void output(double *x,double *y,double *vx,double *vy,double *a){
  static int count=1;
  char filename[128];
  sprintf(filename,"coord_%.d.dat",count);
  ofstream file;
  file.open(filename);
  double temp0=0.0;
  
  for(int i=0;i<NP;i++){
    file << x[i] << " " << y[i]<< " " << a[i] << endl;
    temp0 += 0.5*(vx[i]*vx[i]+vy[i]*vy[i]);
    //  cout <<vx[i]<<endl;
  }
  file.close();

  cout<<"temp="<< temp0/NP <<endl;
  count++;
}*/


int main(int argc, char** argv){
  double *x,*xi, *x_dev, *xi_dev, *x0_dev,*vx,*vx_dev, *y, *y_dev, *yi, *yi_dev, *y0_dev, *vy, *a, *dx_dev,*dy_dev,*vy_dev,*a_dev,*fx_dev,*fy_dev;
  double *theta, *theta_dev; 
  double *x_corr_dev, *y_corr_dev, *x_corr, *y_corr, corr_x=0., corr_y=0., *corr_x_dev, *corr_y_dev;
  int *list_dev,*gate_dev, time_count, init_count;
  double *MSD_dev, *MSD_host, *ISF_dev,*ISF_host;
  double sampling_time, time_stamp=0.;
  int ri=1000, si = 500;
  double delta_r = 0.01;
  double *histogram, *rdf_dev, *rdf_host, *r_dev, *r_host;
  double *Sq_dev, *Sq_host, *q_dev, *q_host;
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

  int position_interval;
  if (omega < 3.0){
    dt = 0.001;
    timemax = 1.e5;
    eq_count = 6;
    position_interval = 100./dt;
  }

  else{
    dt = 0.01;
    timemax = 1.e6;
    eq_count = 60;
    position_interval = 1000./dt;
  }

  position_start = timemax/2.0;

  printf("Pe = %.4f, omega = %.4f, phi = %.4f, potential = %d, dt = %.4f, timemax = %d, eq_count = %d, position_start = %.1f \n", tau_p, omega, rho, select_potential, dt, timemax, eq_count, position_start);

  int time_interval = 10./dt;
  double sec; //measurred time
  double noise_intensity = sqrt(2.*zeta_zero*temp*dt_initial); //Langevin noise intensity.  
  double anglar_noise_intensity = sqrt(2./ tau_p * dt); 
  double LB = sqrt(M_PI*(1.0*1.0)*(double)NP/(4.* rho));  //system size
  int np,myrank; // the variable for MPI
  
  //MPI initialize lines
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  // nodes of MPI take each GPU
  int gpu_id = myrank;
  cudaSetDevice(gpu_id); 

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
  rdf_host  = (double*)malloc(NB*NT*sizeof(double));
  r_host  = (double*)malloc(NB*NT*sizeof(double));
  Sq_host  = (double*)malloc(NB*NT*sizeof(double));
  q_host  = (double*)malloc(NB*NT*sizeof(double));
  theta = (double*)malloc(NB*NT*sizeof(double));
  
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
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&x_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&y_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&corr_x_dev, sizeof(double));
  cudaMalloc((void**)&corr_y_dev, sizeof(double));
  cudaMalloc((void**)&MSD_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&ISF_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&rdf_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&histogram, ri * NP * sizeof(double));
  cudaMalloc((void**)&r_dev,  NB * NT * sizeof(curandState)); 
  cudaMalloc((void**)&Sq_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&q_dev,  NB * NT * sizeof(curandState));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&gate_dev, sizeof(int)); // for update 
  cudaMalloc((void**)&list_dev,  NB * NT * NN* sizeof(int)); 
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState)); 
   
   //for(int k =0 ; k<omega_number; k++)
  {
  time_stamp = 0.;
  sampling_time = 5.*dt;
  time_count = 0;
  
	for (int i=0; i<ri;i++){
        rdf_MPI[i]=0.;
    }
	for (int i=0; i<si;i++){
	    Sq_MPI[i]=0.;
   }

  for(double t=dt;t<timemax;t+=dt){
    if(int(t/dt)== int((sampling_time + time_stamp)/dt)){
	sampling_time *=pow(10,0.1);
	sampling_time=int(sampling_time/dt)*dt;
  //printf("%.5f	%d\n",t, time_count);
	time_count++;
	if(sampling_time > sampling_time_max/pow(10.,0.1)){
	  time_stamp=0.;
	  sampling_time = 5.*dt;
	  break;
      }
    }
  } 
  
  time_stamp = 0.;
  int rdf_count=0;
  int max_count = time_count;
  double measure_time[time_count], MSD[time_count], count[time_count], ISF[time_count], MSD_MPI[time_count], ISF_MPI[time_count];
    //Make the measure time table
    time_count = 0.;
    for(double t=dt;t<timemax;t+=dt){
      if(int(t/dt)== int((sampling_time + time_stamp)/dt)){
        count[time_count] = 0.;
        MSD[time_count] = 0.;
        ISF[time_count] = 0.;
        measure_time[time_count] = t - time_stamp;
    sampling_time *=pow(10,0.1);
    sampling_time=int(sampling_time/dt)*dt;
    //if(myrank==0){printf("%.5f	%d\n", measure_time[time_count], time_count);}
    time_count++;
    if(sampling_time > sampling_time_max/pow(10.,0.1)){
      time_stamp=0.;//reset the time stamp
      sampling_time=5.*dt; //reset the time sampling_time
      break;
        }
      }
    }

	for (int i=0; i<max_count;i++){
		MSD_MPI[i] = 0.;
		ISF_MPI[i] = 0.;;
   	}
  int rn_seed = rand()+myrank; 
  setCurand<<<NB,NT>>>(rn_seed, state); // Construction of the cudarand state.  

  init_array_rand<<<NB,NT>>>(x_dev,LB,state);
  init_array_rand<<<NB,NT>>>(y_dev,LB,state);
  init_binary<<<NB,NT>>>(a_dev,1.0, 1.0);
  init_array<<<NB,NT>>>(vx_dev,0.);
  init_array<<<NB,NT>>>(vy_dev,0.);
  init_array<<<NB,NT>>>(x_corr_dev,0.);
  init_array<<<NB,NT>>>(y_corr_dev,0.);
  init_array<<<NB,NT>>>(ISF_dev,0.);
  init_array<<<NB,NT>>>(MSD_dev,0.);
  ini_gate_kernel<<<1,1>>>(gate_dev,1);//block variable
  cudaMemcpy(a, a_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev); 
  
  for(double t=0;t<timeeq;t+=dt_initial){
    // cout<<t<<endl;
    calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev);
    langevin_kernel<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,state,0.0,LB);
    disp_gate_kernel<<<NB,NT>>>(LB,vx_dev,vy_dev,dx_dev,dy_dev,gate_dev, dt_initial); //for auto-list method
    update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev);
  }
  
  measureTime();
  time_count = 0;
  init_count = 0;

  /*
  if(myrank==0){
    //cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    //cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(x, x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(theta, theta_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    output(x,y,theta,vx,vy,a,0.);
    }
  */

  // cABP simulation start!!!!!!!!!
  char pot_file[128];
  mkdir("potential",0755);
    if(select_potential==0){
    sprintf(pot_file,"potential/LJ_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
  if(select_potential == 1){
    sprintf(pot_file,"potential/WCA_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
 if(select_potential == 2){
    sprintf(pot_file,"potential/HP_Pe%.4f_omega%.4f_rho%.4f.dat",tau_p,omega,rho);
  }
  FILE *fp4;
  fp4 = fopen(pot_file,"w+");



  for(double t=dt;t<timemax;t+=dt){
    if(select_potential == 0) {calc_force_LJ_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev, pot_dev);}
    if(select_potential == 1) {calc_force_WCA_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev,pot_dev);}
    if(select_potential == 2) {calc_force_kernel_HP<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev,pot_dev);}
    langevin_kernel_cABP<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,theta_dev, fx_dev, fy_dev,state, anglar_noise_intensity, LB, omega, dt);
    com_correction<<<NB,NT>>>(x_dev, y_dev, x_corr_dev, y_corr_dev, LB);
    //RDF, Sq measure
    int rounded_t = int(t/dt + 0.5);
    if((int)(t / dt)== int((sampling_time + time_stamp)/dt)){
	    count[time_count]++;//measure count at each logarithmic times
      //cudaMemcpy(x_corr, x_corr_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      //cudaMemcpy(y_corr, y_corr_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      corr_x = thrust::reduce(thrust::device_pointer_cast(x_corr_dev), thrust::device_pointer_cast(x_corr_dev + NB * NT),0.0,thrust::plus<double>());
      corr_y = thrust::reduce(thrust::device_pointer_cast(y_corr_dev), thrust::device_pointer_cast(y_corr_dev + NB * NT),0.0,thrust::plus<double>());
      //calc_com(x_corr, y_corr, &corr_x, &corr_y);
      cudaMemcpy(corr_x_dev, &corr_x, sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(corr_y_dev, &corr_y, sizeof(double),cudaMemcpyHostToDevice);
      //cudaDeviceSynchronize();

      if(init_count >= eq_count){
        if(time_count==0){
          //cudaDeviceSynchronize();
          copy_kernel2<<<NB,NT>>>(xi_dev, yi_dev, x_dev,y_dev, corr_x_dev, corr_y_dev);
        }
        MSD_ISF_device<<<NB,NT>>>(x_dev, y_dev, xi_dev, yi_dev, corr_x_dev, corr_y_dev, MSD_dev, ISF_dev, LB);
        double MSD_temp = thrust::reduce(thrust::device_pointer_cast(MSD_dev), thrust::device_pointer_cast(MSD_dev + NB * NT),0.0,thrust::plus<double>()); //the variable for check in real-time
        double ISF_temp = thrust::reduce(thrust::device_pointer_cast(ISF_dev), thrust::device_pointer_cast(ISF_dev + NB * NT),0.0,thrust::plus<double>()); //If you don't need to check, using just sub-routines
        
        MSD[time_count] += MSD_temp;//reduce the MSD from each particles
        ISF[time_count] += ISF_temp; //reduce the ISF from each particles

        printf("%d %d	%.4f	%.4f  %.4f  %.4f  %.4f\n", time_count, init_count, measure_time[time_count], MSD_temp, ISF_temp, corr_x, corr_y);
        //cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
        //cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
        //double K = calc_K(vx,vy);
        //output_t(x, y, t, time_stamp, corr_x, corr_y, K);

      }

      //else {printf("time = %.4f\n", measure_time[time_count]);}
	    sampling_time *=pow(10,0.1);
	    sampling_time=int(sampling_time/dt)*dt;
	    time_count++;
      
	    if(sampling_time > sampling_time_max/pow(10.,0.1)){
	      time_stamp=t; //memory of initial measure time for logarithmic sampling
	      sampling_time = 5.*dt; //reset the time sampling_time
	      init_count++;
        time_count = 0;
      }
    }

   // if( rounded_t % 2000 == 0 && t >= position_start){
   //       } 

    disp_gate_kernel_cABP<<<NB,NT>>>(LB,vx_dev,vy_dev,dx_dev,dy_dev,gate_dev, select_potential, dt); //for auto-list method
    // cudaDeviceSynchronize(); // for printf in the device.
    update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev);
    // cudaDeviceSynchronize();
    // cout <<t<<endl;
    MPI_Barrier(MPI_COMM_WORLD);

    if( rounded_t % time_interval == 0 && myrank==0){
      printf("time = %.2f\n",t);
      pot = thrust::reduce(thrust::device_pointer_cast(pot_dev), thrust::device_pointer_cast(pot_dev + NB * NT),0.0,thrust::plus<double>());
      
      fprintf(fp4, "%.4f %.6f \n", t, pot);
      }

    
    if( rounded_t % position_interval == 0 && myrank==0 && t >= position_start){
      cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(x, x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(y, y_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(theta, theta_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      output(x,y,theta,vx,vy,a,t, omega,tau_p, rho, select_potential );
	  calculate_rdf<<<NB,NT>>>(x_dev, y_dev, LB, delta_r, r_dev, ri, histogram);
      calculate_structure_factor<<<NB,NT>>>(x_dev, y_dev, LB, q_dev, Sq_dev, si);
      rdf_count++;
      }
    MPI_Barrier(MPI_COMM_WORLD);  
  } 

  fclose(fp4);

  sec = measureTime()/1000.;
  cout<<"time(sec):"<<sec<<endl;
  //cudaMemcpy(x,   x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
 //cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  //cudaMemcpy(y,   y_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  //cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  //cudaMemcpy(a, a_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
 
  // data gather/ng and summation to zero node
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&MSD,&MSD_MPI,max_count,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&ISF,&ISF_MPI,max_count,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

 //
 if(myrank==0){
  for (int i=0; i<max_count;i++){
	MSD[i] = MSD_MPI[i]/(double)np;
	ISF[i] = ISF_MPI[i]/(double)np;
   }
 }

  MPI_Barrier(MPI_COMM_WORLD);
  reduce_rdf<<<NB,NT>>>(ri,r_dev,rdf_dev,histogram, delta_r, rdf_count, rho);
  cudaMemcpy(rdf_host, rdf_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(r_host, r_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(Sq_host, Sq_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(q_host, q_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(Sq_host,&Sq_MPI,si,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(rdf_host,&rdf_MPI,ri,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if(myrank==0){
    for (int i=0; i<ri;i++){
        rdf_host[i] = rdf_MPI[i]/(double)np;
    }
    for (int i=0; i<si;i++){
	    Sq_host[i] = Sq_MPI[i]/(double)np;
   }
  output_Measure(measure_time, MSD, ISF, count, max_count, eq_count, ri, r_host, rdf_host, si, q_host, Sq_host, rdf_count, omega, tau_p, rho, select_potential); 
  } 

} //omega for loop end
  //output(x,y,vx,vy,a);
  
  MPI_Barrier(MPI_COMM_WORLD);
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
  cudaFree(histogram);
  cudaFree(r_dev);
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
  MPI_Finalize(); 
  return 0;
}

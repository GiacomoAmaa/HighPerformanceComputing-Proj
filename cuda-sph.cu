/****************************************************************************
 *
 * sph.c -- Smoothed Particle Hydrodynamics
 *
 * https://github.com/cerrno/mueller-sph
 *
 * Copyright (C) 2016 Lucas V. Schuermann
 * Copyright (C) 2022 Moreno Marzolla
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ****************************************************************************/
/*********************** GIACOMO AMADIO 971304 ******************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLKDIM 1024

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

const float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
const float REST_DENS = 300;    // rest density
const float GAS_CONST = 2000;   // const for equation of state
const float H = 16;             // kernel radius
const float EPS = 16;           // equal to H
const float MASS = 2.5;         // assume all particles have the same mass
const float VISC = 200;         // viscosity constant
const float DT = 0.0007;        // integration timestep
const float BOUND_DAMPING = -0.5;

const int MAX_PARTICLES = 20000;

// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000

const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

/* Particles data structure; stores position, velocity, and force for
 * integration stores density (rho) and pressure values for SPH.
 *
 * Structure of arrays used to coalesce gpu memory accesses
 */
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *press;   // density, pressure
} particles_t;

particles_t h_p, d_p;

int n_particles = 0;    // number of currently active particles

/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Set initial position of nth particle to (x, y); initialize all
 * other attributes to default values (zeros).
 */
void init_particle( int num_p, float x, float y )
{
    h_p.x[num_p] = x;
    h_p.y[num_p] = y;
    h_p.vx[num_p] = h_p.vy[num_p] = 0.0;
    h_p.fx[num_p] = h_p.fy[num_p] = 0.0;
    h_p.rho[num_p] = 0.0;
    h_p.press[num_p] = 0.0;
}

/**
 * Initialize the SPH model with `n` particles. The caller is
 * responsible for allocating `particle` fields arrays of size
 * `MAX_PARTICLES`.
 */
void init_sph( int n )
{
    n_particles = 0;

    printf("Initializing with %d particles\n", n);
    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(n_particles, x+jitter, y);
                n_particles++;
            } else {
                return;
            }
        }
    }
    assert(n_particles == n);
}

__global__ void compute_density_pressure( particles_t p, int n )
{
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
     * to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
     * et al.
     */
    const float POLY6 = 4.0 / (M_PI * pow(H, 8));

    __shared__ float rho[BLKDIM];                   // shared memory for efficient reduction
    __shared__ float pi_x[BLKDIM], pi_y[BLKDIM];    // (each element accessed n times) shared memory is used for more efficient accesses

    const int lindex = threadIdx.x;               // local index to access shared memory 
    const int bstart = blockIdx.x * BLKDIM;       // starting index of particles partition (specific to each block)
    const int bend = (blockIdx.x + 1)* BLKDIM;    // final index of the partition
    int bsize;

    /* every thread contributes to fill up shared memory */
    if( bstart + lindex < n ){
        pi_x[lindex] = p.x[bstart + lindex];
        pi_y[lindex] = p.y[bstart + lindex];
    }

    /* foreach particle of this partition that is in domain */
    for ( int global_i = bstart , local_i = 0; global_i < n && global_i < bend; global_i++, local_i++) {
        /* reset reduction variables*/
        bsize = blockDim.x / 2;
        rho[lindex] = 0.0;

        /* when all threads are done */
        __syncthreads();
        
        /* each thread in domain takes care of computing his partial result*/
        for (int j = lindex; j < n; j += BLKDIM) {

            const float dx = p.x[j] - pi_x[local_i];
            const float dy = p.y[j] - pi_y[local_i];
            const float d2 = dx*dx + dy*dy;

            if (d2 < HSQ) {
                rho[lindex] += MASS * POLY6 * pow(HSQ - d2, 3.0);
            }
        }

        /* wait for all threads to finish the operation */
        __syncthreads();

        /* All threads within the block cooperate to compute the sum of partial results */
        while ( bsize > 0 ) {
            if ( lindex < bsize ) {
                rho[lindex] += rho[lindex + bsize];
            }
            bsize = bsize / 2; 
            /* threads must synchronize before performing the next
            reduction step */
            __syncthreads(); 
        }

        /* only one thread per block writes back the result on global memory */
        if ( 0 == lindex ) {
            p.rho[global_i] = rho[0];
            p.press[global_i] = GAS_CONST * (rho[0] - REST_DENS);
        }
    }
}

__global__ void compute_forces( particles_t p, int n )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
     * to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
     * et al.
     */
    const float SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * pow(H, 5));
    const float EPS = 1e-6;

    /* shared memory for efficient reduction */ 
    __shared__ float fpress_x[BLKDIM], fpress_y[BLKDIM];
    __shared__ float fvisc_x[BLKDIM], fvisc_y[BLKDIM];

    __shared__ float pi_x[BLKDIM], pi_y[BLKDIM]; // (each element accessed n times) shared memory is used for more efficient accesses

    const int lindex = threadIdx.x;             // local index to access shared memory 
    const int bstart = blockIdx.x * BLKDIM;     // starting index of particles partition (specific to each block)
    const int bend = (blockIdx.x + 1)* BLKDIM;  // final index of the partition
    int bsize;

    /* every thread contributes to fill up shared memory */
    if( bstart + lindex < n ){
        pi_x[lindex] = p.x[bstart + lindex];
        pi_y[lindex] = p.y[bstart + lindex];
    }

    /* foreach particle of this partition that is in domain */
    for (int global_i = bstart, local_i = 0; global_i < n && global_i < bend; global_i++, local_i++) {
        /* reset reduction variables*/
        bsize = blockDim.x / 2;
        fpress_x[lindex] = fpress_y[lindex] = fvisc_x[lindex] = fvisc_y[lindex] = 0.0;

        /* when all threads are done */
        __syncthreads();
        
        /* each thread in domain takes care of computing his partial results*/
        for (int j = lindex; j < n; j += BLKDIM) {

            if (global_i == j)
                continue;

            const float dx = p.x[j] - pi_x[local_i];
            const float dy = p.y[j] - pi_y[local_i];
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x[lindex] += -norm_dx * MASS * (p.press[global_i] + p.press[j]) / (2 * p.rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                fpress_y[lindex] += -norm_dy * MASS * (p.press[global_i] + p.press[j]) / (2 * p.rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x[lindex] += VISC * MASS * (p.vx[j] - p.vx[global_i]) / p.rho[j] * VISC_LAP * (H - dist);
                fvisc_y[lindex] += VISC * MASS * (p.vy[j] - p.vy[global_i]) / p.rho[j] * VISC_LAP * (H - dist);
            }
        }

        /* wait for all threads to finish the operation */
        __syncthreads();

        /* All threads within the block cooperate to compute sums of partial results*/
        while ( bsize > 0 ) {
            if ( lindex < bsize ) {
                fpress_x[lindex] += fpress_x[lindex + bsize];
                fpress_y[lindex] += fpress_y[lindex + bsize];
                fvisc_x[lindex] += fvisc_x[lindex + bsize];
                fvisc_y[lindex] += fvisc_y[lindex + bsize];
            }
            bsize = bsize / 2; 
            /* threads must synchronize before performing the next
            reduction step */
            __syncthreads(); 
        }

        /* only one thread per block writes back the result on global memory */
        if( 0 == lindex){
            const float fgrav_x = Gx * MASS / p.rho[global_i];
            const float fgrav_y = Gy * MASS / p.rho[global_i];
            p.fx[global_i] = fpress_x[0] + fvisc_x[0] + fgrav_x;
            p.fy[global_i] = fpress_y[0] + fvisc_y[0] + fgrav_y;
        }
    }
}

__global__ void integrate( particles_t p, int n )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // assigns each thread a different element

    /* if thread is in domain */
    if(idx < n){
        // forward Euler integration
        p.vx[idx] += DT * p.fx[idx] / p.rho[idx];
        p.vy[idx] += DT * p.fy[idx] / p.rho[idx];
        p.x[idx] += DT * p.vx[idx];
        p.y[idx] += DT * p.vy[idx];

        // enforce boundary conditions
        if (p.x[idx] - EPS < 0.0) {
            p.vx[idx] *= BOUND_DAMPING;
            p.x[idx] = EPS;
        }
        if (p.x[idx] + EPS > VIEW_WIDTH) {
            p.vx[idx] *= BOUND_DAMPING;
            p.x[idx] = VIEW_WIDTH - EPS;
        }
        if (p.y[idx] - EPS < 0.0) {
            p.vy[idx] *= BOUND_DAMPING;
            p.y[idx] = EPS;
        }
        if (p.y[idx] + EPS > VIEW_HEIGHT) {
            p.vy[idx] *= BOUND_DAMPING;
            p.y[idx] = VIEW_HEIGHT - EPS;
        }
    }
}

__global__ void avg_velocities( particles_t p, int n, float *result )
{
    __shared__ float temp[BLKDIM];  // shared memory for efficient reduction

    int lindex = threadIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    /* computes the array of results in shared memory to later perform reduction */
    if ( gindex < n ) {
        temp[lindex] = hypot(p.vx[gindex], p.vy[gindex]) / n;
    } else {
        temp[lindex] = 0.0;
    }

    /* wait for all threads to finish the operation */
    __syncthreads();

    /* All threads within the block cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2; 
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads(); 
    }

    if ( 0 == lindex ) {
        atomicAdd(result, temp[0]);
    }
}

void update( void )
{
    const int n_of_blocks = (n_particles + BLKDIM - 1)/BLKDIM;
    const size_t size = n_particles * sizeof(float);

    /* copy the data structure on the device */
    cudaSafeCall( cudaMemcpy(d_p.x, h_p.x, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.y, h_p.y, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.fx, h_p.fx, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.fy, h_p.fy, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.vx, h_p.vx, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.vy, h_p.vy, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.rho, h_p.rho, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p.press, h_p.press, size, cudaMemcpyHostToDevice) );

    compute_density_pressure<<<n_of_blocks, BLKDIM>>>(d_p, n_particles);
    cudaCheckError();

    compute_forces<<<n_of_blocks, BLKDIM>>>(d_p, n_particles);
    cudaCheckError();

    integrate<<<n_of_blocks, BLKDIM>>>(d_p, n_particles);
    cudaCheckError();

    /* copy the data structure on the host */
    cudaSafeCall( cudaMemcpy(h_p.vx, d_p.vx, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.vy, d_p.vy, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.x, d_p.x, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.y, d_p.y, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.fx, d_p.fx, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.fy, d_p.fy, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.rho, d_p.rho, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(h_p.press, d_p.press, size, cudaMemcpyDeviceToHost) );          
}

int main(int argc, char **argv)
{
    srand(1234);
    int n = DAM_PARTICLES;
    int nsteps = 50;
    double tstart, tend;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    if (n > MAX_PARTICLES) {
        fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
        return EXIT_FAILURE;
    }

    const size_t size = n * sizeof(float);

    /* HOST allocation of `particles` data structure */
    h_p.x = (float*)malloc(size); assert( h_p.x != NULL );
    h_p.y = (float*)malloc(size); assert( h_p.y != NULL );
    h_p.vx = (float*)malloc(size); assert( h_p.vx != NULL );
    h_p.vy = (float*)malloc(size); assert( h_p.vy != NULL );
    h_p.fx = (float*)malloc(size); assert( h_p.fx != NULL );
    h_p.fy = (float*)malloc(size); assert( h_p.fy != NULL );
    h_p.rho = (float*)malloc(size); assert( h_p.rho != NULL );
    h_p.press = (float*)malloc(size); assert( h_p.press != NULL );

    /* DEVICE allocation of `particles` data structure */
    cudaSafeCall( cudaMalloc((void**) &(d_p.x), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.y), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.fx), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.fy), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.vx), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.vy), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.rho), size));
    cudaSafeCall( cudaMalloc((void**) &(d_p.press), size));

    const int n_of_blocks = (n + BLKDIM - 1)/BLKDIM;
    float *d_avg;

    /* allocate memory on the device to store the result */
    cudaSafeCall( cudaMalloc((void **)&d_avg, sizeof(*d_avg)) );

    tstart = hpc_gettime();
    init_sph(n);
    tend = hpc_gettime();
    printf("Build time: %f\n", tend - tstart);

    tstart = hpc_gettime();
    for (int s=0; s<nsteps; s++) {
        float avg = 0.0f;

        update();

        /* Copy the initial result (zero) to the device; this is important
         * since the avg_velocities kernel requires *d_avg to be initially zero.
         */
        cudaSafeCall( cudaMemcpy(d_avg, &avg, sizeof(avg), cudaMemcpyHostToDevice) );

        /* the average velocities MUST be computed at each step, even
         * if it is not shown (to ensure constant workload per
         * iteration)
         */
        avg_velocities<<<n_of_blocks, BLKDIM>>>(d_p, n, d_avg);
        cudaCheckError();

        /* Copy the result from device memory to host memory */
        cudaSafeCall( cudaMemcpy(&avg, d_avg, sizeof(avg), cudaMemcpyDeviceToHost) );
        
        if (s % 10 == 0)
            printf("step %5d, avgV=%f\n", s, avg);
    }
    tend = hpc_gettime();
    printf("Elapsed time: %f\n", tend - tstart);

    /* HOST deallocation `particles` and fields */
    free(h_p.x);
    free(h_p.y);
    free(h_p.fx);
    free(h_p.fy);
    free(h_p.vx);
    free(h_p.vy);
    free(h_p.rho);
    free(h_p.press);

    /* HOST deallocation `particles` and fields */
    cudaFree(d_p.x);
    cudaFree(d_p.y);
    cudaFree(d_p.fx);
    cudaFree(d_p.fy);
    cudaFree(d_p.vx);
    cudaFree(d_p.fy);
    cudaFree(d_p.rho);
    cudaFree(d_p.press);

    return EXIT_SUCCESS;
}

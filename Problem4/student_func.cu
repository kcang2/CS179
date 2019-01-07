//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cmath>
#include <algorithm>

using namespace std;

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void predicate_kernel(const unsigned int* const d_in,
				  unsigned int* const d_ones,
				  unsigned int* const d_zeros,
				  const size_t numElems,
				  const unsigned int bit) 
{	
	int mid = threadIdx.x + blockIdx.x * blockDim.x;

	if (mid < numElems) {
		unsigned int mask = 1 << (bit); // mask from 0...01 to 10...0
		unsigned int input = d_in[mid];
		unsigned int isOne = ((input & mask) == mask) ? 1 : 0; // if the result of masking = 1, the bit is 1
		unsigned int isZero = !((input & mask) == mask) ? 1 : 0;; // else it is 0

		d_ones[mid] = isOne;
		d_zeros[mid] = isZero;
	}
	__syncthreads();
}

__global__ // Same as in Problem Set 3, except we don't store the cdf back into the histogram
void cdf_kernel(unsigned int* const d_cdf,
	const unsigned int* const d_histo,
	const size_t numElems,
	const unsigned int limit)
{
	extern __shared__ unsigned int shared[];
	int dataID = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;


	if (tid < min( int (blockDim.x), int (numElems - blockDim.x * blockIdx.x) ) - 1)
		shared[tid + 1] = d_histo[dataID];
		
	if (tid == 0)
		shared[tid] = 0;
	__syncthreads();

	int nb = 1;
	unsigned int temp;

	for (int i = 0; i < limit; ++i) {
		if ((tid - nb) >= 0)
			temp = shared[tid] + shared[tid - nb];
		else
			temp = shared[tid];
		__syncthreads();
		shared[tid] = temp;
		nb *= 2;
		__syncthreads();
	}

	if (dataID < numElems)
		d_cdf[dataID] = shared[tid];
		
	__syncthreads(); 
}

__global__
void combine_kernel(const unsigned int blockID,
					const unsigned int* const d_histo,
					unsigned int* const d_cdf,
					const size_t numElems) {
	int dataID = blockDim.x * blockID + threadIdx.x;
	int tid = threadIdx.x;
	int prevID = blockDim.x * blockID - 1;
	__shared__ unsigned int newValue;

	if (tid == 0)
		newValue = d_cdf[prevID] + d_histo[prevID]; // 1st thread copies the previous block's cumulative value
	__syncthreads();

	if (dataID < numElems )
		d_cdf[dataID] += newValue; // each cdf value in this block is offset by newValue
	__syncthreads();
}


__global__
void sort_kernel(const unsigned int * const d_zeros,
				 const unsigned int * const d_ones,
				 const unsigned int * const d_cdf0,
				 const unsigned int * const d_cdf1,
				 const unsigned int * const d_inputVals,
				 const unsigned int * const d_inputPos,
				 unsigned int * const d_outputVals,
				 unsigned int * const d_outputPos,
				 const size_t numElems) 
{
	__shared__ unsigned int offset;
	int dataID = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	if (tid == 0)
		offset = d_cdf0[numElems - 1] + d_zeros[numElems - 1]; // offset of 1's cdf 
	__syncthreads();

	unsigned int newPos;
	if (dataID < numElems ) {
		newPos = d_cdf0[dataID] * d_zeros[dataID] + (d_cdf1[dataID] + offset) * d_ones[dataID] ;
		d_outputVals[newPos] = d_inputVals[dataID];
		d_outputPos[newPos] = d_inputPos[dataID];
	}
	__syncthreads();
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
	// d_inputPos corresponds to coords which was created in 
	// HW4.cu (a sequence from 0 to numElems - 1)

	int numThreads = 1024;
	int blockSize = numThreads;
	int gridSize = (numElems + blockSize - 1)/blockSize ;
	unsigned int limit = log2(numThreads); // number of iterations for scan

	unsigned int *d_ones, *d_zeros, *d_cdf1, *d_cdf0;

	checkCudaErrors(cudaMalloc(&d_ones, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_zeros, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_cdf1, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_cdf0, numElems * sizeof(unsigned int)));

	for (int bit = 0; bit < 32; ++bit) { // loop thru all 32 bits in the unsigned int

		predicate_kernel << < gridSize, blockSize >> > (d_inputVals, d_ones, d_zeros, numElems, bit); // Get 0s & 1s histogram
		cdf_kernel << < gridSize, blockSize, blockSize * sizeof(unsigned int) >> > (d_cdf0, d_zeros, numElems, limit); // Get 0s cdf
		cdf_kernel << < gridSize, blockSize, blockSize * sizeof(unsigned int) >> > (d_cdf1, d_ones, numElems, limit); // Get 1s cdf
				
		for (int block = 1; block < gridSize; ++block){ // combine the cdfs for 0 & 1, from 2nd block onwards
			combine_kernel << < 1, blockSize >> > (block, d_zeros, d_cdf0, numElems);
			combine_kernel << < 1, blockSize >> > (block, d_ones, d_cdf1, numElems);
		}
				
		sort_kernel << < gridSize, blockSize >> > (d_zeros, d_ones, d_cdf0, d_cdf1, d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
		checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		
	}
	
	checkCudaErrors(cudaFree(d_ones));
	checkCudaErrors(cudaFree(d_zeros));
	checkCudaErrors(cudaFree(d_cdf1));
	checkCudaErrors(cudaFree(d_cdf0));

}

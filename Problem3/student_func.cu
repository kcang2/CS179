/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <algorithm>
#include <cmath>

using namespace std;

__global__
void shmem_minmax(const float* const d_in, 
		  float* const d_out,
		  const bool isMin, const int maxDataSize )
{
	extern __shared__ float shared[];

	int dataID = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	if (dataID<maxDataSize) // Don't go out-of-bounds in data
		shared[tid] = d_in[dataID]; // Store into shared memory
	__syncthreads(); // Wait for all threads to load their data

// Only need (blockSize/2) number of iterations; Decrement by dividing by 2 OR shift-right 1
for (int s = blockDim.x / 2; s > 0; s >>= 1) { 
	if (tid < s) { // Only let left-half threads do work.
		if (isMin) // A left-half thread reduces with it's right-half equivalent
			shared[tid] = min(shared[tid], shared[tid + s]);
		else
			shared[tid] = max(shared[tid], shared[tid + s]);
	}
	__syncthreads(); // Wait for all threads to reduce.
}

if (tid == 0) // Only let 1st thread store the resultant min/max to global memory
	d_out[blockIdx.x] = shared[tid];
}

void reduce(const float* const d_logLuminance,
	float &minmax_logLum,
	bool isMin,
	const size_t numRows,
	const size_t numCols)
{
	const int numThreads = 1024;
	const int dataSize = numRows * numCols;

	float *d_in, *d_temp;

	for (int inputSize = dataSize; inputSize > 0; inputSize >>= 10) {
		int gridSize = max(inputSize >> 10, 1); // Number of blocks we need = data/number of threads. Divide by 2^10 = shift-right 10
		int blockSize = min(numThreads, inputSize); // Either use max number of threads OR enough threads to cover all remaining data in 1 block
		checkCudaErrors(cudaMalloc(&d_temp, gridSize * sizeof(float))); // Allocate memory for output
		if (inputSize == dataSize) // Dont have to copy data from output to input on first iteration
			shmem_minmax << < gridSize, blockSize, blockSize * sizeof(float) >> > (d_logLuminance, d_temp, isMin, inputSize);
		else {
			shmem_minmax << < gridSize, blockSize, blockSize * sizeof(float) >> > (d_in, d_temp, isMin, inputSize);
			checkCudaErrors(cudaFree(d_in)); // Free memory in input since it will not be used in next iteration
		}
		d_in = d_temp; // The output is now handled as input in the next iteration.
	}
	checkCudaErrors(cudaMemcpy(&minmax_logLum, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_temp)); // Free output once done

}

__global__
void histo_kernel(const float* const d_in,
	unsigned int* const d_out,
	const float min_logLum,
	const float range_logLum,
	const int dataSize,
	const size_t numBins)
{
	// Bin counts in shared memory
	extern __shared__ unsigned int shared1[];

	int dataID = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	// Let the first (numBins) threads initialize the block-local histogram (in shared memory) to 0s
	if (tid < numBins)
		shared1[tid] = 0;
	__syncthreads();

	unsigned int bin; // Data managed by this thread belongs to this bin
	if (dataID < dataSize) {
		// Consider data = min. bin = data - min / range *numBins = 0;
		// But if data = max, bin = data - min / range *numBins = numBins; OUT_F_BOUNDS
		// Hence clip data to be numBins - 1
		bin = min(  (unsigned int)(numBins - 1), (unsigned int) ( (d_in[dataID] - min_logLum) / range_logLum * numBins) ) ; 
		atomicAdd(&(shared1[bin]), 1); // Atomically increment shared bin count
	}
	__syncthreads(); 
	
	// Again, let the first (numBins) threads of each block update global bin count atomically using the shared bins
	if (tid < numBins)
		atomicAdd(&d_out[tid], shared1[tid]);

}

__global__
void cdf_kernel(unsigned int* const d_cdf,
				const size_t numBins,
				const unsigned int limit)
{
	extern __shared__ unsigned int shared2[];
	int dataID = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	// Store data into shared memory. Note that they are right-shifted by 1 since 
	// Hillis-Steele is inclusive but histograms are exclusive.
	// We only manipulate the 2nd element onwards. The 1st element is identity (0).
	// We also disregard the count of the last bin since it's exclusive scan
	
	if (dataID < numBins - 1) 
		shared2[tid+1] = d_cdf[dataID];
	if (tid == 0)
		shared2[tid] = 0;
	__syncthreads();
	
	/* //Inclusive Scan
	if (dataID < numBins)
		shared2[tid] = d_cdf[dataID];
	*/

	int nb = 1; // neighbour = 2^i. Since i=0, neighbour = 1
	unsigned int temp; // Iterative output placeholder for the scan. 
	// Cannot write straight away to shared memory since other threads may need the data held by this thread
	for (int i = 0; i < limit; ++i) {
		if ((tid - nb) >= 0) 
			// Hillis-Steele scan adds the neighbour to the current thread's data
			temp = shared2[tid] + shared2[tid - nb];
		else
			temp = shared2[tid]; // If no neighbour, keep as is.
		__syncthreads(); // Wait for all threads to finish computing their new value
		shared2[tid] = temp; // Store into shared memory
		nb *= 2;		//   2^(i+1) = 2^i * 2^1
		__syncthreads(); // Wait for threads to complete
	}
	

	if (dataID < numBins) // Copy from shared memory to global memory
		d_cdf[dataID] = shared2[tid];
	
	__syncthreads(); // Wait for write to global memory to finish
}

void histogram_and_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	const float min_logLum,
	const float range_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	const int dataSize = numRows * numCols;
	const int numThreads = 1024;
	int blockSize = numThreads;
	// Basically (numBins/blockSize) but it will not floor the value (flooring may lead to insufficient blocks)
	int gridSize = (dataSize + blockSize - 1) / blockSize;

	histo_kernel << < gridSize, blockSize, numBins * sizeof(unsigned int) >> > (d_logLuminance, d_cdf, min_logLum, range_logLum, dataSize, numBins);
		
	// If checking of results is required:
	/*
	unsigned int *h_out;
	h_out = (unsigned int *)malloc( numBins*sizeof(unsigned int) );
	unsigned int accum = 0;
	checkCudaErrors(cudaMemcpy(h_out, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < numBins; ++i){
		if (h_out[i] > 0){
			printf("Bin %i Count: %i\n", i, h_out[i]);
			accum += h_out[i];
		}
	}
	printf("Accumulated %i where total is %i\n", accum, dataSize);
	delete[] h_out;
	*/
	
	// Basically (numBins/blockSize) but it will not floor the value (flooring may lead to insufficient blocks)
	gridSize = (numBins + blockSize - 1) / blockSize; 
	unsigned int limit = log2(numBins); // number of iterations for scan
	cdf_kernel << < gridSize, blockSize, numBins * sizeof(unsigned int) >> > (d_cdf, numBins, limit);


	// If checking of results is required:
	/*
	int testGrid = 1;
	unsigned int testLimit = 3;
	size_t testBins = 8;
	unsigned int test_cdf[] = {1,2,3,4,5,6,7,8};
	unsigned int * test_d_cdf;
	checkCudaErrors(cudaMalloc(&test_d_cdf, testBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(test_d_cdf, test_cdf, testBins * sizeof(unsigned int), cudaMemcpyHostToDevice));
	cdf_kernel << < testGrid, testBins, testBins * sizeof(unsigned int) >> > (test_d_cdf, testBins, testLimit);
	checkCudaErrors(cudaMemcpy(test_cdf, test_d_cdf, testBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(test_d_cdf));
	for (int i = 0; i < testBins; ++i)
		printf("%i\n",test_cdf[i]);
	*/
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMins) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	
	
	reduce(d_logLuminance, min_logLum, true, numRows, numCols);
	reduce(d_logLuminance, max_logLum, false, numRows, numCols);

	// If checking of results is required:
	/*
	printf("Min: %f\n", min_logLum);
	printf("Max: %f\n", max_logLum);
	*/

	float range = max_logLum - min_logLum;
	histogram_and_cdf(d_logLuminance, d_cdf, min_logLum, range, numRows, numCols, numBins);


}

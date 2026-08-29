/* gpu_bandwidth.cu — the streaming-copy ceiling every "% of bandwidth" claim
 * in REPORT.md is measured against. It was cited in the docs for two days
 * without existing in the repository, which made those percentages
 * unreproducible. Best case by construction: a pure float4 copy. A
 * read-modify-write kernel cannot reach it, so utilisation figures computed
 * against this number UNDERSTATE how close a real kernel is to its own limit.
 *
 *   nvcc -O3 -arch=native -o gpu_bandwidth gpu_bandwidth.cu && ./gpu_bandwidth
 */
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(const float4* __restrict__ a, float4* __restrict__ b, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) b[i]=a[i];
}
int main(){
  size_t bytes = 512ull<<20; size_t n = bytes/sizeof(float4);
  float4 *a,*b; cudaMalloc(&a,bytes); cudaMalloc(&b,bytes); cudaMemset(a,1,bytes);
  for(int i=0;i<3;i++) k<<<(n+255)/256,256>>>(a,b,n);
  cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); for(int i=0;i<10;i++) k<<<(n+255)/256,256>>>(a,b,n); cudaEventRecord(e);
  cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e);
  printf("achievable copy bandwidth: %.1f GB/s\n", 10.0*2.0*bytes/(ms/1000.0)/1e9);
  return 0;
}

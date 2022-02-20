__kernel void calculate_rank(
    __global ulong * stakes, 
    __global uint * inLinksCount, __global  uint * outLinksCount,
    __global ulong * outLinksIns, __global  ulong * inLinksOuts,
    __global ulong * inLinksUsers, __global ulong * outLinksUsers,
    __global ulong * inLinksStartIndex, __global  ulong * outLinksStartIndex,
    double dampingFactor, double tolerance,
    __global double * rank, __global double * entropy,
    __global double * karma, __global double * lights) {

}
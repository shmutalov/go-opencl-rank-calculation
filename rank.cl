__kernel void calculate_rank(ulong * stakes, 
    uint * inLinksCount, uint * outLinksCount,
    ulong * outLinksIns, ulong * inLinksOuts,
    ulong * inLinksUsers, ulong * outLinksUsers,
    ulong * inLinksStartIndex, ulong * outLinksStartIndex,
    double dampingFactor, double tolerance,
    __global double * rank, __global double * entropy,
    __global double * karma, __global double * lights) {

}
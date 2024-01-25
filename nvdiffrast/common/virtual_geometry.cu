#include "../common/common.h"
#include "../common/rasterize.h"
#include "virtual_geometry.h"


static __device__ __forceinline__ bool IsOutSidePlane(const Plane &plane, float x, float y, float z)
{
    return plane.Normal[0]*x + plane.Normal[1]*y + plane.Normal[2]*z + plane.Distance < 0;
}
static __device__ __forceinline__ bool IsBoxOutSidePlane(const Plane &plane, const BoundingBox &AABB)
{
    return 
        IsOutSidePlane(plane, AABB.XMin, AABB.YMin, AABB.ZMin) && 
        IsOutSidePlane(plane, AABB.XMin, AABB.YMin, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMin, AABB.YMax, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMin, AABB.YMax, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMin, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMin, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMax, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMax, AABB.ZMax);
}

static __device__ __forceinline__ bool IsBoxOutSideFrustum(const ViewFrustum& Frustum, const BoundingBox &AABB)
{
    return 
        IsBoxOutSidePlane(Frustum.Planes[0], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[1], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[2], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[3], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[4], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[5], AABB);
}

__global__ void VirtualGeometryFrustumCullKernal(VirtualGeometryFrustumCullParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z; 
    int cluster_index = px + py*blockDim.x*gridDim.x;
    if (cluster_index >= p.numClusters)
        return;

    const ViewFrustum& frustum = p.frustums[pz];
    // const BoundingBox& AABB = p.AABBs[cluster_index];
    BoundingBox AABB = p.AABBs[cluster_index];

    bool culled = IsBoxOutSideFrustum(frustum, AABB);

    bool* pCulled = p.culled + p.numClusters*pz + cluster_index;
    *pCulled = culled;
}

//------------------------------------------------------------------------
// Accumulate gradients kernel.
__global__ void VirtualGeometryAggregateGradKernel(const VirtualGeometryAccumulateGradParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int group_index = px + py*blockDim.x*gridDim.x;
    if (group_index >= p.numGroups)
        return;

    int offset = p.offsetGroups[group_index];
    int* pGroup = p.matchingVerts + offset;
    int numVerts = (p.offsetGroups[group_index+1] - offset) / 2;

    for (int j = 0; j < p.numAttr; j++)
    {
        float grad = 0.f;
        for (int i = 0; i < numVerts; i++)
        {
            int cluster_index = pGroup[i*2];
            int vertex_index = pGroup[i*2+1];
            int ptr = vertex_index * p.numAttr + j;
            if (p.grad[cluster_index])
                grad += p.grad[cluster_index][ptr];
        }

        for (int i = 0; i < numVerts; i++)
        {
            int cluster_index = pGroup[i*2];
            int vertex_index = pGroup[i*2+1];
            int ptr = vertex_index * p.numAttr + j;
            if (p.grad[cluster_index])
                p.grad[cluster_index][ptr] = grad;
        }
    }
    
}
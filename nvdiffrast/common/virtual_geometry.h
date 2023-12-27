#pragma once
#include "virtual_geometry_partition.h"
struct Plane 
{
    float Normal[3];
    float Distance;
};
struct BoundingBox
{
    float XMin, YMin, ZMin;
    float XMax, YMax, ZMax;
};
struct ViewFrustum
{
    Plane Planes[6];
};

struct VirtualGeometryFrustumCullParams
{
    const ViewFrustum*      frustums;                       // The frustum of each minibatch
    const BoundingBox*      AABBs;                          // The axis-aligned bounding box of each cluster
    bool*                   culled;                         // Outgoing culling result, culled[n][i]==true means that for the n'th frustum the i'th cluster is culled.
    int                     n;                              // Minibatch size.
    int                     numClusters;                    // The number of clusters

};

//------------------------------------------------------------------------
// CUDA forward rasterizer shader kernel params.

struct VirtualGeometryRasterizeCudaFwdShaderParams
{
    const float**   pos;            // Vertex positions for each cluster.
    const int**     tri;            // Triangle indices for each cluster.
    const int*      in_idx;         // Cluster idx and triangle idx buffer from rasterizer.
    const int*      mat_ids;        // Material ID for each cluster.
    float*          out;            // Main output buffer.
    float*          out_db;         // Bary pixel gradient output buffer.
    short*          out_mat;        // Material ID output buffer.
    int             numTriangles;   // Number of triangles for each cluster. All clusters must have the same number of triangles.
    int*            numVertices;    // Number of vertices for each cluster.
    int             numClusters;    // Number of clusters.
    int             width;          // Image width.
    int             height;         // Image height.
    int             depth;          // Size of minibatch.
    int             instance_mode;  // 1 if in instance rendering mode.
    float           xs, xo, ys, yo; // Pixel position to clip-space x, y transform.
};

//------------------------------------------------------------------------
// Gradient CUDA kernel params.

struct VirtualGeometryRasterizeGradParams
{
    const float**   pos;            // Incoming position buffer for each cluster.
    const int**     tri;            // Incoming triangle buffer for each cluster.
    const float*    out;            // Rasterizer output buffer.
    const float*    dy;             // Incoming gradients of rasterizer output buffer.
    const float*    ddb;            // Incoming gradients of bary diff output buffer.
    float**         grad;           // Outgoing position gradients.
    int             numTriangles;   // Number of triangles for each cluster. All clusters must have the same number of triangles.
    int*            numVertices;    // Number of vertices for each cluster.
    int             numClusters;    // Number of clusters.
    int             width;          // Image width.
    int             height;         // Image height.
    int             depth;          // Size of minibatch.
    int             instance_mode;  // 1 if in instance rendering mode.
    float           xs, xo, ys, yo; // Pixel position to clip-space x, y transform.
};

//------------------------------------------------------------------------

struct VirtualGeometryAccumulateGradParams
{
    float**         grad;               // Outgoing position gradients.
    int*            matchingVerts;      // something like |cid0,vid0,cid1,vid1|...The cids and vids between two vertical bar indicates a group of matching vertices
    int*            offsetGroups;       // The offset of each group in matchingVerts
    int             numGroups;
    int             numAttr;
};
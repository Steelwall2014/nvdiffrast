#pragma once
/**
 * Most of the cluster building operations are adapted from Unreal Engine 5.2.
 * The division of mesh uses METIS graph partioning library.
*/ 

#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include "hash_table.h"
#include "metis.h"


struct vec2
{
    float x, y;
    bool operator==(const vec2& Other) const { return x == Other.x && y == Other.y; }
    float& operator[](size_t i) { return (&x)[i]; }
    const float& operator[](size_t i) const { return (&x)[i]; }
    static constexpr size_t length() { return 2; }
    static vec2 nodata;
};

struct vec3
{
    float x, y, z;
    bool operator==(const vec3& Other) const { return x == Other.x && y == Other.y && z == Other.z; }
    float& operator[](size_t i) { return (&x)[i]; }
    const float& operator[](size_t i) const { return (&x)[i]; }
    static constexpr size_t length() { return 3; }
	float dot(const vec3& other) const { return x*other.x + y*other.y + z*other.z; }
    static vec3 nodata;
};
inline vec3 operator+(const vec3& A, const vec3& B)
{
	return vec3{A.x+B.x, A.y+B.y, A.z+B.z};
}
inline vec3 operator/(const vec3& A, float B)
{
	return vec3{A.x/B, A.y/B, A.z/B};
}
inline float distance(const vec3& A, const vec3& B)
{
	return std::sqrt((A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z));
}

struct vec4
{
    float x, y, z, w;
    bool operator==(const vec4& Other) const { return x == Other.x && y == Other.y && z == Other.z && w == Other.w; }
    float& operator[](size_t i) { return (&x)[i]; }
    const float& operator[](size_t i) const { return (&x)[i]; }
    static constexpr size_t length() { return 4; }
    static vec4 nodata;
};

//------------------------------------------------------------------------
// Hash utils
static inline uint32_t MurmurFinalize32(uint32_t Hash)
{
	Hash ^= Hash >> 16;
	Hash *= 0x85ebca6b;
	Hash ^= Hash >> 13;
	Hash *= 0xc2b2ae35;
	Hash ^= Hash >> 16;
	return Hash;
}

static inline uint64_t MurmurFinalize64(uint64_t Hash)
{
	Hash ^= Hash >> 33;
	Hash *= 0xff51afd7ed558ccdull;
	Hash ^= Hash >> 33;
	Hash *= 0xc4ceb9fe1a85ec53ull;
	Hash ^= Hash >> 33;
	return Hash;
}

static inline uint32_t Murmur32( std::initializer_list< uint32_t > InitList )
{
	uint32_t Hash = 0;
	for( auto Element : InitList )
	{
		Element *= 0xcc9e2d51;
		Element = ( Element << 15 ) | ( Element >> (32 - 15) );
		Element *= 0x1b873593;
    
		Hash ^= Element;
		Hash = ( Hash << 13 ) | ( Hash >> (32 - 13) );
		Hash = Hash * 5 + 0xe6546b64;
	}

	return MurmurFinalize32(Hash);
}

static inline uint32_t HashPosition( vec3 Position )
{
	union { float f; uint32_t i; } x;
	union { float f; uint32_t i; } y;
	union { float f; uint32_t i; } z;

	x.f = Position.x;
	y.f = Position.y;
	z.f = Position.z;

	return Murmur32( {
		Position.x == 0.0f ? 0u : x.i,
		Position.y == 0.0f ? 0u : y.i,
		Position.z == 0.0f ? 0u : z.i
	} );
}

//------------------------------------------------------------------------
// Cluster mesh

static inline uint32_t Cycle3( uint32_t Value )
{
	uint32_t ValueMod3 = Value % 3;
	uint32_t Value1Mod3 = ( 1 << ValueMod3 ) & 3;
	return Value - ValueMod3 + Value1Mod3;
}

// Find edge with opposite direction that shares these 2 verts.
/*
	  /\
	 /  \
	o-<<-o
	o->>-o
	 \  /
	  \/
*/
class EdgeHash
{
public:
	EdgeHash()
		: HashTable(1)
	{}
	EdgeHash(size_t Num)
		: HashTable(1 << (int)std::log2( Num ), Num)
	{}
    // std::unordered_multimap<uint32_t, uint32_t> HashTable;
    // folly::AtomicHashMap<uint32_t, std::unique_ptr<uint32_t[]>> HashTable2;
    FHashTable HashTable;

	// template< typename FGetPosition >
	// void Add( idx_t EdgeIndex, FGetPosition&& GetPosition )
	// {
	// 	const vec3& Position0 = GetPosition( EdgeIndex );
	// 	const vec3& Position1 = GetPosition( Cycle3( EdgeIndex ) );
				
	// 	uint32_t Hash0 = HashPosition( Position0 );
	// 	uint32_t Hash1 = HashPosition( Position1 );
	// 	uint32_t Hash = Murmur32( { Hash0, Hash1 } );

	// 	HashTable.Add_Concurrent( Hash, EdgeIndex );
	// 	HashTable.insert( {Hash, EdgeIndex} );
	// }

	template< typename FGetPosition >
	void Add_Concurrent( idx_t EdgeIndex, FGetPosition&& GetPosition )
	{
		const vec3& Position0 = GetPosition( EdgeIndex );
		const vec3& Position1 = GetPosition( Cycle3( EdgeIndex ) );
				
		uint32_t Hash0 = HashPosition( Position0 );
		uint32_t Hash1 = HashPosition( Position1 );
		uint32_t Hash = Murmur32( { Hash0, Hash1 } );

		HashTable.Add_Concurrent( Hash, EdgeIndex );
		// auto iter = HashTable2.find(Hash);
		// if (iter == HashTable2.end())
		// {
		// 	iter = HashTable2.emplace(Hash).first;
		// }
		// iter->second.insertHead(EdgeIndex);
	}

	template< typename FGetPosition, typename FuncType >
	void ForAllMatching( idx_t EdgeIndex, bool bAdd, FGetPosition&& GetPosition, FuncType&& Function )
	{
		const vec3& Position0 = GetPosition( EdgeIndex );
		const vec3& Position1 = GetPosition( Cycle3( EdgeIndex ) );
				
		uint32_t Hash0 = HashPosition( Position0 );
		uint32_t Hash1 = HashPosition( Position1 );
		uint32_t Hash = Murmur32( { Hash1, Hash0 } );
		// auto range = HashTable.equal_range( Hash );
		for( uint32_t OtherEdgeIndex = HashTable.First( Hash ); HashTable.IsValid( OtherEdgeIndex ); OtherEdgeIndex = HashTable.Next( OtherEdgeIndex ) )
		{
            // int32_t OtherEdgeIndex = iter->second;
			if( Position0 == GetPosition( Cycle3( OtherEdgeIndex ) ) &&
				Position1 == GetPosition( OtherEdgeIndex ) )
			{
				// Found matching edge.
				Function( EdgeIndex, OtherEdgeIndex );
			}
		}

		if( bAdd )
			HashTable.Add( Murmur32( { Hash0, Hash1 } ), EdgeIndex );
	}
};

struct Adjacency
{
	std::vector< idx_t >				    Direct;
	std::unordered_multimap< idx_t, idx_t >	Extended;

	Adjacency( size_t Num )
	{
		Direct.resize( Num );
	}

	void	Link( idx_t EdgeIndex0, idx_t EdgeIndex1 )
	{
		if( Direct[ EdgeIndex0 ] < 0 && 
			Direct[ EdgeIndex1 ] < 0 )
		{
			Direct[ EdgeIndex0 ] = EdgeIndex1;
			Direct[ EdgeIndex1 ] = EdgeIndex0;
		}
		else
		{
			Extended.insert( {EdgeIndex0, EdgeIndex1} );
			Extended.insert( {EdgeIndex1, EdgeIndex0} );
		}
	}

	template< typename FuncType >
	void	ForAll( idx_t EdgeIndex, FuncType&& Function ) const
	{
		idx_t AdjIndex = Direct[ EdgeIndex ];
		if( AdjIndex != -1 )
		{
			Function( EdgeIndex, AdjIndex );
		}

		auto range = Extended.equal_range( EdgeIndex );
		for( auto Iter = range.first; Iter != range.second; ++Iter )
		{
			Function( EdgeIndex, Iter->second );
		}
	}
};

static constexpr int DivideAndRoundNearest(int Dividend, int Divisor)
{
    return (Dividend >= 0)
        ? (Dividend + Divisor / 2) / Divisor
        : (Dividend - Divisor / 2 + 1) / Divisor;
}

class GraphPartitioner
{
public:

    GraphPartitioner(idx_t NumTriangles, int MaxPartitionSize=128)
        : MaxPartitionSize(MaxPartitionSize)
    {
        TriIndexes.resize(NumTriangles);
        for (int i = 0; i < NumTriangles; i++)
		{
            TriIndexes[i] = i;
		}
        PartitionIDs.resize(NumTriangles);
        SwappedWith.resize(NumTriangles);
		for (int i = 0; i < NumTriangles; i++)
			SwappedWith[i] = i;
    }

    struct GraphData
    {
        idx_t Offset=0;   // Offset in Indexes
        idx_t Num=0;      // Number of triangles
        std::vector<idx_t> Adjacency;
        std::vector<idx_t> AdjacencyCost;
        std::vector<idx_t> AdjacencyOffset;
    };

    struct Range
    {
        idx_t Begin;
        idx_t End;
		bool operator<( const Range& Other) const { return Begin < Other.Begin; }
    };

	void PartitionStrict( GraphData* Graph, bool MultiThreaded=true );

    void RecursiveBisectGraph(GraphData* Graph)
    {
        GraphData* ChildGraphs[2];
        BisectGraph(Graph, ChildGraphs);
        delete Graph;

        if (ChildGraphs[0] && ChildGraphs[1])
        {
            RecursiveBisectGraph(ChildGraphs[0]);
            RecursiveBisectGraph(ChildGraphs[1]);
        }
    }

	void AddPartition( idx_t Offset, idx_t Num );

    void BisectGraph(GraphData* Graph, GraphData* ChildGraphs[2]);

	std::atomic<int32_t> NumPartitions = 0;
    std::vector<Range> Ranges;      // Each Range indicates a cluster
    std::vector<idx_t> TriIndexes;  // Indexes of triangles

	std::vector< idx_t >		PartitionIDs;
	std::vector< idx_t >		SwappedWith;

	int32_t		MaxPartitionSize = 128;

};

struct VertexData
{
    vec3* Positions = nullptr;

    idx_t* Indices = nullptr;

	idx_t NumTriangles = 0;
	idx_t NumVertices = 0;
};

class Cluster
{
public:
	Cluster() { }

    // std::vector<float> Positions;
    // std::vector<float> TexCoords;
    // std::vector<float> Normals;
    // std::vector<float> Tangents;
    std::vector<int32_t> Indices;
    std::vector<idx_t> OldTriangleIndices;	// old indices for every triangle
	int32_t NumVertices;

};

struct MatchingVertex
{
    int32_t ClusterIndex;
    int32_t AttrIndexInCluster;
    bool operator<(const MatchingVertex& Other) const
    {
        if (ClusterIndex < Other.ClusterIndex)
            return true;
        if (ClusterIndex > Other.ClusterIndex)
            return false;
        if (AttrIndexInCluster < Other.AttrIndexInCluster)
            return true;
        return false;
    }
};

struct ClusterResult
{
    std::vector<Cluster> Clusters;
};

void ClusterTriangles(
        const VertexData& Data, 
        ClusterResult& OutResult,
        int MaxPartitionSize=128);



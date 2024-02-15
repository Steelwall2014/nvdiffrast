#include <algorithm>
#include <float.h>
#include <functional>
#include <iostream>
#include <metis.h>
#include <mutex>
#include <queue>
#include "virtual_geometry_partition.h"
#include "parallel.hpp"

using std::vector;
using std::set;
using std::function;
using std::tuple;

vec2 vec2::nodata = vec2{FLT_MAX, FLT_MAX};
vec3 vec3::nodata = vec3{FLT_MAX, FLT_MAX, FLT_MAX};
vec4 vec4::nodata = vec4{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

std::vector<std::vector<idx_t>> ConnectedComponents(vector<idx_t>& Adjacency, vector<idx_t>& AdjacencyOffset);

struct ProgressBar
{
    std::atomic<int32_t> last_pos = -1;
    void update(float progress)
    {
        int barWidth = 70;

        int pos = barWidth * progress;
        if (pos != last_pos)
        {
            static std::mutex m;
            std::lock_guard<std::mutex> lock(m);
            last_pos = pos;
            std::cout << "[";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            if (pos == barWidth)
                std::cout << "\n";
            std::cout.flush();
        }
    }
};

template<typename LAMBDA>
class TYCombinator
{
	LAMBDA Lambda;

public:
	constexpr TYCombinator(LAMBDA&& InLambda): Lambda(std::move(InLambda)) 
	{}

	constexpr TYCombinator(const LAMBDA& InLambda): Lambda(InLambda) 
	{}

	template<typename... ARGS>
	constexpr auto operator()(ARGS&&... Args) const -> decltype(Lambda(static_cast<const TYCombinator<LAMBDA>&>(*this), std::forward<ARGS>(Args)...))
	{
		return Lambda(static_cast<const TYCombinator<LAMBDA>&>(*this), std::forward<ARGS>(Args)...);
	}

	template<typename... ARGS>
	constexpr auto operator()(ARGS&&... Args) -> decltype(Lambda(static_cast<TYCombinator<LAMBDA>&>(*this), std::forward<ARGS>(Args)...))
	{
		return Lambda(static_cast<TYCombinator<LAMBDA>&>(*this), std::forward<ARGS>(Args)...);
	}
};
template<typename LAMBDA>
constexpr auto MakeYCombinator(LAMBDA&& Lambda)
{
	return TYCombinator<std::decay_t<LAMBDA>>(std::forward<LAMBDA>(Lambda));
}

void GraphPartitioner::PartitionStrict(GraphData* Graph, bool MultiThreaded)
{
    int32_t NumPartitionsExpected = std::ceil( (float)Graph->Num / MaxPartitionSize );
	// NumPartitions = 0;
    // Bar.update(0.0);
    if (NumPartitionsExpected > 4 && MultiThreaded)
    {
        static BS::thread_pool pool;
        pool.push_task(MakeYCombinator([this](auto Self, GraphData* Graph) -> void
            {
                GraphData* ChildGraphs[2];
                BisectGraph(Graph, ChildGraphs);
                delete Graph;

                if (ChildGraphs[0] && ChildGraphs[1])
                {
					// Only spawn add a worker thread if remaining work is expected to be large enough
                    if (ChildGraphs[0]->Num > 32768)
                    {
                        pool.push_task(Self, ChildGraphs[0]);
                    }
                    else 
                    {
                        Self(ChildGraphs[0]);
                    }
                    Self(ChildGraphs[1]);
                }
            }), Graph);
        pool.wait_for_tasks();
    }
    else
    {
        RecursiveBisectGraph(Graph);
    }
}

void GraphPartitioner::AddPartition(idx_t Offset, idx_t Num)
{
    int32_t num_partitions = NumPartitions.fetch_add(1);
    Range& Range = Ranges[num_partitions];
    Range.Begin	= Offset;
    Range.End	= Offset + Num;
}

// 由于METIS库的图分割算法有时候会把一个连通图分割成两个不是连通图的子图，所以需要重新合并连通分量
// 合并之后两个子图就变成了连通图，不过两个子图的节点数量很可能是不平衡的，不过这并不重要
static void ReconnectComponents(idx_t* PartitionIDs, GraphPartitioner::GraphData* Graph)
{
    std::vector<std::vector<idx_t>> ComponentsAfterPartition;
    std::vector<std::pair<idx_t, idx_t>> Boundary;

    // 首先找到划分之后，两个子图加起来一共有多少个连通分量（两个子图之间视为不连通）
    {
        idx_t NumNodes = Graph->AdjacencyOffset.size()-1;
        std::queue<idx_t> q;
        std::vector<bool> visited(NumNodes, false);
        idx_t seed = 0;
        while (seed < NumNodes)
        {
            std::vector<idx_t> Component;
            q.push(seed);
            visited[seed] = true;
            while (!q.empty())
            {
                idx_t node = q.front(); q.pop();
                Component.push_back(node);
                for (int i = Graph->AdjacencyOffset[node]; i < Graph->AdjacencyOffset[node+1]; i++)
                {
                    idx_t adj = Graph->Adjacency[i];
                    if (!visited[adj])
                    {
                        if (PartitionIDs[adj] == PartitionIDs[node])
                        {
                            q.push(adj);
                            visited[adj] = true;
                        }
                        else 
                        {
                            Boundary.push_back({adj, node});
                        }
                    }
                }
            }
            ComponentsAfterPartition.push_back(Component);
            while (seed < NumNodes && visited[seed])
                seed++;
        }
    }

    // 接着，如果发现有大于2个连通分量，就意味着至少其中一个子图不是一个连通图
    if (ComponentsAfterPartition.size() > 2)
    {
        // 将连通分量视为节点，连通分量之间的连接视为边，建立一个关于连通分量的图
        int NumComponents = ComponentsAfterPartition.size();
        std::vector<int> ComponentIdx(Graph->Num);
        std::vector<std::vector<bool>> ComponentAdjacency(NumComponents, std::vector<bool>(NumComponents, false));
        for (int ComponentId = 0; ComponentId < ComponentsAfterPartition.size(); ComponentId++)
        {
            for (idx_t tri : ComponentsAfterPartition[ComponentId])
            {
                ComponentIdx[tri] = ComponentId;
            }
        }
        for (auto [adj, node] : Boundary)
        {
            ComponentAdjacency[ComponentIdx[adj]][ComponentIdx[node]] = true;
            ComponentAdjacency[ComponentIdx[node]][ComponentIdx[adj]] = true;
        }

        // 找到具有最少三角形数量的连通分量
        int min_triangles_num = ComponentsAfterPartition[0].size();
        int index = 0;
        for (int i = 1; i < NumComponents; i++)
        {
            if (ComponentsAfterPartition[i].size() < min_triangles_num)
            {
                min_triangles_num = ComponentsAfterPartition[i].size();
                index = i;
            }
        }

        // 从具有最少三角形的连通分量开始，广度优先遍历连通分量组成的图
        // 把遍历到的连通分量合并，直到最后只剩下两个节点（一个原来的，一个是由很多节点合并而成的）
        std::queue<int> q;
        std::vector<bool> visited(NumComponents, false);
        std::set<int> MergedComponent;
        q.push(index);
        visited[index] = true;
        while (!q.empty())
        {
            int comp = q.front(); q.pop();
            if (MergedComponent.size() < NumComponents-1)
                MergedComponent.insert(comp);
            else
                break;
            for (int adj = 0; adj < NumComponents; adj++)
            {
                if (ComponentAdjacency[adj][comp] && !visited[adj])
                {
                    q.push(adj);
                    visited[adj] = true;
                }
            }
        }

        // 重新修改来自METIS的PartitionIDs
        for (int i = 0; i < Graph->Num; i++)
            PartitionIDs[i] = 0;
        for (int CompIdx : MergedComponent)
        {
            for (idx_t TriIndex : ComponentsAfterPartition[CompIdx])
            {
                PartitionIDs[TriIndex] = 1;
            }
        }
    }
}

void GraphPartitioner::BisectGraph(GraphData* Graph, GraphData* ChildGraphs[2])
{
    ChildGraphs[0] = nullptr;
    ChildGraphs[1] = nullptr;

    if( Graph->Num <= MaxPartitionSize )
    {
        AddPartition( Graph->Offset, Graph->Num );
        return;
    }

    const int32_t TargetPartitionSize = MaxPartitionSize;
    const int32_t TargetNumPartitions = std::max( 2, DivideAndRoundNearest( Graph->Num, TargetPartitionSize ) );

    real_t PartitionWeights[] = {
        0.5,
        0.5
    };

    idx_t NumConstraints = 1;
    idx_t NumParts = 2;
    idx_t EdgesCut = 0;
    idx_t Options[ METIS_NOPTIONS ];
    METIS_SetDefaultOptions( Options );
    Options[ METIS_OPTION_UFACTOR ] = 1;
    // Options[ METIS_OPTION_SEED ] = 42;

    int r = METIS_PartGraphRecursive(
        &Graph->Num, 
        &NumConstraints, 
        Graph->AdjacencyOffset.data(), 
        Graph->Adjacency.data(), 
        NULL, 
        NULL, 
        Graph->AdjacencyCost.data(), 
        &NumParts, 
        PartitionWeights, 
        NULL, 
        Options, 
        &EdgesCut, 
        PartitionIDs.data() + Graph->Offset);

    ReconnectComponents(PartitionIDs.data() + Graph->Offset, Graph);
    
    if (r == METIS_OK)
    {

        int Front = Graph->Offset;
        int Back = Graph->Offset + Graph->Num - 1;
        while (Front <= Back)
        {
            while (Front <= Back && PartitionIDs[Front] == 0)
            {
                SwappedWith[Front] = Front;
                Front++;
            }

            while (Front <= Back && PartitionIDs[Back] == 1)
            {
                SwappedWith[Back] = Back;
                Back--;
            }

            if (Front < Back)
            {
                std::swap(TriIndexes[Front], TriIndexes[Back]);
                SwappedWith[Front] = Back;
                SwappedWith[Back] = Front;
                Front++;
                Back--;
            }
        }

        idx_t Split = Front;

        idx_t Num[2];
        Num[0] = Split - Graph->Offset;
        Num[1] = Graph->Offset + Graph->Num - Split;

        if( Num[0] <= MaxPartitionSize && Num[1] <= MaxPartitionSize )
        {
            AddPartition( Graph->Offset,	Num[0] );
            AddPartition( Split,			Num[1] );
        }
        else
        {
            for( int32_t i = 0; i < 2; i++ )
            {
                ChildGraphs[i] = new GraphData;
                ChildGraphs[i]->Adjacency.reserve( Graph->Adjacency.size() >> 1 );
                ChildGraphs[i]->AdjacencyCost.reserve( Graph->Adjacency.size() >> 1 );
                ChildGraphs[i]->AdjacencyOffset.reserve( Num[i] + 1 );
                ChildGraphs[i]->Num = Num[i];
            }

            ChildGraphs[0]->Offset = Graph->Offset;
            ChildGraphs[1]->Offset = Split;

            for( idx_t i = 0; i < Graph->Num; i++ )
            {
                GraphData* ChildGraph = ChildGraphs[ i >= ChildGraphs[0]->Num ];

                ChildGraph->AdjacencyOffset.push_back( ChildGraph->Adjacency.size() );
                
                idx_t OrgIndex = SwappedWith[ Graph->Offset + i ] - Graph->Offset;
                for( idx_t AdjIndex = Graph->AdjacencyOffset[ OrgIndex ]; AdjIndex < Graph->AdjacencyOffset[ OrgIndex + 1 ]; AdjIndex++ )
                {
                    idx_t Adj     = Graph->Adjacency[ AdjIndex ];
                    idx_t AdjCost = Graph->AdjacencyCost[ AdjIndex ];

                    // Remap to child
                    Adj = SwappedWith[ Graph->Offset + Adj ] - ChildGraph->Offset;

                    // Edge connects to node in this graph
                    if( 0 <= Adj && Adj < ChildGraph->Num )
                    {
                        ChildGraph->Adjacency.push_back( Adj );
                        ChildGraph->AdjacencyCost.push_back( AdjCost );
                    }
                }
            }
            ChildGraphs[0]->AdjacencyOffset.push_back( ChildGraphs[0]->Adjacency.size() );
            ChildGraphs[1]->AdjacencyOffset.push_back( ChildGraphs[1]->Adjacency.size() );

        }
    }
}

std::vector<std::vector<idx_t>> ConnectedComponents(vector<idx_t>& Adjacency, vector<idx_t>& AdjacencyOffset)
{
    idx_t NumNodes = AdjacencyOffset.size()-1;
    std::queue<idx_t> q;
    std::vector<bool> visited(NumNodes, false);
    std::vector<std::vector<idx_t>> Components;
    idx_t seed = 0;
    while (seed < NumNodes)
    {
        std::vector<idx_t> Component;
        q.push(seed);
        visited[seed] = true;
        while (!q.empty())
        {
            idx_t node = q.front(); q.pop();
            Component.push_back(node);
            for (int i = AdjacencyOffset[node]; i < AdjacencyOffset[node+1]; i++)
            {
                idx_t adj = Adjacency[i];
                if (!visited[adj])
                {
                    q.push(adj);
                    visited[adj] = true;
                }
            }
        }
        Components.push_back(Component);
        while (visited[seed] && seed < NumNodes)
            seed++;
    }
    return Components;
}

std::tuple<std::vector<std::vector<idx_t>>, std::vector<std::vector<idx_t>>>
GetSmallComponents(const vector<vector<idx_t>>& Components, int MaxPartitionSize)
{
    std::vector<std::vector<idx_t>> SmallComponents;
    std::vector<std::vector<idx_t>> BigComponents;
    for (int i = 0; i < Components.size(); i++)
    {
        if (Components[i].size() < MaxPartitionSize)
        {
            SmallComponents.push_back(Components[i]);
        }
        else 
        {
            BigComponents.push_back(Components[i]);
        }
    }
    return { SmallComponents, BigComponents };
}


std::vector<std::vector<idx_t>>
MergeSmallComponents(const vector<vector<idx_t>>& SmallComponents, int MaxPartitionSize, function<vec3(idx_t)> GetCenter, float threshold=FLT_MAX)
{
    threshold = std::max(0.f, threshold);
    std::vector<vec3> CenterOfSmallComponents;
    for (int i = 0; i < SmallComponents.size(); i++)
    {
        vec3 center{0.f, 0.f, 0.f};
        for (idx_t TriIdx : SmallComponents[i])
        {
            center = center + GetCenter(TriIdx);
        }
        center = center / float(SmallComponents[i].size());
        CenterOfSmallComponents.push_back(center);
    }

    int NumSmallComponents = SmallComponents.size();
    std::vector<std::vector<float>> Distances(NumSmallComponents, std::vector<float>(NumSmallComponents));
    for (int i = 0; i < NumSmallComponents; i++)
    {
        for (int j = 0; j < NumSmallComponents; j++)
        {
            Distances[i][j] = distance(CenterOfSmallComponents[i], CenterOfSmallComponents[j]);
        }
    }
    std::vector<bool> merged(SmallComponents.size(), false);
    std::vector<std::vector<idx_t>> MergedComponents;
    for (int i = 0; i < NumSmallComponents; i++)
    {
        if (merged[i])
            continue;
        std::vector<std::pair<float, int>> Dist_Comp_pair;
        for (int j = 0; j < NumSmallComponents; j++)
        {
            float dist = Distances[i][j];
            Dist_Comp_pair.push_back({dist, j});
        }
        // sort by the distances from this small component to all other small component (and itself, the distance is obviously 0)
        std::sort(Dist_Comp_pair.begin(), Dist_Comp_pair.end());
        idx_t NumTriangles = 0;
        std::vector<idx_t> MergedComponent;
        for (int j = 0; j < Dist_Comp_pair.size(); j++)
        {
            float dist = Dist_Comp_pair[j].first;
            if (dist > threshold)
                break;
            int comp = Dist_Comp_pair[j].second;
            if (merged[comp])
                continue;
            
            if (NumTriangles+SmallComponents[comp].size() < MaxPartitionSize)
            {
                for (idx_t TriIdx : SmallComponents[comp])
                {
                    MergedComponent.push_back(TriIdx);
                }
                merged[comp] = true;
                NumTriangles += SmallComponents[comp].size();
            }
            
        }
        if (!MergedComponent.empty())
            MergedComponents.push_back(MergedComponent);
    }

    // MergedComponents.insert(MergedComponents.end(), BigComponents.begin(), BigComponents.end());

    return MergedComponents;
}

Adjacency GenerateGraph(
    idx_t NumIndices,
    GraphPartitioner::GraphData* Graph,
    std::function<vec3(idx_t)> GetPosition)
{
    size_t NumEdges = NumIndices;
    size_t NumTriangles = NumEdges / 3;

    int thread_num = 1;
    if (std::thread::hardware_concurrency() > 0)
        thread_num = std::thread::hardware_concurrency();

    EdgeHash edge_hash(NumEdges);
    ParallelFor(NumEdges, thread_num, [&](idx_t EdgeIndex)
    {
        edge_hash.Add_Concurrent(EdgeIndex, GetPosition);
    });

    Graph->Num = NumTriangles;
    Graph->AdjacencyOffset.resize(NumTriangles+1);
    Adjacency Adjacency{NumEdges};

    ParallelFor(NumEdges, thread_num, [&](idx_t EdgeIndex)
    {
        idx_t AdjIndex = -1;
        idx_t AdjCount = 0;
        edge_hash.ForAllMatching(EdgeIndex, false, GetPosition, 
            [&](idx_t EdgeIndex, idx_t OtherEdgeIndex) {
                AdjIndex = OtherEdgeIndex;
                AdjCount++;
            });
        if (AdjCount > 1)
            AdjIndex = -2;
        Adjacency.Direct[EdgeIndex] = AdjIndex;
    });

	for( idx_t EdgeIndex = 0; EdgeIndex < NumEdges; EdgeIndex++ )
	{
		if( Adjacency.Direct[ EdgeIndex ] == -2 )
		{
			vector<idx_t> Edges;
			edge_hash.ForAllMatching(EdgeIndex, false, GetPosition,
				[&]( idx_t EdgeIndex0, idx_t EdgeIndex1 )
				{
					Edges.push_back(EdgeIndex1);
				} );
			std::sort(Edges.begin(), Edges.end());
			
			for(idx_t Edge : Edges)
			{
				Adjacency.Link(EdgeIndex, Edge);
			}
		}

	}

    ProgressBar Bar;
    std::atomic<idx_t> count = 0;
    std::vector<std::vector<idx_t>> TriAdjacency(NumTriangles);
    ParallelFor(NumTriangles, thread_num, [&](idx_t TriangleIndex)
    {
        Bar.update(float(count.fetch_add(1)+1) / (NumTriangles*2));
        TriAdjacency[TriangleIndex].reserve(3);
        for (int i = 0; i < 3; i++)
        {
            idx_t EdgeIndex = TriangleIndex*3 + i;
            edge_hash.ForAllMatching(EdgeIndex, false, GetPosition, 
                [&](idx_t EdgeIndex, idx_t OtherEdgeIndex) {
                    idx_t OtherTriangleIndex = OtherEdgeIndex / 3;
                    TriAdjacency[TriangleIndex].push_back(OtherTriangleIndex);
                });
        }
    });
    Graph->Adjacency.reserve(NumTriangles * 3);
    Graph->AdjacencyCost.reserve(NumTriangles * 3);
    for (idx_t TriangleIndex = 0; TriangleIndex < NumTriangles; TriangleIndex++)
    {
        Bar.update(float(count.fetch_add(1)+1) / (NumTriangles*2));
        Graph->AdjacencyOffset[TriangleIndex] = Graph->Adjacency.size();
        for (idx_t OtherTriangleIndex : TriAdjacency[TriangleIndex])
        {
            Graph->Adjacency.push_back(OtherTriangleIndex);
            Graph->AdjacencyCost.push_back(260);
        }
    }
    Graph->AdjacencyOffset[NumTriangles] = Graph->Adjacency.size();

    return Adjacency;
}

Adjacency ClusterTrianglesImpl(
        const VertexData& Data,
        int MaxPartitionSize, std::vector<idx_t>& TriIndexes, std::vector<GraphPartitioner::Range>& Ranges)
{
    auto& Positions = Data.Positions;
    auto& Indices = Data.Indices;

    auto GetPosition = [&](idx_t EdgeIndex) {
        return Positions[Indices[EdgeIndex]];
    };
    size_t NumTriangles = Data.NumTriangles;
    size_t NumEdges = Data.NumTriangles * 3;

    int thread_num = 1;
    if (std::thread::hardware_concurrency() > 0)
        thread_num = std::thread::hardware_concurrency();

    std::cout << "Building graph...\n";
    using GraphData = GraphPartitioner::GraphData;

    GraphPartitioner::GraphData* Graph = new GraphPartitioner::GraphData;
    Adjacency Adjacency = GenerateGraph(Data.NumTriangles*3, Graph, GetPosition);

    auto GetTriCenter = [&](idx_t TriIdx) {
        idx_t v0 = Indices[TriIdx*3];
        idx_t v1 = Indices[TriIdx*3+1];
        idx_t v2 = Indices[TriIdx*3+2];
        vec3 pos = (Positions[v0]+Positions[v1]+Positions[v2]) / 3.f;
        return pos;
    };

    std::cout << "Partitioning graph connected components...\n";
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    x_min = y_min = z_min = FLT_MAX;
    x_max = y_max = z_max = -FLT_MAX;
    for (idx_t i = 0; i < Data.NumVertices; i++)
    {
        x_min = std::min(x_min, Data.Positions[i].x);
        y_min = std::min(y_min, Data.Positions[i].y);
        z_min = std::min(z_min, Data.Positions[i].z);
        x_max = std::max(x_max, Data.Positions[i].x);
        y_max = std::max(y_max, Data.Positions[i].y);
        z_max = std::max(z_max, Data.Positions[i].z);
    }
    float diagonal_length = distance(vec3{x_min, y_min, z_min}, vec3{x_max, y_max, z_max});
    float threshold = diagonal_length * 0.05;
    std::cout << "Small components merge threshold: " << threshold << "\n";

    std::vector<std::vector<idx_t>> Components;
    {
        // Separate the connected components, merge the small components and keep the big enough components for partitioning
        std::vector<std::vector<idx_t>> components = ConnectedComponents(Graph->Adjacency, Graph->AdjacencyOffset);
        auto [SmallComponents, BigComponents] = GetSmallComponents(components, MaxPartitionSize);
        auto MergedComponents = MergeSmallComponents(SmallComponents, MaxPartitionSize, GetTriCenter, threshold);
        Components = MergedComponents;
        Components.insert(Components.end(), BigComponents.begin(), BigComponents.end());
    }

    GraphPartitioner Partitioner(NumTriangles, MaxPartitionSize);
    int32_t NumPartitionsExpected = std::ceil( (float)NumTriangles / MaxPartitionSize );
	Partitioner.Ranges.resize( NumPartitionsExpected * 10 );

    vector<idx_t> SortedTo(Partitioner.TriIndexes.size());
    auto Partition = [&](idx_t* PartitionIDs, idx_t Back, idx_t ComponentId) {

        idx_t Front = 0;
        while (Front <= Back)
        {
            while (Front <= Back && PartitionIDs[Front] != ComponentId)
            {
                Front++;
            }

            while (Front <= Back && PartitionIDs[Back] == ComponentId)
            {
                Back--;
            }

            if (Front < Back)
            {
                std::swap(Partitioner.TriIndexes[Front], Partitioner.TriIndexes[Back]);
                std::swap(PartitionIDs[Front], PartitionIDs[Back]);
                Front++;
                Back--;
            }
        }

        return Front;
    };
    {
        std::vector<GraphData*> Graphs;
        ParallelFor(Components.size(), thread_num, [&](idx_t CompIdx) {
            auto& Component = Components[CompIdx];
            for (idx_t TriIndex : Component)
            {
                Partitioner.PartitionIDs[TriIndex] = CompIdx;
            }
        });
        idx_t Num = NumTriangles;
        for (int CompIdx = 0; CompIdx < Components.size(); CompIdx++)
        {
            idx_t Split = Partition(Partitioner.PartitionIDs.data(), Num-1, CompIdx);
            GraphData* ChildGraph = new GraphData();
            Graphs.push_back(ChildGraph);
            ChildGraph->Offset = Split;
            ChildGraph->Num = Num - Split;
            Num = Split;
        }
        ParallelFor(NumTriangles, thread_num, [&](idx_t i) {
            idx_t TriIndex = Partitioner.TriIndexes[i];
            SortedTo[TriIndex] = i;
        });
        std::atomic<idx_t> count = 0;
        ProgressBar Bar;
        ParallelFor(Graphs.size(), thread_num, [&](idx_t i)
        {
            GraphData* ChildGraph = Graphs[i];
            for (int i = ChildGraph->Offset; i < ChildGraph->Offset+ChildGraph->Num; i++)
            {
                Bar.update(float(count.fetch_add(1)+1) / NumTriangles);
                ChildGraph->AdjacencyOffset.push_back(ChildGraph->Adjacency.size());
                idx_t TriIndex = Partitioner.TriIndexes[i];
                for (int j = Graph->AdjacencyOffset[TriIndex]; j < Graph->AdjacencyOffset[TriIndex+1]; j++)
                {
                    idx_t Adj = Graph->Adjacency[j];
                    idx_t AdjCost = Graph->AdjacencyCost[j];
                    if (ChildGraph->Offset <= SortedTo[Adj] && SortedTo[Adj] < ChildGraph->Offset+ChildGraph->Num)
                    {
                        ChildGraph->Adjacency.push_back(SortedTo[Adj]-ChildGraph->Offset);
                        ChildGraph->AdjacencyCost.push_back(AdjCost);
                    }
                }
            }
            ChildGraph->AdjacencyOffset.push_back(ChildGraph->Adjacency.size());
        });

        std::cout << "Partitioning each components...\n";
        for (GraphData* Graph : Graphs)
            Partitioner.PartitionStrict(Graph, true);

        Bar.update(1.0);
    }


    Partitioner.Ranges.resize(Partitioner.NumPartitions);
    std::sort(Partitioner.Ranges.begin(), Partitioner.Ranges.end());

    TriIndexes = std::move(Partitioner.TriIndexes);
    Ranges = std::move(Partitioner.Ranges);

    return Adjacency;

}

void ClusterTriangles(
        const VertexData& Data,
        ClusterResult& OutResult,
        int MaxPartitionSize)
{
    auto& Positions = Data.Positions;
    auto& Indices = Data.Indices;

    std::vector<idx_t> TriIndexes;
    std::vector<GraphPartitioner::Range> Ranges;
    Adjacency Adjacency = ClusterTrianglesImpl(Data, MaxPartitionSize, TriIndexes, Ranges);

    size_t NumTriangles = Data.NumTriangles;
    size_t NumEdges = Data.NumTriangles * 3;

    int thread_num = 1;
    if (std::thread::hardware_concurrency() > 0)
        thread_num = std::thread::hardware_concurrency();

    std::cout << "Constructing clusters...\n";
    std::atomic<int32_t> NumClusters = 0;
    vector<std::vector<idx_t>> OldToNewMappings(Ranges.size());
    vector<std::vector<idx_t>> ClusterExternalEdges(Ranges.size());

    auto& clusters = OutResult.Clusters;
    vector<idx_t> SortedTo(TriIndexes.size());
    clusters.resize(Ranges.size());
    ParallelFor(NumTriangles, thread_num, [&](idx_t i) {
        idx_t TriIndex = TriIndexes[i];
        SortedTo[TriIndex] = i;
    });
    ProgressBar Bar;
    ParallelFor(Ranges.size(), thread_num, [&](idx_t ClusterIndex) 
    {
        auto& Range = Ranges[ClusterIndex];
        idx_t TriStart = Range.Begin;
        idx_t TriEnd = Range.End;

        Cluster& cluster = clusters[ClusterIndex];
            
        set<idx_t> OldIndexesSet;
        for (int i = TriStart; i < TriEnd; i++)
        {
            int TriIndex = TriIndexes[i];
            for (int j = 0; j < 3; j++)
                OldIndexesSet.insert(Indices[TriIndex*3 + j]);
        }
        cluster.NumVertices = OldIndexesSet.size();
        vector<idx_t>& OldToNewMapping = OldToNewMappings[ClusterIndex];
        OldToNewMapping = vector<idx_t>(OldIndexesSet.begin(), OldIndexesSet.end());
        std::sort(OldToNewMapping.begin(), OldToNewMapping.end());
        cluster.Indices.reserve((TriEnd-TriStart) * 3);
        for (idx_t i = TriStart; i < TriEnd; i++)
        {
            idx_t TriIndex = TriIndexes[i];
            for (int j = 0; j < 3; j++)
            {
                idx_t EdgeIndex = TriIndex*3 + j;
                idx_t OldVertIndex = Indices[EdgeIndex];
                int32_t NewVertIndex = std::lower_bound(OldToNewMapping.begin(), OldToNewMapping.end(), OldVertIndex) - OldToNewMapping.begin();
                cluster.Indices.push_back(NewVertIndex);
            }
            cluster.OldTriangleIndices.push_back(TriIndex);
        }
            
        for (idx_t i = Range.Begin; i < Range.End; i++)
        {
            idx_t TriIndex = TriIndexes[i];
            for (int j = 0; j < 3; j++)
            {
                idx_t EdgeIndex = TriIndex*3 + j;
                Adjacency.ForAll(EdgeIndex, 
                    [&](idx_t, idx_t OtherEdgeIndex) {
                        idx_t OtherTriIndex = OtherEdgeIndex / 3;
                        if (SortedTo[OtherTriIndex] < Range.Begin || SortedTo[OtherTriIndex] >= Range.End)
                        {
                            ClusterExternalEdges[ClusterIndex].push_back(EdgeIndex);
                        }
                    });
            }
        }
        int32_t num_clusters = NumClusters.fetch_add(1);
        Bar.update(float(num_clusters+1) / Ranges.size());
    });
}
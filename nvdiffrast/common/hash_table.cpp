#include "hash_table.h"

std::atomic<uint32_t> FHashTable::EmptyHash[1] = { ~0u };

void FHashTable::Resize( uint32_t NewIndexSize )
{
	if( NewIndexSize == IndexSize )
	{
		return;
	}

	if( NewIndexSize == 0 )
	{
		Free();
		return;
	}

	if( IndexSize == 0 )
	{
		HashMask = (uint16_t)(HashSize - 1);
		Hash = new std::atomic<uint32_t>[ HashSize ];
		std::memset( Hash, 0xff, HashSize * 4 );
	}

	uint32_t* NewNextIndex = new uint32_t[ NewIndexSize ];

	if( NextIndex )
	{
		std::memcpy( NewNextIndex, NextIndex, IndexSize * 4 );
		delete[] NextIndex;
	}
	
	IndexSize = NewIndexSize;
	NextIndex = NewNextIndex;
}

float FHashTable::AverageSearch() const
{
	uint32_t SumAvgSearch = 0;
	uint32_t NumElements = 0;
	for( uint32_t Key = 0; Key < HashSize; Key++ )
	{
		uint32_t NumInBucket = 0;
		for( uint32_t i = First( (uint16_t)Key ); IsValid( i ); i = Next( i ) )
		{
			NumInBucket++;
		}

		SumAvgSearch += NumInBucket * ( NumInBucket + 1 );
		NumElements  += NumInBucket;
	}
	return (float)( SumAvgSearch >> 1 ) / (float)NumElements;
}
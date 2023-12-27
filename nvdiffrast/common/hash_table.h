#pragma once
#include <atomic>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cmath>

/*-----------------------------------------------------------------------------
	Adapted from Unreal Engine 5.2

	Dynamically sized hash table, used to index another data structure.
	Vastly simpler and faster than TMap.

	Example find:

	uint32_t Key = HashFunction( ID );
	for( uint32_t i = HashTable.First( Key ); HashTable.IsValid( i ); i = HashTable.Next( i ) )
	{
		if( Array[i].ID == ID )
		{
			return Array[i];
		}
	}
-----------------------------------------------------------------------------*/
class FHashTable
{
public:
					FHashTable( uint32_t InHashSize = 1024, uint32_t InIndexSize = 0 );
					FHashTable( const FHashTable& Other );
					~FHashTable();

	void			Clear();
	void			Clear( uint32_t InHashSize, uint32_t InIndexSize = 0 );
	void			Free();
	void	Resize( uint32_t NewIndexSize );

	// Functions used to search
	uint32_t			First( uint32_t Key ) const;
	uint32_t			Next( uint32_t Index ) const;
	bool			IsValid( uint32_t Index ) const;
	
	void			Add( uint32_t Key, uint32_t Index );
	void			Add_Concurrent( uint32_t Key, uint32_t Index );
	void			Remove( uint32_t Key, uint32_t Index );

	// Average # of compares per search
	float	AverageSearch() const;

protected:
	// Avoids allocating hash until first add
	static std::atomic<uint32_t> EmptyHash[1];

	uint32_t			HashSize;
	uint32_t			HashMask;
	uint32_t			IndexSize;

	uint32_t*			NextIndex;
	std::atomic<uint32_t>* Hash;
};

inline FHashTable::FHashTable( uint32_t InHashSize, uint32_t InIndexSize )
	: HashSize( InHashSize )
	, HashMask( 0 )
	, IndexSize( InIndexSize )
	, Hash( EmptyHash )
	, NextIndex( nullptr )
{
	assert( HashSize > 0 );
	assert(  (HashSize & (HashSize-1)) == 0 );
	
	if( IndexSize )
	{
		HashMask = HashSize - 1;
		
		Hash = new std::atomic<uint32_t>[ HashSize ];
		NextIndex = new uint32_t[ IndexSize ];

		std::memset( Hash, 0xff, HashSize * 4 );
	}
}

inline FHashTable::FHashTable( const FHashTable& Other )
	: HashSize( Other.HashSize )
	, HashMask( Other.HashMask )
	, IndexSize( Other.IndexSize )
	, Hash( EmptyHash )
{
	if( IndexSize )
	{
		Hash = new std::atomic<uint32_t>[ HashSize ];
		NextIndex = new uint32_t[ IndexSize ];

		std::memcpy( Hash, Other.Hash, HashSize * 4 );
		std::memcpy( NextIndex, Other.NextIndex, IndexSize * 4 );
	}
}

inline FHashTable::~FHashTable()
{
	Free();
}

inline void FHashTable::Clear()
{
	if( IndexSize )
	{
		std::memset( Hash, 0xff, HashSize * 4 );
	}
}

inline void FHashTable::Clear( uint32_t InHashSize, uint32_t InIndexSize )
{
	Free();

	HashSize = InHashSize;
	IndexSize = InIndexSize;

	assert( HashSize > 0 );
	assert( (HashSize & (HashSize-1)) == 0 );

	if( IndexSize )
	{
		HashMask = HashSize - 1;
		
		Hash = new std::atomic<uint32_t>[ HashSize ];
		NextIndex = new uint32_t[ IndexSize ];

		std::memset( Hash, 0xff, HashSize * 4 );
	}
}

inline void FHashTable::Free()
{
	if( IndexSize )
	{
		HashMask = 0;
		IndexSize = 0;
		
		delete[] Hash;
		Hash = EmptyHash;
		
		delete[] NextIndex;
		NextIndex = nullptr;
	}
} 

// First in hash chain
inline uint32_t FHashTable::First( uint32_t Key ) const
{
	Key &= HashMask;
	return Hash[ Key ];
}

// Next in hash chain
inline uint32_t FHashTable::Next( uint32_t Index ) const
{
	assert( Index < IndexSize );
	assert( NextIndex[Index] != Index ); // check for corrupt tables
	return NextIndex[ Index ];
}

inline bool FHashTable::IsValid( uint32_t Index ) const
{
	return Index != ~0u;
}

inline uint32_t RoundUpToPowerOfTwo(uint32_t value) {
    if ((value & (value - 1)) == 0) {
        return value;
    }

    uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }

    return result;
}

inline void FHashTable::Add( uint32_t Key, uint32_t Index )
{
	if( Index >= IndexSize )
	{
		Resize( std::max< uint32_t >( 32u, RoundUpToPowerOfTwo( Index + 1 ) ) );
	}

	Key &= HashMask;
	NextIndex[ Index ] = Hash[ Key ];
	Hash[ Key ] = Index;
}

// Safe for many threads to add concurrently.
// Not safe to search the table while other threads are adding.
// Will not resize. Only use for presized tables.
inline void FHashTable::Add_Concurrent( uint32_t Key, uint32_t Index )
{
	assert( Index < IndexSize );

	Key &= HashMask;
	NextIndex[ Index ] = Hash[ Key ].exchange(Index);
}

inline void FHashTable::Remove( uint32_t Key, uint32_t Index )
{
	if( Index >= IndexSize )
	{
		return;
	}

	Key &= HashMask;

	if( Hash[Key] == Index )
	{
		// Head of chain
		Hash[Key] = NextIndex[ Index ];
	}
	else
	{
		for( uint32_t i = Hash[Key]; IsValid(i); i = NextIndex[i] )
		{
			if( NextIndex[i] == Index )
			{
				// Next = Next->Next
				NextIndex[i] = NextIndex[ Index ];
				break;
			}
		}
	}
}
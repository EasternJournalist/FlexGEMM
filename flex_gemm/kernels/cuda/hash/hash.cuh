namespace flex_gemm {
namespace hash {

// 32 bit Murmur3 hash
__forceinline__ __device__ size_t hash(uint32_t k, size_t N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}


// 64 bit Murmur3 hash
__forceinline__ __device__ size_t hash(uint64_t k, size_t N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}


template<typename K>
__forceinline__ __device__ void linear_probing_insert(
    K* hashmap_keys,
    const K key,
    const size_t N
) {
    size_t slot = hash(key, N);
    // Bound the probe count by ``N``: a non-full table needs at most ``N-1``
    // probes; if we hit ``N`` probes the map is full and ``key`` is absent,
    // which can only mean the caller mis-sized the hashmap. Trap so the host
    // sees a CUDA error instead of an infinite spin.
    for (size_t probes = 0; probes < N; ++probes) {
        K prev = atomicCAS(&hashmap_keys[slot], std::numeric_limits<K>::max(), key);
        if (prev == std::numeric_limits<K>::max() || prev == key) {
            return;
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
    printf("flex_gemm::linear_probing_insert: hashmap full (N=%zu) -- aborting.\n", N);
    __trap();
}


template<>
__forceinline__ __device__ void linear_probing_insert(
    uint64_t* hashmap_keys,
    const uint64_t key,
    const size_t N
) {
    size_t slot = hash(key, N);
    for (size_t probes = 0; probes < N; ++probes) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hashmap_keys[slot]),
            static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()),
            static_cast<unsigned long long>(key)
        );
        if (prev == std::numeric_limits<uint64_t>::max() || prev == key) {
            return;
        }
        slot = (slot + 1) % N;
    }
    printf("flex_gemm::linear_probing_insert<uint64>: hashmap full (N=%zu) -- aborting.\n", N);
    __trap();
}


template<typename K, typename V>
__forceinline__ __device__ void linear_probing_insert(
    K* hashmap_keys,
    V* hashmap_values,
    const K key,
    const V value,
    const size_t N
) {
    size_t slot = hash(key, N);
    for (size_t probes = 0; probes < N; ++probes) {
        K prev = atomicCAS(&hashmap_keys[slot], std::numeric_limits<K>::max(), key);
        if (prev == std::numeric_limits<K>::max() || prev == key) {
            hashmap_values[slot] = value;
            return;
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
    printf("flex_gemm::linear_probing_insert (kv): hashmap full (N=%zu) -- aborting.\n", N);
    __trap();
}


template<typename V>
__forceinline__ __device__ void linear_probing_insert(
    uint64_t* hashmap_keys,
    V* hashmap_values,
    const uint64_t key,
    const V value,
    const size_t N
) {
    size_t slot = hash(key, N);
    for (size_t probes = 0; probes < N; ++probes) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hashmap_keys[slot]),
            static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()),
            static_cast<unsigned long long>(key)
        );
        if (prev == std::numeric_limits<uint64_t>::max() || prev == key) {
            hashmap_values[slot] = value;
            return;
        }
        slot = (slot + 1) % N;
    }
    printf("flex_gemm::linear_probing_insert<uint64> (kv): hashmap full (N=%zu) -- aborting.\n", N);
    __trap();
}


template<typename K, typename V>
__forceinline__ __device__ V linear_probing_lookup(
    const K* hashmap_keys,
    const V* hashmap_values,
    const K key,
    const size_t N
) {
    size_t slot = hash(key, N);
    // Same N-probe bound as the insert side: if the key is present it must be
    // found within ``N-1`` probes; otherwise we treat it as not-found. This
    // also prevents an infinite loop if the map is full and the key is absent.
    for (size_t probes = 0; probes < N; ++probes) {
        K prev = hashmap_keys[slot];
        if (prev == std::numeric_limits<K>::max()) {
            return std::numeric_limits<V>::max();
        }
        if (prev == key) {
            return hashmap_values[slot];
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
    return std::numeric_limits<V>::max();
}

} // namespace hash
} // namespace flex_gemm

# vLLM and Paged Attention

PagedAttention, used by vLLM, solves the problem of inefficient GPU memory management for the key-value (KV) cache in large language model (LLM) serving. The KV cache stores attention key and value tensors generated during the autoregressive decoding process, which are kept in GPU memory to generate subsequent tokens. This cache is large (e.g., up to 1.7GB for a single sequence in LLaMA-13B) and dynamic, as its size depends on unpredictable sequence lengths. Traditional systems waste 60-80% of KV cache memory due to:

1. **Internal Fragmentation**: Allocated memory slots for a sequence are reserved but unused because the system cannot predict how many tokens will be generated.
2. **External Fragmentation**: Gaps between fixed memory blocks go unused, as requests vary widely in size.
3. **Over-Reservation**: Systems allocate contiguous memory blocks for the maximum possible sequence length, leading to unused memory.

PagedAttention addresses these issues by:

- **Dynamic Memory Allocation**: Inspired by virtual memory and paging in operating systems, it partitions the KV cache into fixed-sized blocks, allocating memory only as needed during inference. This eliminates the need to reserve large contiguous memory blocks upfront, reducing waste to under 4%.[](https://blog.runpod.io/introduction-to-vllm-and-how-to-run-vllm-on-runpod-serverless/)[](https://arxiv.org/abs/2309.06180)[](https://scalingknowledge.substack.com/p/an-introduction-to-vllm-and-pagedattention)
- **Non-Contiguous Storage**: It stores KV pairs in non-contiguous memory blocks, managed via block tables that map logical to physical blocks, minimizing fragmentation.[](https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm)[](https://hackernoon.com/pagedattention-and-vllm-explained-what-are-they)
- **Efficient Memory Sharing**: For tasks like parallel sampling or beam search, where multiple output sequences share the same input prompt, PagedAttention enables sharing of KV cache blocks using a copy-on-write mechanism and reference counting. This reduces memory overhead by up to 55%, making complex sampling algorithms more practical.[](https://vllm.ai/)[](https://www.hopsworks.ai/dictionary/pagedattention)[](https://medium.com/%40ashishsingh.chunar2017/vllm-easy-fast-and-cheap-llm-serving-with-pagedattention-28fa109ef919)

These optimizations allow vLLM to achieve near-optimal memory utilization, enabling larger batch sizes, higher throughput (up to 24x compared to HuggingFace Transformers), and reduced latency, especially for longer sequences and larger models.[](https://arxiv.org/abs/2309.06180)[](https://gigazine.net/gsc_news/en/20230622-vllm-paged-attention/)[](https://www.hopsworks.ai/dictionary/vllm)
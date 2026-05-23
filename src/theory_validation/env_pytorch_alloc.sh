# PyTorch 2.x CUDA allocator (avoid deprecated PYTORCH_CUDA_ALLOC_CONF / max_split_size_mb).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF

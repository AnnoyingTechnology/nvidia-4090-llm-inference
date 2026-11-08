# Isolated upgrade of the working CUDA 12.9 image; host driver and models stay unchanged.
FROM ulmus/qwen38-vllm:0.27.1-cu129
WORKDIR /next
COPY source/ .
COPY requirements-cu129.txt ./requirements-cu129.txt
RUN /app/venv/bin/pip uninstall -y vllm \
    && /app/venv/bin/python -c "import shutil; shutil.rmtree('/app/venv/lib/python3.12/site-packages/vllm', ignore_errors=True)" \
    && /app/venv/bin/pip install -r requirements-cu129.txt \
    && /app/venv/bin/pip check
RUN ln -s /app/venv /next/venv \
    && SP=$(/app/venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1) \
    && bash patches/apply.sh "$SP" && PY=/app/venv/bin/python bash kvarn/install.sh
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN PY=/app/venv/bin/python bash xgrammar/install.sh
RUN PY=/app/venv/bin/python bash verify.sh --install \
    && /app/venv/bin/python bench/test_offload_mamba_swa_boundaries.py \
    && /app/venv/bin/python bench/test_dflash_context_capacity.py \
    && /app/venv/bin/python bench/mq3d_scratch_pool_test.py \
    && /app/venv/bin/python bench/test_async_mamba_kv_sizing.py \
    && /app/venv/bin/python bench/test_kv_cache_seat_budget.py
ENV PATH=/app/venv/bin:$PATH CUDA_HOME=/usr/local/cuda LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/local/cuda/lib64
LABEL ulmus.hyperqwen.commit=6ab9bd05cb84b538a6595d634f3e337ac8a80856
ENTRYPOINT ["/bin/bash", "/next/single-user/start_qwen.sh"]

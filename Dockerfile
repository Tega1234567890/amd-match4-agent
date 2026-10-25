# AMD AI League Match 4 -- web retrieval agent.
# Base = the ROCm image mandated for Matches 2-3 (same FROM as the MC3 starter kit). Re-check the tag once the
# Match 4 spec is published: the base is verified by LAYER IDENTITY, so never squash / flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Weights ship in the image (Qwen2.5-1.5B-Instruct ~3 GiB bf16; swap for a bigger model if VRAM/size allow).
# Fetched into the build context before `docker build` (CI does this; locally:
#   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/llm )
COPY models/llm/ /models/llm/

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    || pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY app/ /app/
RUN mkdir -p /app/output /app/corpus

ENV M4_MODEL_DIR=/models/llm \
    M4_OUTPUT_DIR=/app/output \
    M4_DEADLINE_S=25 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# No server, no port: the harness execs into the running container.
#   warm-up : python3 /app/app.py --index          (loads model once, resident unix-socket daemon)
#   query   : python3 /app/app.py --query-id query_01 --query "..." [--url https://...]
#   output  : /app/output/query_01_output.json
CMD ["sleep", "infinity"]

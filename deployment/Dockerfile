FROM python:3.11.15-slim-bookworm AS ffmpeg-builder
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY deployment/build_ffmpeg.sh deployment/check_ffmpeg.py /tmp/xedit-ffmpeg/
RUN bash /tmp/xedit-ffmpeg/build_ffmpeg.sh

FROM python:3.11.15-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg-builder /opt/xedit-ffmpeg /opt/xedit-ffmpeg
ENV PATH="/opt/xedit-ffmpeg/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY x_editor ./x_editor
COPY configs ./configs
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["x-edit"]
# Control plane only. GPU models run in isolated backend environments.

ARG CUDA_VERSION="11.8.0"
ARG UBUNTU_VERSION="22.04"
ARG BASE_TAG="latest"

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu$UBUNTU_VERSION

ARG PYTHON_VERSION="3.9"
ENV PYTHON_VERSION=$PYTHON_VERSION

RUN useradd -m -u 1001 appuser
RUN apt-get update && \
    apt-get install -y wget git && rm -rf /var/lib/apt/lists/* && \
    mkdir /runpod-volume && \
    chown appuser:appuser /runpod-volume && \
    mkdir /workspace && \
    chown appuser:appuser /workspace && \
    mkdir /app && \
    chown appuser:appuser /app

USER appuser

ENV HOME /home/appuser
ENV PATH="${HOME}/miniconda3/bin:${PATH}"
WORKDIR /app
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ${HOME}/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="${HOME}/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}:/app"

ADD requirements.txt .
ADD entrypoint.sh .

RUN pip3 install -r requirements.txt && \
    rm requirements.txt

# Add your file
ADD handler.py .

ENV GPTQ_REPO=""
ENV GPTQ_FILE=""
ENV GPTQ_BASENAME=""
ENV GPTQ_REVISION="main"
ENV GGML_TYPE=""
ENV GGML_LAYERS=""

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Call your file when your container starts
CMD [ "python3", "-u", "handler.py" ]
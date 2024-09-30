FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install torch torchvision

#FROM kpis_temp

WORKDIR /app

COPY . /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["python3", "inference_patch.py", "--data_dir", "/input", "--output_dir", "/output"]
CMD ["--batch_size", "64"]

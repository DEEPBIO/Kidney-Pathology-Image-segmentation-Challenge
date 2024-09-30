FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install torch torchvision 
RUN pip3 install tifffile tqdm

RUN apt-get update
RUN apt-get -y dist-upgrade
RUN apt-get -y install openslide-tools
ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-openslide
RUN pip3 install openslide-python

#FROM kpis_temp

WORKDIR /app

COPY . /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["python3", "inference_slide.py", "--data_dir", "/input", "--output_dir", "/output"]
CMD ["--batch_size", "16"]

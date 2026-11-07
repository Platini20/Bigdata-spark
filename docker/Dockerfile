FROM spark:4.2.0-python3
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libblas3 liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir numpy==2.2.6
USER spark

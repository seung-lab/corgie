FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN apt-get clean && apt-get -y update && apt-get install -y locales && locale-gen en_US.UTF-8

ENV LC_CTYPE en_US.UTF-8
ENV LANG en_US.UTF-8

ADD . /opt/corgie
WORKDIR /opt/corgie

RUN apt-get update \
  # Install dependencies
  && apt-get install -y --no-install-recommends \
      libgtk2.0-dev language-pack-en build-essential \
  && pip install setuptools wheel \
  && pip install --no-cache-dir -r requirements-release.txt \
  && pip install -e . \
  # Cleanup apt
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \

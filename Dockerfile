# Base container: python 3.7.13
FROM python:3.7.12-slim


# 2) change to root to install packages
USER root

# RUN apt update

# 3) install packages using notebook user
# USER jovyan

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]
FROM python:3.11-slim

# Install system dependencies including X11, browser, and additional programming languages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    sudo \
    # C/C++ development tools
    gcc \
    g++ \
    cmake \
    gdb \
    make \
    clang \
    # Go installation
    golang \
    # Additional useful development tools
    vim \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/*


# Add Go to PATH
ENV PATH=$PATH:/usr/local/go/bin
ENV GOPATH=$HOME/go
ENV PATH=$PATH:$GOPATH/bin


ENV USERNAME=jiatong
ENV HOME=/home/$USERNAME
RUN useradd -m -s /bin/bash -d $HOME $USERNAME && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR $HOME

RUN mkdir -p app
WORKDIR $HOME/app

COPY --chown=$USERNAME:$USERNAME utils.py main.py cost.txt ./
COPY --chown=$USERNAME:$USERNAME configs/ ./configs/
COPY --chown=$USERNAME:$USERNAME data/ ./data/
COPY --chown=$USERNAME:$USERNAME prompts/ ./prompts/
COPY --chown=$USERNAME:$USERNAME verifier_agent/ ./verifier_agent/
COPY --chown=$USERNAME:$USERNAME skills/ ./skills/
COPY --chown=$USERNAME:$USERNAME requirements.txt .
COPY --chown=$USERNAME:$USERNAME engine/ ./engine/
RUN pip install --user -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/app:${PYTHONPATH}"

# ENV DISPLAY_NUM=1
# ENV DISPLAY=:${DISPLAY_NUM}
# ENV HEIGHT=768
# ENV WIDTH=1024

CMD ["python", "main.py", "--config", "configs/eval.yaml"]
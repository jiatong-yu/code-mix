FROM python:3.11-slim

# Install system dependencies including X11 and browser
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    sudo \
    # X11 and display requirements
    xvfb \
    scrot \
    imagemagick \
    firefox-esr \
    x11-apps \
    xdotool \
    # Add Mozilla team PPA for Firefox
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to a non-root user
ENV USERNAME=verifier
ENV HOME=/home/$USERNAME
RUN useradd -m -s /bin/bash -d $HOME $USERNAME && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR $HOME

# Create application directory
RUN mkdir -p app
WORKDIR $HOME/app

# Copy your application files
COPY --chown=$USERNAME:$USERNAME verify.py loop.py ./
COPY --chown=$USERNAME:$USERNAME tools/ ./tools/
COPY --chown=$USERNAME:$USERNAME utils.py ./

# Install Python dependencies
COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --user -r requirements.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/app:${PYTHONPATH}"

# Display settings
ENV DISPLAY_NUM=1
ENV DISPLAY=:${DISPLAY_NUM}
ENV HEIGHT=768
ENV WIDTH=1024

# Create a script to initialize the virtual display and run your command
RUN echo '#!/bin/bash\n\
Xvfb :${DISPLAY_NUM} -screen 0 ${WIDTH}x${HEIGHT}x24 &\n\
sleep 2\n\
python main.py --config configs/eval.yaml\n\
' > /home/$USERNAME/start.sh && \
    chmod +x /home/$USERNAME/start.sh

# Command to run the verifier with virtual display
CMD ["/home/data_generator/start.sh"]
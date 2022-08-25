FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0" \
    DEBIAN_FRONTEND=noninteractive
ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"

ARG PYTHON_VERSION=3.9

ARG USERNAME=user
ARG UID=1000
ARG GID=${UID}
RUN groupadd --gid ${GID} $USERNAME \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} \
    && apt-get update \
    && $APT_INSTALL \
    sudo \
    dialog \
    apt-utils \
    build-essential \
    software-properties-common \
    ca-certificates \
    curl \
    unzip \
    wget \
    pkg-config \
    git \
    libgl1-mesa-glx \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
ENV HOME /home/${USERNAME}
WORKDIR ${HOME}

# SSL config
ARG CERT_ZIP="[]"
COPY Dockerfile $CERT_ZIP /tmp/
RUN if [ ${CERT_ZIP} != "[]" ] && [ ${CERT_ZIP} != "" ]; \
    then unzip /tmp/"$(basename ${CERT_ZIP})" -d /usr/share/ca-certificates/ \
    && echo "$(ls -a /usr/share/ca-certificates/ | grep .crt)"  >> /etc/ca-certificates.conf \
    && update-ca-certificates \
    && echo "SSL certificates installed"; \
    fi \
    && cd /tmp/ && ls -A1 | xargs rm -rf
ENV PIP_CERT=/etc/ssl/certs/

USER ${USERNAME}

# Install conda
ARG CONDA_ENV="firewood"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ~/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && sudo ln -s ~/miniconda3/bin/conda /usr/local/bin/conda \
    && conda create -q -n $CONDA_ENV python=${PYTHON_VERSION} \
    && conda init bash \
    && echo "conda activate ${CONDA_ENV}" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=$CONDA_ENV

# Install zsh for interactive container
ARG DEFAULT_SHELL="bash"
ENV DEFAULT_SHELL=$DEFAULT_SHELL
RUN if [ ${DEFAULT_SHELL} = "zsh" ]; \
    then sudo apt-get update \
    && sudo $APT_INSTALL zsh locales \
    && sudo locale-gen en_US.UTF-8 \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended > /dev/null \
    && git clone -q https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && git clone -q https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && sed -i 's/^ZSH_THEME.*/ZSH_THEME="agnoster"/' ~/.zshrc \
    && sed -i 's/^plugins.*/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/' ~/.zshrc \
    && sudo chsh -s /usr/bin/zsh \
    && conda init zsh \
    && echo "conda activate ${CONDA_ENV}" >> ~/.zshrc \
    && sudo rm -rf /var/lib/apt/lists/* \
    ; \
    fi

COPY . ${HOME}/torch-firewood/
RUN sudo chmod -R 777 ${HOME}/torch-firewood/

# Install python packages by pip and conda
ARG USE_JUPYTER="false"
SHELL ["conda", "run", "--no-capture-output", "/bin/bash", "-c"]
RUN $PIP_INSTALL \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    -r ${HOME}/torch-firewood/requirements-dev.txt \
    # conda only packages
    # && conda install -y -c conda-forge
    && if [ "$USE_JUPYTER" == "true" ]; \
    then $PIP_INSTALL jupyterlab; \
    fi

ARG BUILD="true"
RUN if [ ${BUILD} = "true" ]; \
    then cd ${HOME}/torch-firewood/ \
    && bash ./install.sh \
    ; \
    fi
RUN sudo chmod -R 755 ${HOME}/torch-firewood/ && sudo chmod 777 ${HOME}/torch-firewood/

EXPOSE 6006 8888

SHELL ["/bin/bash", "-c"]
CMD ["bash", "-c", "$DEFAULT_SHELL"]

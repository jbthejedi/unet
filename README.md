# unet
Education implementation of unet
Start runpod with A100 and 50G of volume container

# Update package installer, install vim and screen
apt update && apt install vim -y && apt install screen -y

# Copy contents to file install_gh.sh
# Contents of install_gh.sh
```
(type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
	&& mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y
```
# Install gh client with script install_gh.sh
touch install_gh.sh
chmod +x install_gh.sh
./install_gh.sh
gh auth login
# Clone repo
gh repo clone <repo_name>

# Contents of install_mc.sh
```
mkdir -p /workspace/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda3/miniconda.sh
bash /workspace/miniconda3/miniconda.sh -b -u -p /workspace/miniconda3
rm /workspace/miniconda3/miniconda.sh

```

chmod +x install_mc.sh
# install miniconda
using install_mc.sh

# install env
/workspace/miniconda3/bin/conda init && source ~/.bashrc

# Create env
conda create -n unet python=3.10 && conda activate unet
pip install -r requirements.txt

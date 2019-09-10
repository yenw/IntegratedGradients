# Integrated Gradients for PhoenixGo
* Integrated Gradients: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
* Integrated Gradients in RL: [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.17857)
* PhoenixGo: https://github.com/Tencent/PhoenixGo
* GoGUI: https://github.com/Remi-Coulom/gogui

# Usage:
* 1: pip install tensorflow
* 2: wget https://github.com/Tencent/PhoenixGo/releases/download/trained-network-20b-v1/trained-network-20b-v1.tar.gz 
* 3: tar -xzvf trained-network-20b-v1.tar.gz && cd trained-network-20b-v1/ckpt
* 4: wget https://github.com/yenw/IntegratedGradients/blob/master/IG_GTP.py
* 5: wget https://github.com/yenw/IntegratedGradients/blob/master/IG_GTP.sh && chmod +x IG_GTP.sh
* 6: download and install GoGUI
* 7: let IG_GTP.sh be the GTP engine of GoGUI

# Screenshot
![GoGui](../master/IG.png?raw=true)

# Version
2018.09.17 V0.1

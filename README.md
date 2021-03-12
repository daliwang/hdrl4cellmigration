# Hierarchical deep reinforcement learning reveals novel mechanism of cell movement

## Introduction
This is accompany code and data associated with the paper submission 'Hierarchical deep reinforcement learning reveals novel mechanism of cell movement'.

## Package requirements
  Python3 <br />
  PyTorch 0.2 (tested on 0.2.0_3) <br />
  Mesa <br />
  PIL 4.2.1 <br />
  scikit-learn 0.19.1 <br />
  numpy (tested on 1.15.0) <br />
  
## File structure
  ./nuclei_data/: folder with textual data of nuclei <br />
  ./nuclei_data/nuclei_cpaaa_RL: nuclei that used for HDRL training <br />
  ./trained_models/:folder with all the pre-trained models
  ./trained_models/dqn_eval_net_pretrained.pkl: checkpoint of the trained lower-level DQN <br />
  ./trained_models/motion_model.pkl: checkpoint of the trained Motion Model <br />
  ./trained_models/neighbor_model_800k_train.p: checkpoint of the trained Neighbor Relationship Model <br />
  ./Embryo.py: functions for parsing the embryo data
  ./draw_plane.py: visualization
  ./model.py: agent-based model
  ./run.py: simulation running

## Usage
1. Explore the successful scenarios: python3 run_dqn.py <br />

## Citation
add after paper submitted to biorxiv.


# Hierarchical deep reinforcement learning reveals novel mechanism of cell movement

## Introduction
This is accompany code and data associated with the paper submission 'Hierarchical deep reinforcement learning reveals novel mechanism of cell movement'. And it was cloned and updated from https://github.com/zwang84/hdrl4cellmigration/

## Package requirements
  Python3 (tested on 3.6.2) <br />
      Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf<br />
  PyTorch 0.2 (tested on 0.2.0_3 (old) and 0.4.1) <br />
  Mesa (tested on 0.8.1) <br />
  PIL 4.2.1 <br />
  scikit-learn 0.19.1 <br />
  numpy (tested on 1.15.0) <br />
  
## File structure
  ./data/: folder with textual data of nuclei <br />
  ./data/data_description.txt: a brief description of the input textual embryonic data. <br />
  ./data/Cpaaa_0: embryonic data for Cpaaa migration training and evaluation. <br />
  ./data/Cpaaa_[1-3]: embryonic data for Cpaaa test case. <br />
  ./data/mu_int_R_CANL_[1-2]: embryonic data for mu_int_R and CANL case. <br />
  
  ./trained_models/:folder with all the pre-trained models. <br />
  ./trained_models/hdrl_llmodel.pkl: checkpoint of the trained lower-level DQN <br />
  ./trained_models/motion_model.pkl: checkpoint of the trained Motion Model <br />
  ./trained_models/neighbor_model.p: checkpoint of the trained Neighbor Relationship Model <br />
  ./trained_models/TMM.pkl.link: the download link for the TMM <br />
  
  ./saved_data/: folder that used for saving the output data when exploring the successful scenarios. (see below) <br />
  
  ./embryo.py: functions for parsing the embryo data <br />
  ./draw_plane.py: visualization <br />
  ./model.py: agent-based model for cell migration using Mesa <br />
  ./model_obs_cpaaa.py: a cell migration environment that uses TMM to detect emerging features in the Cpaaa case. <br />
  ./model_obs_mu.py: a cell migration environment that uses TMM to detect emerging features in the mu_int_R and CANL cases. <br />
  ./run.py: an HDRL Model for cell migration using the model.py environment <br />

## Usage
1.  Explore the successful scenarios: 
   Command: python3 run.py <br />
   Three Files are generated in the 'saved_data' folder after the evaluation: <br />
     - movement_index.pkl: the movement index (1 for directional movement and 0 for random movement) of Cpaaa at each time step. <br />
     - cpaaa_locations.pkl: location of Cpaaa at each time step. <br />
     - target_locations.pkl: location of the target cell (ABarpaapp) at each time step. If ABarpaapp is not born, [0,0,0] is used as a placeholder. <br />

2. Test the Cpaaa migration movement index in three different embryos: <br />
  (1). Download the observational data and the TMM checekpoint in the google drive: <br /> 
  observational data: https://drive.google.com/drive/folders/1_w0p7t_dmTha8ODTgosGXZO9gMRmYd3N?usp=sharing  <br />
  TMM checkpoint: https://drive.google.com/file/d/172FC8-8074mxotD8JSZemeRcA3v57ZF6/view?usp=sharing <br />
  (2) Command: python3 model_obs_cpaaa.py --emb [1,2,3] <br />
  Movment index will print out after the program is done. The results are used for Figure 5B,5C, and 5D<br />

3. Test the movement index of the mu_int_R and CANL migration in two embroys: <br />
  (1). Download the observational data  in the google drive: <br /> 
  observational data: https://drive.google.com/drive/folders/12JOhhz9LxvNig4BgcOidWTUqjrfW08t-?usp=sharing <br />
  (2) Command: python3 model_obs_mu.py --cell [mu_int_r, canl] --emb [1,2] <br />
  Movment index will print out after the program is done. The results are used for Figure 6C, and Figure 6D <br />



## Citation
Will update after paper submission.


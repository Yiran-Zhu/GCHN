# GCHN
PyTorch implementation of "Graph Convolutional Hourglass Network for Skeleton-based Action Recognition".

<img src="figures/framework.pdf" width="90%">

# Dependencies
- Python 3.6
- PyTorch 1.2.0
- PyYAML, tqdm, tensorboard

# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 
        -data\  
          -kinetics_raw\  
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`
     
# Training & Testing




    `python main.py --config ./config/ntu-cross-view/train_joint.yaml`

    `python main.py --config ./config/ntu-cross-view/train_bone.yaml`
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

    `python main.py --config ./config/ntu-cross-view/test_joint.yaml`

    `python main.py --config ./config/ntu-cross-view/test_bone.yaml`

Then combine the generated scores with: 

    `python ensemble.py`
     
# Contact
For any questions, please feel free to contact: `yiranupup@gmail.com`

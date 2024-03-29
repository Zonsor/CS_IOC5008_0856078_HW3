# CS_IOC5008_0856078_HW3 
# Note
　The uploaded files do not contain Detectron2. You should build Detectron2 first. If your OS is linux, follow Detecron2 GitHub directly. If your OS is win10, you can follow the "Findings or Summary". Those steps is basically from conansherry's GitHub. I just change a little bit to get work on my computer. After building Detectron2, place the files of this repository into Detectron2 main folder.
 
* ***demo.py:*** This .py file is based on the Colab Notebook of Detecton2. It only contains the code of predicted part. After building, you can run this file to check whether Detectron2 can work or not.  
* ***detectron2_dataset.py:*** This file is for constructing annotaion from SVHN dataset format (.mat file) to Detectron2 format(.json). Because some variables are not JSON serializable, I convert them to list first, and transform them back in HW3.py for convenience.  
* ***HW3.py:*** This is the main file for HW3. It contains training part and testing part. You can tune some hyperparameters here. Amd make sure the file path is modified correctly by you.  
* ***HW3_inference_time.ipynb:*** This file format is belong to Jupyter Notebook. You should open it with Google Colab. And you can see the inference time.(You should unpload the weight file and testing images first)  
* ***inference_time.png:*** the result of inference time
# Reference from GitHub:
Detectron2: https://github.com/facebookresearch/detectron2  
Detectron2 on windows: https://github.com/conansherry/detectron2

# Speed benchmark:
  ![image](https://github.com/Zonsor/CS_IOC5008_0856078_HW3/blob/master/inference_time.png)
　Because I use detectron2 in this homework, the inference code is only one line. And TA said “we only need to choose one image to do inference.”. However, I am still not sure whether this is valid format or not. The result is about ***65.5 ms***.

# Brief introduction:
　The task of this homework is to do digits detection on Street View House Numbers (SVHN) dataset. There are 33402 training images and 13068 test images in this dataset. I use detectron2 to train faster R-CNN with a ResNet50+FPN (feature pyramid networks) backbone. The result of mAP is ***0.46726***.

# Methodology:
　This homework I did is mainly follow the tutorial of Detectron2. Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. It provides many kinds of models like faster R-CNN and RetinaNet. Besides, this codebase can do not only object detection but also instance segmentation and so on. I choose faster R-CNN with a ResNet50+FPN backbone to complete this task because the trade off between precision and speed.

# Findings or Summary:
　To be honest, I don’t make much effort to get better accuracy. However, building environment and processing annotation really take much time. Therefore, I want to explain how I build Detectron2 on Windows. I believe this report will help me build Detectron2 on Windows successfully again in future. Detecron2 is mainly for Linux. If we build it on Linux, I think it is not difficult. Nevertheless, when I build it on Windows, facing many kinds of difficulty. Even though I build it successfully, the program can not still run correctly. The following statements are building steps.  
1. Open the GitHub website “Detectron2 on windows” I mentioned in reference.  
2. Make sure your python environment meet the “Requirements” mentioned in GitHub.(Install git command if you don’t have it)  
3. VS2019 is not necessary. The necessary things is visual studio toolbox. However, I still recommend you to install VS2019 community directly for convenience.  
4. “Change two files manually in pytorch package” mentioned in GitHub  
5. Follow “Build detectron2” cell, Type some commands. Usually, you can build the environment successfully now. However, if you can not run the demo program correctly (get errors), follow step 6.  
6. Check whether you install full version of CUDA. If you just install the cudatoolkit in pytorch, you may get error like me. Thus, install full version of CUDA from NVIDIA website. Restart the computer and rebuild Detectron2 again.

After finishing these steps, I think we can run the code from Detectrin2 correctly. By the way, I am stuck on step 6 for several days.  
　In addition, I also find that we usually need to build the environment to finish the tasks like object detection and instance segmentation. I have fined many source codes on GitHub. If you want to use state-of-the-art methods, building environment is usually necessary. And they are all mainly for Linux. Processing annotation is really annoying. Different source codes have different formats. It takes much time if we want to try difference source codes.

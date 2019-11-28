# CS_IOC5008_0856078_HW3 
# Note

# Reference from GitHub:
Detectron2: https://github.com/facebookresearch/detectron2  
Detectron2 on windows: https://github.com/conansherry/detectron2

# Speed benchmark:

Figure 1. Inference time on Colab (65.5 ms/image)  
  Because I use detectron2 in this homework, the inference code is only one line. And TA said “we only need to choose one image to do inference.”. However, I am still not sure whether this is valid format or not. The result is about 65.5 ms.

# Brief introduction:
  The task of this homework is to do digits detection on Street View House Numbers (SVHN) dataset. There are 33402 training images and 13068 test images in this dataset. I use detectron2 to train faster R-CNN with a ResNet50+FPN (feature pyramid networks) backbone. The result of mAP is 0.46726.

# Methodology:
  This homework I did is mainly follow the tutorial of Detectron2. Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. It provides many kinds of models like faster R-CNN and RetinaNet. Besides, this codebase can do not only object detection but also instance segmentation and so on. I choose faster R-CNN with a ResNet50+FPN backbone to complete this task because the trade off between precision and speed.

# Findings or Summary:
  To be honest, I don’t make much effort to get better accuracy. However, building environment and processing annotation really take much time. Therefore, I want to explain how I build Detectron2 on Windows. I believe this report will help me build Detectron2 on Windows successfully again in future. Detecron2 is mainly for Linux. If we build it on Linux, I think it is not difficult. Nevertheless, when I build it on Windows, facing many kinds of difficulty. Even though I build it successfully, the program can not still run correctly. The following statements are building steps.  
1. Open the GitHub website “Detectron2 on windows” I mentioned in reference.  
2. Make sure your python environment meet the “Requirements” mentioned in GitHub.  
  (Install git command if you don’t have it)  
3. VS2019 is not necessary. The necessary things is visual studio toolbox. However, I still recommend you to install VS2019 community directly for convenience.  
4. “Change two files manually in pytorch package” mentioned in GitHub  
5. Follow “Build detectron2” cell, Type some commands. Usually, you can build the environment successfully now. However, if you can not run the demo program correctly (get errors), follow step 6.  
6. Check whether you install full version of CUDA. If you just install the cudatoolkit in pytorch, you may get error like me. Thus, install full version of CUDA from NVIDIA website. Restart the computer and rebuild Detectron2 again.

After finishing these steps, I think we can run the code from Detectrin2 correctly. By the way, I am stuck on step 6 for several days.  
In addition, I also find that we usually need to build the environment to finish the tasks like object detection and instance segmentation. I have fined many source codes on GitHub. If you want to use state-of-the-art methods, building environment is usually necessary. And they are all mainly for Linux. Processing annotation is really annoying. Different source codes have different formats. It takes much time if we want to try difference source codes.

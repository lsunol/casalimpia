# casalimpia
**AI-powered project to clean and stage room images by separating furniture from architectural features.**

## Summary and Introduction

In this project, we explore the application of Stable Diffusion \+ LoRA for inpainting tasks focused on removing objects from indoor spaces while maintaining structural integrity. Our primary goal is to generate "empty room" images by eliminating furniture and other objects, and preserving the original walls, floors, and ceilings without introducing unwanted artifacts. This can be particularly useful for real estate visualization, where presenting uncluttered spaces can enhance the perception of available space.

To achieve this, we trained the model using a dataset of 13,000 empty room images, applying masks to simulate object removal. Additionally, we leveraged images containing furniture to visually assess the model's performance. For evaluation, we used PSNR, SSIM, and LPIPS metrics.

## Running the Code and Requirements

The project is organized into two main folders:
* `back`: Contains all the fine-tuning training and inference logic  
* `model-and-research`: Contain the experiments and the research process written on Google Colab.

### Setup and execution

The steps to execute this code follows:
1. Download the code from git repo:
    ```
    $ git clone https://github.com/lsunol/casalimpia.git
    ```
2. Download the dataset (optional) from our [Google Drive](https://drive.google.com/file/d/1B_qSXrrq3ibDPmFt4FDFQsz_DGjcp3GD/view?usp=sharing)  
3. Create python environment  
   ```bash  
   conda create -n casalimpia python=3.12  
   conda activate casalimpia
   ```  
4. Install dependencies  
   ```bash
   cd back  
   pip install -r requirements.txt
   ```  
5. Run main.py (see section below)

### Dependencies

#### Cuda drivers in python environment

Before installing `requirements.txt` make sure to install CUDA-enabled versions of PyTorch and Torchvision.

#### Weights & Biases Setup
This project uses the Weights & Biases service, so an account is needed to view results:
1. Create an account at wandb.ai
2. Login through terminal:
    ```bash  
    $ wandb login
    ```

### Machine resources

* **GPU**: Minimum 8 GB VRAM (tested on NVIDIA GeForce RTX 3060\)  
* **Shared RAM**: Minimum 8GB
* **Storage**: \~10 GB for model weights and datasets

### Repository structure

```
casalimpia/
├── back/  
│   ├── data/  
│   │   ├── emptyRooms/             # Empty room images  
│   │   ├── oneformerMasks/         # Mask images  
│   │   └── output/                 # Generated during training  
│   ├── main.py  
│   └── requirements.txt  
├── models-and-research/  
│   ├── dataset_watermarks          
│   │   └── automatic-removal.sh    # Script to remove watermarks from dataset images  
└── docs/                           # Folder containing documentation resources
```

### Running main.py

Key parameters:

| Parameter                  | Description |
|----------------------------|-------------|
| `--empty-rooms-dir`        | Directory containing empty room images, split into train/validation/test |
| `--masks-dir`              | Directory containing corresponding mask images |
| `--epochs`                 | Number of training epochs (default: 20) |
| `--batch-size`             | Number of images per training batch (default: 1) |
| `--output-dir`             | Directory where training outputs will be saved |
| `--lora-rank`              | Rank of LoRA layers (default: 32, controls capacity) |
| `--lora-alpha`             | Scaling factor for LoRA layers (default: 16) |
| `--lora-dropout`           | Dropout probability in LoRA layers (default: 0.1) |
| `--lora-target-modules`    | Which attention modules to apply LoRA to (default: "to_q to_v to_out.0") |
| `--initial-learning-rate`  | Starting learning rate (default: 1e-3) |
| `--lr-scheduler`           | Learning rate schedule ("linear", "cosine", "constant", etc.) |
| `--overfitting`            | Disabled if unspecified: uses one image for the whole training to ensure the network is capable of training (i.e., memorizing) |
| `--save-epoch-tensors`     | Save model weights after each epoch |
| `--shape-warmup-epochs-pct`| Percentage of epochs to use synthetic shape masks (0.0-1.0) |

### Execution examples

#### Getting Help

To view all available parameters and their descriptions, simply run:

```bash  
$ python back/main.py --help  
```

This will display a list of options to customize the training process according to your needs.

#### Training

To train the model, you can run the following command:

```bash
python back/main.py \  
    --empty-rooms-dir back/data/emptyRooms \  
    --masks-dir back/data/oneformerMasks \  
    --output-dir back/data \  
    --model stability-ai \  
    --epochs 20 \  
    --batch-size 1 \  
    --initial-learning-rate 1e-3 \  
    --lora-rank 32 \  
    --lora-alpha 16 \  
    --lora-dropout 0.1 \  
    --lora-target-modules "to_k to_q to_v to_out.0" \  
    --img-size 512 \  
    --dtype float32 \  
    --lr-scheduler cosine \  
    --guidance-scale 1
```

> [!IMPORTANT] 
> The Stable Diffusion model will be downloaded automatically on the first run (\~4GB).

> [!TIP]
> This setup uses a small embedded dataset (10 images) included in the repository, meant for testing the architecture and verifying functionality.

### Colabs (model-and-research/) 

Most of the researches were done by writing notebooks with google collabs. Here you can find the mask generation, infenrece pipeline and training.

```
models-and-resarch/  
├── dataset\_watermarks  
│   └── automatic-removal.sh  
├── inference  
│   ├── sd\_no\_fine-tuning  
│   │   ├── inference\_sd.ipynb  
│   │   ├── inference\_sd\_with\_oneFormer.ipynb  
│   │   ├── inference\_with\_ade20k-resnetdilated-ppm.ipynb  
│   │   └── resources  
│   │       ├── notes.md  
│   │       ├── room\_object\_masks.npz  
│   │       ├── room\_objects.png  
│   │       └── room\_objects\_foreground\_mask.png  
│   └── sd\_with\_LoRA\_fine-tuning  
│       └── inference\_sd\_with\_LoRA\_weights.ipynb  
├── large\_files\_info  
│   └── LARGE\_FILES.md  
├── masks  
│   ├── OneFormerPipeline.ipynb  
│   └── ade20k-resnet50dilated-ppm.ipynb  
└── training-LoRA-sd  
    └── training\_with\_Lora.ipynb
```

## Motivation and References 

Our project addresses a significant gap in the real estate market. When homeowners list properties with furnished rooms, many potential buyers struggle to visualize the true dimensions and potential of these spaces. The furniture may distract from architectural features, make rooms appear smaller, or simply not match a buyer's taste, potentially causing them to overlook promising properties.

While AI solutions exist to virtually furnish empty rooms, we discovered a surprising lack of tools that perform the reverse function - removing existing furniture to show clean, empty spaces. This is of course tied to the inherent limitations of diffusion models and how they are trained to create/hallucinate objects or change style of current objects, but not delete them.

Teaching a model the concept of “empty space” is much harder than teaching what a single object means from a dataset perspective. For example, it’s easy to find many pictures of horses in different contexts on the internet, but in order to show what “empty means” we need pictures of empty space or a comparison (same dataset empty and not empty) \- which is much more scarce.

## Goal

Our goal is to develop an AI-powered de-furnishing tool that transforms images of furnished rooms into realistic empty spaces, enabling real estate sellers to better showcase property potential.

## Paper reference "An Empty Room is All We Want: Automatic Defurnishing of Indoor Panoramas”

From our research, we identified the paper *[An Empty Room is All We Want: Automatic Defurnishing of Indoor Panoramas](https://arxiv.org/abs/2405.03682)*, which shares a similar approach to ours. The key difference is that the authors already had access to a dataset of over 160,000 panoramic unfurnished room images and utilized rendering tools to place objects into these empty spaces.  
Unfortunately, this paper remains solely an academic publication, as the authors have not released any code or dataset to the community. Nevertheless, their research serves as a significant reference for our project goals and methodology.

The following image presents their pipeline:

![][image1]  

### Conclusions obtained from the paper

After carefully studying the methodology in the paper, we found that the authors reached similar conclusions to what we were already considering regarding architecture: using Diffusion models with LoRA fine-tuning. 

The key insight gained from the paper was identifying that the main challenge is avoiding hallucinations. Even the smallest artifacts can cause the Diffusion model to generate objects regardless of the prompt, even when explicitly using phrases like "empty room, no furniture" and other variants.

The authors of the paper employed various strategies to prevent model hallucinations, with their primary approach being the artificial addition of 3D object artifacts (shadows, lighting effects, etc.) from a 3D model library called *Objaverse*. By incorporating these artifacts into their training dataset, they successfully taught the model to generate empty spaces regardless of whether these artifacts were present. They also implemented additional refinements, such as initializing images with 97% noise to provide the model with more context before generation began. Despite these efforts and access to a substantial dataset of high-quality 360° images, the authors still encountered hallucination issues in some cases, highlighting that this remains a significant challenge in the field.

> [!NOTE]
> Objaverse is a massive dataset of annotated 3D objects.

### Experiments definition

Based on insights from the paper, we've decided to gather empty room images from real estate portals (360° imagery being beyond our scope). We'll then use furnished room images and an off-the-shelf segmentation model (OneFormer) to create a repository of realistic object masks that can be overlaid onto empty room images, teaching the model to generate empty space. For inference, this same segmentation model would detect foreground objects in a room and mask them, creating input for the U-Net diffusion model to predict pixels within the masked areas—ideally as empty space.

### Scope of the Project (what we want to achieve)

- We focus exclusively on interior room images, specifically living room and bedroom spaces.  
- We work with images from Spanish real estate portals, recognizing that image quality and camera angles may present challenges for our final results.  
- We will not utilize Blender tools or the Objaverse database, as we would need to invest significant time in projecting furniture into rooms in a realistic manner.

## Framework

Our framework or strategy is based on the references we have obtained, including the research and testing of the current state-of-the-art with inpainting models for solving our leading problem, **being able to de-furnish an image of a room with furniture and obtain or predict the empty room without altering the dimensions and quality of the room space**

In the future sections of “Experiments and Learning,” we explain why we must fine-tune the inpainting model.

### Data

#### What kind of data do we need for this problem?
  
We need a dataset containing images of rooms with furniture as the input and their counterparts as the target of the same rooms without any furniture, more than fixed elements of the rooms such as windows, doors, heaters, and the bare walls, ceiling, and floor.

Unfortunately, we do not have the set of images needed or access to the dataset used in the paper reference, where they had empty room images. With the help of 3D-blending tools and the Objaverse dataset, they rendered objects in the unfurnished spaces. This allows them to have synthetic rooms with objects as input and as targets for the original unfurnished rooms.  
  
We needed a similar strategy or idea of *rendering* images in the unfurnished spaces. Then our plan for the dataset is to fetch one empty room image and another unrelated room with furniture, calculate the binary mask of the room with furniture, and overlay the mask in the room.   
  
We decided to experiment with bedrooms and living room spaces, since these are the places where most of the furniture can be found in a flat or house.  
  
![][image2]  
Then we need:  
* **Input** (synthetic data): Unfurnished room image with a binary mask of an unrelated room with furniture on top of the empty room image.  
* **Masks**: The mask of a unrelated furnished room for the overlaid images.  
* **Target**: The original empty room.

#### Data Collection (how did we get this data?) 

##### Extract the Data from public resources on the Internet and remove any watermark

We needed the most unfurnished and furnished room images available on the internet to be downloaded. In Kaggle, we found specific datasets, but there was a lack of unfurnished rooms, and for furnished rooms, the quantity was not enough, and the quality or angle of the images was not relevant to our cases.   
     
Hence, we decided to focus on real estate portals in Spain that contain a mix of empty and furnished rooms, with a manual curation of the data we yielded 13000 authentic images of bedrooms and living rooms. The advantage of this approach is that it allows us to obtain images with similar proportions and styles since the data collection was only in Spain.   
     
The next challenge was the watermarks on the images that such portals attached to the photos. The issue with those watermarks is that they can provoke hallucinations in the model during the training and inference stages of inpainting.  
     
As a solution, we wrote a script to remove the watermarks of the downloaded images obtained from such portals. You can find it under the folder:  
```
models-and-research/dataset_watermarks/automatic-removal.sh
```  
Finally, we managed to get over 13,000 images of unfurnished and furnished bedrooms and living rooms without any watermarks

#### Semantic Mask

Once we have the dataset, we need to generate the masks for the inpainting process, therefore we need a Semantic Segmentation model able to segment the objects found in a room and identify them (Semantic Process) for recognizing which objects should be considered background and which ones foreground.
- Background: any object representing the composition of the room, such as walls, doors, ceiling lamps, windows, heaters, ceiling, structures, and the floors  
- Foreground: objects considered furniture in a room such as sofas, tables, paintings, beds, pillows, carpets, etc. 

The generation of these masks would serve for two purposes:

1) **TRAINING**: overlay mask in the unfurnished room images during the training process. In this way, we were able to generate the Input Synthetic data and generate the conditioning context for predicting the proper noise.  
   ![][image3]

2) **INFERENCE**: semantic masks for the inference process for the furnished room images. There will be the traditional approach of using the masks for conditioning the output

In both cases, we expect to generate automatic masks to identify only the foreground objects, also known as the *room's furniture*.

For selecting the most appropriate Semantic Segmentation Model, we tried:

- **[Segment Anything Model](https://github.com/facebookresearch/segment-anything)** (*SAM*) by Meta is a universal segmentation that produces high-quality masks for any object and can generate masks automatically without additional training. 

- **[ADE20K-ResNet50](https://github.com/CSAILVision/semantic-segmentation-pytorch)** dilated is a semantic segmentation model built on a ResNet50 backbone that employs dilated convolutions to increase the receptive field without losing resolution. Trained on the ADE20K dataset with 150 semantic categories, it efficiently performs pixel-wise classification for scene parsing tasks.

- **[OneFormer](https://cvpr.thecvf.com/virtual/2023/poster/23168)** is a transformer-based model that unifies semantic, instance, and panoptic segmentation into one framework. OneFormer uses task-conditioned queries and multi-task training to handle all these three types of segmentation.

Under the section *Experiments and Learning*, we show the different results obtained from the three semantic segmentation model approaches.

So far, our data collection and mask generation strategy can be resumed in the following diagram.

![][image4]

#### Train, test and validation

We divided our dataset into three subsets to ensure proper training, validation, and testing. Specifically, **10,000 images** were allocated to the training set, while **1,500 images** were used for validation and another **1,500 images** for testing. This split allowed us to monitor the model's generalization and avoid overfitting. Although masks were essential for training, they were not considered part of the ground truth. Therefore, no distinction was made between them across the different subsets.

### Architecture 

Since our problem falls under the GEN AI Inpainting Problems category, we will use the Stable Diffusion model for inpainting, also known as a latent diffusion model. In a nutshell, this kind of model is a composition of different models working in conjunction, therefore it is considered a pipeline. The main relevance of this workflow is that the process or analysis is happening at the latent space level, meaning that computational resources are lower than if the process occurs at the pixel level.

![][image5]

#### Element Responsibilities

* The **Variational Autoencoder** (*VAE*) is used in our project to **convert images into a latent representation** before feeding them into the UNet. It compresses the input image into a lower-dimensional latent space, reducing computational requirements while preserving key visual features. We use the pre-trained **VAE** without modifications, as we do not need to fine-tune it for our purposes. Although we initially used the **VAE decoder** to verify that it was functioning correctly, it is not a key component in our training process beyond its role in encoding the images.  
* The **CLIP** model is used in our project to **extract text embeddings from prompts**, providing a way to condition the model on textual descriptions. We use a pre-trained CLIP model without modifications or fine-tuning. Its role is strictly limited to processing textual input and transforming it into meaningful numerical embeddings, which are then used as conditioning signals for the UNet during training and inference.  
* The **UNet** is the core of our inpainting pipeline, responsible for denoising the latent space and reconstructing missing parts of the image. We have not modified the UNet's architecture beyond adding **LoRA** (Low-Rank Adaptation) modules, which allow for more efficient fine-tuning. The UNet receives as input:  
  * Four latent channels of the masked input image.  
  * Four latent channels of the ground truth (target image).  
  * One additional mask channel, downscaled to 64x64, which matches the UNet's expected input size.

  Rather than training the entire UNet, our focus has been on fine-tuning the LoRA layers, optimizing only a small subset of the model's parameters to adapt it to the inpainting task efficiently

* **SCHEDULER**: It has a straightforward role used in the training and inference process. For training, the scheduler determines how noise is progressively added to furnished room images according to a specific schedule, providing the timestep parameters that control noise levels at each training step. During inference, the scheduler orchestrates the step-by-step denoising process, calculating precisely how much noise to remove from masked areas at each timestep based on the UNet's predictions.

#### Current state-of-the-art for diffusion models (how are they falling?)

Diffusion models such as Stable Diffusion Inpainting and LaMA are focusing on inpainting problems where a region of an image (the mask) is replaced with a target element. This behaviour is excellent for scenarios where we can replace objects in an image, but for our case we want to use inpainting for removing objects and predicting their background.

Our goal is going in another direction, where inpainting diffusion models were not trained, however, we need to test how good or bad the diffusion models can respond to such requests.

Therefore, we decided to test the stable diffusion model for inpainting since, by the paper reference and other research on the internet, the SD model seems to bring the best outcomes.

For the testing, we created a notebook with an interactive loop where by each iteration five outputs were generated, and each one of them will count with different variations in the hyperparameters, particularly for the guidance scale and prompts, since we are trying mainly to guide the model to the best outcome. In the following image, we can see the main idea for testing the current state-of-the-art

![][image6]  

> [!TIP]
> You can play around with the two different masks and interactive inference process in the following colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3cLX94FvBr9BxmHG3IK45w4aRLmfr-E?usp=sharing)

Results for different variations in different iterations follow:

| Input image      | Mask          | Results on the first iteration
| :---:            | :---:         | :---:
| ![][image7]      | ![][image8]   | ![][image9] 

From the image above, we can see that only 1 out of 5 output images is going in the proper direction, this is a challenge for our goal since we are expecting the first iteration of the five images to be unfurnished and keeping the room dimensions and not altering their background. On the other side, in variation 3, we can see that the shadows are another problem where future mask generations are not able to capture any objects in the room.

Finally, we can conclude that the shadows of the images can provoke hallucinations in the result, and the values of the best hyperparameters are not deterministic. 

Hence, we hypothesize that fine-tuning of the SD could bring better results. The LoRA (Low-Rank Adaptation) concept was decided to be used as a technique since it does not involve fine-tuning all the parameters of the model, only to add more parameters and train only the new ones. The image below shows the concept of LoRA:

![][image10]

#### Our pipelines for inpainting (training vs inference)

##### Training

Pre-processing Input Image  
![][image11]

Our input data is a synthetic furnished room (the overlay unrelated furniture room binary mask image put over the unfurnished room space).  
The image above indicates how we prepared the input image.

###### Forward Process

![][image12]

* The input image and mask are encoded through the VAE into the latent space.  
* A random timestep (t) is selected per each training iteration, where a Gaussian noise (ε) is sampled with the same shape as the latent representation. The intensity of noise is determined by the noise schedule. Later, the noise is applied only to the latent image, resulting in a noisy latent that will be fed into the U-Net.

###### Complete Fine-Tuning Training Process

![][image13]

Once the latent image with noise and latent mask are generated, the conditioning part will be made of text prompts, including negative prompts, the latent mask and the scheduler that will inform the U-Net the intensity of noise added by each step.  
The U-Net will be fed with the latent image with noise (*noisy latent*) and the conditioning attributes guiding the U-Net to predict the added noise. During this learning process only the layers of LoRa mainly attached to the cross-attentions U-Net layers will be called for predicting the noise. Therefore, they will be affected in the backpropagation process.
After the predicted noise is calculated, the loss calculation is triggered where the backpropagation is called and the LoRa weights are adjusted.


##### Inference

###### Initialization

![][image14]

![][image15]

In this stage, the original image (furnished room image) is concatenated with its mask and encoded to the latent space.  
Pure random noise is only added to the mask region of the concatenated image, meaning that the unmasked region retains its original latent values.

###### Complete inference process

![][image16]  
After initializing, the process, we will need to define the hyperparameters such as time steps and prompt, which will serve as conditional signals that will be fed into the U-Net along with the noisy latent  
During the Denoising process, the U-Net using the LoRa pre-trained weights will remove the added noise of the mask region from the Noisy latent input. This will be a loop with an end of *t* inference steps. By each *t* step, the previous output of the U-Net is used as Input for the next Step.  
During this recurrent process, only the mask region should be altered, the rest of the latent values are preserved.  
Once the loop is concluded, the decoding process should bring back the output to the pixel space where the desired result of the unfurnished room image is generated.  
> [!NOTE]
>  "pixel space" refers to the final visual representation of an image with actual RGB pixel values that can be displayed, in contrast to the latent space where diffusion models operate during their processing steps.
    
## Experiments and learnings

### Experimental Setup 

#### Development and testing environments

Throughout our project, we leveraged different computing environments based on the needs of each phase:

1. Our primary development and training took place in **local environments** using **Visual Studio Code** (VSC) and high-performance GPUs, such as NVIDIA RTX series. This setup enabled us to conduct most of the experimentation, including mask generation with OneFormer, data type optimization (float16, float32, mixed precision), and multiple iterative LoRA training runs. The advantage of working locally was the ability to perform faster iteration cycles and in-depth debugging, which was crucial for refining our approach.  
2. **Google Colab** was used for lightweight trials and early-stage testing, particularly when team members didn't have immediate access to high-performance local machines. This platform allowed us to run basic model inference and small-scale mask generation experiments, using Colab’s GPU resources to validate initial concepts with minimal computational overhead  
3. For the most computationally demanding tasks, such as large-scale training, we utilized **Google Cloud**. With access to powerful GPUs like the A100, we were able to run extended training sessions, including our full 13,000-image dataset. This ensured scalability and efficiency, particularly in the final stages of model convergence, where mixed precision and OneFormer-generated masks were fully leveraged to optimize performance.

#### Experiment tracking/monitoring

To track and analyze our training runs, we used **Weights & Biases (W\&B)**, a powerful tool for experiment tracking and visualization. Our project, documented on W\&B at [this link](https://wandb.ai/lsunol-pgai/casalimpia), includes **over 500 runs**, allowing us to monitor progress and refine our approach.

As the project evolved, we incrementally added more insights, logging not only **quantitative metrics** like PSNR, SSIM, and LPIPS, but also **qualitative outputs** by uploading generated images of inferences on real furnished rooms. Additionally, we tracked the **model weights** from different training sessions, making it easier to compare versions and assess improvements.

We also leveraged **W&B Sweeps**, a feature designed to automate hyperparameter optimization. Sweeps enabled us to systematically explore different configurations and identify correlations between hyperparameters and model performance, helping us fine-tune our approach more effectively.

#### Metrics

In our project, we evaluated the performance of our inpainting model using three key metrics: PSNR, SSIM, and LPIPS. 

* *Peak Signal-to-Noise Ratio* (PSNR) measures the pixel-wise difference between the generated and target images, with higher values indicating better reconstruction fidelity.   
* *Structural Similarity Index* (SSIM) assesses perceptual similarity by comparing structural information such as luminance, contrast, and texture.   
* *Learned Perceptual Image Patch Similarity* (LPIPS) goes beyond pixel-based comparisons by using deep network embeddings to estimate perceptual differences, making it more aligned with human visual perception.

During training, we applied these metrics to images that were originally empty, ensuring that our model accurately reconstructed the structural elements of the rooms. However, we also conducted a perceptual evaluation using inpainting on furnished rooms to observe how the model handled object removal. Since we do not have ground truth images of these rooms without furniture, this evaluation was purely qualitative, allowing us to visually assess the network’s performance in real-world scenarios.

### **Experiment 1: Mask Generation Techniques**

#### Hypothesis  
Selecting the optimal method for auto-generating high-quality masks to isolate objects (furniture) from room images.
#### Setup
**Firstly**, we began with an initial code using the **MIT ADE20K** model, implementing six basic mask generation methods to refine segmentation:  
|   |   |
|---|---|
|**Raw Mask** | Directly derived from the segmentation output, serving as the baseline with unprocessed foreground isolation. |
|**Cleaned Mask** | Applied `clean_mask` with a 7x7 kernel to remove small isolated noise regions using binary opening, though it often left gaps.|
|**Smoothed Mask** | Used `smooth_mask` with a 7x7 Gaussian filter and a 0.3 threshold to reduce noise, but it blurred edges excessively.|
|**Closed Mask**| Employed `close_mask` with a 9x9 kernel for binary closing and dilation to fill internal holes, yet it sometimes over-expanded boundaries.|
| **Refined Mask**| Utilized `refine_contours` with OpenCV to apply convex hulls to contours, aiming to correct shapes, but it failed to capture fine details like chair legs.  |
| **Combined Approaches**|We experimented with combinations (e.g., cleaned \+ closed \+ smoothed and raw \+ refined) to enhance results, adjusting kernel sizes and thresholds.|

Despite these efforts, the masks consistently failed to segment fine structures accurately, a limitation tested with both COCO and ADE20K datasets, prompting further exploration.   

**Secondly**, we tested the **ckpt/ade20k-resnet50dilated-ppm_deepsup** checkpoint, loading it into the segmentation module to improve mask quality. However, this approach also proved inadequate, producing masks with noticeable holes and misplaced spots, failing to address the fine detail issue.  
     
**Thirdly**, and finally, we adopted **OneFormer**, integrating its pre-trained model and tailoring the pipeline to its output format. This shift enabled the generation of high-quality masks that successfully captured fine details, leading to the creation of the definitive mask set for training the inpainting pipeline.

> [!IMPORTANT]
> While automating mask generation, we manually created some masks to support the initial stages of the project, allowing progress on the inpainting task before an automated solution was fully developed.

> [!IMPORTANT] 
> To handle varying image resolutions, we initially unified mask dimensions by adding borders, which caused misplaced segmentation, but resolved this by applying zoom-based resizing for consistent and accurate mask dimensions.

#### Results

| Example using **MIT ADE20K**|
| --- |
| ![][image100] |

| Example using **ckpt/ade20k-resnet50dilated-ppm_deepsup**|
|---|
|![][image101]|

| Example using **OneFormer** |
|---|
|![][image102]|

| Example of the mask from image with border|
|---|
|![][image17] ![][image18]|

The initial ADE20K and COCO-based methods, even with extensive post-processing, produced masks with significant gaps and failed to detect fine details like chair legs, rendering them unsuitable for inpainting. The ade20k-resnet50dilated-ppm\_deepsup checkpoint offered marginal improvements but introduced holes and misplaced spots, further highlighting its limitations. In contrast, OneFormer delivered precise masks that captured intricate structures, forming the basis for the final training dataset and proving superior for local training and evaluation.

#### Conclusions  
The journey from initial segmentation to OneFormer underscored the need for advanced tools to handle fine details in mask generation, a process that required significant effort through numerous unmentioned hypotheses and experiments to achieve satisfactory results. Addressing mask dimension inconsistencies via zoom-based resizing proved effective, eliminating artifacts seen with border padding. While manual masks facilitated early progress, OneFormer emerged as the best solution, offering robust and accurate segmentation. For future training on platforms like Google Cloud, OneFormer’s scalability and precision, combined with zoom-based resizing, make it the recommended approach, potentially enhanced with automated tuning to optimize large-scale mask production, building on the extensive groundwork laid to overcome the initial challenges.

### Experiment 2: Overfitting

#### Hypothesis

If our training pipeline is correct, the model should be able to overfit perfectly on a single image given enough iterations. By doing this, we ensure that:

* The LoRA adaptation is working correctly.
* The VAE and latent processing are coherent.
* The masking and conditioning process is properly aligned.
* The optimizer and learning rate are not preventing convergence.

A perfect overfit means achieving a very high PSNR and low losses, indicating that the model memorized the target image.

#### Setup

We used a single input image with its corresponding mask and target output, running thousands of training epochs while adjusting LoRA rank, alpha, and other parameters to analyze their impact on convergence. The goal was to achieve the highest PSNR possible, ensuring the model could reconstruct the target image with maximum fidelity.

#### Results 

The best overfitting run reached a `PSNR` of 30.12, with the model visually reproducing the target image by epoch 1200. Consequently, the model correctly reconstructed fine details such as the wall socket and ceiling lights, even though they were masked out in the input. This suggests that the model learned to infer missing details based on structural patterns and prior knowledge embedded in Stable Diffusion's training data.   
To illustrate the results of this experiment, we present the following 2 images.   
These are 3 samples of the best `PSNR` results:  
![][image19]  
The visual result of the best run, achieved at the epoch 1200
| Masked image | Ground truth | Inferred result |
|<td colspan="3"> ![][image20]</td> |

#### Conclusions

The experiment confirmed that our pipeline functions correctly, as the model successfully memorized the image, proving the integration of LoRA and VAE. Additionally, the model demonstrated a capacity for plausible detail reconstruction beyond what was provided in the input, indicating that inpainting benefits from learned priors. While overfitting ensures the system works, real training must avoid excessive memorization to maintain generalization.

### Experiment 3: Margin Mask**

#### Hypothesis
Despite training with empty room photos and object masks, the model still hallucinates objects during inference on previously furnished images. We believe this occurs because masks fail to capture certain artifacts that trigger hallucinations—including undetected object fragments, shadows, lighting effects, and other object-related visual cues.

#### Setup
To address this efficiently, we collected a 20px colored margin around each segmentation mask from furnished rooms. These margin masks likely contain the problematic artifacts. During training, before merging masks with empty room photos, we overlaid these margins onto the empty rooms as real pixels, creating artifact "contours" around the core masks. Our goal was to teach the model to generate empty space even when artifacts are present.

#### Results
This approach produced better results than without margin masks, suggesting the model is learning in the right direction. However, we still haven't achieved complete object removal, likely due to model underfitting as evidenced by flat training loss.

#### Conclusions
Though imperfect, this approach shows promise. With different optimizers, increased learning capacity through U-Net training, and other refinements, this strategy could potentially succeed.

### Experiment 4: Different Type of Data (float16 / float32 / mixed)

#### Hypothesis

Our hypothesis centered on selecting the optimal dtype for training the Stable Diffusion (SD) inpainting model, particularly after encountering errors such as NaN gradients and incorrect PSNR calculations (e.g., PSNR \= 0). We anticipated that float16 could reduce memory demands but might exacerbate these issues due to numerical instability, while float32 would offer stability at a higher memory cost, and mixed precision could mitigate errors while maintaining efficiency.

#### Setup

We began by configuring the SD model with LoRA, initially setting all components (VAE, UNet, text encoder) to float32. Upon encountering numerical errors and inconsistencies in logs (e.g., PSNR stuck at 0), we shifted to experimenting with float16 by updating the pipeline to torch\_dtype=torch.float16 and enabling torch.autocast to leverage mixed precision dynamically during training and inference steps. Recognizing type mismatches between the UNet (still in float32) and the pipeline, we realigned the UNet's dtype to float16, adjusting its loading to match the pipeline's precision. 

To address gradient underflow and NaN issues observed with float16, we integrated GradScaler to scale gradients appropriately, experimenting with different scaling factors to stabilize training. We also modified the training loop to include explicit checks for NaN values in loss calculations and adjusted the learning rate scheduler to adapt to the lower precision, ensuring compatibility with the LoRA attention modules. Throughout, we iteratively refined the setup by monitoring logs and WandB for signs of OOM errors, numerical instability, or degraded image quality, making incremental changes to the dtype configuration to optimize performance.

#### Results

Full float32 training was stable but memory-intensive, often causing OOM errors, while float16 reduced memory usage but initially led to NaN errors, requiring a lower learning rate for stability. Mixed precision with autocast balanced memory savings and stability, minimizing errors like incorrect PSNR values. After adjustments, float16 with GradScaler became the primary choice for local training, effectively reducing memory demands while maintaining acceptable stability and quality.

#### Conclusions 

While float16 proved viable for local training with careful tuning, mixed precision emerged as the most promising approach for future training scenarios, such as on Google Cloud. Leveraging torch.autocast and GradScaler in mixed precision offers a robust solution, combining memory efficiency, training speed, and numerical stability, making it ideal for scaling up on cloud platforms with more demanding computational resources.

### Experiment 4: Finding the best hyperparameters for the training

#### Hypothesis
By systematically tuning hyperparameters, we aim to identify an optimal configuration that maximizes model performance while maintaining training efficiency. We expect that selecting appropriate values will lead to improved inpainting quality while preventing overfitting or underfitting.

#### Setup

1. Based on preliminary overfitting runs, we established a **base configuration** of hyperparameters as a starting point for tuning.

2. **float16** precision was chosen as the most optimal setting for local training, improving both memory efficiency and computational speed.

3. Using **early Stopping**, we determined that 100 epochs was the ideal number, as only a few runs could arrive at the end.

4. Hyperparameter search was performed iteratively, following a **custom-made selection guide**, modifying one parameter at a time while keeping others constant. 

5. The **evaluation metrics** included PSNR, training loss, validation loss and visual inspection, ensuring a balance between reconstruction quality and training stability.

#### Results

As an example, the guide of the search and the best results (light green) can be observed in the next table:

![][image21]

On the one hand, the basic selection criterion has been fundamental metrics such as `PSNR` or losses. Below is an example of optimizer selection based on PSNR calculated at each epoch, where Adagrad (red one) emerged as the standout performer, successfully completing all 100 epochs unlike the other optimizers.

![][image22]

On the other hand, as already mentioned, one of the definitive selection criteria has been a visual inspection of the results, since metrics can be misleading, especially due to the effect of shadows. An example of one of the best runs of `guidance scale` (value of 7.5) can be observed in the following samples illustration.

| input image           | inferred image     | ground truth image |
| ---                   | ---                | ---                |
|<td colspan="3"> ![][image23] </td>|

In addition, we also leveraged **Weights & Biases Sweeps** to get automated results. This allowed us to systematically explore different configurations and identify which parameters had the strongest correlation with PSNR.

We conducted nearly 200 runs, varying the following parameters:

|   |
|---|
| Learning rate scheduler (lr-scheduler)  |
| Initial learning rate (initial-learning-rate)  |
| LoRA alpha (lora-alpha)  |
| LoRA rank (lora-rank)  |
| LoRA dropout (lora-dropout)  |
| LoRA target modules (lora-target-modules)  |
| Guidance scale (guidance-scale)  |

After analysing the results, we found that initial learning rate, learning rate scheduler, and LoRA dropout had the highest impact on overfitting performance, showing the strongest correlation with final PSNR.  
The following figure illustrates the PSNR obtained across different runs, highlighting the influence of these parameters:

![][image24]  

Additionally, we computed correlation coefficients to determine which parameters influenced PSNR the most. The following visualization demonstrates the impact of each hyper-parameter on the final results:

![][image25]

Our hyper-parameter analysis revealed that LoRA dropout, LoRA alpha, and initial learning rate had the most significant impact on training performance, showing similar importance levels. While guidance scale also influenced PSNR, it is an inference parameter rather than a training factor, meaning it improves output quality but does not directly affect model convergence.

#### Conclusions

The hyperparameter selection process was an **extensive and time-consuming** task, requiring a **total of 46 manual runs (8 days)** and **190 automated runs**. While we achieved a stable and well-performing configuration, the results are **not yet ideal**, as the training was limited to **only 100 epochs on manual tests and 200 epochs on automated _wandb_ sweeps**.

However, these experiments provided a **solid starting point** for the next phase, where higher computational power and longer training allows further improvements. The selected hyperparameters will serve as the **baseline** for fine-tuning on the full dataset of **12,000 images**, aiming for better inpainting quality and generalization.


## What is next?

The project has laid a strong foundation for inpainting empty rooms, but several avenues remain to be explored to enhance performance and applicability. The following tasks outline our planned future directions:

1. **Testing More Sophisticated Optimizers**: While optimizers like Agrad were used in the current training pipeline, exploring more advanced options such as **prodigy** could potentially improve convergence speed and stability. These optimizers could better handle the high-dimensional parameter space of diffusion models, especially during long training runs on Google Cloud, offering opportunities to achieve higher PSNR values and more robust inpainting results with less computational overhead.  
     
2. **Refining Mask Generation Strategy**: The current mask generation approach covers a large surface area by masking all objects, including non-movable elements like built-in wardrobes, which may not align with practical use cases such as preparing a room for sale. A revised strategy will focus on masking only movable furniture (e.g., tables, sofas, lamps) while preserving fixed structures, reducing the inpainting area and potentially improving the model’s focus on structural consistency, which can be tested using OneFormer with updated foreground detection criteria.  
     
3. **Mitigating Shadow Effects**: Shadows in images, not covered by masks, can mislead the diffusion model into interpreting them as cues for object placement, contrary to our goal of generating empty rooms. Future work will explore shadow detection and removal techniques, such as integrating shadow segmentation models or preprocessing images to minimize shadow effects, ensuring the model focuses solely on structural elements and improves the realism of inpainted backgrounds.  
     
4. **Incorporating edge detection**: Inspired by [EdgeConnect](https://medium.com/data-science/lines-first-color-next-an-inspirational-deep-image-inpainting-approach-b2d980efb364#:~:text=%E2%80%9CLines%20First%2C%20Color%20Next%E2%80%9D%20An,The%20first%20generator%20is), we plan to enhance inpainting by first generating edge maps that define room boundaries, such as transitions between walls, floors, ceilings, windows, and non-removable objects. By creating these structural edges before inpainting, the model can better preserve the room’s geometry and focus inpainting efforts on the masked regions, potentially leading to more seamless and contextually accurate results, which will be tested on Google Cloud with the refined dataset.
![][image103]  

5. **Training UNet directly without LoRA on GoogleCloud**: The current approach uses LoRA for efficient fine-tuning, but directly training the UNet without LoRA on Google Cloud could unlock higher performance by allowing full parameter optimization. This resource-intensive experiment will leverage Google Cloud’s A100 GPUs to train the entire UNet on the 13,000-image dataset, potentially improving inpainting quality at the cost of increased computational demands, offering a benchmark to compare against the LoRA-based results.  
     
6. **Exploring Newer Stable Diffusion Versions**: Currently, the project utilizes Stable Diffusion V2 for inpainting, but newer versions (e.g., XL, Turbo or V3) offer improved performance and enhanced feature sets. We plan to experiment with these updated models to assess potential gains in image quality, training efficiency, and adaptability to our empty room task, leveraging Google Cloud’s resources to fine-tune and evaluate these versions with our dataset, potentially surpassing the capabilities of V2 in structural preservation and inpainting accuracy.

## Best Results

Our initial results were poor due to an implementation issue that affected how the inpainting mask was applied. Specifically, there was a misalignment between the masked area in the input image and the mask provided to the model. This caused the model to attempt inpainting over an unintended region, often leading to unnatural borders and artifacts. Although the training loss was decreasing, the model struggled to overcome this inconsistency:  

| masked image | inferred image | ground truth|
|<td colspan="3"> ![][image26] </td>|

Once we identified and fixed this bug, achieving overfitting on a single image became much easier. By training the model with only one image and its corresponding mask, we were able to reach a state where the model almost perfectly reconstructed the target. This proved that our training pipeline was functioning correctly and that the LoRA adaptation was working as expected:  
| masked image | ground truth| inferred image |
|<td colspan="3"> ![][image27] </td>|

However, throughout our training process, we noticed that the masks used did not always align with the background structure of the room in terms of geometry, size, and perspective. This led us to experiment with an initial warmup phase where, instead of using complex object-shaped masks, we applied simple geometric shapes. This approach allowed the model to first learn to fill in basic missing regions before handling more intricate inpainting tasks. The results from this warmup phase were promising, as the model was able to handle these simpler cases quite effectively:  

| masked image | ground truth| inferred image |
|<td colspan="3"> ![][image28] </td>|


Finally, we tested our model on real furnished rooms where objects were naturally placed within the scene rather than overlaid. Here, the model often failed to produce the desired results. Instead of removing objects completely, it tended to replace them with stylistically similar elements, rather than generating an empty space. This suggests that the model is strongly biased toward object reconstruction rather than true object removal:  
| masked image | ground truth| inferred image |
|<td colspan="3"> ![][image29] </td>|

## Conclusions

This project highlighted key challenges in training a diffusion model for object removal, emphasizing dataset quality, model behaviour and training limitations. Below are our main takeaways:

1. Dataset quality is crucial. Since no datasets exist with the same rooms both furnished and unfurnished, we had to create our own approach. The lack of direct supervision made it difficult to prevent the model from hallucinating objects where none should exist.  
2. Diffusion models naturally hallucinate, making object removal difficult. These models are designed to generate content, not erase it. LoRA helped adjust their, but fully suppressing object reconstruction proved challenging. The key issue was ensuring that empty areas remained empty.  
3. Pre-trained models have biases we cannot fully control. The Stable Diffusion model likely distinguishes originally empty rooms from those with removed objects based on subtle cues like lighting and perspective. These biases made it difficult to override certain behaviours without deeper modifications.  
4. Training showed progress, but more time was needed. While results improved, time constraints limited hyperparameter exploration. Testing different optimizers, adjusting learning rates, and even training the UNet itself could have led to better performance. Further fine-tuning remains an open avenue for improvement.


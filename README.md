# segment-anything

- This project focuses on enhancing the [Segment Anything Model (SAM)](https://segment-anything.com/) for remote sensing applications, specifically for segmenting sidewalks from satellite imagery. 

- The work involves setting up a development environment with GIS tools, replicating SAM implementation for satellite data, and then finetuning the model on a specialized sidewalk dataset. Key challenges include selecting appropriate loss functions and segmentation metrics for narrow, often occluded features like sidewalks. 

- The project culminates in developing a web application using Shiny for Python and Hugging Face Spaces, allowing users to upload satellite images and receive instant sidewalk segmentations. 

- This effort combines cutting-edge computer vision techniques with practical applications in urban planning and mapping, contributing to the rapidly growing field of remote sensing. The project demonstrates the potential of large foundational models in computer vision and their impact on various domains including urban planning, agriculture, and environmental monitoring.

- Model trained on High Performance Computing (hpc) is now on [Hugging Face](https://huggingface.co/ttd22/segment-anything/blob/main/sam_model.pth).

- [Test your satellite images](https://huggingface.co/spaces/ttd22/segment-anything-model) here or use sample satellite images to segment sidewalks.
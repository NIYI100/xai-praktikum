# Uncertainty Handling for VLMs in Robot Navigation

This repository contains the project developed for the Explainable AI practical in winter semester 24/25 at the Karlsruhe Institute of Technology (KIT). The project investigates how uncertainty can be estimated for Vision Language Models (VLMs) used in robotic pointing and navigation tasks.

## Use Case

VLMs can guide a robot by identifying relevant objects or locations in an image. In safety-critical settings, however, an incorrect and overconfident prediction can lead to unsafe actions. This project evaluates uncertainty measures for two robotic VLMs, **Robopoint** and **Molmo**.

The experiments analyze output-token probabilities, the spatial spread of predictions, clustering with DBSCAN, and Multiple-Choice Question Answering (MCQA). The results indicate that token probabilities and prediction spread provide limited information about uncertainty, while clustering combined with MCQA shows promising potential for distinguishing relevant from irrelevant pointing options.

## Used Vision Language Models

**[RoboPoint](https://arxiv.org/abs/2406.10721)** is a vision-language model designed to predict spatial affordances, such as image keypoints, from language instructions for robotic tasks. It uses instruction tuning and synthetic training data to support applications including robot navigation and manipulation.

**[Molmo](https://arxiv.org/abs/2409.17146)** is a family of open-weight vision-language models developed together with the open PixMo datasets. Its training includes detailed image captions, visual question answering, and 2D pointing data, making it suitable for spatial visual reasoning.

## Data and Datasets

The experiments use robotic manipulation data from BridgeData V2 and the `kit_irl_real_kitchen` dataset collected by the Intuitive Robots Lab at KIT. The datasets contain images and trajectories from toy kitchen environments with natural-language pick and pick-and-place tasks. In addition, derived experiment sets were created to focus on ambiguous tasks where multiple image regions could provide a valid solution.

## Repository Contents

- `molmo/` - Jupyter notebooks for experiments and analyses using Molmo.
- `robopoint/` - Jupyter notebooks for experiments and analyses using Robopoint.
- `data/` - Experiment images, trajectories, and derived datasets.
- `molmo_utils.py`, `robopoint_utils.py`, `utils.py` - Shared helper functions for model inference, data processing, visualization, and evaluation.
from PIL import Image
import requests
import torch
import os
import matplotlib.pyplot as plt
import re
from ast import literal_eval
import numpy as np
import json

def extract_tasks_and_images(path_to_directory):
    tasks = []
    images = []

    for experiment in os.listdir(path_to_directory):
        if experiment.startswith("."):
            continue
        subdir_path = os.path.join(path_to_directory, experiment)
        task_file_path = os.path.join(subdir_path, "lang.txt")

        # Get Task
        with open(task_file_path, "r") as lang_file:
            task = lang_file.read()
            tasks.append(task)

        # Get image
        for file in os.listdir(subdir_path):
                    if file.startswith("im_") and file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(subdir_path, file)
                        #image = Image.open(image_path)
                        images.append(image_path)

    return tasks, images 

def extract_all(path_to_directory):
    tasks = []
    images = []
    objects_list = []
    groundtruths = []

    for experiment in os.listdir(path_to_directory):
        if experiment.startswith("."):
            continue

        subdir_path = os.path.join(path_to_directory, experiment)
        task_file_path = os.path.join(subdir_path, "lang.txt")
        objects_file_path = os.path.join(subdir_path, "objects.txt")
        groundtruth_file_path = os.path.join(subdir_path, "groundtruth.txt")

        # Get Task
        with open(task_file_path, "r") as lang_file:
            task = lang_file.read()
            tasks.append(task)

        with open(objects_file_path, "r") as lang_file:
            content = lang_file.read().strip()
            if content:  # Check if the file is not empty
                objects = json.loads(content)
            else:
                objects = []
            objects_list.append(objects)

        # Get Groundtruth
        with open(groundtruth_file_path, "r") as gt_file:
            line = gt_file.read().strip()
            groundtruth = literal_eval(line)  # Safely evaluate the string as a Python literal
            groundtruths.append(groundtruth)

        # Get image
        for file in os.listdir(subdir_path):
                    if file.startswith("im_") and file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(subdir_path, file)
                        #image = Image.open(image_path)
                        images.append(image_path)

    return tasks, images, groundtruths, objects_list

def extract_all_trajectories(path_to_directory):
    tasks = []
    images = []
    objects_list = []
    groundtruths = []

    for experiment in os.listdir(path_to_directory):
        if experiment.startswith("."):
            continue

        subdir_path = os.path.join(path_to_directory, experiment)
        task_file_path = os.path.join(subdir_path, "lang.txt")
        objects_file_path = os.path.join(subdir_path, "objects.txt")
        groundtruth_file_path = os.path.join(subdir_path, "groundtruth.txt")

        # Get Task
        with open(task_file_path, "r") as lang_file:
            task = lang_file.read()
            tasks.append(task)

        with open(objects_file_path, "r") as lang_file:
            content = lang_file.read().strip()
            if content:  # Check if the file is not empty
                objects = json.loads(content)
            else:
                objects = []
            objects_list.append(objects)

        # Get Groundtruth
        with open(groundtruth_file_path, "r") as gt_file:
            line = gt_file.read().strip()
            groundtruth = literal_eval(line)  # Safely evaluate the string as a Python literal
            groundtruths.append(groundtruth)


        # Filter files that start with a number and have the correct extension
        pictures = [
            f for f in os.listdir(subdir_path)
            if f.split('.')[0].isdigit() and f.endswith((".png", ".jpg", ".jpeg"))
        ]
    
        # Sort files numerically by the number in the name
        image_trajectory = []
        for file in sorted(pictures, key=lambda x: int(x.split('.')[0])):
            image_path = os.path.join(subdir_path, file)
            #image = Image.open(image_path)
            image_trajectory.append(image_path)
        images.append(image_trajectory)
            
    return tasks, images, groundtruths, objects_list

def visualize_points_on_image(image_path, labels, list_of_coordinates, title="Coordinates on Image"):
    with Image.open(image_path) as image:
        plt.imshow(image, alpha=1)
        image_width, image_height = image.size
    
        for label, coordinates in zip(labels, list_of_coordinates):
            # Extract and plot the points
            x_coords = [x for x, y in coordinates]
            y_coords = [y for x, y in coordinates]
            plt.scatter(x_coords, y_coords, marker='o', label=label)
        
        # Add labels and show the plot
        plt.title(title)
        plt.legend(loc="lower right")
        plt.axis("on")  # Show axes
        plt.show()
        plt.close()

def plot_euclidean_bplot(labels, list_of_coordinates, ground_truths, title="Euclidean Distance Boxplots"):
    data = []
    for i in range(len(labels)):
        distances = []
        ground_truth = ground_truths[i]
        coordinates = list_of_coordinates[i]
        for point in coordinates:
            euc_dist = np.sqrt(np.square(ground_truth[0] - point[0]) + np.square(ground_truth[1] - point[1]))
            distances.append(euc_dist)
        data.append(distances)

    # Create the boxplots side by side
    plt.boxplot(data)
    plt.xticks(np.arange(1, len(labels) + 1), labels)  # Set x-axis labels
    plt.ylabel('Euclidean distance from ground truth')
    plt.title(title)
    plt.show()
    plt.close()

def plot_loglikelihood_bplot():
    raise NotImplementedError

def close_all_images(root_dir):
    closed_images = 0
    # Walk through all subdirectories and files in the given root directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Check if the file is an image based on its extension
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(subdir, file)
                try:
                    image = Image.open(image_path)
                    image.close()
                    closed_images += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
    print(f"Found and closed {closed_images} images.")
    
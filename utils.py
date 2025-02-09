from PIL import Image
import requests
import torch
import os
import matplotlib.pyplot as plt
import re
from ast import literal_eval
import numpy as np
import json
import math
import itertools
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

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

def visualize_points_on_image(image, labels, list_of_coordinates, title="Coordinates on Image"):
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
            euc_dist = calculate_euclidian_distance(ground_truth, point)
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


"""
labels: One label per task
list_of_probs: [[task1_prob1, task1_prob2], [task2_prob1, task2_prob2], ...]
list_of_distances: [[task1_dist1, task1_diat2], [task2_dist1, task2_dist2], ...]
"""
def plot_scatter(labels, list_of_probs, list_of_distances, title="Scatterplot", x_label="Distance to Groundtruth", y_label="Probabilities"):
    for label, probs, distances in zip(labels, list_of_probs, list_of_distances):
        plt.scatter(distances, probs, marker='o', label=label)
    
    # Adding labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc=(1.04, 0))
    
    # Show the plot
    plt.show()
    plt.close()


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

def calculate_euclidian_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calculate_normalized_euclidian_distance(coord1, coord2, width, height):
    norm1 = (coord1[0] / width, coord1[1] / height)
    norm2 = (coord2[0] / width, coord2[1] / height)
    return calculate_euclidian_distance(norm1, norm2)

def calculate_spread(coordinates, width, height):
    max_dist = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = calculate_normalized_euclidian_distance(coordinates[i], coordinates[j], width, height)
            max_dist = max(max_dist, dist)
    
    return max_dist


def cluster_data(coordinates, epsilon=10, min_samples=3):
    coordinates = np.array(coordinates)
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coordinates)
    labels = db.labels_

    # Calculate number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if (n_clusters == 0):
        print("Didnt find any cluster")
        return n_clusters, 0, 0, 0, 0
    else:    
        # Calculate majority_cluster
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        majority_cluster_points = coordinates[labels == majority_label]

        # Calculate noisy points
        noisy_majority_points = coordinates[labels == -1]

        # Calculate centroid and diameter of majority cluster
        centroid = np.mean(majority_cluster_points, axis=0)
        diameter = np.max(pdist(majority_cluster_points)) if len(majority_cluster_points) > 1 else 0

        # Calculate number of noisy points
        n_noise = len(noisy_majority_points)
        
        # Print results
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noisy points: {n_noise}")
        print(f"Centroid of majority cluster: {centroid}")
        print(f"Diameter of majority cluster: {diameter}") 

        return n_clusters, majority_cluster_points, noisy_majority_points, centroid, diameter

def show_cluster(image, majority_cluster_points, noisy_majority_points, centroid, diameter):  
    plt.imshow(image, alpha=1)

    # Plot majority cluster points
    plt.scatter(majority_cluster_points[:, 0], majority_cluster_points[:, 1], c='blue', label='Majority Cluster')

    # Plot noisy points 
    plt.scatter(noisy_majority_points[:, 0], noisy_majority_points[:, 1], c='black', label='Noise', alpha=0.6)
   
    # Plot centroid
    plt.scatter(centroid[0], centroid[1], c='red', marker='x', label='Centroid')

    # Draw cluster boundary
    circle = plt.Circle(centroid, diameter / 2, color='blue', fill=False, linestyle='dashed', linewidth=1)
    plt.gca().add_patch(circle)

    plt.legend()
    plt.show()
    plt.close()

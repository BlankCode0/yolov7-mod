import os
import json
import random
import shutil

from pathlib import Path
from PIL import Image


image_folder = "D:\\vs projects\\yolov7-mod\\data\\bdd100k_split"
output_dir = "D:\\vs projects\\yolov7-mod\\custom_data"
label_dir = "D:/vs projects/bdd100k_labels_release/bdd100k/labels"

_10k_path = "D:\\vs projects\\bdd100k\\bdd100k\\images\\10k"

train_json_path = os.path.join(label_dir, "bdd100k_labels_images_train.json")
val_json_path = os.path.join(label_dir, "bdd100k_labels_images_val.json")

train_images_path = os.path.join(_10k_path, "train")
val_images_path = os.path.join(_10k_path, "val")

train_op = os.path.join(output_dir, "labels", "train")
val_op = os.path.join(output_dir, "labels", "val")

classes = [
    "bike",
    "bus",
    "car",
    "drivable area",
    "lane",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
]
class_mappings = {}
for i in classes:
    class_mappings[i] = classes.index(i)

for i in ["images", "labels"]:
    os.makedirs(os.path.join(output_dir, i), exist_ok=True)
    for j in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, i, j), exist_ok=True)

# Function to write labels to TXT file


def write_labels(image_set, output_folder, annotations_dict, mode):

    for img_name in image_set:
        # Load the image to get its dimensions
        img_path = os.path.join(_10k_path, mode, img_name)
        with Image.open(img_path) as img:
            img_width, img_height = img.size  # Get actual image dimensions

        # Get the annotations for the current image
        if img_name in annotations_dict:

            labels = annotations_dict[img_name]["labels"]
            with open(
                os.path.join(output_folder, f"{img_name[:-4]}.txt"), "w"
            ) as label_file:
                for label in labels:
                    category = label["category"]
                    if "box2d" in label:
                        box = label["box2d"]
                        x_center = (box["x1"] + box["x2"]) / 2
                        y_center = (box["y1"] + box["y2"]) / 2
                        width = box["x2"] - box["x1"]
                        height = box["y2"] - box["y1"]
                        # Normalize values to the image size
                        x_center /= img_width
                        y_center /= img_height
                        width /= img_width
                        height /= img_height
                        # Write to label file
                        class_id = class_mappings[category]
                        label_file.write(
                            f"{class_id} {x_center} {y_center} {width} {height}\n"
                        )

                    elif "poly2d" in label:
                        for polygon in label["poly2d"]:
                            vertices = polygon["vertices"]
                            x_coords = [v[0] for v in vertices]
                            y_coords = [v[1] for v in vertices]
                            x_min = min(x_coords)
                            y_min = min(y_coords)
                            x_max = max(x_coords)
                            y_max = max(y_coords)
                            x_center = (x_min + x_max) / 2
                            y_center = (y_min + y_max) / 2
                            width = x_max - x_min
                            height = y_max - y_min
                            # Normalize values to the image size
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height
                            # Write to label file
                            class_id = class_mappings[category]
                            label_file.write(
                                f"{class_id} {x_center} {y_center} {width} {height}\n"
                            )

            # Copy the image to the correct directory
            shutil.copy(img_path, os.path.join(output_dir, "images", mode, img_name))


def create_labels():
    # Load JSON annotations
    with open(train_json_path, "r") as f:
        train_annotations = json.load(f)

    with open(val_json_path, "r") as f:
        val_annotations = json.load(f)

    # Concatenate the annotations
    all_annotations = train_annotations + val_annotations

    # Create a dictionary to store the labels
    annotations_dict = {img["name"]: img for img in all_annotations}

    img_list = os.listdir(train_images_path)
    train_images = random.sample(img_list, 1500)

    img_list = os.listdir(val_images_path)
    val_images = random.sample(img_list, 800)


    # print(len(train_images),len(val_images))
    # Write labels for train and val sets
    write_labels(train_images, train_op, annotations_dict, "train")
    print("Labels created successfully!")

    write_labels(val_images, val_op, annotations_dict, "val")
    print("Labels created successfully!")

def move_to_val():
    images_dir = os.path.join(output_dir,"images","train")

    labels_dir = os.path.join(output_dir,"labels")

    images_to_move = list(Path(images_dir).glob('*.jpg'))[:100]  # Assuming image format is .jpg

    # Move the images and collect their annotations
    for image_path in images_to_move:
        image_name = image_path.name

        only_name = Path(image_name).stem
        shutil.move(str(image_path),os.path.join(output_dir,"images","val"))

        shutil.move(os.path.join(labels_dir,"train",only_name+".txt"),os.path.join(labels_dir,"val",only_name+".txt"))


# create_labels()
move_to_val()

# # Used code


# output_train_dir = os.path.join(output_dir, "train")
# output_val_dir = os.path.join(output_dir, "val")
# output_test_dir = os.path.join(output_dir, "test")


# Paths to your dataset and JSON file
# json_file_path = 'path/to/your/annotations.json'  # Update this path


# # Create output folders if they don't exist
# output_train = Path('D:/vs projects/bdd100k/bdd100k/labels/train')
# output_val = Path('D:/vs projects/bdd100k/bdd100k/labels/val')
# output_train.mkdir(parents=True, exist_ok=True)
# output_val.mkdir(parents=True, exist_ok=True)


# # Make sure output folders exist
# os.makedirs(output_train_dir, exist_ok=True)
# os.makedirs(output_val_dir, exist_ok=True)
# os.makedirs(output_test_dir, exist_ok=True)


# # Number of images to extract
# num_train = 1000
# num_val = 500
# num_test = 500


# # Save the selected annotations to a new file
# def save_annotations(selected_img_names, annotations, output_json_path):
#     selected_annotations = [
#         ann for ann in annotations if ann["name"] in selected_img_names
#     ]
#     with open(output_json_path, "w") as f:
#         json.dump(selected_annotations, f)


# # Select random images and copy them to the destination folder
# def select_and_copy_images(src_dir, dest_dir, num_images, annotations):
#     img_list = os.listdir(src_dir)
#     selected_imgs = random.sample(img_list, num_images)

#     for img_name in selected_imgs:
#         img_src_path = os.path.join(src_dir, img_name)
#         img_dest_path = os.path.join(dest_dir, img_name)
#         shutil.copy(img_src_path, img_dest_path)

#     # Get the corresponding image names without extensions for the annotation search
#     selected_img_names = [os.path.splitext(img_name)[0] for img_name in selected_imgs]

#     return selected_img_names


# # Split the train, val, test images
# def split_dataset():
#     # Train
#     train_annotations = load_json(train_json_path)
#     selected_train_imgs = select_and_copy_images(
#         train_img_dir, output_train_dir, num_train, train_annotations
#     )
#     save_annotations(
#         selected_train_imgs,
#         train_annotations,
#         os.path.join(output_dir, "train_labels.json"),
#     )

#     print("train comp")
#     # Val
#     val_annotations = load_json(val_json_path)
#     selected_val_imgs = select_and_copy_images(
#         val_img_dir, output_val_dir, num_val, val_annotations
#     )
#     save_annotations(
#         selected_val_imgs, val_annotations, os.path.join(output_dir, "val_labels.json")
#     )

#     print("val comp")
#     # # Test (no labels for test set)
#     # select_and_copy_images(test_img_dir, output_test_dir, num_test, None)

#     print("Dataset split complete!")


# # Load JSON annotations
# def load_json(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     return data

# def find_cls(annotation_file):
#     # Load annotations from the JSON file
#     with open(annotation_file, 'r') as f:
#         annotations = json.load(f)

#     # Set to collect unique classes
#     unique_classes = set()

#     # Loop through each annotation and extract category names
#     for annotation in annotations:
#         for label in annotation.get('labels', []):
#             category = label['category']
#             unique_classes.add(category)

#     # Convert set to list and sort for better readability
#     unique_classes = sorted(list(unique_classes))

#     # Print the unique classes and their count
#     print(f"Unique Classes in BDD Dataset: {unique_classes}")
#     print(f"Total Number of Classes: {len(unique_classes)}")

#     return unique_classes

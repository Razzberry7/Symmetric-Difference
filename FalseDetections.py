import sys
import os.path
import ModelTools as mt
import json
import itertools
from alive_progress import alive_bar


path = "C:\\Users\\razbo\\OneDrive\\Documents\\Class Documents\\Research\\" # Change this to your Research folder
model_file = f"{path}Weights/Merged_best.pt" # Change this to where you keep weights 
model_conf = 0.1
model_iou = 0.3
dataset_names = ["macrie_dji_phantom_3_6-3-2022_picked_bushes.v2-validation-set-10v.yolov5pytorch",
                 "Macrie_iPhoneThanh_6_2022_DUKE.v3-validation-set-10v_2t.yolov5pytorch",
                 "Blueberry_Berry_Merged_80_7-25-2023"] # Change this to dataset names
times_reannotated = 0
model = mt.Model(model_file, conf=model_conf, iou=model_iou, agnostic=False)


for dataset_name in dataset_names:
    answer = str(input(f"Continue with {dataset_name}? (y/n) \n>"))
    if answer.lower() == "n":
        continue

    dataset_path = f"{path}/Processed Datasets/{dataset_name}/train/images/" # Change this to wherever you keep your datasets
    if not os.path.exists(dataset_path):
        dataset_path = f"{path}/Processed Datasets/{dataset_name}/valid/images/" # Change this to wherever you keep your datasets

    result_directory = f"./false_predictions_results/re-annotated_{times_reannotated}_times/{dataset_name}/c{model_conf}_i{model_iou}/"

    # Create resulting directory if it doesn't already exist
    if not os.path.exists(f"{result_directory}"):
        os.makedirs(f"{result_directory}")
    else:   
        print(f"The directory at {result_directory} already exists!")


    results = {}
    with alive_bar(len(os.listdir(dataset_path))) as bar:
        for image in os.listdir(dataset_path):
            image_path = dataset_path + image
            label_path = os.path.dirname(dataset_path[:-1]) + "/labels/" + image[:-3] + "txt"
            false_negatives, false_positives, total_gt_count, total_predictions = model.find_nonoverlapping(ground_truth_path=label_path, image_path=image_path, tiled=True)
            output = []

            for box in false_negatives:
                box[5] = 2
                output.append(box)
            for box in false_positives:
                # Print to a file
                # Convert to YOLO LATER
                box[5] = 1
                output.append(box)
            
            true_positives = (total_predictions - len(false_positives))
            results[str(image)] = {
                "GT" : total_gt_count,
                "Predictions" : total_predictions,
                "TP" : true_positives,
                "FP" : len(false_positives),
                "FN" : len(false_negatives),
                "Precision" :  true_positives / total_predictions,
                "Recall" : true_positives / (true_positives + len(false_negatives))
            }

            # Export images with the new bounding boxes displayed
            mt.export_annotated_image(output, result_directory, image, image_path=image_path, label_thickness=1, conf_text=False)

            # Update progress bar
            bar()
            
    # Save everything to the new JSON file
    with open(f"{result_directory}results.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # Create a summary from the results.json
    with open(f"{result_directory}results.json", 'r') as orig_json_file:
        results_from_json = json.load(orig_json_file)
        
        # Summary txt file  
        with open(f"{result_directory}results_summary.json", 'w') as sum_json_file:
            # Sort the dictionary by age in descending order
            sorted_dict = dict(sorted(results_from_json.items(), key=lambda x: x[1]["FP"], reverse=True))
            
            num = 10
            out = dict(itertools.islice(sorted_dict.items(), num))
            json.dump(out, sum_json_file, indent=4)

    

    


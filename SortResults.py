import json
import itertools
import statistics

times_reannotated = 0
dataset_names = ["macrie_dji_phantom_3_6-3-2022_picked_bushes.v2-validation-set-10v.yolov5pytorch",
                 "Macrie_iPhoneThanh_6_2022_DUKE.v3-validation-set-10v_2t.yolov5pytorch",
                 "Blueberry_Berry_Merged_80_7-25-2023"]
model_conf = 0.1
model_iou = 0.3

for dataset_name in dataset_names:
        result_directory = f"./false_predictions_results/re-annotated_{times_reannotated}_times/{dataset_name}/c{model_conf}_i{model_iou}/"

        with open(f"{result_directory}results.json", 'r') as file:
                results_from_json = json.load(file)

                # Summary txt file  
                with open(f"{result_directory}results_summary.json", 'w') as json_file:
                        # Sort the dictionary by age in descending order
                        sorted_dict = dict(sorted(results_from_json.items(), key=lambda x: x[1]["FP_GT_Ratio"], reverse=True))
                        
                        num = 20
                        out = dict(itertools.islice(sorted_dict.items(), num))
                        json.dump(out, json_file, indent=4)
                
                results = [[],[],[],[],[]]
                # sums = [0, 0, 0, 0, 0]
                count = 0
                means = []
                stdevs = []

                with open(f"{result_directory}results_summary.txt", 'w') as txt_file:
                        for result in results_from_json.items():
                                results[0].append(int(result[1]["GT_Count"]))
                                results[1].append(int(result[1]["FN_Count"]))
                                results[2].append(int(result[1]["FP_Count"]))
                                results[3].append(float(result[1]["FN_GT_Ratio"]))
                                results[4].append(float(result[1]["FP_GT_Ratio"]))
                        #         sums[0] += int(result[1]["GT_Count"])
                        #         sums[1] += int(result[1]["FN_Count"])
                        #         sums[2] += int(result[1]["FP_Count"])
                        #         sums[3] += float(result[1]["FN_GT_Ratio"])
                        #         sums[4] += float(result[1]["FP_GT_Ratio"])
                        #         count += 1
                        
                        # for result in results:
                        #         means.append(sum/count)
                                
                        for result in results:
                                means.append(statistics.mean(result))
                                stdevs.append(statistics.stdev(result))

                        print(f"Means: {means}", file=txt_file)
                        print(f"Standard Deviations: {stdevs}", file=txt_file)



        

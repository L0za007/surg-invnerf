import os
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def load_vid_list_txt(dataset_path):
    video_list_file = Path(dataset_path) / 'videos_list.txt'
    output_folder_file = Path(dataset_path) / 'output_folders.txt'
    video_type_file = Path(dataset_path) / 'video_type.txt'
    video_list = open(video_list_file, 'r').readlines()
    output_folders = open(output_folder_file, 'r').readlines()
    video_type = open(video_type_file, 'r').readlines()
    video_list = [v.strip() for v in video_list]
    output_folders = [o.strip() for o in output_folders]
    video_type = [t.strip() for t in video_type]
    assert len(output_folders) == len(video_type), "Mismatch in lengths of output folders and video types"
    assert len(video_list) == len(output_folders), "Mismatch in lengths of video list and output folders"
    return np.array(video_list), np.array(output_folders), np.array(video_type)

def summarize_json_files(json_file_paths):
    # Define the keys to include
    main_keys = ["Experiment","Date (aa-mm-dd)","Starting time(hrs-min-sec)","Training time(hrs)","Video Path","Video Type","recon mse","recon psnr","recon ssim"]
    error_keys = {
        "error 3DptsA": ["avgL1", "avgL2"],
        "error 3DptsB": ["avgL1", "avgL2"],
        "error 2DptsB": ["avgL1", "avgL2"]
    }

    rows = []
    for filepath in json_file_paths:
        with open(filepath, 'r') as f:
            data = json.load(f)
        row = {}
        # Extract main fields
        for key in main_keys:
            row[key] = data.get(key, "")
        # Extract error fields 
        for err_key, sub_keys in error_keys.items():
            err_data = data.get(err_key, {})
            for sub in sub_keys:
                values = err_data.get(sub, [])
                col_name = f"{err_key} {sub}"
                row[col_name] = values if values else float('nan')

        rows.append(row)
    return pd.DataFrame(rows)

def summarize_pts_in_json_files(json_file_paths, out_dir):
    # main_key = "Video Path"

    test= {}
    for filepath in json_file_paths:
        with open(filepath, 'r') as f:
            data = json.load(f)
        ptsA3D = data.get('Hat 3DptsA', [])
        ptsB3D = data.get('Hat 3DptsB', [])
        ptsB2D = data.get('Hat 2DptsB', [])
        video_tag = data.get('Video Path', "")
        current_test = data.get('Experiment', "")
        if current_test not in test.keys():
            test[current_test] = {'3DptsA': {}, '3DptsB': {}, '2DptsB': {}}
        if video_tag in test[current_test]['3DptsA'].keys():
            raise ValueError(f"Duplicate video tag found: {video_tag} in test {current_test}")
        test[current_test]['3DptsA'][video_tag] = ptsA3D
        test[current_test]['3DptsB'][video_tag] = ptsB3D
        test[current_test]['2DptsB'][video_tag] = ptsB2D
    # create a jsonfile per pts type
    pts_type = ["3DptsA", "3DptsB", "2DptsB"]
    for test_name, pts_data in test.items():
        for pts in pts_type:
            output_file = os.path.join(out_dir, f"InvNeRF_{test_name}_{pts}.json")
            with open(output_file, 'w') as f:
                json.dump(pts_data[pts], f, indent=4)
            print(f"Saved {pts} data for {test_name} to {output_file}")


def gather_information_from_all_folders(dataset_path):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset folder does not exist: {dataset_path}")
    
    _, output_folders, video_types = load_vid_list_txt(dataset_path)
    output_lines = []
    for i,subfolder in enumerate(output_folders):
        subfolder_path = os.path.join(dataset_path, subfolder+'/out')
        if not os.path.exists(subfolder_path):
            print(f"Subfolder does not exist: {subfolder_path}")
            continue
        print(f"Processing subfolder: {subfolder_path} (video type: {video_types[i]})")
        for folder_name in sorted(os.listdir(subfolder_path)):
            eval_file = os.path.join(subfolder_path, folder_name, 'evaluation.json')
            if not os.path.isfile(eval_file):
                print(f"Evaluation file does not exist: {eval_file}")
                # raise ValueError(f"Evaluation file does not exist: {eval_file}")
                continue
            output_lines.append(eval_file)
    # Summarize all JSON files
    df = summarize_json_files(output_lines)
    # Save the DataFrame to a CSV file
    output_csv = os.path.join(dataset_path, 'evaluation_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")
    summarize_pts_in_json_files(output_lines, dataset_path)
            
def gather_information_from_folders(data_path):
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset folder does not exist: {data_path}")
    output_lines = []
    for subfolder in sorted(os.listdir(data_path)):
        eval_file = os.path.join(data_path, subfolder, 'evaluation.json')
        if not os.path.isfile(eval_file):
            # raise ValueError(f"Evaluation file does not exist: {eval_file}")
            continue
        
        output_lines.append(eval_file)
    # Summarize all JSON files
    df = summarize_json_files(output_lines)
    # Save the DataFrame to a CSV file
    output_csv = os.path.join(data_path, 'evaluation_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")
    summarize_pts_in_json_files(output_lines, data_path)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate tracker on each video os the STIR dataset')
    parser.add_argument('--out_dir', '-o', type=str, help='Path to output directory')
    parser.add_argument('--all', action='store_true', help='Gather information from all folders in the dataset')
    args = parser.parse_args()
    out_dir = args.out_dir
    if not out_dir:
        raise ValueError("Output directory must be specified with --out_dir or -o")
    if not os.path.exists(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")
    if args.all:
        gather_information_from_all_folders(out_dir)
    else:
        gather_information_from_folders(out_dir)
import sys
import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def main():
    csv_folder = Path("analysis_data/comparsion/result")
    output_folder = Path("analysis_data/comparsion/nor")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        files = [file for file in csv_folder.iterdir() if file.suffix == ".npy"]
        files.sort()
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return

    for video in files:
        video_name = video.stem
        print(f"--- {video_name} start ---")
        df_n = np.load(video)
        df_n = normalize(df_n)
        df_n = df_n[df_n[:, 1] == 0]
        df_n = df_n[:, 2:]
        df_no_conf = np.delete(df_n, np.arange(2, df_n.shape[1], 3), axis=1)
        print(f"final:{np.isnan(df_no_conf).sum()}")
        print(f"final:{np.isinf(df_no_conf).sum()}")

        try: 
            np.save(output_folder / f"{video_name}.npy", df_no_conf)
            logger.success(f"Processing completed for video: {video_name}")
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {e}")

def interp_zero(col):
    n = len(col)
    if np.count_nonzero(col) < 2:
        return
    i =0
    while i < n:
        if col[i] == 0:
            start = i - 1
            j = i
            while j < n and col[j] == 0:
                j += 1
            end = j

            if start >= 0 and end < n:
                x = [start, end]
                y = [col[start], col[end]]
                interp_vals = np.interp(np.arange(start + 1, end), x, y)
                col[start + 1:end] = interp_vals

            i = end
        else:
            i += 1
    return col

def normalize(df):
    df_n = np.copy(df)
    pre_dist = 0
    col = [5, 6, 8, 9, 17, 18]
    for c in col:
        df_n[:, c] = interp_zero(df_n[:, c])
        print(f"check: {np.isnan(df_n).sum()}")
    mask = (df_n == 0)

    pre = np.zeros(df_n[0].shape)
    for i in range(df.shape[0]):
        RShoulder_x = df_n[i, 8]
        RShoulder_y = df_n[i, 9]
        LShoulder_x = df_n[i, 17]
        LShoulder_y = df_n[i, 18]
        
        dist = np.sqrt((LShoulder_x - RShoulder_x) **2 + (LShoulder_y - RShoulder_y) **2)
        if dist == 0. and pre_dist != 0:
            dist = (pre_dist * 0.8) + 0.2 * dist
            print(f"dist: {dist, RShoulder_x, RShoulder_y, LShoulder_x, LShoulder_y}")
        pre_dist = dist
        
        center_x = df_n[i, 5]
        center_y = df_n[i, 6]

        for j in range(2, df.shape[1], 3):
            df_n[i, j] = df_n[i, j] - center_x
        for j in range(3, df.shape[1], 3):
            df_n[i, j] -= center_y
        for j in range(2, df.shape[1], 3):
            df_n[i, j] = df_n[i, j] / dist
        for j in range(3, df.shape[1], 3):
            df_n[i, j] = df_n[i, j] / dist
    
    df_n[mask] = 0.
    return df_n





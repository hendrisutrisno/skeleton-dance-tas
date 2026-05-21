# skeleton-dance-tas

Dataset and annotations for skeleton-based temporal action segmentation in dance from multi-person video sequences.

## Overview

This repository provides the dataset used in the paper:

**A Comparative Study of Temporal Models and Skeleton Representations for Temporal Action Segmentation in Dance**

The dataset is designed for **temporal action segmentation (TAS)** from **video-derived skeleton sequences**. It contains annotated dance action clips collected from a public dance tutorial and supplementary participant recordings. Each performer is treated as an independent sample, including cases extracted from multi-person scenes.

The repository is intended to support research on:

- temporal action segmentation
- skeleton-based action analysis
- dance motion understanding
- fine-grained human motion analysis

## Dataset summary

- **Task**: Temporal Action Segmentation (frame-level labeling)
- **Domain**: Dance motion
- **Input type**: Video-derived skeleton sequences
- **Skeleton settings**: BODY-25 and COCO-18
- **Number of action classes**: 7
- **Sources**:
  - one YouTube dance tutorial sequence
  - supplementary participant-recorded sequences
- **Special cases**:
  - multi-person scenes
  - performer-level extraction
  - duet performance sequences

## Data characteristics

The dataset focuses on continuous dance motion with smooth transitions between actions. It is intended for studying how pose representation and temporal modeling affect segmentation quality.

Each sample corresponds to a **single performer sequence**, even when the original source video contains multiple people. The dataset includes more controlled tutorial examples as well as participant-recorded examples with greater variation in execution and recording conditions.

## Repository structure

```text
skeleton-dance-tas/
├── README.md
├── LICENSE
├── Data_processing/
│   ├── Extraction.py
│   ├── SORT.py
│   └── Normalize.py
├── metadata/
│   ├── class_labels.txt
├── annotations/
│   ├── train/
│   └── test/
├── skeletons/
│   ├── body25/
│   └── coco18/
├── videos/
│   ├── source_links.md

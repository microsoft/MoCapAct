#!/bin/bash

dataset_paths=$1

export clips=""
export train=""
{
  while IFS=, read -r clip; do
    export clips="$clips,$clip"
    export train="$train,$dataset_path/$clip.hdf5"
  done
} < clip_splits/locomotion_clips.txt
export clips=${clips:1}
export train=${train:1}

export metrics=$dataset_path/dataset_metrics.npz

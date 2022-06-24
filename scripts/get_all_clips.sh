#!/bin/bash

dataset_path=$1

export clips=""
export train=""
{
  while IFS=, read -r clip; do
    export clips="$clips,$clip"
    export train="$train,$dataset_path/$clip.hdf5"
  done
} < clip_splits/train_clips.txt
export train=${train:1}

export val=""
{
  while IFS=, read -r clip; do
    export clips="$clips,$clip"
    export val="$val,$dataset_path/$clip.hdf5"
  done
} < clip_splits/val_clips.txt
export val=${val:1}
export clips=${clips:1}

export snippets=""
{
  while IFS=, read -r snippet; do
    export snippets="$snippets,$snippet"
  done
} < clip_splits/all_snippets.txt
export snippets=${snippets:1}

export metrics=$dataset_path/dataset_metrics.npz

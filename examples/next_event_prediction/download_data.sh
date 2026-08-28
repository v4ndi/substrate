#!/bin/bash

mkdir -p data && cd data && hdfs dfs -get /user/team/team_ai_avatar/avatar_fm/examples/next_event_prediction/processed_sequences

directory="processed_sequences"

valid_dir="${directory}/valid"
train_dir="${directory}/train"

mkdir -p "$valid_dir"
mkdir -p "$train_dir"

files=($(find "$directory" -maxdepth 1 -type f -name "*.parquet"))

if [ ${#files[@]} -eq 0 ]; then
    echo "No parquet files found in $directory"
    exit 0
fi

shuffled_files=($(shuf -e "${files[@]}"))

for ((i=0; i<10 && i<${#shuffled_files[@]}; i++)); do
    mv "${shuffled_files[i]}" "$valid_dir/"
    echo "Moved ${shuffled_files[i]} to $valid_dir"
done

for ((i=10; i<${#shuffled_files[@]}; i++)); do
    mv "${shuffled_files[i]}" "$train_dir/"
    echo "Moved ${shuffled_files[i]} to $train_dir"
done

echo "Done. Moved 10 files to $valid_dir and $((${#shuffled_files[@]} - 10)) files to $train_dir"

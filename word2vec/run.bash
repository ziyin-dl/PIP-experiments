#!/bin/bash
for (( i=1; i < 100; ++i)); do
  dim=$((1*i))
  echo "$dim"
  python word2vec_optimized.py --min_count 120 --embedding_size $dim --train_data=text8   --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/
done

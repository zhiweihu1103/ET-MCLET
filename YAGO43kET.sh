export LOG_PATH=./logs/YAGO43kET_baseline.txt
export SAVE_NAME=YAGO43kET_baseline
export DATASET=YAGO43kET
export LIGHTGCN_LAYER=1
export CL_TEMPERATURE=0.6
export CL_LOSS_WEIGHT=0.001
export NUM_HEADS=5


CUDA_VISIBLE_DEVICES=5 python run.py --dataset $DATASET --load_ET --load_KG --load_TC --load_EC --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --beta 2.0 --cuda --save_name $SAVE_NAME \
--lightgcn_layer $LIGHTGCN_LAYER --cl_temperature $CL_TEMPERATURE --cl_loss_weight $CL_LOSS_WEIGHT --num_heads $NUM_HEADS \
  > $LOG_PATH 2>&1 &

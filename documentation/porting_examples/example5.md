# Transfer example

```bash
# Set up these paths according to your env
WORKING_DIR=...    # Choose a working dir (better in scratch)
NYU_JIANT_DIR=...  # Where you downloaded https://github.com/jiant-dev/nyu-jiant

MODELS_DIR=${WORKING_DIR}/models
DATA_DIR=${WORKING_DIR}/data
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/transfer_example
OUTPUT_DIR=${WORKING_DIR}/output_dir/transfer_example
MODEL_TYPE=roberta-large

# Download model
python jiant/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

# Move data into location
# Ping Haokun on slack, if you have access issues
cp -r /scratch/hl3232/shared/transfer_pilot_data ${WORKING_DIR}/data/data


# Prepare data configs
python ${NYU_JIANT_DIR}/documentation/porting_examples/example5_assets/write_data_configs.py \
    --output_base_path ${DATA_DIR}/


# Tokenize and cache datasets
for TASK_NAME in mnli ccg squadv1 cosmosqa rte cola boolq wic
do
    python jiant/proj/simple/tokenize_and_cache.py \
        --task_config_path ${DATA_DIR}/configs/${TASK_NAME}.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
        --phases train,val \
        --max_seq_length 256 \
        --do_iter \
        --smart_truncate \
        --output_dir ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME}
done

# Generate run configs
declare -A TASK_EPOCHS=(
  ["mnli"]=3
  ["ccg"]=3
  ["squadv1"]=3
  ["cosmosqa"]=3
  ["rte"]=20
  ["cola"]=20
  ["boolq"]=20
  ["wic"]=20
)
declare -A GPUS=(
  ["mnli"]=1
  ["ccg"]=1
  ["squadv1"]=1
  ["cosmosqa"]=4
  ["rte"]=1
  ["cola"]=1
  ["boolq"]=1
  ["wic"]=1
)
for TASK_NAME in mnli ccg squadv1 cosmosqa rte cola boolq wic
do
    python ${NYU_JIANT_DIR}/documentation/porting_examples/example4_assets/make_config.py \
        --task_config_path ${DATA_DIR}/configs/${TASK_NAME}.json \
        --task_cache_base_path ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME} \
        --train_batch_size 16 \
        --epochs ${TASK_EPOCHS[${TASK_NAME}]} \
        --output_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json
done

# Train single task
for TASK_NAME in mnli ccg squadv1 cosmosqa rte cola boolq wic
do
    COMMAND="python \
        jiant/proj/main/runscript.py \
        run \
        --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json \
        --model_load_mode from_transformers \
        --learning_rate 1e-5 \
        --force_overwrite \
        --do_train --do_val \
        --do_save \
        --eval_every_steps 2000 \
        --no_improvements_for_n_evals 30 \
        --save_checkpoint_every_steps 10000 \
        --output_dir ${OUTPUT_DIR}/${TASK_NAME}/" sbatch ~/j2_g${GPUS[${TASK_NAME}]}.sbatch
done

# Train target task from source task
for SOURCE_TASK in mnli ccg squadv1 cosmosqa
do
    for TARGET_TASK in rte cola boolq wic
    do
        COMMAND="python \
            jiant/proj/main/runscript.py \
            run \
            --ZZoverrides model_path \
            --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
            --jiant_task_container_config_path ${RUN_CONFIG_DIR}/${TARGET_TASK}.json \
            --model_load_mode partial \
            --model_path ${OUTPUT_DIR}/${SOURCE_TASK}/best_model.p \
            --learning_rate 1e-5 \
            --force_overwrite \
            --do_train --do_val \
            --do_save \
            --eval_every_steps 5000 \
            --no_improvements_for_n_evals 30 \
            --save_checkpoint_every_steps 10000 \
            --output_dir ${OUTPUT_DIR}/${SOURCE_TASK}__${TARGET_TASK}/" sbatch ~/j2_g${GPUS[${TARGET_TASK}]}.sbatch
    done
done


grep major ${OUTPUT_DIR}/*/val_metrics.json
```

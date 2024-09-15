# MLSM-isiXhosa
This repository contains the source code and resources for training and evaluating a Masked Latent Semantic Modeling (MLSM) model specifically for the isiXhosa language. This work extends the research presented in the ACL Findings paper [_Masked Latent Semantic Modeling: an Efficient Pre-training Alternative to Masked Language Modeling_](https://aclanthology.org/2023.findings-acl.876/) with the code available at the [MLSM repository](https://github.com/szegedai/MLSM). It also utilizes the Masakhane Natural Language Understanding (NLU) evaluation sets for [Named Entity Recognition (NER)](https://github.com/masakhane-io/masakhane-ner), [Text Classification (NEWS)](https://github.com/masakhane-io/masakhane-news) and [Part of Speech tagging (POS)](https://github.com/masakhane-io/masakhane-pos)

## Creation of an isiXhosa model
These are the steps followed to train and evaluate our isiXhosa MLSM model. Some minor changes were made to the "pretrainer.py" file from the MLSM repository, as well as the "train_ner.py", "train_textclass.py", "train_pos.py" from their respective Masakhane repositories. Therefore, we recommend any emulation of these results to utilize the versions of these files uploaded here.

### Step 1: Creating a custom tokenizer

This step creates a custom tokenizer on the data that the model the model will be pre-trained on.  

``` 
python train_tokenizer.py --vocab_size $VOCAB_SIZE
                          --folder $INPUT_DATA_FOLDER
                          --out_folder $TOKENIZER_NAME
                          --cased
```

```$VOCAB_SIZE``` = 25000

```$INPUT_DATA_FOLDER``` = "wura-xh.tar.gz".

### Step 2: Creating an auxiliary model

This step creates an isiXhosa language model trained on a traditional Masked Language Modelling (MLM) task with the bert-base architecture. This model provides semantic information for steps 3 and 4.

```
python pretrainer.py --transformer ${MODEL} \\
                     --reinit \\
                     --tokenizer ${TOKENIZER_NAME} \\
                     --out_dir ${AUXILIARY_MODEL_LOCATION} \\
                     --data_location ${PRETRAINING_DATA} \\
                     --training_seqs 25600000 \\
                     --batch 64 --grad_accum 16
```

```${MODEL}``` = "google-bert/bert-base-cased", the architecture to be used for this model.

```${TOKENIZER_NAME}``` refers to the tokenizer created in step 1

```${PRETRAINING_DATA}``` = "wura-xh.tar.gz", same as in step 1

### Step 3: Learning the dictionary matrix 

```
python train_dict.py --transformer ${AUXILIARY_MODEL} \\
                     --corpus ${PRETRAINING_CORPUS} \\
                     --output ${SAVE_LOCATION} \\
                     --layer ${LAYER_TO_USE} \\
```

```${AUXILIARY_MODEL}``` refers to the location of the model trained in step 1

```${PRETRAINING_CORPUS}``` = "wura-xh.tar.gz", same as in previous steps

```$LAYER_TO_USE``` = 11, this specifies the hidden layer of the auxiliary model used for dictionary learning. It was set to 11 here since bert-base has 12 layers.

### Step 4: Pre-training using MLSM

The following script indicates how the final MLSM model was trained:

```
python pretrainer.py --transformer ${MLSM_MODEL} \\
                     --reinit \\
                     --tokenizer ${TOKENIZER_NAME} \\
                     --out_dir ${MODEL_LOCATION} \\
                     --data_location ${PRETRAINING_DATA} \\
                     --training_seqs 37171200 \\
                     --batch 64 --grad_accum 16 \\
                     --transformer2 ${AUXILIARY_MODEL} \\
                     --dict_file ${DICTIONARY_LOCATION} \\
                     --layer 11
```

```${MLSM_MODEL}``` = "google-bert/bert-base-cased", the architecture used for this model

```${TOKENIZER_NAME}``` refers to the tokenizer created in step 1

```${PRETRAINING_DATA}``` = "wura-xh.tar.gz" as in previous steps

```training_seqs``` = 36300 x 16 x 64 = 37171200, where 36300 is the amount of training steps to complete 200 epochs on the data

```${AUXILIARY_MODEL}``` refers to the model constructed in step 2

```${DICTIONARY_LOCATION}``` refers to the dictionary matrix constructed in step 3

```layer``` = 11, this matches the layer used to construct the dictionary in step 3

### Step 4.5: Pre-training from a checkpoint using MLSM

The following indicates how to continue training from a saved checkpoint:

```
python pretrainer.py --transformer ${MLSM_MODEL} \
                     --not-reinit \
                     --cnt_train ${CHECKPOINT_OPTIM_STATE} \
                     --out_dir ${MODEL_LOCATION} \
                     --data_location ${PRETRAINING_DATA} \
                     --training_seqs 37171200 \
                     --batch 64 --grad_accum 16 \
                     --transformer2 ${AUXILIARY_MODEL} \
                     --tokenizer ${TOKENIZER_NAME} \
                     --dict_file ${DICTIONARY_LOCATION} \
                     --layer 11
```

```not-reinit``` keeps the values pre-trained by the checkpoint model

```${CHECKPOINT_OPTIM_STATE}``` is a ".pkl" file saved in the model's folder

With the values not mentioned here being identical to step 4.

### Step 5: Evaluation

The following scripts indicate how to evaluate the pre-trained model on the NLU tasks.

Named Entity Recognition

```
python -u train_ner.py --data_dir $DATA_DIR \
	--model_type bert \
	--model_name_or_path $MODEL \
	--output_dir $OUTPUT_DIR \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict \
	--overwrite_output_dir
```

Text Classification

```
python -u train_textclass.py --data_dir $DATA_DIR \
	--model_type bert\
	--model_name_or_path $MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--per_gpu_eval_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--gradient_accumulation_steps 2 \
	--do_train \
	--do_eval \
	--do_predict \
	--overwrite_output_dir
```

Part of Speech

```
python -u train_pos.py --data_dir $DATA_DIR \
	--model_type bert \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--gradient_accumulation_steps 2 \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict \
	--overwrite_output_dir
```

```$NUM_EPOCHS``` = 20

```$BATCH_SIZE``` = 32

```$SAVE_STEPS``` = 10000

```$SEED``` = j for j in range(5)+1, since each NLU task was run 5 times the seed was 1,2,3,4,5 incrementing for each run.

These values were kept consistent for each NLU task.

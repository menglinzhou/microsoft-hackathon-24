# ML-LoRA-E5-samll

The raw model we picked in this file is e5-samll model (https://huggingface.co/intfloat/e5-small). We train the LoRA fine-tuning with mixed dataset with twitter and raid data. All other settings are the same as the training for twitter dataset.

- twitter_raid_train.csv.zip: the first version of mixed data with all twitter data and some generated data from raid.

- raid_twitter_train.csv: the second version of mixed data with 10,000 twitter, 10,000 rewritten twitter, 80,000 human-generated text from raid (10,000 each domain), 128,000 AI-generated text from raid (2000 each combination of domain and model).  The details of data merging is provided in data_merge.ipynb.

- raid_twitter_LoRA_e5: contains 3 checkpoints saved during training (each per epoch). The last checkpoint is the final model. The training is using twitter_raid_train.csv.zip.

- **results_LoRA_e5: contains 3 checkpoints saved during training (each per epoch). The peformance decrease during the third epoch. We can take the second epoch checkpoint as the last model (checkpoint-36480). The training is using raid_twiiter_train.csv on (https://drive.google.com/file/d/1UJobJLH3M8DotuO0Hu5HZwPShomVKb39/view?usp=sharing).**

- **resilts_raw_e5: contains 3 checkpoints saved during training (each per epoch). The last checkpoint is the final model. We train the last output layer of the raw model, as the output layer is added for the task of sequence classification. When initialized, the weights in the output layer is randomly assigned. We use raid_twiiter_train.csv.zip to train those weights in the output layer. The checkpoint folders are zipped.**

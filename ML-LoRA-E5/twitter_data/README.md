# ML-LoRA-E5-samll

The raw model we picked in this file is e5-samll model (https://huggingface.co/intfloat/e5-small). We train the LoRA fine-tuning with twitter dataset which consists of 200,000 twitters and 90,000 rewritten twitters with GPT-4o-mini (200 versus 90 showing mild imbalance). The dataset is splitted into training set and validation/test set. The rank in the LoRA model is set as 8 and the alpha is selected as 16 (following the LoRA paper). The target modules to add adapter are chosen as query, key and value modules in attension layers. Focal loss is utilized to address the problem of imbalance.

- results_LoRA_e5: contains 3 checkpoints saved during training (each per epoch). The last checkpoint is the final model.

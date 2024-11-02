# ML-LoRA-E5-samll

The raw model we picked in this file is e5-samll model (https://huggingface.co/intfloat/e5-small). We train the LoRA fine-tunning with mixed dataset which consists of samples from Kaggle, Yahoo news downloaded from Yahoo and rewritten Yahoo news with GPT-4o-mini. The dataset is splitted into training set and validation/test set. The rank in the LoRA model is set as 8 and the alpha is selected as 16 (following the LoRA paper). The target modules to add adapter are chosen as query and key modules in attension layers.

- final_model: the final model saved after 10 epoches.
- results_LoRA_e5: contains checkpoints saved during training (every 500 batches). We can load any checkpoints we want to make inference or continue training. The last checkpoint equals to the final model.

* Note: the performance of the validation dataset do not increase after the third epoch (which is between checkout-16000 and checkout-16500). We can early stop the training as later-on training epoches may introduce overfitting problem.

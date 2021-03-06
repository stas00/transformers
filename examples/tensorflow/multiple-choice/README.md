<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multiple Choice

## Fine-tuning on SWAG

```bash
export SWAG_DIR=/path/to/swag_data_dir
python ./examples/multiple-choice/run_tf_multiple_choice.py \
--task_name swag \
--model_name_or_path bert-base-cased \
--do_train \
--do_eval \
--data_dir $SWAG_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir models_bert/swag_base \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--logging-dir logs \
--gradient_accumulation_steps 2 \
--overwrite_output
```

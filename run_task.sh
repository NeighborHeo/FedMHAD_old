python main.py --model_name vit_tiny_patch16_224 --task multilabel --sublossmode mha --distill_heads 3 --lambda_kd 0.1 \
&& python main.py --model_name vit_tiny_patch16_224 --task multilabel --sublossmode None --distill_heads 0 --lambda_kd 0 \
&& python main.py --model_name vit_tiny_patch16_224 --task multilabel --sublossmode mha --distill_heads 1 --lambda_kd 0.1 \
&& python main.py --model_name vit_tiny_patch16_224 --task multilabel --sublossmode mha --distill_heads 2 --lambda_kd 0.1
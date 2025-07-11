python pred.py --model llama3.1-8b-128k --compress --cascading --score --pred_name add_entropy --device 4 --cache_size 1024 --window_size 32

python eval.py --model llama3.1-8b-128k --cache_size 1024 --eval_avg --dir_path pred_result/cache1024

win12 == win8
python pred.py --model mistral-0.3-7b-32k --compress --cascading --score --pred_name ex16_win12 --device 4 --cache_size 256 --window_size 8
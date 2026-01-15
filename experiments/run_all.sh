cd /data/shenggui/projects/spec-decoding/SpecForge

bash ./experiments/run_llama_70b_custom.sh |& tee -a ./logs/llama_70b_custom.log

bash ./experiments/run_qwen_30b_sglang.sh |& tee -a ./logs/qwen_30b_sglang.log
bash ./experiments/run_qwen_30b_custom.sh |& tee -a ./logs/qwen_30b_custom.log
bash ./experiments/run_qwen_30b_hf.sh |& tee -a ./logs/qwen_30b_hf.log

bash ./experiments/run_qwen_235b_sglang.sh |& tee -a ./logs/qwen_235b_sglang.log
bash ./experiments/run_qwen_235b_custom.sh |& tee -a ./logs/qwen_235b_custom.log
bash ./experiments/run_qwen_235b_hf.sh |& tee -a ./logs/qwen_235b_hf.log
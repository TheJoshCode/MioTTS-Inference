
cd ~/Downloads/llama.cpp/build/bin && ./llama-server -hf Aratako/MioTTS-GGUF -hff  MioTTS-2.6B-Q8_0.gguf -c 8192 --cont-batching --batch_size 8 --port 8000

python run_server.py --llm-base-url http://localhost:8000/v1

python run_gradio.py

python batch_infra.py kjv.txt --reference-audio ./mf.wav -o ./out/

watch -n 0.5 nvidia-smi
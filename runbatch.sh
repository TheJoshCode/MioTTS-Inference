#!/bin/bash

LLAMA_URL="http://localhost:8000"

echo "Starting llama-server..."
gnome-terminal --title="llama-server" -- bash -c "
cd ~/Downloads/llama.cpp/build/bin && \
./llama-server -hf Aratako/MioTTS-GGUF -hff MioTTS-1.2B-Q4_K_M.gguf -c 8192 --cont-batching --batch_size 8 --port 8000
"

echo "Waiting for llama-server to become healthy..."

until curl -s "$LLAMA_URL/health" > /dev/null 2>&1 || \
      curl -s "$LLAMA_URL/v1/models" > /dev/null 2>&1
do
    sleep 1
done

echo "llama-server is up."

gnome-terminal --title="run_server.py" -- bash -c "source .venv/bin/activate && python run_server.py --llm-base-url http://localhost:8000/v1"
sleep 2
#gnome-terminal --title="gradio" -- bash -c "source .venv/bin/activate && python run_gradio.py"
sleep 1
gnome-terminal --title="nvidia-smi" -- bash -c "watch -n 0.5 nvidia-smi"
sleep 7
gnome-terminal --title="batch_infra" -- bash -c "source .venv/bin/activate && python batch_infra.py kjv.txt --reference-audio ./mf.wav -o ./out/ --workers 4"

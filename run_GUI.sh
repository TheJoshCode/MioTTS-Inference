#!/bin/bash

LLAMA_URL="http://localhost:8000"

echo "Starting llama-server..."
gnome-terminal --title="llama-server" -- bash -c "cd ~/Downloads/llama.cpp/build/bin && ./llama-server -hf Aratako/MioTTS-GGUF -hff MioTTS-1.7B-Q8_0.gguf -c 1024 --cont-batching --batch_size 8 --port 8000"

echo "Waiting for llama-server to become healthy..."

until curl -s "$LLAMA_URL/health" > /dev/null 2>&1 || \
      curl -s "$LLAMA_URL/v1/models" > /dev/null 2>&1
do
    sleep 1
done

echo "llama-server is up."
sleep 1
gnome-terminal --title="run_server.py" -- bash -c "source .venv/bin/activate && python run_server.py --best-of-n-enabled --llm-base-url http://localhost:8000/v1"
sleep 1
gnome-terminal --title="gradio" -- bash -c "source .venv/bin/activate && python run_gradio.py"
##gnome-terminal --title="batch_infra" -- bash -c "source .venv/bin/activate && python batch_infra.py kjv.txt --reference-audio ./mf.wav -o ./out/ --use-file-upload"

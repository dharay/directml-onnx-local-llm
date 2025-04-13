
## setup
.venv/scripts/activate
pip install -r req.txt

## compatible models:
https://huggingface.co/models?other=DML&sort=created
https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html#run-with-directml

## run
python onnxApp.py -m models\directml-int4-awq-block-128 -e dml
python onnxApp.py -m models\phi3vfp16 -e dml


huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include directml/* --local-dir .

## WIP
converting gguf or safetensors to onnx dml
chat app
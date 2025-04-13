from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_directml as tdml
import torch.nn as nn
# Set the device to DirectML
dml_device = tdml.device(0)  # DirectML device
device = torch.device("cpu")  # CPU device


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

# Move the model to the DirectML device

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.chat_template = '<start_of_turn>user\n{system_prompt}{input}<end_of_turn>\n<start_of_turn>model\n'
tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
model = model.to(device)
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
print("Chatbot is ready! Type 'exit' to quit.\n")
model.generate(**formatted_chat)
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# Disable warnings about padding_side that cannot be rectified with current software:
logging.set_verbosity_error()

model_names = ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large", "microsoft/phi-2"]
use_model_index = 3  # Change 0: small model, 1: medium, 2: large model (requires most resources!)
model_name = model_names[use_model_index]
          
tokenizer = AutoTokenizer.from_pretrained(model_name) # , padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)


# The chat function: received a user input and chat-history and returns the model's reply and chat-history:
def reply(input_text, history=None):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), chat_history_ids

history = None
while True:
    input_text = input("> ")
    if input_text in ["", "bye", "quit", "exit"]:
        break
    reply_text, history_new = reply(input_text, history)
    history=history_new
    if history.shape[1]>80:
        old_shape = history.shape
        history = history[:,-80:]
        print(f"History cut from {old_shape} to {history.shape}")
    # history_text = tokenizer.decode(history[0])
    # print(f"Current history: {history_text}")
    print(f"D_GPT: {reply_text}")
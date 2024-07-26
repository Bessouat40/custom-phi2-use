import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")

class Phi2:

    def __init__(self) -> None:
        self.model, self.tokenizer = self.load_phi2()
        self.prompt_begin = "Instruct: "
        self.prompt_end = "Provide detailed instructions for cooking including quantities of ingredients and cooking steps."
        self.prompt = None
        self.inputs = None
        
    @staticmethod
    def load_phi2():
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True) #torch.float32 for cpu
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        return model, tokenizer
    
    def format_inputs(self, user_input):
        self.prompt = self.prompt_begin + user_input + self.prompt_end
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt", return_attention_mask=False)

    def generate_outputs(self):
        output_ids = self.model.generate(**self.inputs, max_length=500, do_sample=True, top_p=0.95, top_k=60)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def generate_outputs_streaming(self):
        # Generate outputs in a streaming manner (character by character)
        output_ids = self.model.generate(**self.inputs, max_length=500, do_sample=True, top_p=0.95, top_k=60)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        for char in output_text:
            yield char

    def talk_to_me(self):
        prompt = input('>> Message FoodGPT: ')
        self.format_inputs(prompt)
        print("FoodGPT: ", end="", flush=True)

        for char in self.generate_outputs_streaming():
            print(char, end="", flush=True)
        print()  # Print a newline after the response is complete
        return self.talk_to_me()

# Example usage:
if __name__ == "__main__":
    bot = Phi2()
    bot.talk_to_me()

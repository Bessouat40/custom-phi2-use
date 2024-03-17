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
    def load_phi2() :
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True) #torch.float3é for cpu
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        return model, tokenizer
    
    def format_inputs(self, user_input) :
        self.prompt = self.prompt_begin + user_input + self.prompt_end
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt", return_attention_mask=False)

    def generate_outputs(self):
        # outputs = self.model.generate(**self.inputs, max_length=500)
        # return self.tokenizer.batch_decode(outputs)[0]
        output = self.model.generate(**self.inputs, max_length=500)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def talk_to_me(self) :
        prompt = input('>> Message FoodGPT : ')
        self.format_inputs(prompt)
        print(self.generate_outputs())
        return self.talk_to_me()

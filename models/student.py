from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional
from .base import BaseModel
from dotenv import load_dotenv
import os
from peft import PeftModel 
load_dotenv()

MODEL_NAME = os.getenv("STUDENT_MODEL_NAME")
LORA_PATH = os.getenv("STUDENT_LORA_PATH")

class StudentModel(BaseModel):
    """Student model from Hugging Face."""
    
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        hf_token = os.getenv("HF_TOKEN", None)
        lora_path = os.getenv("STUDENT_LORA_PATH", None)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading student base model {model_name} on {self.device}...")
        if lora_path:
            print(f"  with LoRA adapter from {lora_path}")
        else:
            print("  without LoRA adapter (pure base model)")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

        # Base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        # LoRA
        if lora_path:
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            self.model = base_model

        # pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Student model loaded successfully")
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs['input_ids'][i].shape[0]
            response_tokens = output[input_len:]  
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "type": "student",
            "device": self.device,
            "parameters": "1.5B"
        }
import os
from dotenv import load_dotenv
from .teacher import TeacherModel

load_dotenv()

class JudgeModel(TeacherModel):
    """Judge model for LLM-as-a-Judge score"""

    def __init__(self, base_url=None, model_name=None, token=None, **kwargs):
        super().__init__(
            base_url=base_url or os.getenv("JUDGE_URL"),
            model_name=model_name or os.getenv("JUDGE_MODEL"),
            token=token or os.getenv("JUDGE_TOKEN", ""),
            **kwargs
        )

    def score_response(self, prompt: str, response: str, max_score: int = 5) -> float:
        judge_prompt = (
            f"You are an expert evaluator. Rate the quality of the following response "
            f"to the given question on a scale from 1 to {max_score}.\n\n"
            f"Question: {prompt}\n\n"
            f"Response: {response}\n\n"
            f"Provide only a single integer score from 1 to {max_score}. "
            f"Score:"
        )
        raw = self.generate([judge_prompt], temperature=0.0, max_tokens=8)[0].strip()

        import re
        match = re.search(r"\b([1-5])\b", raw)
        if match:
            score = int(match.group(1))
            return (score - 1) / (max_score - 1)  
        return 0.5  

    def score_batch(self, prompts: list, responses: list, max_score: int = 5) -> list:
        return [
            self.score_response(p, r, max_score)
            for p, r in zip(prompts, responses)
        ]
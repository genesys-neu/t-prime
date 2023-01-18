from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
)  # doctest: +SKIP
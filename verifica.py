from ctransformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    model_type="llama"
)

print(model("Ciao, come posso aiutarti?"))
print("1. Python script started...")
from flask import Flask, request, jsonify, render_template

print("2. Flask imported successfully. Importing PyTorch (this may take a moment)...")
import torch
import torch.nn.functional as F

print("3. PyTorch imported. Importing Tiktoken...")
import tiktoken
from model import GPT, GPTConfig

app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"4. Booting Wall-E on: {device}")

print("5. Initiating Tiktoken vocabulary download/load...")
enc = tiktoken.get_encoding("gpt2")
eot_token = enc._special_tokens['<|endoftext|>']

print("6. Loading 124M parameter weights into memory (~500MB)...")
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load("gpt2_wall_e_assistant.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()
print("7. Model successfully loaded! Starting local server...")

def generate_text(prompt_text, max_new_tokens=150, temperature=0.4, top_k=40, repetition_penalty=1.2):
    formatted_prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt_text}\n\n"
        "### Response:\n"
    )
    
    input_tokens = enc.encode_ordinary(formatted_prompt)
    idx = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            
            # --- Repetition Penalty ---
            # Penalize tokens that have already appeared in the output
            generated_tokens = set(idx[0, len(input_tokens):].tolist())
            for token_id in generated_tokens:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

            # --- Top-K Filtering ---
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == eot_token:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)

    full_output = enc.decode(idx[0].tolist())
    response_only = full_output.split("### Response:\n")[-1].strip()
    return response_only

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400
        
    # Intercept basic greetings
    clean_input = user_input.lower()
    if clean_input in ["hi", "hello", "hey", "hi there", "hello walle"]:
        return jsonify({"reply": "Hello! I am Wall-E, a custom 124M parameter model. Please give me a specific instruction or task to complete!"})
        
    # Route actual tasks to the neural network
    reply = generate_text(user_input, temperature=0.4)
    return jsonify({"reply": reply})
if __name__ == '__main__':
    app.run(debug=True, port=5000)
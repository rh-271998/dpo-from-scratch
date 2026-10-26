from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json
from tqdm import tqdm
import sys

torch.manual_seed(0)

def get_model_answer(question):
    messages = [
        {"role": "system", "content": "Answer with a short phrase only, no explanation."},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,                       # greedy: its single best guess
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]    # slice off the prompt
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def get_model_answers_batch(questions):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role":"system","content":"Answer with a short phrase only, no explanation."},
             {"role":"user","content":q}],
            tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]          # left-pad → same start index for all
    return [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]


def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)          # drop punctuation
    s = re.sub(r'\b(a|an|the)\b', ' ', s)      # drop articles
    return re.sub(r'\s+', ' ', s).strip()

def contains(answer_norm, phrase_norm):
    if not phrase_norm: return False
    return re.search(r'\b' + re.escape(phrase_norm) + r'\b', answer_norm) is not None

def knows_answer(model_ans, correct, distractors):
    ans = norm(model_ans)
    if not contains(ans, norm(correct)):            # correct answer must appear
        return False
    if any(contains(ans, norm(d)) for d in distractors):  # ...and no distractor
        return False
    return True

if __name__ == "__main__":

    modelName = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        dtype=torch.float32,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)

    ds = load_dataset('allenai/sciq')

    known = []
    batchSize = 8

    for i in tqdm(range(0, len(ds['validation']), batchSize)):
        ex = ds['validation'][i:i+batchSize]
        q = ex['question']
        correct = ex['correct_answer']
        d1, d2, d3 = [ex['distractor1'], ex['distractor2'], ex['distractor3']]

        answers = get_model_answers_batch(q)
        
        for j in range(len(answers)):
            dists = [d1[j], d2[j], d3[j]]
            if knows_answer(answers[j], correct[j], dists):
                known.append({"question": q[j], "correct": correct[j], "distractors": dists})

    json.dump(known, open("../data/sciq_known_val.json", "w"), indent=2)
    print(f"kept {len(known)} / {len(ds['validation'])}")
            # print(f"knows={knows_answer(ans, correct, distractors)!s:5} | correct={correct!r} | model={ans[:50]!r} | Q={q[:55]}")
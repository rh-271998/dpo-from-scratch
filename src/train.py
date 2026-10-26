import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from loraLinear import get_lora_model, LoRALinear
from torch.optim import AdamW
import torch.nn.functional as F

torch.manual_seed(0)

def build_side(messages, response, tokenizer):

    promptText = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    promptLen = len(tokenizer(promptText, add_special_tokens=False)['input_ids'])

    fullText = tokenizer.apply_chat_template(messages + [{'role': "assistant",
                                                          'content': response}], tokenize=False)

    ids = tokenizer(fullText, add_special_tokens=False)['input_ids']

    responseMask = [0]*promptLen + [1]*(len(ids)-promptLen)
    return ids, responseMask

def pad_and_batch(data, tokenizer, maxLen):

    cIdArr = []
    cAttnArr = []
    cRespArr = []

    rIdArr = []
    rAttnArr = []
    rRespArr = []

    for i in range(len(data)):

        chosenIds, chosenMask = build_side(data[i]['messages'], data[i]['chosen'], tokenizer)
        chosenAttn = [1] * len(chosenIds)
        
        rejectedIds, rejectedMask = build_side(data[i]['messages'], data[i]['rejected'], tokenizer)
        rejectedAttn = [1] * len(rejectedIds)

        padLen = maxLen - len(chosenIds)

        assert padLen >= 0, f"maxLen {maxLen} is not sufficient. Increase maxLen to >= {len(chosenIds)}"

        if padLen > 0:
            chosenIds += [tokenizer.pad_token_id] * padLen
            chosenMask += [0] * padLen
            chosenAttn += [0] * padLen

        padLen = maxLen - len(rejectedIds)

        assert padLen >= 0, f"maxLen {maxLen} is not sufficient. Increase maxLen to >= {len(rejectedIds)}"

        if padLen > 0:
            rejectedIds += [tokenizer.pad_token_id] * padLen
            rejectedMask += [0] * padLen
            rejectedAttn += [0] * padLen

        cIdArr.append(chosenIds)
        cAttnArr.append(chosenAttn)
        cRespArr.append(chosenMask)

        rIdArr.append(rejectedIds)
        rAttnArr.append(rejectedAttn)
        rRespArr.append(rejectedMask)

    dataset = TensorDataset(torch.tensor(cIdArr), torch.tensor(cAttnArr), torch.tensor(cRespArr),
                            torch.tensor(rIdArr), torch.tensor(rAttnArr), torch.tensor(rRespArr))

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    return loader

def set_adapters(model, on):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = on

def seq_logprob(model, inputIds, attn, respMask):

    logits = model(input_ids=inputIds, attention_mask=attn).logits
    sl = logits[:, :-1, :]
    targets = inputIds[:, 1:]
    tokenLogp = sl.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(sl, dim=-1)
    return (tokenLogp * respMask[:, 1:]).sum(dim=1)

if __name__ == '__main__':

    maxLen = 160

    modelName = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('../data/sycophancyData.json', 'r') as f:
        data = json.load(f)

    loader = pad_and_batch(data, tokenizer, maxLen)

    model = get_lora_model(modelName)

    set_adapters(model, True)

    device = next(model.parameters()).device

    model.train()

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    beta = 0.1

    for epoch in range(3):

        lossEpoch = []
        accEpoch = []

        for cI, cA, cR, rI, rA, rR in loader:
            cI = cI.to(device)
            cA = cA.to(device)
            cR = cR.to(device)
            rI = rI.to(device)
            rA = rA.to(device)
            rR = rR.to(device)

            set_adapters(model, True)
            polW = seq_logprob(model, cI, cA, cR)
            polL = seq_logprob(model, rI, rA, rR)

            set_adapters(model, False)
            with torch.no_grad():
                refW = seq_logprob(model, cI, cA, cR)
                refL = seq_logprob(model, rI, rA, rR)

            set_adapters(model, True)

            margin = beta * ((polW - refW) - (polL - refL))
            
            loss = -F.logsigmoid(margin).mean()
            acc = (margin > 0).float().mean()
            
            lossEpoch.append(loss.item())
            accEpoch.append(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}, loss: {sum(lossEpoch)/len(lossEpoch):.4f}, acc: {sum(accEpoch)/len(accEpoch):.4f}")


    adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(adapters, f'../result/dpo.pt')
    print(f'saved checkpoint!')
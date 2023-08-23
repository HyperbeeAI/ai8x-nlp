###########################################################################
# NLP demo software by HyperbeeAI.                                        #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
license_statement = "NLP demo software by HyperbeeAI. Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai"
print("imported utils.py")
print(license_statement)
print("")

import torch
import layers
from tokenizers import Tokenizer
import time, torch, datasets
from tqdm import tqdm 

tokenizer_en = None 
tokenizer_es = None

def tokenize_es(text):
    return tokenizer_es.encode(text).ids[:48 - 2]

def tokenize_en(text):
    return tokenizer_en.encode(text).ids[:48 - 1]

def translate_sentence(sentence, src_field, trg_field, model, device):

    model.eval()
    if isinstance(sentence, str):
        tokens = tokenize_es(sentence)
    else:
        tokens = sentence

    tokens = [src_field.init_token] + tokens + [src_field.eos_token] + [src_field.pad_token] * (48 - 2 - len(tokens)) 
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_out, _ = model.encoder(src_tensor)

    trg_indexes = [trg_field.init_token, ] + [trg_field.pad_token] * (48 - 1) 

    for i in range(48 - 1):
        start_idx = max(0, i - 7)
        
        trg_tensor = torch.LongTensor(trg_indexes[start_idx:start_idx + 8]).unsqueeze(0).to(device)

        with torch.no_grad():
            output, _, _ = model.decoder(trg_tensor, enc_out, max(0, i - 7))

        pred_token = output.argmax(2)[:, min(i, 7)].item()
        trg_indexes[i + 1] = pred_token
        if pred_token == trg_field.eos_token:
            break

    try:
        trg_indexes = trg_indexes[1:trg_indexes.index(trg_field.eos_token)]
    except ValueError: 
        trg_indexes = trg_indexes[1:]

    trg_tokens = tokenizer_en.decode(trg_indexes, skip_special_tokens=False)

    return trg_tokens


def calculate_bleu(data, src_field, trg_field, model, device, spiece=False, output_file = f"test.{time.time()}.out"):
    
    if spiece:
        from tokenizers import pre_tokenizers
        pre_tokenizer = pre_tokenizers.Digits(individual_digits=True)
    else:
        pre_tokenizer = tokenizer_en.pre_tokenizer

    trgs = []
    pred_trgs = []
    print('Evaluate on bleu:')
    for src, trg in tqdm(zip(open("news-comm-v15/news-comm-v15-all-test.es"), open("news-comm-v15/news-comm-v15-all-test.en"))):

        if len(src) < 3 or len(trg) < 3:
            continue

        normalized = pre_tokenizer.pre_tokenize_str(tokenizer_en.normalizer.normalize_str(trg))

        if len(normalized) > 48:
            continue

        trgs.append([ " ".join(map(lambda x: x[0], normalized)) ])

        pred_trg = translate_sentence(src, src_field, trg_field, model, device)
        pred_trgs.append(pred_trg)


    with open(output_file, "w") as fo:
        fo.write("\n".join(pred_trgs))

    sacrebleu = datasets.load_metric('sacrebleu')
    return sacrebleu.compute(predictions=pred_trgs, references=trgs)

tokenizer_es = Tokenizer.from_file(f"assets/es.json")
tokenizer_en = Tokenizer.from_file(f"assets/en.json")
TRG_PAD_IDX  = tokenizer_en.token_to_id("<PAD>")

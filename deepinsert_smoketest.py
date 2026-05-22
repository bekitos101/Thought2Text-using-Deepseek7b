import torch
from model import EEGModelForCausalLM

print("Loading model...", flush=True)
model = EEGModelForCausalLM.from_pretrained("all_models/deepseek-llm-7b-chat_all")
model.eeg_encoder.cuda()
model.mm_proj.cuda()
model.eval()

print(f"token_inject={model.token_inject}  mm_proj.weight.shape={model.mm_proj.weight.shape}", flush=True)

B = 1
device = "cuda"

# Build mm_embeds_raw matching the projector input shape
if model.token_inject:
    mm_embeds_raw = torch.randn(B, 10, 50, device=device)        # (B, 10, out_channels)
else:
    mm_embeds_raw = torch.randn(B, model.mm_proj.weight.shape[1], device=device)  # (B, emb_size)

input_ids1 = torch.full((B, 30), model.padding_token_id, dtype=torch.long, device=device)
input_ids1[0, 20:] = 1   # 10 real prompt tokens
input_ids2 = torch.full((B, 20), model.padding_token_id, dtype=torch.long, device=device)
input_ids2[0, 15:] = 2   # 5 real response tokens

# Project and unsqueeze to (B, E, H) — same steps model.forward() does
mm_embeds = model.mm_proj(mm_embeds_raw)
if len(mm_embeds.shape) == 2:
    mm_embeds = mm_embeds.unsqueeze(1)
print(f"mm_embeds shape after proj: {mm_embeds.shape}", flush=True)

print("Testing _deepinsert_forward at injection_layer=21 ...", flush=True)
model.train()
out, labels = model._deepinsert_forward(input_ids1, input_ids2, mm_embeds, injection_layer=21)
print(f"  loss={out.loss.item():.4f}  logits={out.logits.shape}  labels={labels.shape}", flush=True)

print("Testing _deepinsert_generate at injection_layer=21 ...", flush=True)
model.eval()
mm_embeds2 = model.mm_proj(mm_embeds_raw)
if len(mm_embeds2.shape) == 2:
    mm_embeds2 = mm_embeds2.unsqueeze(1)
gen_ids, _ = model._deepinsert_generate(input_ids1, input_ids2, mm_embeds2,
                                          injection_layer=21, max_new_tokens=5)
print(f"  generated shape: {gen_ids.shape}  tokens: {gen_ids[0].tolist()}", flush=True)

print("PASS", flush=True)

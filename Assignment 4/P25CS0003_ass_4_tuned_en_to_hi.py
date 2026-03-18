# -*- coding: utf-8 -*-
"""
Assignment 4 - Optimizing Transformer Translation with Ray Tune & Optuna
English → Hindi Neural Machine Translation
Submit as: rollno_ass_4_tuned_en_to_hi.py / .ipynb
"""

# ─────────────────────────────────────────────
# 0.  Installs  (uncomment if running fresh)
# ─────────────────────────────────────────────
# !pip install ray[tune] optuna

# ─────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────
import os, math, time, pickle, warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.grid": True, "grid.color": "#dee2e6", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
PALETTE = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3", "#4895ef"]
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.show()

# =============================================================================
# 2.  DATA LOADING & EDA
# =============================================================================
print("\n" + "="*60)
print("SECTION 2 — DATA LOADING & EDA")
print("="*60)

df = pd.read_csv("English-Hindi.tsv", sep="\t", header=None,
                 names=["id1", "en", "id2", "hi"])
df = df[["en", "hi"]].dropna().reset_index(drop=True)

df["en_len"]   = df["en"].apply(lambda x: len(str(x).split()))
df["hi_len"]   = df["hi"].apply(lambda x: len(str(x).split()))
df["en_chars"] = df["en"].apply(len)
df["hi_chars"] = df["hi"].apply(len)

print(f"Total sentence pairs : {len(df)}")
print("\nEnglish stats:\n", df["en_len"].describe().round(2))
print("\nHindi stats:\n",   df["hi_len"].describe().round(2))

# ── GRAPH 1: EDA Dashboard ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("EDA Dashboard — English–Hindi Dataset", fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df["en_len"], bins=40, kde=True, color=PALETTE[0], ax=ax1, alpha=0.75)
ax1.set_title("English Sentence Length"); ax1.set_xlabel("Words")

ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(df["hi_len"], bins=40, kde=True, color=PALETTE[1], ax=ax2, alpha=0.75)
ax2.set_title("Hindi Sentence Length"); ax2.set_xlabel("Words")

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df["en_len"], df["hi_len"], alpha=0.15, s=4, color=PALETTE[2])
m, b = np.polyfit(df["en_len"], df["hi_len"], 1)
xr = np.linspace(df["en_len"].min(), df["en_len"].max(), 100)
ax3.plot(xr, m*xr+b, color=PALETTE[3], linewidth=2, label=f"y={m:.2f}x+{b:.2f}")
ax3.set_title("EN vs HI Length (scatter)"); ax3.set_xlabel("EN words"); ax3.set_ylabel("HI words")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 0])
ax4.boxplot([df["en_len"], df["hi_len"]], labels=["English","Hindi"], patch_artist=True,
            boxprops=dict(facecolor=PALETTE[0],alpha=0.5),
            medianprops=dict(color="red",linewidth=2))
ax4.set_title("Length Boxplot"); ax4.set_ylabel("Words")

ax5 = fig.add_subplot(gs[1, 1])
ratio = df["hi_len"] / df["en_len"].replace(0, np.nan)
sns.histplot(ratio.dropna(), bins=40, kde=True, color=PALETTE[4], ax=ax5, alpha=0.75)
ax5.axvline(ratio.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean={ratio.mean():.2f}")
ax5.set_title("HI/EN Length Ratio"); ax5.legend(fontsize=8)

ax6 = fig.add_subplot(gs[1, 2])
all_en = " ".join(df["en"]).lower().split()
top_en = Counter(all_en).most_common(10)
wds,cnts = zip(*top_en)
ax6.barh(list(wds)[::-1], list(cnts)[::-1], color=PALETTE[0])
ax6.set_title("Top-10 EN Words"); ax6.set_xlabel("Count")

ax7 = fig.add_subplot(gs[2, 0])
for col,lbl,c in [("en_len","English",PALETTE[0]),("hi_len","Hindi",PALETTE[1])]:
    vals = np.sort(df[col].values)
    ax7.plot(vals, np.arange(1,len(vals)+1)/len(vals), label=lbl, color=c, linewidth=2)
ax7.axvline(50, color="red", linestyle="--", linewidth=1.5, label="max_len=50")
ax7.set_title("Cumulative Length Coverage")
ax7.set_xlabel("Sentence Length"); ax7.set_ylabel("Fraction"); ax7.legend(fontsize=8)

ax8 = fig.add_subplot(gs[2, 1])
sns.kdeplot(df["en_chars"], color=PALETTE[0], label="English", fill=True, alpha=0.4, ax=ax8)
sns.kdeplot(df["hi_chars"], color=PALETTE[1], label="Hindi",   fill=True, alpha=0.4, ax=ax8)
ax8.set_title("Character Length KDE"); ax8.set_xlabel("Characters"); ax8.legend(fontsize=8)

ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
summary = [["Metric","English","Hindi"],
           ["Total pairs",f"{len(df):,}",f"{len(df):,}"],
           ["Mean length",f"{df['en_len'].mean():.1f}",f"{df['hi_len'].mean():.1f}"],
           ["Median",f"{df['en_len'].median():.1f}",f"{df['hi_len'].median():.1f}"],
           ["Max",str(df['en_len'].max()),str(df['hi_len'].max())],
           ["<=50 words",f"{(df['en_len']<=50).mean()*100:.1f}%",
            f"{(df['hi_len']<=50).mean()*100:.1f}%"]]
tbl = ax9.table(cellText=summary[1:], colLabels=summary[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.6)
for (r,c),cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor("#4361ee"); cell.set_text_props(color="white")
    elif r%2: cell.set_facecolor("#eef2ff")
ax9.set_title("Dataset Summary", pad=12)

plt.tight_layout()
savefig("01_eda_dashboard.png")

for i in range(5):
    print(f"EN: {df.loc[i,'en']}\nHI: {df.loc[i,'hi']}\n---")

# =============================================================================
# 3.  VOCABULARY
# =============================================================================
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0:"<pad>",1:"<sos>",2:"<eos>",3:"<unk>"}
        self.stoi = {"<pad>":0,"<sos>":1,"<eos>":2,"<unk>":3}
        self.idx = 4; self.word_freq = Counter()

    def build_vocab(self, sentence_list):
        for s in sentence_list:
            for w in self.tokenize(s): self.word_freq[w] += 1
        for w,f in self.word_freq.items():
            if f >= self.freq_threshold:
                self.stoi[w] = self.idx; self.itos[self.idx] = w; self.idx += 1

    def tokenize(self, s): return str(s).lower().strip().split()
    def numericalize(self, s):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in self.tokenize(s)]
    def __len__(self):        return len(self.stoi)
    def __getitem__(self, t): return self.stoi.get(t, self.stoi["<unk>"])

en_vocab = Vocabulary(freq_threshold=2)
hi_vocab = Vocabulary(freq_threshold=2)
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())
print(f"\nEnglish vocab: {len(en_vocab)} | Hindi vocab: {len(hi_vocab)}")

# ── GRAPH 2: Vocabulary Analysis ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("Vocabulary Analysis", fontsize=15, fontweight="bold")

ax = axes[0,0]
ax.loglog(range(1,501), sorted(en_vocab.word_freq.values(),reverse=True)[:500],
          color=PALETTE[0], linewidth=1.5)
ax.set_title("English Word Frequency (Zipf)"); ax.set_xlabel("Rank"); ax.set_ylabel("Freq")

ax = axes[0,1]
ax.loglog(range(1,501), sorted(hi_vocab.word_freq.values(),reverse=True)[:500],
          color=PALETTE[1], linewidth=1.5)
ax.set_title("Hindi Word Frequency (Zipf)"); ax.set_xlabel("Rank"); ax.set_ylabel("Freq")

ax = axes[0,2]
thresh = [1,2,3,5,10,20]
ax.plot(thresh,[sum(1 for f in en_vocab.word_freq.values() if f>=t) for t in thresh],
        "o-", color=PALETTE[0], label="English", linewidth=2)
ax.plot(thresh,[sum(1 for f in hi_vocab.word_freq.values() if f>=t) for t in thresh],
        "s-", color=PALETTE[1], label="Hindi",   linewidth=2)
ax.axvline(2,color="red",linestyle="--",linewidth=1.5,label="threshold=2")
ax.set_title("Vocab Size vs Threshold"); ax.set_xlabel("Freq Threshold"); ax.legend(fontsize=9)

special = {"<pad>","<sos>","<eos>","<unk>"}
ax = axes[1,0]
top20 = [(w,f) for w,f in en_vocab.word_freq.most_common(25) if w not in special][:20]
wl,fl = zip(*top20)
ax.barh(list(wl)[::-1], list(fl)[::-1], color=PALETTE[0], alpha=0.85)
ax.set_title("Top-20 English Words"); ax.set_xlabel("Frequency")

ax = axes[1,1]
top20h = [(w,f) for w,f in hi_vocab.word_freq.most_common(25) if w not in special][:20]
wh,fh = zip(*top20h)
ax.barh(list(wh)[::-1], list(fh)[::-1], color=PALETTE[1], alpha=0.85)
ax.set_title("Top-20 Hindi Words"); ax.set_xlabel("Frequency")

ax = axes[1,2]
buckets = [0,5,10,15,20,30,50]
oov_r = []
for lo,hi_b in zip(buckets,buckets[1:]):
    sub = df[(df["en_len"]>=lo)&(df["en_len"]<hi_b)]
    tot = oov = 0
    for s in sub["en"]:
        for tok in en_vocab.tokenize(s):
            tot += 1
            if tok not in en_vocab.stoi: oov += 1
    oov_r.append(oov/tot*100 if tot else 0)
ax.bar([f"{lo}-{hi_b}" for lo,hi_b in zip(buckets,buckets[1:])], oov_r,
       color=PALETTE[2], alpha=0.85)
ax.set_title("EN OOV Rate by Sentence Length"); ax.set_xlabel("Length bucket"); ax.set_ylabel("OOV %")

plt.tight_layout()
savefig("02_vocab_analysis.png")

# =============================================================================
# 4.  ENCODING HELPER
# =============================================================================
def encode_sentence(sentence, vocab, max_len=50):
    toks = ([vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len-2]
            + [vocab.stoi["<eos>"]])
    return toks + [vocab.stoi["<pad>"]] * (max_len - len(toks))

# ── GRAPH 3: Token encoding heatmap ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
fig.suptitle("Token Encoding Heatmap (10 sample sentences, max_len=50)",
             fontsize=13, fontweight="bold")
for ax,(sentences,vocab,lang) in zip(axes, [
        (df["en"].head(10).tolist(), en_vocab, "English"),
        (df["hi"].head(10).tolist(), hi_vocab, "Hindi")]):
    mat = np.array([encode_sentence(s,vocab) for s in sentences])
    im  = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_title(f"{lang} Encoded Token IDs"); ax.set_xlabel("Position"); ax.set_ylabel("Sentence #")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
savefig("03_encoding_heatmap.png")

# =============================================================================
# 5.  MODEL ARCHITECTURE
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model,self.num_heads,self.d_k = d_model,num_heads,d_model//num_heads
        self.Wq=nn.Linear(d_model,d_model); self.Wk=nn.Linear(d_model,d_model)
        self.Wv=nn.Linear(d_model,d_model); self.Wo=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(0.1)
    def forward(self,q,k,v,mask=None):
        B=q.size(0)
        Q=self.Wq(q).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        K=self.Wk(k).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        V=self.Wv(v).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        sc=torch.matmul(Q,K.transpose(-2,-1))/(self.d_k**0.5)
        if mask is not None: sc=sc.masked_fill(mask==0,-1e9)
        o=self.drop(torch.softmax(sc,dim=-1)).matmul(V)
        return self.Wo(o.transpose(1,2).contiguous().view(B,-1,self.d_model))

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(),
                               nn.Dropout(dropout),nn.Linear(d_ff,d_model))
    def forward(self,x): return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.g=nn.Parameter(torch.ones(d_model)); self.b=nn.Parameter(torch.zeros(d_model)); self.eps=eps
    def forward(self,x):
        m,s=x.mean(-1,keepdim=True),x.std(-1,keepdim=True)
        return self.g*(x-m)/(s+self.eps)+self.b

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,num_heads); self.ffn=FeedForward(d_model,d_ff,dropout)
        self.n1,self.n2=LayerNorm(d_model),LayerNorm(d_model); self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        x=self.n1(x+self.drop(self.attn(x,x,x,mask))); return self.n2(x+self.drop(self.ffn(x)))

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.s_attn=MultiHeadAttention(d_model,num_heads); self.c_attn=MultiHeadAttention(d_model,num_heads)
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.n1,self.n2,self.n3=LayerNorm(d_model),LayerNorm(d_model),LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,enc,sm=None,tm=None):
        x=self.n1(x+self.drop(self.s_attn(x,x,x,tm)))
        x=self.n2(x+self.drop(self.c_attn(x,enc,enc,sm)))
        return self.n3(x+self.drop(self.ffn(x)))

class Encoder(nn.Module):
    def __init__(self,vsz,d_model,nl,nh,dff,ml,drop):
        super().__init__()
        self.emb=nn.Embedding(vsz,d_model); self.pos=PositionalEncoding(d_model,ml)
        self.lyrs=nn.ModuleList([EncoderLayer(d_model,nh,dff,drop) for _ in range(nl)])
        self.drop=nn.Dropout(drop)
    def forward(self,x,mask=None):
        x=self.drop(self.pos(self.emb(x)))
        for l in self.lyrs: x=l(x,mask)
        return x

class Decoder(nn.Module):
    def __init__(self,vsz,d_model,nl,nh,dff,ml,drop):
        super().__init__()
        self.emb=nn.Embedding(vsz,d_model); self.pos=PositionalEncoding(d_model,ml)
        self.lyrs=nn.ModuleList([DecoderLayer(d_model,nh,dff,drop) for _ in range(nl)])
        self.drop=nn.Dropout(drop)
    def forward(self,x,enc,sm=None,tm=None):
        x=self.drop(self.pos(self.emb(x)))
        for l in self.lyrs: x=l(x,enc,sm,tm)
        return x

class Transformer(nn.Module):
    def __init__(self,sv,tv,d_model=512,nl=6,nh=8,dff=2048,ml=100,drop=0.1):
        super().__init__()
        self.enc=Encoder(sv,d_model,nl,nh,dff,ml,drop)
        self.dec=Decoder(tv,d_model,nl,nh,dff,ml,drop)
        self.fc=nn.Linear(d_model,tv)
    def pad_mask(self,seq,idx): return (seq!=idx).unsqueeze(1).unsqueeze(2)
    def sub_mask(self,sz): return torch.tril(torch.ones(sz,sz)).bool().to(next(self.parameters()).device)
    def forward(self,src,tgt,sp,tp):
        sm=self.pad_mask(src,sp)
        tm=self.pad_mask(tgt,tp)&self.sub_mask(tgt.size(1))
        return self.fc(self.dec(tgt,self.enc(src,sm),sm,tm))

# ── GRAPH 4: Architecture insights ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Transformer Architecture Insights", fontsize=14, fontweight="bold")

ax = axes[0]
d_,ml_ = 64, 50
pe = np.zeros((ml_,d_))
p  = np.arange(ml_)[:,None]
dv = np.exp(np.arange(0,d_,2)*(-np.log(10000.0)/d_))
pe[:,0::2]=np.sin(p*dv); pe[:,1::2]=np.cos(p*dv)
im=ax.imshow(pe.T,cmap="RdBu",aspect="auto",origin="lower")
plt.colorbar(im,ax=ax)
ax.set_title("Positional Encoding (d_model=64)"); ax.set_xlabel("Position"); ax.set_ylabel("Dim")

ax = axes[1]
configs=[("3L-4H-1024",3,4,1024),("3L-8H-2048",3,8,2048),
         ("4L-4H-2048",4,4,2048),("6L-8H-2048",6,8,2048),("6L-8H-4096",6,8,4096)]
pm=[]
for _,nl,nh,dff in configs:
    m=Transformer(len(en_vocab),len(hi_vocab),512,nl,nh,dff,50,0.1)
    pm.append(sum(p.numel() for p in m.parameters())/1e6); del m
bars=ax.bar([c[0] for c in configs], pm, color=PALETTE[:len(configs)])
ax.set_title("Parameter Count by Config"); ax.set_ylabel("Parameters (M)")
for bar,val in zip(bars,pm):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{val:.1f}M", ha="center", fontsize=8)
plt.xticks(rotation=15); plt.tight_layout()
savefig("04_architecture_insights.png")

# =============================================================================
# 6.  DATASET & COLLATE
# =============================================================================
class TranslationDataset(Dataset):
    def __init__(self,df,ev,hv,ml=50):
        self.en=df["en"].tolist(); self.hi=df["hi"].tolist(); self.ev,self.hv,self.ml=ev,hv,ml
    def __len__(self): return len(self.en)
    def __getitem__(self,i):
        return (torch.tensor(encode_sentence(self.en[i],self.ev,self.ml)),
                torch.tensor(encode_sentence(self.hi[i],self.hv,self.ml)))

def collate_fn(batch):
    src,tgt=zip(*batch); src,tgt=torch.stack(src),torch.stack(tgt)
    # .contiguous() is critical — slicing creates non-contiguous tensors
    # which crash inside Ray forked worker processes
    return src.contiguous(), tgt[:,:-1].contiguous(), tgt[:,1:].contiguous()

# =============================================================================
# 7.  TRANSLATE & BLEU
# =============================================================================
VAL_DATASET = [
    ("I love you.",                  "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?",                 "आप कैसे हैं?"),
    ("You should sleep.",            "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.",  "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.",             "मुझे टॉम को बताने दीजिए।"),
]

def translate_sentence(model,sent,ev,hv,device,ml=50):
    model.eval()
    src=torch.tensor(encode_sentence(sent,ev,ml)).unsqueeze(0).to(device)
    toks=[hv["<sos>"]]
    with torch.no_grad():
        for _ in range(ml):
            tgt=torch.tensor(toks).unsqueeze(0).to(device)
            out=model(src,tgt,ev["<pad>"],hv["<pad>"])
            nxt=out[0,-1].argmax().item(); toks.append(nxt)
            if nxt==hv["<eos>"]: break
    return " ".join(hv.itos[i] for i in toks[1:-1])

def evaluate_bleu(model,val_data,ev,hv,device,ml=50):
    refs,hyps=[],[]
    for en_s,hi_s in val_data:
        hyps.append(translate_sentence(model,en_s,ev,hv,device,ml).split())
        refs.append([hi_s.split()])
    return corpus_bleu(refs,hyps,smoothing_function=SmoothingFunction().method4)

# =============================================================================
# 8.  PART 1 — BASELINE
# =============================================================================
def run_baseline():
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ML,BS=50,60; SP=en_vocab["<pad>"]; TP=hi_vocab["<pad>"]
    dataset=TranslationDataset(df,en_vocab,hi_vocab,ML)
    loader=DataLoader(dataset,batch_size=BS,shuffle=True,collate_fn=collate_fn,num_workers=0)
    model=Transformer(len(en_vocab),len(hi_vocab),512,6,8,2048,ML,0.1).to(DEVICE)
    crit=nn.CrossEntropyLoss(ignore_index=TP)
    opt=optim.Adam(model.parameters(),lr=1e-4)
    losses,bleu_hist=[],[]
    start=time.time()
    for epoch in range(100):
        model.train(); ep=0.0
        for src,ti,to in tqdm(loader,desc=f"Baseline {epoch+1}/100",leave=False):
            src,ti,to=src.to(DEVICE),ti.to(DEVICE),to.to(DEVICE)
            out=model(src,ti,SP,TP).contiguous().reshape(-1,len(hi_vocab))
            loss=crit(out,to.contiguous().reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); ep+=loss.item()
        avg=ep/len(loader); losses.append(avg)
        if (epoch+1)%10==0:
            bl=evaluate_bleu(model,VAL_DATASET,en_vocab,hi_vocab,DEVICE)
            bleu_hist.append((epoch+1,bl))
            print(f"Epoch {epoch+1:3d} | Loss: {avg:.4f} | BLEU: {bl*100:.2f}")
    total=time.time()-start
    torch.save(model.state_dict(),"transformer_translation_final.pth")
    _plot_baseline_curves(losses,bleu_hist,total)
    print(f"\n{'='*50}\n  Time: {total:.1f}s | Loss: {losses[-1]:.4f} | BLEU: {bleu_hist[-1][1]*100:.2f}\n{'='*50}")
    return {"time":total,"loss":losses[-1],"bleu":bleu_hist[-1][1] if bleu_hist else 0,
            "loss_curve":losses,"bleu_history":bleu_hist,"model":model,"device":DEVICE}

def _plot_baseline_curves(losses, bleu_hist, total_time):
    # ── GRAPH 5: Baseline training curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Baseline Training — 100 Epochs  ({total_time/60:.1f} min)",
                 fontsize=14, fontweight="bold")

    ax=axes[0]
    ax.plot(range(1,len(losses)+1),losses,color=PALETTE[0],linewidth=2)
    ax.fill_between(range(1,len(losses)+1),losses,alpha=0.15,color=PALETTE[0])
    ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    ax=axes[1]
    w=5; smooth=np.convolve(losses,np.ones(w)/w,mode="valid")
    ax.plot(range(1,len(losses)+1),losses,color=PALETTE[0],alpha=0.3,linewidth=1,label="Raw")
    ax.plot(range(w,len(losses)+1),smooth,color=PALETTE[3],linewidth=2,label=f"Smooth w={w}")
    ax.set_title("Loss Raw vs Smoothed"); ax.legend()

    ax=axes[2]
    if bleu_hist:
        ep,bl=zip(*bleu_hist)
        ax.plot(ep,[b*100 for b in bl],"o-",color=PALETTE[1],linewidth=2,markersize=6)
        ax.fill_between(ep,[b*100 for b in bl],alpha=0.15,color=PALETTE[1])
        ax.set_title("BLEU Progress"); ax.set_xlabel("Epoch"); ax.set_ylabel("BLEU (%)")

    plt.tight_layout()
    savefig("05_baseline_training_curves.png")

# =============================================================================
# 9.  PART 2 — RAY TUNE + OPTUNA
# =============================================================================
def train_tune(config):
    from ray import tune as ray_tune
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ML=50; SP=en_vocab["<pad>"]; TP=hi_vocab["<pad>"]
    lr,bs,nh,dff,drop,nl,ne=(config["lr"],config["batch_size"],config["num_heads"],
                               config["d_ff"],config["dropout"],config["num_layers"],config["num_epochs"])
    dataset=TranslationDataset(df,en_vocab,hi_vocab,ML)
    loader=DataLoader(dataset,batch_size=bs,shuffle=True,collate_fn=collate_fn,num_workers=0)
    model=Transformer(len(en_vocab),len(hi_vocab),512,nl,nh,dff,ML,drop).to(DEVICE)
    crit=nn.CrossEntropyLoss(ignore_index=TP)
    opt=optim.Adam(model.parameters(),lr=lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ne)
    avg=None
    for epoch in range(ne):
        model.train(); ep=0.0
        for src,ti,to in loader:
            src,ti,to=src.to(DEVICE),ti.to(DEVICE),to.to(DEVICE)
            out=model(src,ti,SP,TP).contiguous().reshape(-1,len(hi_vocab))
            loss=crit(out,to.contiguous().reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); ep+=loss.item()
        avg=ep/len(loader); sched.step()
        ray_tune.report({"loss": avg, "epoch": epoch+1})
    bleu=evaluate_bleu(model,VAL_DATASET,en_vocab,hi_vocab,DEVICE)
    ray_tune.report({"loss": avg, "bleu": bleu, "epoch": ne})

# 6 hyperparameters — satisfies >=4 requirement
search_space = {
    "lr":         tune.loguniform(1e-5, 1e-3),   # 1
    "batch_size": tune.choice([32, 64, 128]),      # 2
    "num_heads":  tune.choice([4, 8]),             # 3  (512 divisible by both)
    "d_ff":       tune.choice([1024, 2048, 4096]), # 4
    "dropout":    tune.uniform(0.1, 0.4),          # 5
    "num_layers": tune.choice([3, 4, 6]),          # 6
    "num_epochs": 15,
}

optuna_search  = OptunaSearch(metric="loss", mode="min")
asha_scheduler = ASHAScheduler(metric="loss", mode="min",
                                max_t=15, grace_period=3, reduction_factor=2)

def run_tuning(num_samples=20):
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=16,          # limit total CPUs Ray can use
            object_store_memory=2_000_000_000  # 2GB object store limit
        )
    tuner=tune.Tuner(
        tune.with_resources(train_tune, resources={"cpu": 2}),  # 1 CPU per trial
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=num_samples,
            max_concurrent_trials=8   # run 4 trials at once
        ),
        param_space=search_space)
    results=tuner.fit()
    best=results.get_best_result(metric="loss",mode="min")
    print("\n"+"="*60+"\nBEST CONFIG")
    for k,v in best.config.items(): print(f"  {k:15s}: {v}")
    print(f"  Best loss: {best.metrics['loss']:.4f}\n"+"="*60)
    _plot_tuning_analysis(results)
    return results, best

def _plot_tuning_analysis(results):
    # ── GRAPH 6: Tuning sweep analysis ───────────────────────────────────────
    rows=[]
    for r in results:
        cfg=r.config; m=r.metrics or {}
        rows.append({"lr":cfg.get("lr",np.nan),"batch_size":cfg.get("batch_size",np.nan),
                     "num_heads":cfg.get("num_heads",np.nan),"d_ff":cfg.get("d_ff",np.nan),
                     "dropout":cfg.get("dropout",np.nan),"num_layers":cfg.get("num_layers",np.nan),
                     "loss":m.get("loss",np.nan),"bleu":m.get("bleu",np.nan)})
    tdf=pd.DataFrame(rows).dropna(subset=["loss"])
    if tdf.empty: print("  [info] No trial data yet."); return

    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.suptitle("Ray Tune + Optuna Sweep Analysis",fontsize=14,fontweight="bold")

    ax=axes[0,0]
    ax.hist(tdf["loss"],bins=15,color=PALETTE[0],edgecolor="white",alpha=0.85)
    ax.axvline(tdf["loss"].min(),color="red",linestyle="--",linewidth=2,
               label=f"Best={tdf['loss'].min():.4f}")
    ax.set_title("Trial Loss Distribution"); ax.set_xlabel("Loss"); ax.legend()

    ax=axes[0,1]
    sc=ax.scatter(tdf["lr"],tdf["loss"],c=tdf["loss"],cmap="RdYlGn_r",
                  s=60,alpha=0.8,edgecolors="grey",linewidths=0.4)
    plt.colorbar(sc,ax=ax,label="Loss")
    ax.set_xscale("log"); ax.set_title("Learning Rate vs Loss")
    ax.set_xlabel("LR (log scale)"); ax.set_ylabel("Loss")

    ax=axes[0,2]
    bins_=[0.1,0.2,0.3,0.4]
    grps=[tdf[(tdf["dropout"]>=lo)&(tdf["dropout"]<hi)]["loss"].values
          for lo,hi in zip(bins_,bins_[1:])]
    grps=[g for g in grps if len(g)>0]
    bp=ax.boxplot(grps,labels=["0.1-0.2","0.2-0.3","0.3-0.4"][:len(grps)],patch_artist=True)
    for patch,c in zip(bp["boxes"],PALETTE): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title("Dropout vs Loss")

    ax=axes[1,0]
    for hp,vals,c in [("num_heads",[4,8],PALETTE[0]),
                       ("num_layers",[3,4,6],PALETTE[1]),
                       ("batch_size",[32,64,128],PALETTE[2])]:
        xr=[f"{hp}={v}" for v in vals]
        yl=[tdf[tdf[hp]==v]["loss"].mean() for v in vals]
        ax.bar(xr,yl,color=c,alpha=0.75)
    ax.set_title("Mean Loss by Categorical HP"); ax.set_ylabel("Mean Loss")
    plt.setp(ax.get_xticklabels(),rotation=25,ha="right")

    ax=axes[1,1]
    for dff,c in zip([1024,2048,4096],PALETTE[:3]):
        sub=tdf[tdf["d_ff"]==dff]["loss"]
        if len(sub): ax.scatter([dff]*len(sub),sub,color=c,s=50,alpha=0.7,label=f"d_ff={dff}")
    ax.set_title("d_ff vs Loss"); ax.set_xlabel("d_ff"); ax.set_ylabel("Loss"); ax.legend()

    ax=axes[1,2]
    top10=tdf.nsmallest(10,"loss").reset_index(drop=True)
    for _,row in top10.iterrows():
        nlr=(np.log10(row["lr"])-np.log10(1e-5))/(np.log10(1e-3)-np.log10(1e-5))
        ndr=(row["dropout"]-0.1)/0.3
        nls=1-(row["loss"]-tdf["loss"].min())/(tdf["loss"].max()-tdf["loss"].min()+1e-9)
        ax.plot([0,1,2],[nlr,ndr,nls],color=plt.cm.RdYlGn(nls),alpha=0.7,linewidth=1.5)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["LR","Dropout","Loss(inv)"])
    ax.set_title("Parallel Coords — Top-10 Trials")

    plt.tight_layout()
    savefig("06_tuning_analysis.png")

# =============================================================================
# 10.  PART 3 — RETRAIN BEST
# =============================================================================
def retrain_best(best_config, baseline_metrics=None, save_path="rollno_ass_4_best_model.pth"):
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ML=50; SP=en_vocab["<pad>"]; TP=hi_vocab["<pad>"]
    lr,bs,nh,dff,drop,nl,ne=(best_config["lr"],best_config["batch_size"],
                               best_config["num_heads"],best_config["d_ff"],
                               best_config["dropout"],best_config["num_layers"],
                               best_config.get("num_epochs",40))
    dataset=TranslationDataset(df,en_vocab,hi_vocab,ML)
    loader=DataLoader(dataset,batch_size=bs,shuffle=True,collate_fn=collate_fn,num_workers=0)
    model=Transformer(len(en_vocab),len(hi_vocab),512,nl,nh,dff,ML,drop).to(DEVICE)
    crit=nn.CrossEntropyLoss(ignore_index=TP)
    opt=optim.Adam(model.parameters(),lr=lr)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ne)
    BASE_BLEU=(baseline_metrics or {}).get("bleu", 0.50)
    # Normalize — if stored as percentage (>1), convert to 0-1
    if BASE_BLEU > 1: BASE_BLEU = BASE_BLEU / 100.0
    losses,bleu_hist=[],[]; matched=None; start=time.time()
    best_bleu_seen=0.0; best_epoch_seen=0
    for epoch in range(ne):
        model.train(); ep=0.0
        for src,ti,to in tqdm(loader,desc=f"Best {epoch+1}/{ne}",leave=False):
            src,ti,to=src.to(DEVICE),ti.to(DEVICE),to.to(DEVICE)
            out=model(src,ti,SP,TP).contiguous().reshape(-1,len(hi_vocab))
            loss=crit(out,to.contiguous().reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); ep+=loss.item()
        avg=ep/len(loader); losses.append(avg); sched.step()
        if (epoch+1)%5==0 or epoch==ne-1:
            bl=evaluate_bleu(model,VAL_DATASET,en_vocab,hi_vocab,DEVICE)
            bleu_hist.append((epoch+1,bl))
            print(f"Epoch {epoch+1:3d} | Loss: {avg:.4f} | BLEU: {bl*100:.2f}")
            if bl>=BASE_BLEU and matched is None:
                matched=epoch+1; print(f"  Baseline BLEU matched at epoch {matched}!")
            if bl > best_bleu_seen:
                best_bleu_seen=bl; best_epoch_seen=epoch+1
                torch.save(model.state_dict(), save_path)
                print(f"  Star New best BLEU {bl*100:.2f}% at epoch {epoch+1} — checkpoint saved!")
    total=time.time()-start
    final_bleu=best_bleu_seen
    print(f"  Best BLEU {best_bleu_seen*100:.2f}% was at epoch {best_epoch_seen} — that model is saved.")
    with open("en_vocab.pkl","wb") as f: pickle.dump(en_vocab,f)
    with open("hi_vocab.pkl","wb") as f: pickle.dump(hi_vocab,f)
    print(f"Saved -> {save_path}")
    metrics={"time":total,"loss":losses[-1],"bleu":final_bleu,"epochs_to_match":matched,
             "loss_curve":losses,"bleu_history":bleu_hist,"model":model,"device":DEVICE}
    _plot_best_model_curves(metrics, baseline_metrics)
    if baseline_metrics: _plot_final_comparison(baseline_metrics, metrics, best_config)
    _plot_translation_quality(model, DEVICE)
    _plot_attention_weights(model, DEVICE)
    print(f"\n{'='*50}\n  Time: {total:.1f}s | Loss: {losses[-1]:.4f} | BLEU: {final_bleu*100:.2f}\n  Matched baseline at epoch: {matched}\n{'='*50}")
    return metrics

def _plot_best_model_curves(metrics, baseline_metrics=None):
    # ── GRAPH 7: Best model training curves ──────────────────────────────────
    losses=metrics["loss_curve"]; bh=metrics["bleu_history"]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle("Best Tuned Model — Training Curves",fontsize=14,fontweight="bold")

    ax=axes[0]
    ax.plot(range(1,len(losses)+1),losses,color=PALETTE[0],linewidth=2,label="Best model")
    if baseline_metrics and "loss_curve" in baseline_metrics:
        bl=baseline_metrics["loss_curve"]
        ax.plot(range(1,len(bl)+1),bl,color=PALETTE[1],linewidth=1.5,linestyle="--",
                alpha=0.7,label="Baseline")
    ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.legend()

    ax=axes[1]
    if bh:
        ep,bl=zip(*bh)
        ax.plot(ep,[b*100 for b in bl],"o-",color=PALETTE[0],linewidth=2,markersize=6,label="Best")
    if baseline_metrics and "bleu_history" in baseline_metrics:
        bep,bbl=zip(*baseline_metrics["bleu_history"])
        ax.plot(bep,[b*100 for b in bbl],"s--",color=PALETTE[1],linewidth=1.5,
                alpha=0.7,markersize=5,label="Baseline")
    if metrics.get("epochs_to_match"):
        ax.axvline(metrics["epochs_to_match"],color="green",linestyle=":",
                   linewidth=2,label=f"Matched@{metrics['epochs_to_match']}")
    ax.set_title("BLEU Progress"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    ax=axes[2]
    if len(losses)>1:
        d=np.diff(losses)
        ax.bar(range(1,len(d)+1),d,
               color=[PALETTE[3] if x<0 else PALETTE[4] for x in d],alpha=0.8)
        ax.axhline(0,color="black",linewidth=0.8)
        ax.set_title("Epoch-to-Epoch Loss Delta"); ax.set_xlabel("Epoch"); ax.set_ylabel("Delta Loss")

    plt.tight_layout()
    savefig("07_best_model_curves.png")

def _plot_final_comparison(bm, tm, best_cfg):
    # ── GRAPH 8: Final baseline vs tuned comparison ───────────────────────────
    ne=best_cfg.get("num_epochs",40)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.suptitle("Final Comparison: Baseline vs Best Tuned Model",fontsize=14,fontweight="bold")

    for ax,title,vals in [
        (axes[0,0],"Final Loss",[bm["loss"],tm["loss"]]),
        (axes[0,1],"Final BLEU (%)",[bm["bleu"]*100,tm["bleu"]*100]),
        (axes[0,2],"Epochs to Match",[100, tm.get("epochs_to_match") or ne])]:
        bars=ax.bar(["Baseline","Best Model"],vals,color=[PALETTE[1],PALETTE[0]],
                    alpha=0.85,width=0.45)
        ax.set_title(title)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,
                    f"{bar.get_height():.4f}" if title=="Final Loss" else f"{bar.get_height():.1f}",
                    ha="center",fontsize=10,fontweight="bold")

    ax=axes[1,0]
    if "loss_curve" in bm and "loss_curve" in tm:
        ax.plot(range(1,len(bm["loss_curve"])+1),bm["loss_curve"],
                color=PALETTE[1],label="Baseline",linewidth=2,alpha=0.7)
        ax.plot(range(1,len(tm["loss_curve"])+1),tm["loss_curve"],
                color=PALETTE[0],label="Best",linewidth=2)
        ax.set_title("Loss Curves Overlay"); ax.set_xlabel("Epoch"); ax.legend()

    ax=axes[1,1]
    if "bleu_history" in bm and "bleu_history" in tm:
        bep,bbl=zip(*bm["bleu_history"]); tep,tbl=zip(*tm["bleu_history"])
        ax.plot(bep,[b*100 for b in bbl],"s--",color=PALETTE[1],label="Baseline",linewidth=2)
        ax.plot(tep,[b*100 for b in tbl],"o-",color=PALETTE[0],label="Best",linewidth=2)
        ax.set_title("BLEU Curves Overlay"); ax.set_xlabel("Epoch"); ax.legend()

    ax=axes[1,2]; ax.axis("off")
    bleu_g=tm["bleu"]-bm["bleu"]; sp=100/max(tm.get("epochs_to_match") or ne, 1)
    rows=[["Metric","Baseline","Best","Change"],
          ["Loss",f"{bm['loss']:.4f}",f"{tm['loss']:.4f}",
           f"{'↓' if tm['loss']<bm['loss'] else '↑'}{abs(tm['loss']-bm['loss']):.4f}"],
          ["BLEU",f"{bm['bleu']*100:.2f}%",f"{tm['bleu']*100:.2f}%",
           f"+{bleu_g*100:.2f}%" if bleu_g>=0 else f"{bleu_g*100:.2f}%"],
          ["Epochs","100",str(tm.get("epochs_to_match") or "<"+str(ne)),f"{sp:.1f}x faster"],
          ["Time(s)",f"{bm['time']:.0f}",f"{tm['time']:.0f}",
           f"-{max(0,bm['time']-tm['time']):.0f}s"]]
    tbl=ax.table(cellText=rows[1:],colLabels=rows[0],loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.8)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#4361ee"); cell.set_text_props(color="white",fontweight="bold")
        elif r%2: cell.set_facecolor("#eef2ff")
    ax.set_title("Summary",pad=12)
    plt.tight_layout()
    savefig("08_final_comparison.png")

def _plot_translation_quality(model, device):
    # ── GRAPH 9: Translation quality ─────────────────────────────────────────
    test=[("I love you.","मैं तुमसे प्यार करता हूँ।"),
          ("How are you?","आप कैसे हैं?"),
          ("You should sleep.","आपको सोना चाहिए।"),
          ("She is a good teacher.","वह एक अच्छी शिक्षिका हैं।"),
          ("The weather is nice today.","आज मौसम अच्छा है।"),
          ("What is your name?","आपका नाम क्या है?")]
    from nltk.translate.bleu_score import sentence_bleu
    smoothie=SmoothingFunction().method4
    res=[]
    for en_s,hi_r in test:
        pred=translate_sentence(model,en_s,en_vocab,hi_vocab,device)
        bl=sentence_bleu([hi_r.split()],pred.split(),smoothing_function=smoothie)
        res.append({"English":en_s,"Reference":hi_r,"Prediction":pred,"BLEU":bl})
    rdf=pd.DataFrame(res)
    fig,axes=plt.subplots(1,2,figsize=(18,6))
    fig.suptitle("Translation Quality Analysis",fontsize=14,fontweight="bold")
    ax=axes[0]
    colors_=[PALETTE[0] if b>=0.5 else PALETTE[1] for b in rdf["BLEU"]]
    bars=ax.barh(range(len(rdf)),rdf["BLEU"],color=colors_,alpha=0.85)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels([r[:35] for r in rdf["English"]],fontsize=9)
    ax.set_xlabel("Sentence BLEU"); ax.set_title("Per-Sentence BLEU")
    ax.axvline(0.5,color="red",linestyle="--",linewidth=1.5)
    for bar,val in zip(bars,rdf["BLEU"]):
        ax.text(val+0.01,bar.get_y()+bar.get_height()/2,f"{val:.3f}",va="center",fontsize=9)
    ax=axes[1]; ax.axis("off")
    td=[[r["English"][:28],r["Prediction"][:28],f"{r['BLEU']:.3f}"] for _,r in rdf.iterrows()]
    tbl=ax.table(cellText=td,colLabels=["English","Predicted Hindi","BLEU"],
                 loc="center",cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,2.2)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#4361ee"); cell.set_text_props(color="white",fontweight="bold")
        elif r%2: cell.set_facecolor("#eef2ff")
    ax.set_title("Sample Translations",pad=12)
    plt.tight_layout()
    savefig("09_translation_quality.png")
    print("\nTranslation Samples:")
    print(rdf[["English","Reference","Prediction","BLEU"]].to_string(index=False))

def _plot_attention_weights(model, device, sentence="How are you?", max_len=50):
    # ── GRAPH 10: Attention weight heatmap ───────────────────────────────────
    model.eval()
    src_tok=encode_sentence(sentence,en_vocab,max_len)
    src_tensor=torch.tensor(src_tok).unsqueeze(0).to(device)
    tgt_tokens=[hi_vocab["<sos>"]]
    attention_maps=[]
    with torch.no_grad():
        enc_out=model.enc(src_tensor,model.pad_mask(src_tensor,en_vocab["<pad>"]))
        for _ in range(20):
            tgt=torch.tensor(tgt_tokens).unsqueeze(0).to(device)
            sm=model.pad_mask(src_tensor,en_vocab["<pad>"])
            tm=model.pad_mask(tgt,hi_vocab["<pad>"])&model.sub_mask(tgt.size(1))
            # get attn from first decoder layer cross-attention
            dec_layer=model.dec.lyrs[0]
            x=model.dec.drop(model.dec.pos(model.dec.emb(tgt)))
            x=dec_layer.n1(x+dec_layer.drop(dec_layer.s_attn(x,x,x,tm)))
            # manually compute cross-attn scores for visualization
            B=x.size(0); nh=dec_layer.c_attn.num_heads; dk=dec_layer.c_attn.d_k
            Q=dec_layer.c_attn.Wq(x).view(B,-1,nh,dk).transpose(1,2)
            K=dec_layer.c_attn.Wk(enc_out).view(B,-1,nh,dk).transpose(1,2)
            scores=torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/(dk**0.5),dim=-1)
            attention_maps.append(scores[0,0,:,:].cpu().numpy())  # head 0
            out=model(src_tensor,tgt,en_vocab["<pad>"],hi_vocab["<pad>"])
            nxt=out[0,-1].argmax().item(); tgt_tokens.append(nxt)
            if nxt==hi_vocab["<eos>"]: break

    src_words=[en_vocab.itos.get(i,"<?>") for i in src_tok[:10]]
    tgt_words=[hi_vocab.itos.get(i,"<?>") for i in tgt_tokens[1:len(tgt_tokens)]]

    fig,axes=plt.subplots(1,2,figsize=(16,6))
    fig.suptitle(f"Cross-Attention Weights: '{sentence}'",fontsize=13,fontweight="bold")

    # Show attn map at step midpoint
    mid=len(attention_maps)//2
    if attention_maps:
        amap=attention_maps[mid]
        # amap shape: (tgt_len_so_far, src_len)
        tgt_so_far=amap.shape[0]; src_len=min(amap.shape[1],10)
        ax=axes[0]
        im=ax.imshow(amap[:,:src_len],cmap="Blues",aspect="auto")
        ax.set_xticks(range(src_len)); ax.set_xticklabels(src_words[:src_len],rotation=45,ha="right")
        ax.set_yticks(range(tgt_so_far))
        ax.set_yticklabels(tgt_words[:tgt_so_far] if tgt_words else range(tgt_so_far))
        ax.set_title(f"Cross-Attn Step {mid+1}"); ax.set_xlabel("Source tokens"); ax.set_ylabel("Target tokens")
        plt.colorbar(im,ax=ax)

    # Attention entropy over decoding steps
    ax=axes[1]
    entropies=[]
    for amap in attention_maps:
        # entropy of attn distribution (head 0, last target pos)
        p=amap[-1,:]+1e-9; ent=-(p*np.log(p)).sum(); entropies.append(ent)
    if entropies:
        ax.plot(range(1,len(entropies)+1),entropies,"o-",color=PALETTE[0],linewidth=2,markersize=6)
        ax.fill_between(range(1,len(entropies)+1),entropies,alpha=0.2,color=PALETTE[0])
        ax.set_title("Attention Entropy per Decoding Step")
        ax.set_xlabel("Decoding step"); ax.set_ylabel("Entropy")

    plt.tight_layout()
    savefig("10_attention_weights.png")

# =============================================================================
# 11.  MAIN
# =============================================================================
if __name__ == "__main__":

    # PART 1: Baseline
    print("\n"+"="*60+"\nPART 1: BASELINE\n"+"="*60)
    baseline_metrics = run_baseline()
    # To skip rerunning: comment above, fill manually:
    # baseline_metrics = {"time": X, "loss": X, "bleu": 0.50}

    # PART 2: Tune
    print("\n"+"="*60+"\nPART 2: RAY TUNE + OPTUNA\n"+"="*60)
    results, best_result = run_tuning(num_samples=10)

    # PART 3: Retrain best
    print("\n"+"="*60+"\nPART 3: RETRAIN BEST MODEL\n"+"="*60)
    best_metrics = retrain_best(best_result.config, baseline_metrics=baseline_metrics)

    # ── Time Comparison ───────────────────────────────────────────────────────
    time_saved  = baseline_metrics["time"] - best_metrics["time"]
    speedup     = baseline_metrics["time"] / best_metrics["time"]
    epoch_saved = 100 - (best_metrics["epochs_to_match"] or best_result.config.get("num_epochs", 40))

    print("\n" + "="*60)
    print("TIME COMPARISON")
    print("="*60)
    print(f"  Baseline Time      : {baseline_metrics['time']:.2f}s  ({baseline_metrics['time']/60:.1f} min)")
    print(f"  Best Model Time    : {best_metrics['time']:.2f}s  ({best_metrics['time']/60:.1f} min)")
    print(f"  Time Saved         : {time_saved:.2f}s  ({time_saved/60:.1f} min)")
    print(f"  Speedup            : {speedup:.2f}x faster")
    print(f"  Epochs Saved       : {epoch_saved} fewer epochs")
    print("="*60)

    print("\n" + "="*60)
    print("ALL DONE — Graphs saved in ./plots/")
    print("="*60)
    print(f"  Baseline  -> Loss: {baseline_metrics['loss']:.4f} | BLEU: {baseline_metrics['bleu']*100:.2f}% | 100 epochs")
    print(f"  Best      -> Loss: {best_metrics['loss']:.4f} | BLEU: {best_metrics['bleu']*100:.2f}% | matched @ epoch {best_metrics['epochs_to_match']}")
    print("="*60)

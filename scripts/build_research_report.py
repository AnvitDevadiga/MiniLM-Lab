"""Build a conventional black-and-white empirical research paper."""
import json
import statistics
from itertools import pairwise
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"docs/data/science_run.json"; OUT=ROOT/"output/pdf/MiniLM-Lab-Research-Paper.pdf"
W,H=A4; M=46; GAP=18; COL=(W-2*M-GAP)/2; BLACK=colors.black; GRAY=colors.HexColor("#555555"); LIGHT=colors.HexColor("#cccccc")

def t(c,x,y,s,n=9,f="Times-Roman",col=BLACK,a="left"):
    c.setFont(f,n); c.setFillColor(col)
    (c.drawCentredString if a=="center" else c.drawRightString if a=="right" else c.drawString)(x,y,s)
def wrap(c,s,x,y,w,n=9,lead=12,f="Times-Roman",col=BLACK):
    line=""
    for word in s.split():
        q=(line+" "+word).strip()
        if stringWidth(q,f,n)<=w: line=q
        else: t(c,x,y,line,n,f,col); y-=lead; line=word
    if line: t(c,x,y,line,n,f,col); y-=lead
    return y
def head(c,num,title,x,y,w):
    t(c,x,y,f"{num}  {title}",11,"Times-Bold"); c.setStrokeColor(BLACK); c.line(x,y-4,x+w,y-4); return y-18
def cap(c,num,s,x,y,w): return wrap(c,f"Figure {num}. {s}",x,y,w,8,10,"Times-Italic",GRAY)
def foot(c,p):
    c.setStrokeColor(LIGHT); c.line(M,31,W-M,31); t(c,M,20,"MiniLM Lab | Anvit Devadiga",7,col=GRAY); t(c,W-M,20,str(p),7,col=GRAY,a="right")
def tbl(c,x,y,widths,rows,rh=20):
    for ri,row in enumerate(rows):
        xx=x
        for ci,val in enumerate(row):
            if ri==0: c.setFillColor(colors.HexColor("#eeeeee")); c.rect(xx,y-rh,widths[ci],rh,fill=1,stroke=0)
            c.setStrokeColor(BLACK if ri==0 else LIGHT); c.setLineWidth(.35); c.rect(xx,y-rh,widths[ci],rh,fill=0,stroke=1)
            t(c,xx+4,y-rh+6,str(val),7.5,"Times-Bold" if ri==0 else "Times-Roman")
            xx+=widths[ci]
        y-=rh
    return y
def learning(c,x,y,w,h,d):
    l,b,r,top=x+30,y+22,x+w-8,y+h-10; c.setStrokeColor(BLACK); c.rect(l,b,r-l,top-b,fill=0)
    for v in (2.5,3,3.5):
        yy=b+(v-2.5)*(top-b); c.setStrokeColor(LIGHT); c.line(l,yy,r,yy); t(c,l-5,yy-2,f"{v:.1f}",7,col=GRAY,a="right")
    final=d["metrics"][-1]["step"]
    for key,dash in (("train_loss",[]),("validation_loss",[3,2])):
        pts=[]
        for row in d["metrics"]:
            xx=l+row["step"]/final*(r-l); yy=b+(row[key]-2.5)*(top-b); pts.append((xx,yy)); t(c,xx,b-12,str(row["step"]),7,col=GRAY,a="center")
        c.setStrokeColor(BLACK); c.setDash(dash); c.setLineWidth(1.1)
        for a,z in pairwise(pts): c.line(a[0],a[1],z[0],z[1])
        c.setDash([])
        for xx,yy in pts: c.circle(xx,yy,2.3,fill=1,stroke=0)
    t(c,l+4,top+3,"solid: train   dashed: held-out",7,"Times-Italic",GRAY); t(c,l,y+2,"optimizer updates",7,col=GRAY)
def latency(c,x,y,w,h,d):
    vals=[[statistics.mean(v)*1000,statistics.stdev(v)*1000] for v in (d["benchmark"]["uncached_seconds_samples"],d["benchmark"]["cached_seconds_samples"])]
    base=y+22; top=y+h-12; maxv=max(v[0] for v in vals)*1.2; slot=(w-50)/2
    c.setStrokeColor(BLACK); c.line(x+28,base,x+w,base)
    for i,(label,(mean,sd)) in enumerate(zip(("full recompute","KV cache"),vals)):
        xx=x+42+i*slot; bh=mean/maxv*(top-base); c.setFillColor(colors.HexColor("#777777" if i==0 else "#bbbbbb")); c.rect(xx,base,28,bh,fill=1,stroke=1)
        mid=xx+14; e=sd/maxv*(top-base); c.setStrokeColor(BLACK); c.line(mid,base+bh-e,mid,base+bh+e); c.line(mid-4,base+bh-e,mid+4,base+bh-e); c.line(mid-4,base+bh+e,mid+4,base+bh+e)
        t(c,mid,base-12,label,7,col=GRAY,a="center"); t(c,mid,base+bh+e+4,f"{mean:.1f}",7.5,"Times-Bold",a="center")
    t(c,x+28,top+3,"milliseconds; bars mean, whiskers +/- 1 SD",7,"Times-Italic",GRAY)
def trade(c,x,y,w,h,d):
    q=d["quantization"]; series=[("stored tensors (MB)",[q["original_stored_bytes"]/1e6,q["int8_stored_bytes"]/1e6]),("32 forward passes (ms)",[q["original_32_forward_seconds"]*1000,q["int8_32_forward_seconds"]*1000])]
    for j,(title,vals) in enumerate(series):
        x0=x+j*w/2; c.setStrokeColor(BLACK); c.rect(x0,y,w/2-8,h,fill=0); t(c,x0+8,y+h-14,title,8,"Times-Bold")
        mx=max(vals)*1.25
        for i,(lab,val) in enumerate(zip(("FP32","INT8"),vals)):
            yy=y+h-44-i*34; bw=val/mx*(w/2-65); c.setFillColor(colors.HexColor("#777777" if i==0 else "#bbbbbb")); c.rect(x0+32,yy,bw,14,fill=1,stroke=1); t(c,x0+8,yy+3,lab,7.5); t(c,x0+38+bw,yy+3,f"{val:.2f}",7.5)
def main():
    d=json.loads(DATA.read_text()); c=canvas.Canvas(str(OUT),pagesize=A4); c.setTitle("MiniLM Lab Research Paper"); c.setAuthor("Anvit Devadiga")
    # p1
    t(c,W/2,772,"MINILM LAB",9,"Times-Bold",GRAY,"center"); t(c,W/2,725,"A Measured Decoder-Only Transformer",23,"Times-Bold",a="center"); t(c,W/2,697,"Learning, inference, and compression trade-offs on a laptop-scale budget",11,"Times-Italic",GRAY,"center"); t(c,W/2,668,"Anvit Devadiga | Technical Research Report | 06 October 2026",9,col=GRAY,a="center"); c.line(72,628,W-72,628)
    y=598; t(c,72,y,"Abstract",12,"Times-Bold"); y-=18; y=wrap(c,"We study whether a compact decoder-only Transformer can learn held-out scientific prose while exposing measurable systems trade-offs. The implementation includes byte-level tokenization, causal self-attention, RMSNorm, SwiGLU, tied embeddings, deterministic evaluation, autoregressive generation, KV caching, and a portable weight-only INT8 reference. The primary run uses approximately 1 MB of public-domain text from Darwin and Einstein, one fixed seed, and CPU measurements. It reaches 17.58 held-out perplexity on 112,111 next-byte targets. KV caching reduces the recorded 96-token decode median from 119.4 ms to 33.4 ms. INT8 reduces stored model tensors by 53.6% but is slower in this reference implementation.",72,y,W-144,10,14); y-=8; wrap(c,"These results establish a reproducible laptop-scale baseline. They do not establish broad language ability, scientific understanding, statistical significance, or Apple M4/MPS performance.",72,y,W-144,10,14,"Times-Bold"); y-=42; t(c,72,y,"Headline measurements",12,"Times-Bold"); tbl(c,72,y-10,[130,110,210],[["Measure","Result","Protocol"],["Held-out perplexity","17.58","112,111 next-byte targets"],["KV-cache speedup","3.58x","CPU; seven timed repeats"],["Stored tensor reduction","53.6%","MLP weight-only INT8"],["CI verification","16 tests pass","Ruff + pytest on macOS CI"]],22); t(c,72,174,"Code, raw JSON, corpus hashes, figures, and reproduction commands: github.com/AnvitDevadiga/MiniLM-Lab",9,"Times-Italic",GRAY); foot(c,1); c.showPage()
    # p2
    y=790; y=head(c,"1","Introduction",M,y,W-2*M); y=wrap(c,"Small language-model repositories often demonstrate a forward pass or a training script, but leave the measurement protocol implicit. MiniLM Lab treats the model as a systems object: the corpus, split, optimizer steps, evaluator, cache path, and compression path are explicit. The goal is not to compete with large language models. The goal is to make the full byte-to-generation path inspectable and falsifiable on ordinary hardware.",M,y,W-2*M,9.5,13); y=head(c,"2","Research questions",M,y-4,W-2*M); y=wrap(c,"(Q1) Does the compact model reduce held-out next-byte loss during training on a reproducible science corpus?",M,y,W-2*M,9.5,13); y=wrap(c,"(Q2) Does reusing past key/value states reduce autoregressive decode latency under a fixed CPU protocol?",M,y,W-2*M,9.5,13); y=wrap(c,"(Q3) What is the quality, storage, and latency trade-off of a portable weight-only INT8 reference?",M,y,W-2*M,9.5,13); y=head(c,"3","Data and protocol",M,y-4,W-2*M); y=wrap(c,"The corpus consists of Project Gutenberg editions of Darwin's On the Origin of Species and Einstein's Relativity: The Special and General Theory. Boilerplate is removed. Approximately the first 90% of each document is used for training; the remainder is held out at a paragraph boundary. Source URLs, raw SHA-256 hashes, prepared split hashes, and byte counts are stored in data/science/manifest.json.",M,y,W-2*M,9.5,13); tbl(c,M,y-4,[150,160,170],[["Protocol item","Published value","Interpretation"],["Training bytes","1,008,977","Approximately 1 MB"],["Held-out bytes","112,112","Same books; later passages"],["Seed","42","One reported run"],["Optimizer updates","1,000","Four evaluation checkpoints"],["Tokenizer","UTF-8 bytes","256-value vocabulary"]],20); foot(c,2); c.showPage()
    # p3
    y=790; y=head(c,"4","Model architecture",M,y,W-2*M); y=wrap(c,"The model is a decoder-only Transformer with four blocks, embedding width 192, six attention heads, context length 128, learned positions, RMSNorm, SwiGLU feed-forward layers, residual connections, and tied input/output embeddings. Training uses AdamW with learning rate 3e-4, microbatch size 8, four-way gradient accumulation, and seed 42.",M,y,W-2*M,9.5,13); x1,x2=M,M+COL+GAP; a=head(c,"5","Evaluation",x1,y-6,COL); a=wrap(c,"The evaluator scores every next token once in non-overlapping context windows. The first token of each window supplies left context. Perplexity is exp(mean negative log-likelihood). Bits per byte is computed from the byte-normalized loss and the exact scored-byte denominator.",x1,a,COL,8.8,12); a=head(c,"6","Implementation controls",x1,a-4,COL); wrap(c,"Checkpoint selection uses deterministic full-split validation. Resume checkpoints include model state, optimizer state, CPU RNG state, configuration, corpus hash, and metrics. Cache outputs are checked against full-forward outputs for learned positions and RoPE.",x1,a,COL,8.8,12); b=head(c,"7","Parameterization",x2,y-6,COL); tbl(c,x2,b,[90,105],[["Component","Configuration"],["Attention","Causal MHA"],["Norm","RMSNorm"],["Activation","SwiGLU"],["Position","Learned / RoPE"],["Output","Tied embeddings"]],21); y=head(c,"8","Architecture flow",M,420,W-2*M); tbl(c,M,y-4,[120,120,220],[["Stage","Representation","Purpose"],["Input","UTF-8 bytes","Corpus interface"],["Transformer","Causal hidden states","Next-byte prediction"],["Output","Vocabulary logits","Autoregressive sampling"]],22); foot(c,3); c.showPage()
    # p4
    y=790; y=head(c,"9","Results: learning behavior",M,y,W-2*M); y=wrap(c,"The held-out curve decreases at the final checkpoint. Because only four checkpoints and one seed are reported, the figure is descriptive rather than a statistical comparison.",M,y,W-2*M,9.5,13); learning(c,M,y-175,W-2*M,160,d); y-=190; y=cap(c,1,"Training and held-out loss across four optimizer checkpoints. The held-out series covers 112,111 targets.",M,y,W-2*M); tbl(c,M,y-4,[130,105,205],[["Metric","Value","Definition"],["Validation loss",f"{d['evaluation']['validation_loss']:.4f} nats/token","Mean held-out negative log-likelihood"],["Perplexity",f"{d['evaluation']['perplexity']:.2f}","exp(validation loss)"],["Bits per byte",f"{d['evaluation']['bits_per_byte']:.3f}","Byte-normalized loss"],["Targets",f"{d['evaluation']['validation_tokens']:,}","Unique next-byte targets"]],22); y-=135; y=head(c,"10","Per-source evaluation",M,y,W-2*M); tbl(c,M,y-4,[150,100,105,85],[["Source","Targets","Perplexity","Bits/byte"],["Darwin","93,405","17.76","4.150"],["Einstein","18,703","16.68","4.060"]],22); wrap(c,"These are within-book continuation tests, not unseen-author generalization tests.",M,y-80,W-2*M,8,10,"Times-Italic",GRAY); foot(c,4); c.showPage()
    # p5
    y=790; y=head(c,"11","Results: inference and compression",M,y,W-2*M); y=wrap(c,"For the cache benchmark, prompt, sampling settings, generated length, warm-up count, and repeat count are fixed. Bars show the mean of seven timed runs; whiskers show plus or minus one standard deviation.",M,y,W-2*M,9.5,13); latency(c,M,y-155,W-2*M,140,d); y-=170; y=cap(c,2,"Decode latency for 96 generated tokens after a 21-token prompt. The cached path is 3.58x faster by median on the recorded CPU run.",M,y,W-2*M); trade(c,M,y-120,W-2*M,100,d); y-=138; y=cap(c,3,"Portable weight-only INT8 reference. Storage decreases while the unoptimized dequantization path increases latency.",M,y,W-2*M); tbl(c,M,y-4,[125,95,95,120],[["Measure","FP32","INT8","Difference"],["Stored tensors","9.84 MB","4.56 MB","-53.6%"],["Validation loss","2.866917","2.867067","+0.000150"],["32 forward passes","55.6 ms","65.9 ms","INT8 slower"]],21); foot(c,5); c.showPage()
    # p6
    y=790; y=head(c,"12","Discussion and limitations",M,y,W-2*M); y=wrap(c,"The evidence supports a compact, inspectable language-model systems baseline. It demonstrates the full path from public text to model training, held-out scoring, autoregressive generation, cache reuse, and a compression trade-off. It does not support claims of broad language ability, scientific understanding, production serving, or superiority over larger open-source training systems.",M,y,W-2*M,9.5,13); y=head(c,"13","Threats to validity",M,y-6,W-2*M); tbl(c,M,y-4,[120,280],[["Threat","Consequence"],["One seed","No uncertainty estimate for model quality or method superiority."],["One model shape","No scaling-law or architecture comparison."],["Within-book holdout","Style and repeated phrases may cross the split."],["Small corpus","Perplexity does not imply general knowledge."],["CPU environment","Does not establish Apple M4/MPS performance."],["Reference INT8 path","Storage result is not accelerated deployment evidence."]],24); y-=180; y=head(c,"14","Required next experiments",M,y,W-2*M); y=wrap(c,"A stronger flagship study should add three or more matched seeds, a third-author test book excluded from training, an M4 MPS run recording throughput and memory, matched learned-position versus RoPE and context-length comparisons, and an accelerated quantization backend where hardware supports it.",M,y,W-2*M,9.5,13); y=head(c,"15","Conclusion",M,y-6,W-2*M); wrap(c,"MiniLM Lab is valuable as an evidence-first engineering study. Its differentiator is the explicit connection between implementation, protocol, raw measurements, tests, and limitations. The M4 benchmark and multi-seed/unseen-author experiments are the remaining work required for a stronger research claim.",M,y,W-2*M,9.5,13); foot(c,6); c.showPage()
    # p7
    y=790; y=head(c,"16","Reproducibility and references",M,y,W-2*M); y=wrap(c,"The repository includes the implementation, 16 correctness tests, pinned corpus manifest, raw experiment record, protocol, generated figures, and this report. The published run can be reproduced with the commands in README.md.",M,y,W-2*M,9.5,13); y=head(c,"17","Reference paper used as a pattern",M,y-6,W-2*M); y=wrap(c,"Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805v2. The reference informed the separation of objective, architecture, protocol, results, and limitations. MiniLM Lab does not claim BERT-like scale, capability, or benchmark performance.",M,y,W-2*M,9.5,13); y=head(c,"18","Data sources",M,y-6,W-2*M); wrap(c,"Project Gutenberg eBook #1228, Darwin's On the Origin of Species, and eBook #30155, Einstein's Relativity: The Special and General Theory. URLs and SHA-256 hashes are recorded in data/science/manifest.json.",M,y,W-2*M,9.5,13); foot(c,7); c.save(); print(OUT)
if __name__=="__main__": main()

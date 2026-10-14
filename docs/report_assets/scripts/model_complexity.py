"""Analytical parameter / MAC count of CAFN from the layer definitions in src/models/*.py (no torch needed)."""
import json
d_model=128; heads=4; ffn=256; ncls=3
def conv3(cin,cout,k=3): return cin*cout*k**3
def conv1(cin,cout,k): return cin*cout*k
def bn(c): return 2*c
comp={}; macs={}
# spatial: input 64^3
vox=lambda n: n**3
p=conv3(1,16)+bn(16); m=conv3(1,16)*vox(64); comp['Stem conv']=p; macs['Stem conv']=m
def res(cin,cout,out_n):
    p=conv3(cin,cout)+bn(cout)+conv3(cout,cout)+bn(cout)
    m=conv3(cin,cout)*vox(out_n)+conv3(cout,cout)*vox(out_n)
    if cin!=cout:
        p+=cin*cout+bn(cout); m+=cin*cout*vox(out_n)
    return p,m
for name,(a,b,n) in {'Residual block 1 (16→32, stride 2)':(16,32,32),'Residual block 2 (32→64, stride 2)':(32,64,16),'Residual block 3 (64→128, stride 2)':(64,128,8)}.items():
    comp[name],macs[name]=res(a,b,n)
spatial=sum(comp.values())
# temporal: L=500 ; lengths 500,250,125,63
t=[(1,16,7,500),(16,32,5,250),(32,64,5,125),(64,128,3,63)]
tp=sum(conv1(a,b,k)+bn(b) for a,b,k,_ in t); tm=sum(conv1(a,b,k)*L for a,b,k,L in t)
comp['Temporal encoder (4 × Conv1d+BN)']=tp; macs['Temporal encoder (4 × Conv1d+BN)']=tm
lin=lambda i,o: i*o+o
attn=4*lin(d_model,d_model)
ln=2*d_model
ff=lin(d_model,ffn)+lin(ffn,d_model)
comp['Cross-attention (W_Q, W_K, W_V, W_O)']=attn
comp['Feed-forward block (128→256→128)']=ff
comp['LayerNorm ×2']=2*ln
heads_p=3*lin(d_model,ncls)
comp['Classification heads ×3']=heads_p
# MACs for fusion
S,T=64,16
macs['Cross-attention (W_Q, W_K, W_V, W_O)']=S*d_model*d_model*2 + T*d_model*d_model*2 + heads*S*T*(d_model//heads)*2
macs['Feed-forward block (128→256→128)']=S*(d_model*ffn*2)
macs['Classification heads ×3']=3*d_model*ncls
total=sum(comp.values())
out={'params':comp,'macs':macs,'total_params':total,'total_macs':sum(macs.values()),
     'spatial_encoder_params':spatial,'temporal_encoder_params':tp,'fusion_params':attn+ff+2*ln,'head_params':heads_p,
     'attn_scores_per_sample': heads*S*T}
json.dump(out,open('figs/complexity.json','w'),indent=1)
for k,v in comp.items(): print(f'{k:45s} {v:>10,}  MACs {macs.get(k,0):>14,}')
print('TOTAL params',total,'MACs',sum(macs.values()))
print('spatial',spatial,'temporal',tp,'fusion',attn+ff+2*ln,'heads',heads_p)

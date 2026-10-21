"""Generate publication figures exclusively from archived experiment records."""
from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from experiment import ROOT,SEEDS
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
COLORS={'full':'#426682','rasc':'#ae563d','prevalence':'#8a8a75'}

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

def records():return [json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
def select(runs,dataset,algo,mask,alpha=.5,clients=5,rounds=40,seeds=None):
    rr=[r for r in runs if all(r['config'][k]==v for k,v in dict(dataset=dataset,algorithm=algo,mask=mask,alpha=alpha,clients=clients,rounds=rounds).items())]
    return sorted([r for r in rr if seeds is None or r['config']['seed'] in seeds],key=lambda r:r['config']['seed'])

def main():
    rr=records()
    fig,ax=plt.subplots(figsize=(7.0,4.9));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,title,body):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor='#f3f5f6',edgecolor='#4d5961',linewidth=1.1))
        ax.text(x+w/2,y+h-.26,title,ha='center',va='top',fontsize=10.1,fontweight='bold')
        ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=9.4,linespacing=1.25)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#4d5961',linewidth=1.1))
    ax.text(0,6.8,'A  One shared feature mask',fontweight='bold',fontsize=11)
    box(.1,4.9,2.8,1.45,'Clients','Training-only bootstrap\nClass effects and support')
    box(3.6,4.9,2.8,1.45,'Consensus server','Reserve class slots\nRank remaining features')
    box(7.1,4.9,2.75,1.45,'Common mask','Same ordered support\nFixed feature budget')
    arrow((2.95,5.6),(3.55,5.6));arrow((6.45,5.6),(7.05,5.6))
    ax.annotate('Broadcast to all clients',xy=(1.5,4.8),xytext=(8.3,4.28),ha='center',fontsize=9.3,
      arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=180,angleB=270,rad=8',color='#4d5961'))
    ax.text(0,3.63,'B  One fixed classifier within each dataset',fontweight='bold',fontsize=11)
    box(.1,1.1,2.,1.7,'Masked input','z = x ⊙ m\nBoT 37 dimensions\nCIC 46 dimensions')
    box(2.9,2.1,3.,1.0,'Dense branch','tanh 24 → tanh 12')
    box(2.9,.5,3.,1.0,'Convolution branch','8 filters → max pool → 12')
    box(6.7,1.25,3.,1.4,'Gated fusion','12 shared units\nSoftmax over C classes')
    arrow((2.15,2.1),(2.85,2.6));arrow((2.15,1.75),(2.85,1.0));arrow((5.95,2.6),(6.65,2.2));arrow((5.95,1.0),(6.65,1.65))
    ax.text(5,.05,'Aggregation changes the optimizer; input and model shapes stay common.',ha='center',fontsize=9.2)
    fig.tight_layout();save(fig,'figure_01_method')

    algs=['fedavg','fedprox','scaffold','fedadam','fedyogi'];labels=['FedAvg','FedProx','SCAFFOLD','FedAdam','FedYogi']
    fig,ax=plt.subplots(figsize=(6.9,4.2));x=np.arange(5);width=.32
    for offset,mask,label in [(-width/2,'full','All 37 features'),(width/2,'rasc','RASC 22 features')]:
        vals=[[r['test']['macro_f1'] for r in select(rr,'bot_primary',a,mask)] for a in algs]
        ax.bar(x+offset,[np.mean(v) for v in vals],width,yerr=[np.std(v,ddof=1) for v in vals],capsize=3,label=label,color=COLORS[mask])
    prev=[r['test']['macro_f1'] for r in select(rr,'bot_primary','fedavg','prevalence')]
    ax.axhline(np.mean(prev),color=COLORS['prevalence'],linestyle=':',label='Prevalence + FedAvg')
    ax.set(xticks=x,xticklabels=labels,ylabel='Test macro F1',ylim=(0,1.02));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=9,loc='upper left');fig.tight_layout();save(fig,'figure_02_optimizers')

    methods=[('fedavg','full','FedAvg all','#426682'),('fedavg','prevalence','Prevalence','#8a8a75'),('fedavg','rasc','RASC + FedAvg','#ae563d'),('scaffold','full','SCAFFOLD all','#6b7c51'),('fedadam','full','FedAdam all','#6f5d8c')]
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.8))
    for ax,settings in zip(axes,[[(a,5) for a in [.1,.3,.5,1.]],[(.5,k) for k in [5,10,20]]]):
        for algo,mask,label,color in methods:
            vals=[[r['test']['macro_f1'] for r in select(rr,'bot_primary',algo,mask,a,k,seeds=SEEDS[:3])] for a,k in settings]
            if any(len(v)!=3 for v in vals):raise RuntimeError('incomplete sensitivity cell')
            xx=[a if len(settings)==4 else k for a,k in settings]
            ax.plot(xx,[np.mean(v) for v in vals],marker='o',markersize=3.5,label=label,color=color,linewidth=1.5)
        ax.set(ylabel='Test macro F1',ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].set(xlabel='Dirichlet α at five clients',xticks=[.1,.3,.5,1.]);axes[1].set(xlabel='Clients at α = 0.5',xticks=[5,10,20])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8.3)
    fig.tight_layout(rect=(0,.16,1,1));save(fig,'figure_03_sensitivity')

    fig,ax=plt.subplots(figsize=(6.9,4.2));algs=algs[:4];labels=labels[:4] if len(labels)==5 else ['FedAvg','FedProx','SCAFFOLD','FedAdam'];x=np.arange(4)
    for offset,mask,label in [(-width/2,'full','All 46 features'),(width/2,'rasc','RASC 28 features')]:
        vals=[[r['test']['macro_f1'] for r in select(rr,'cic_primary',a,mask)] for a in algs]
        if any(len(v)!=5 for v in vals):raise RuntimeError('incomplete external cell')
        ax.bar(x+offset,[np.mean(v) for v in vals],width,yerr=[np.std(v,ddof=1) for v in vals],capsize=3,label=label,color=COLORS[mask])
    prev=[r['test']['macro_f1'] for r in select(rr,'cic_primary','fedavg','prevalence')]
    ax.axhline(np.mean(prev),color=COLORS['prevalence'],linestyle=':',label='Prevalence + FedAvg')
    ax.set(xticks=x,xticklabels=['FedAvg','FedProx','SCAFFOLD','FedAdam'],ylabel='Test macro F1',ylim=(0,1.02));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,'figure_04_external')

    fig,ax=plt.subplots(figsize=(6.9,4.1))
    for algo,mask,color,style,label in [('fedavg','full','#426682','-','FedAvg all'),('fedavg','rasc','#426682','--','RASC + FedAvg'),('fedadam','full','#ae563d','-','FedAdam all'),('fedadam','rasc','#ae563d','--','RASC + FedAdam')]:
        runs=select(rr,'bot_primary',algo,mask,rounds=200,seeds=SEEDS)
        if len(runs)!=10:raise RuntimeError('incomplete extended-budget cell')
        values=np.asarray([[l['validation_macro_f1'] for l in r['round_log']] for r in runs]);mean=values.mean(0)
        ax.plot(np.arange(1,201),mean,color=color,linestyle=style,label=label)
    ax.axvline(40,color='#999999',linewidth=1,linestyle=':');ax.text(42,.04,'40 rounds',color='#666666',fontsize=8.5)
    ax.set(xlabel='Communication round',ylabel='Mean validation macro F1',ylim=(0,1.02),xlim=(1,200));ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.tight_layout();save(fig,'figure_05_training_budget')

if __name__=='__main__':main()

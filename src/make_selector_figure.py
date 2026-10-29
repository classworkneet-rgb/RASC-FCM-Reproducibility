"""Publication plot of all ten fixed-budget selector contrasts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import ROOT

def main():
    obj=json.loads((ROOT/'results/selector_extension_inference.json').read_text())
    names=['no_stability','uniform_class_weights','no_reservation','mutual_information','random']
    labels=['Without stability','Uniform class weights','Without reserved slots','Mutual information','Random mask']
    by={(r['algorithm'],r['mask']):r for r in obj['contrasts']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.1),sharey=True)
    xmin=min(r['ci95_low'] for r in obj['contrasts'])-.025
    xmax=max(r['ci95_high'] for r in obj['contrasts'])+.07
    for ax,algo,title in zip(axes,['fedavg','fedadam'],['FedAvg','FedAdam']):
        for i,name in enumerate(names):
            r=by[algo,name];mean=r['mean_difference']
            ax.errorbar(mean,i,xerr=[[mean-r['ci95_low']],[r['ci95_high']-mean]],
                fmt='o',color='#214C72',ecolor='#52728C',capsize=3,markersize=5,linewidth=1.4)
            ax.text(xmax,i,f"{r['holm_p_ten_contrasts']:.3f}",ha='right',va='center',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
        ax.axvline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set_xlim(xmin,xmax);ax.set_title(title,fontweight='bold',pad=17)
        ax.set_xlabel('RASC minus alternative macro F1')
        ax.grid(axis='x',color='#E0E0E0',linewidth=.6)
        ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.text(1,1.025,'Holm p',transform=ax.transAxes,ha='right',fontsize=9)
    axes[0].set_yticks(np.arange(5),labels);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.255,right=.985,bottom=.19,top=.82,wspace=.18)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'figure_06_selector_controls.pdf',bbox_inches='tight',metadata={'Title':'Fixed-budget selector contrasts','Author':'RASC-FCM authors','Subject':'Paired-seed macro-F1 differences with pointwise bootstrap intervals'})
    fig.savefig(out/'figure_06_selector_controls.png',dpi=250,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()

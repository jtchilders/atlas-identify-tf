#!/usr/bin/env python
import argparse,logging,json,glob,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
logger = logging.getLogger(__name__)


def excepthook(*args):
  logging.getLogger().error('Uncaught exception:', exc_info=args)


sys.excepthook = excepthook


def main():
   logging.basicConfig(level=logging.INFO)
   ''' simple starter program that can be copied for use when starting a new script. '''
   parser = argparse.ArgumentParser(description='plot fullimg mean/sigma files on same plot from bcs')
   parser.add_argument('--bq', '-b',
                       help='bquark numpy file.',required=True)
   parser.add_argument('--cq', '-c',
                       help='cquark numpy file.',required=True)
   parser.add_argument('--sq', '-s',
                       help='squark numpy file.',required=True)

   args = parser.parse_args()


   # profiling data
   layer_histo_bins = 110
   num_layers = 15
   # num_eta = 1153
   # num_phi = 50
   # labels = ['s','c','b']
   color = {'bc':'green','bs':'red','cs':'blue'}
   marker = {'bc':'o','bs':'^','cs':'x'}

   bdata = np.load(args.bq)
   bdata = {'mean':bdata['mean'],'sigma':bdata['sigma']}

   cdata = np.load(args.cq)
   cdata = {'mean':cdata['mean'],'sigma':cdata['sigma']}

   sdata = np.load(args.sq)
   sdata = {'mean':sdata['mean'],'sigma':sdata['sigma']}

   # data = {'b':bdata,'c':cdata,'s':sdata}

   diffdata = {
      'bc': bdata['mean'] - cdata['mean'],
      'bs': bdata['mean'] - sdata['mean'],
      'cs': cdata['mean'] - sdata['mean'],
   }

   layer_det = {0:'pix',1:'pix',2:'pix',3:'pix',
                4:'sct',5:'sct',6:'sct',7:'sct',
                8:'lar',9:'lar',10:'lar',11:'lar',
                12:'til',13:'til',14:'til',15:'til',
   }

   for diff in diffdata:

      d = diffdata[diff]
      vmin = np.min(d)
      vmax = np.max(d)
      norms = {
         'pix': colors.Normalize(vmin=-0.001,vmax=0.001),
         'sct': colors.Normalize(vmin=vmin,vmax=vmax),
         'lar': colors.Normalize(vmin=-0.001,vmax=0.001),
         'til': colors.Normalize(vmin=vmin,vmax=vmax),
      }
      plt.figure(1,figsize=(12,16))
      plots = {'pix':[],'sct':[],'lar':[],'til':[]}
      axes = {'pix':[],'sct':[],'lar':[],'til':[]}
      for layer in range(num_layers):
         
         plt.subplot(num_layers,1,num_layers-layer)
         plt.gca().text(-0.1,0.5,'%s' % layer,fontsize=14,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
         p = plt.imshow(d[layer],aspect='auto')

         axes[layer_det[layer]].append(plt.gca())
         p.set_norm(norms[layer_det[layer]])
         plots[layer_det[layer]].append(p)
         p.callbacksSM.connect('changed',update)
         plt.gca().label_outer()
         # plt.ylim(1,1e5)


      for det in plots:
         plt.gcf().colorbar(plots[det][0], ax=axes[det], orientation='vertical', fraction=.1,aspect=30)
      
      axes['til'][-1].text(-0.15,1.0,'layer',fontsize=15,transform=axes['til'][-1].transAxes)

      plt.gcf().savefig('quarkflavor_%s_fullimg.png' % diff)
      plt.close('all')




def update(changed_image,images):
   for im in images:
      if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
         im.set_cmap(changed_image.get_cmap())
         im.set_clim(changed_image.get_clim())


if __name__ == '__main__':
   main()

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

   for diff in diffdata:

      d = diffdata[diff]
      vmin = np.min(d)
      vmax = np.max(d)
      #norm = colors.Normalize(vmin=vmin, vmax=vmax)
      norm = colors.Normalize(vmin=-0.001,vmax=0.001)
      plt.figure(1,figsize=(12,16))
      plots = []
      ax = []
      for layer in range(num_layers):
      
         plt.subplot(num_layers,1,num_layers-layer)
         plt.gca().text(-0.1,0.5,'%s' % layer,fontsize=14,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
         p = plt.imshow(d[layer],aspect='auto')
         p.set_norm(norm)
         p.callbacksSM.connect('changed',update)
         plots.append(p)
         ax.append(plt.gca())
         plt.gca().label_outer()
         # plt.ylim(1,1e5)


      plt.gcf().colorbar(plots[0], ax=ax, orientation='vertical', fraction=.1,aspect=30)
      plt.gcf().text(0.01,0.95,'layer',fontsize=15)

      plt.gcf().savefig('quarkflavor_%s_fullimg.png' % diff)
      plt.close('all')




def update(changed_image,images):
   for im in images:
      if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
         im.set_cmap(changed_image.get_cmap())
         im.set_clim(changed_image.get_clim())


if __name__ == '__main__':
   main()

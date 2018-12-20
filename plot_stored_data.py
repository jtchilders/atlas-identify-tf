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
   parser = argparse.ArgumentParser(description='plot layerhist mean/sigma files on same plot from bcs')
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
   color = {'s':'green','b':'red','c':'blue'}
   marker = {'s':'o','b':'^','c':'x'}

   bdata = np.load(args.bq)
   bdata = {'mean':bdata['mean'],'sigma':bdata['sigma']}

   cdata = np.load(args.cq)
   cdata = {'mean':cdata['mean'],'sigma':cdata['sigma']}

   sdata = np.load(args.sq)
   sdata = {'mean':sdata['mean'],'sigma':sdata['sigma']}

   data = {'b':bdata,'c':cdata,'s':sdata}

   plt.figure(1,figsize=(16,12))

   for layer in range(num_layers):
      
      plt.subplot(num_layers,1,num_layers-layer)
      
      for qflav in data:

         mean = data[qflav]['mean'][layer]
         sigma = data[qflav]['sigma'][layer]

         layer_data_y = mean
         layer_data_x = [-0.095 + i*0.01 for i in range(layer_histo_bins)]
         layer_error_x = [0.005]*layer_histo_bins
         layer_error_y = sigma
         
         plt.errorbar(layer_data_x,layer_data_y,xerr=layer_error_x,yerr=layer_error_y,
                      markerfacecolor=color[qflav],marker=marker[qflav],alpha=0.6)
      
      plt.ylim(1,1e5)
      plt.yscale('log')

   plt.gcf().savefig('quarkflavor_layerhist.png')
   plt.close('all')



def update(changed_image,images):
   for im in images:
      if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
         im.set_cmap(changed_image.get_cmap())
         im.set_clim(changed_image.get_clim())


if __name__ == '__main__':
   main()

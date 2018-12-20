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


class NPMean:
   def __init__(self,shape):
      self.sum = np.zeros(shape)
      self.sum2 = np.zeros(shape)
      self.n = 0.

   def __iadd__(self,other):
      assert self.sum.shape == other.shape,'shape missmatch: %s != %s' % (self.sum.shape,other.shape)
      self.sum = self.sum + other
      self.sum2 = self.sum2 + other*other
      self.n += 1.
      return self

   def mean(self):
      assert self.n > 2.,'mean must have more than 2 entries'
      return self.sum / self.n

   def sigma(self):
      assert self.n > 2.,'mean must have more than 2 entries'
      mean = self.mean()
      return np.sqrt((1./self.n) * self.sum2 - mean * mean)



def main():
   logging.basicConfig(level=logging.INFO)
   ''' simple starter program that can be copied for use when starting a new script. '''
   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')


   args = parser.parse_args()

   # load configuration
   config_file = json.load(open(args.config_file))

   # get file list
   filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   np.random.shuffle(filelist)
   logger.info('found %s input files',len(filelist))

   # profiling data
   layer_histo_bins = 110
   num_layers = 15
   num_eta = 1153
   num_phi = 50
   labels = ['u','s','c','b','g']
   profile_data = {'b':{'layerhist':NPMean([num_layers,110]),
                        'etaphi':NPMean([num_phi,num_eta]),
                        'fullimg':NPMean([num_layers,num_phi,num_eta])},
                   'c':{'layerhist':NPMean([num_layers,110]),
                        'etaphi':NPMean([num_phi,num_eta]),
                        'fullimg':NPMean([num_layers,num_phi,num_eta])},
                   's':{'layerhist':NPMean([num_layers,110]),
                        'etaphi':NPMean([num_phi,num_eta]),
                        'fullimg':NPMean([num_layers,num_phi,num_eta])},
                  }

   image_counter = 0
   for filename in filelist:
      logger.info('[%s] plotting file: %s',image_counter,filename)
      data = np.load(filename)
      images = data['raw']
      batch_size,nchannels,bins_phi,bins_eta = images.shape
      truth  = data['truth']
      

      for i in range(images.shape[0]):

         plt.figure(1,figsize=(16,12))

         logger.debug('     index: %s',i)
         image = images[i]
         y = truth[i][0]
         label = labels[np.argmax(y[5:])]
         logger.debug('%s %s',image.shape,y.shape)
         logger.debug('%s %s',y,label)

         profile_data[label]['fullimg'] += image


         plt.subplot(221)
         histo_data = image.flatten()
         n, bins, patches = plt.hist(histo_data,100)
         plt.yscale('log')

         plt.subplot(222)
         eta_phi = np.sum(image,axis=0)
         p = plt.imshow(eta_phi,aspect='auto')
         plt.colorbar(p,ax=plt.gca(),orientation='vertical')
         plt.gca().invert_yaxis()
         profile_data[label]['etaphi'] += eta_phi

         plt.subplot(223)
         depth_eta = np.sum(image,axis=1)
         p = plt.imshow(depth_eta,aspect='auto')
         plt.colorbar(p,ax=plt.gca(),orientation='vertical')
         plt.gca().invert_yaxis()

         plt.subplot(224)
         depth_phi = np.sum(image,axis=2)
         p = plt.imshow(depth_phi,aspect='auto')
         plt.colorbar(p,ax=plt.gca(),orientation='vertical')
         plt.gca().invert_yaxis()


         plt.gcf().savefig('%010d_overview.png' % image_counter)
         plt.close('all')

         
         fig,ax = plt.subplots(num_layers)
         fig.set_figwidth(12)
         fig.set_figheight(16)

         vmin = np.min(image)
         vmax = np.max(image)
         logger.debug('min = %s; max = %s',vmin,vmax)
         norm = colors.LogNorm(vmin=1e-6, vmax=vmax)
         plots = []
         for layer in range(image.shape[0]):
            logger.debug('plot layer: %s',layer)
            axid = image.shape[0]-(layer+1)
            p = ax[axid].imshow(image[layer],aspect='auto')
            ax[axid].text(-0.1,0.5,'%s' % layer,fontsize=14,horizontalalignment='center',verticalalignment='center',transform=ax[axid].transAxes)
            p.set_norm(norm)
            p.callbacksSM.connect('changed',update)
            plots.append(p)
            ax[layer].label_outer()
            # figB.colorbar(p,ax=axB[layer],ticks=[0,1])

         fig.colorbar(plots[0], ax=ax, orientation='vertical', fraction=.1,aspect=30)
         fig.text(0.01,0.95,'layer',fontsize=15)

         #plt.show()

         fig.savefig('%010d_layer.png' % image_counter)
         plt.close('all')

         plt.figure(1,figsize=(16,12))

         layer_histos = []
         for layer in range(image.shape[0]):
            plt.subplot(num_layers,1,image.shape[0]-layer)
            layer_histo = image[layer].flatten()
            n, bins, patches = plt.hist(layer_histo,bins=layer_histo_bins,range=(-0.1,1.0),alpha=0.5,label='%s' % layer)
            layer_histos.append(n)
            plt.ylim(1,1e5)
            plt.yscale('log')
            logger.debug('layer %s max %s',layer,np.max(layer_histo))
         profile_data[label]['layerhist'] += np.array(layer_histos)

         plt.gcf().savefig('%010d_layerhist.png' % image_counter)
         plt.close('all')


         image_counter += 1

      if image_counter > 2500:
         break

   for quarkflavor in profile_data:

      image = profile_data[quarkflavor]['fullimg']

      mean = image.mean()
      sigma = image.sigma()

      np.savez_compressed('quarkflavor_%s_fullimg.npz' % quarkflavor,mean=mean,sigma=sigma)

      fig,ax = plt.subplots(num_layers)
      fig.set_figwidth(12)
      fig.set_figheight(16)

      vmin = np.min(mean)
      vmax = np.max(mean)
      logger.debug('min = %s; max = %s',vmin,vmax)
      norm = colors.LogNorm(vmin=1e-6, vmax=vmax)
      plots = []
      for layer in range(mean.shape[0]):
         logger.debug('plot layer: %s',layer)
         axid = mean.shape[0]-(layer+1)
         p = ax[axid].imshow(mean[layer],aspect='auto')
         ax[axid].text(-0.1,0.5,'%s' % layer,fontsize=14,horizontalalignment='center',verticalalignment='center',transform=ax[axid].transAxes)
         p.set_norm(norm)
         p.callbacksSM.connect('changed',update)
         plots.append(p)
         ax[layer].label_outer()
         # figB.colorbar(p,ax=axB[layer],ticks=[0,1])

      fig.colorbar(plots[0], ax=ax, orientation='vertical', fraction=.1,aspect=30)
      fig.text(0.01,0.95,'layer',fontsize=15)

      plt.gcf().savefig('quarkflavor_%s_fullimg.png' % quarkflavor)
      plt.close('all')



      plt.figure(1,figsize=(16,12))

      image = profile_data[quarkflavor]['layerhist']

      mean = image.mean()
      sigma = image.sigma()

      np.savez_compressed('quarkflavor_%s_layerhist.npz' % quarkflavor,mean=mean,sigma=sigma)

      layer_histos = []
      for layer in range(mean.shape[0]):
         plt.subplot(num_layers,1,mean.shape[0]-layer)
         layer_data_y = mean[layer]
         layer_data_x = [-0.095 + i*0.01 for i in range(layer_histo_bins)]
         layer_error_x = [0.005]*layer_histo_bins
         layer_error_y = sigma[layer]
         plt.errorbar(layer_data_x,layer_data_y,xerr=layer_error_x,yerr=layer_error_y)
         plt.ylim(1,1e5)
         plt.yscale('log')

      plt.gcf().savefig('quarkflavor_%s_layerhist.png' % quarkflavor)
      plt.close('all')



def update(changed_image,images):
   for im in images:
      if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
         im.set_cmap(changed_image.get_cmap())
         im.set_clim(changed_image.get_clim())


if __name__ == '__main__':
   main()

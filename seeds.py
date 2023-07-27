'''
This file modify into block-level update and change the comparison to cost-effective version,
Add boundary penalty
updating block are not limited within one superpixel
'''
import math
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os
from skimage.segmentation import mark_boundaries, find_boundaries
import cv2

class Cluster(object):
    cluster_index = 1

    def __init__(self, num_pixels=0, pixels=[]):
        '''
        if num_pixels=0, pixels should be []
        '''
        self.num_pixels = num_pixels
        
        self.pixels = pixels
        self.color_hist = []
        self.boundary_hist = []
        #self.boundarypixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def __str__(self):
        return "{},{},{} {} {} ".format(self.h, self.w, self.d, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()



class SEEDSProcessor(object): 
    
    
    def __init__(self, image, K, N, N_L, bin,alpha, gama):
        self.K = K # total number of superpixels 
        self.N = N
        self.N_step = int(N/2)
        self.colorbin = bin
        self.alpha = alpha
        self.gama = gama
        self.num_levels = N_L
        self.iter_pixel = 0

        self.data = image
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_depth = self.data.shape[2]
        self.Size = self.image_height * self.image_width * self.image_depth
        #self.S = int(math.sqrt(self.N / self.K))
        self.S = int(max(self.image_height,self.image_width,self.image_depth)/math.pow(self.K, 1/3)) ## grid size
        self.block_size = []
        a = self.S/3
        for i in range(self.num_levels):
            self.block_size.append(int(a))
            a = a/2

        self.block_size = np.array(self.block_size)
        self.block_size = np.array([11,5,3,1])
        print("grid size", self.S)
        print("block size", self.block_size)
        self.clusters = []
        self.label = {}
        self.hist = np.zeros((self.image_height, self.image_width, self.image_depth))
        self.Segment = np.zeros((self.image_height, self.image_width, self.image_depth))
        self.boundary = np.ones((self.image_height, self.image_width, self.image_depth))
        self.boundary_energy = np.zeros((self.image_height, self.image_width, self.image_depth))
        
    def make_cluster(self, h,w,d):
        N_step = self.N_step
        img = np.ones((self.image_height,self.image_width,self.image_depth))
        ## init all clusters to create grids..
        pixels = []
        h_ = min(h+self.S,self.image_height)
        w_ = min(w+self.S,self.image_width)
        d_ = min(d+self.S,self.image_depth)
        
        #print(h,w,d, h_,w_,d_)
        idxs = np.argwhere(img[h:h_, w:w_, d:d_])
        #print(idxs[0]) 
        num_pixels = idxs.shape[0]
        #print("number pixels of this cluster",num_pixels)
        
        #for pixel_idx in range(num_pixels):
        #    pixels.append(idxs[pixel_idx]+[h,w,d])

        pixels = list(idxs+np.array([h,w,d]))
        return Cluster(num_pixels, pixels)
    
    def init_clusters(self):
        ## initialize the grids in 3d
        #num_clusters = (len(a)-1)*(len(b)-1)*(len(c)-1)
        #h, w, d = [N_step, N_step, N_step]
        h, w, d = [0, 0, 0]
        flag= 0
        while h < self.image_height:
            while w < self.image_width:
                while d < self.image_depth: 
                    self.clusters.append(self.make_cluster(h=h,w=w,d=d))
                    d += self.S
                    flag +=1
                d = 0
                w += self.S
            w=0
            h += self.S

        self.hist= np.round_(self.colorbin/255 *self.data,0) + 1
        for no, cluster in enumerate(self.clusters):
            for p in cluster.pixels:
                
                self.Segment[tuple(p)] = no+1
            cluster.color_hist = np.histogram(self.hist[self.Segment==no+1].flatten(),bins= self.colorbin,range=[1,self.colorbin+1])[0]
            #cluster.color_hist, cluster.color_bins = list(b),list(a)
        self.boundary = find_boundaries(self.Segment)
        #self.iter_pixel = int(np.count_nonzero(self.boundary)*self.alpha)

    def boundary_h(self,h,w,d, N_step):
        #N_step = self.N_step
        flag = False
        block = self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1]

        #neigh = self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1].flatten()
        boundary_hist = np.unique(self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1].flatten(),return_counts=True)
        num_s = np.count_nonzero(find_boundaries(block))

        return boundary_hist[0],boundary_hist[1] ,num_s
     
    def neigh_hist(self, h,w,d, label, N_step):

        block = self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1]
        #c_neigh = self.hist[np.argwhere(block==label)+np.array([h-N_step,w-N_step,d-N_step])]
        c_neigh = self.hist[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1]
        c_neigh_label = c_neigh[np.where(block==label)]
        num_n = len(c_neigh_label)

        #c_neigh = self.hist[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1].flatten()
        #color_hist,color_bins= np.histogram(c_neigh,bins=self.colorbin,range=[1,self.colorbin+1])
        #color_hist= np.histogram(c_neigh,bins=self.colorbin,range=[1,self.colorbin+1])[0] 
        color_hist= np.histogram(c_neigh_label,bins=self.colorbin,range=[1,self.colorbin+1])[0] 
        color = np.array([max(color_hist),np.argmax(color_hist)])
        
        #boundary_hist[0] = boundary_hist[0]/(self.N * self.N * self.N)
        #return np.sum(boundary_hist*boundary_hist)
        return block, color_hist,num_n
    
    def energy(self, hist, NM):
        '''
        histogram, NM normalized weight
        '''
        hist = np.array(hist)/NM
        
        return np.sum(hist*hist)

    
    def assignment(self, number_level):
        ## find random boundary pixels, update the superpixel label for them
        N = self.block_size[number_level]
        N_step = int(N/2)
        gama = self.gama
        if N > self.N:
            gama=0

        num_b = self.N * self.N * self.N
        
        N_step_B = int(self.N/2)
        #N_step = self.N_step
        num_N = N * N * N

        self.boundary_pixels = np.argwhere(self.boundary)
        #num_b = self.boundary_pixels.shape[0]

        pixel_idxs = np.random.choice(range(self.boundary_pixels.shape[0]),int(len(self.boundary_pixels)*self.alpha))
        #pixel_idxs = np.random.choice(range(self.boundary_pixels.shape[0]),self.iter_pixel)
        #print(self.boundary_pixels(pixel_idxs))
        
        for h,w,d in self.boundary_pixels[pixel_idxs]:
        #for h,w,d in self.boundary_pixels:
            label = int(self.Segment[h,w,d])
            
            if label == 0 or h not in range(N_step,self.image_height-N_step) or w not in range(N_step,self.image_width-N_step) or d not in range(N_step,self.image_depth-N_step):
                continue
            #if label == 0 or h not in range(N,self.image_height-N) or w not in range(N,self.image_width-N) or d not in range(N,self.image_depth-N):
            #    continue
            
            cluster = self.clusters[label-1]
            num_p = cluster.num_pixels
            
            color_hist= cluster.color_hist
            expo_clusters, boundary_hist ,num_s= self.boundary_h(h,w,d,N_step_B)
            #boundary_hist = boundary_hist/num_b
            
            num_label = boundary_hist[np.where(expo_clusters==label)]
            block, colorH, num_n= self.neigh_hist(h,w,d,label,N_step)
        
            #hs1 = self.energy(color_hist, num_p)
            #gs1 = boundary_hist[np.where(expo_clusters==label)]
            #gs1 = np.count_nonzero(find_boundaries(self.Segment))/num_b
            gs1 = num_s/num_b
            for i,expo_cluster in enumerate(expo_clusters):
                expo_cluster =  int(expo_cluster)
                
                if expo_cluster ==0 or expo_cluster == label:
                    continue
                num_exo = boundary_hist[i]

                #self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1] = expo_cluster
                #gs2 = np.count_nonzero(find_boundaries(self.Segment))/num_b
                gs2 = num_n/num_N
                #self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1] = label
                #print(label, expo_cluster)
                # calculate current energy value
                #print(gs1,gs2)
                cluster_index = expo_cluster-1
                c_hist2 = self.clusters[cluster_index].color_hist
                num_p2 = self.clusters[cluster_index].num_pixels
                hs1 = self.energy(color_hist, num_p+num_p2)
                hs2 = self.energy(c_hist2,num_p2+num_p)
                
                ## use observation1 to compare color histogram term
                #Hs_ = np.min([c_hist2[colorAkl[1]], colorAkl[0]])
                Hs = hs1 + hs2
                #print(c_hist2)
                Hs_ = self.energy(color_hist-colorH, num_p+num_p2) +self.energy(c_hist2 +colorH, num_p+num_p2) 
                #print(Hs_, Hs)
                #print(c_hist2)
                #print(num_label,num_exo)
                Es = Hs + num_label*gama
                Es_ = Hs_ + num_exo*gama
                #if Hs_ > Hs :#or Hs_ == Hs:
                if Es_ > Es:
                    #print("updated")
                    #Hs2 = self.energy(c_hist2, num_p2)
                    
                    self.Segment[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1][block==label] = expo_cluster
                    #self.boundary_energy[h,w,d] = Gs_

                    cluster.num_pixels -= num_n
                    self.clusters[cluster_index].num_pixels += num_n
                    
                    cluster.color_hist -= colorH
                    self.clusters[cluster_index].color_hist += colorH
                break
                    
        self.boundary = find_boundaries(self.Segment)

    def save_lab_image(self, path, lab_arr,slice):
        """
        save the gray image
        :param path:
        :param lab_arr:
        :return:
        """
        abs_path = os.path.join("outputs_test_0725",path)
        gray_arr = lab_arr[slice,:,:]
        
        io.imsave(abs_path, gray_arr)


    def save_current_image(self, name,slice):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][p[2]] = cluster.l # use intensity of origin to represent whole window
                
            image_arr[cluster.h][cluster.w][cluster.d] = 0 # mark the origin as dark point
            
        #print(self.clusters[0].pixels)
        
        self.save_lab_image(name, image_arr, slice)

    

    def iterate_times(self, iter):
        slice = 25
        self.init_clusters()
        print("init cluster number",len(self.clusters))
        print("init seg max",np.max(self.Segment.flatten()))

        plt.figure()
        plt.imshow(mark_boundaries(self.data[slice,:,:],self.Segment[slice,:,:],mode='inner')) 
        plt.axis('off')
        print(np.count_nonzero(find_boundaries(self.Segment)))
        name = 'ceus3d_init.png'
        self.save_lab_image(name, self.Segment, slice)
        print("SEEDS initilization completed")
        N_L = self.num_levels
        iters = iter/N_L

        for i in trange(iter):
            #print(int(i/iters))
            self.assignment(int(i/iters))
            #print('iteration{} finished'.format(i))
            
        #name = 'a{alpha}L{level}K{k}N{n}B{b}G{g}.png'.format(alpha=self.alpha,level=self.num_levels, k=self.K,n=self.N,b=self.colorbin,g=self.gama)
        #self.save_lab_image(name, self.Segment, slice)
        #name = 'a{alpha}L{level}K{k}N{n}B{b}G{g}_boundary.png'.format(alpha=self.alpha,level=self.num_levels, k=self.K,n=self.N,b=self.colorbin,g=self.gama)
        #self.save_lab_image(name, self.boundary, slice)
        plt.figure()
        plt.imshow(mark_boundaries(self.data[slice,:,:],self.Segment[slice,:,:],mode='inner')) 
        plt.axis('off')


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
        self.color_bins = [] 
        #self.boundarypixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def __str__(self):
        return "{},{},{} {} {} ".format(self.h, self.w, self.d, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()



class SEEDSProcessor(object): 
    
    
    def __init__(self, image, K, N, bin):
        self.K = K # total number of superpixels 
        self.N = N
        self.N_step = int(N/2)
        self.colorbin = bin
        self.alpha = 0

        self.data = image
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_depth = self.data.shape[2]
        self.Size = self.image_height * self.image_width * self.image_depth
        #self.S = int(math.sqrt(self.N / self.K))
        self.S = int(max(self.image_height,self.image_width,self.image_depth)/math.pow(self.K, 1/3)) ## grid size
        #print("grid size", self.S)
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
        N_step = self.N_step       
        #num_clusters = (len(a)-1)*(len(b)-1)*(len(c)-1)
        #h, w, d = [N_step, N_step, N_step]
        h, w, d = [0, 0, 0]
        flag= 0
        while h < self.image_height:
            while w < self.image_width:
                while d < self.image_depth: 
                    self.clusters.append(self.make_cluster(self=self,h=h,w=w,d=d))
                    d += self.S
                    flag +=1
                d = 0
                w += self.S
            w=0
            h += self.S

        self.hist= np.round_(self.colorbin/255 *self.data,0) + 1
        for no, cluster in enumerate(self.clusters):
            for p in cluster.pixels:
                ## init the segmentation map
                #self.hist[tuple(p)] = int(self.data[tuple(p)]/self.colorbin)
                self.Segment[tuple(p)] = no+1
                #self.boundary[tuple(p)] = self.boundary_term(p[0],p[1],p[2])
            #print("one cluster finished")
            
            a,b = np.unique(self.hist[self.Segment==no+1],return_counts=True)
            cluster.color_hist, cluster.color_bins = list(b),list(a)
        #print("final cluster number", no)
        #self.Segment = self.Segment.astype(np.uint8)
        self.boundary = find_boundaries(self.Segment)

    def boundary_hist(self,h,w,d, Seg):
        ## 
        N_step = self.N_step
        neigh = Seg[h-N_step:h+N_step+1,w-N_step:w+N_step+1,d-N_step:d+N_step+1].flatten()
        boundary_hist = np.unique(neigh,return_counts=True)
        #boundary_hist[0] = boundary_hist[0]/(self.N * self.N * self.N)
        #return np.sum(boundary_hist*boundary_hist)
        
        return boundary_hist

    def energy(self, hist, NM):
        '''
        histogram, NM normalized weight
        '''

        hist = np.array(hist)/NM
        return np.sum(hist*hist)
        
    
    
    def assignment(self):
        ## find random boundary pixels, change them
        N_step = self.N_step
        num_n = self.N * self.N * self.N
        self.boundary_pixels = np.argwhere(self.boundary)

        pixel_idxs = np.random.choice(range(self.boundary_pixels.shape[0]),int(len(self.boundary_pixels)*0.01))
        #print(self.boundary_pixels(pixel_idxs))
        for h,w,d in self.boundary_pixels[pixel_idxs]:
        #for h,w,d in self.boundary_pixels:

            label = int(self.Segment[h,w,d])

            if label == 0 or h not in range(N_step,self.image_height-N_step) or w not in range(N_step,self.image_width-N_step) or d not in range(N_step,self.image_depth-N_step):
                continue
            
            cluster = self.clusters[label-1]
            num_p = cluster.num_pixels
            
            color =  self.hist[h,w,d]
            color_hist, color_bins= cluster.color_hist, cluster.color_bins
            
            expo_clusters, boundary_hist = self.boundary_hist(h,w,d,self.Segment)
            
            ##boundary term of pixel
            if self.boundary_energy[h,w,d] == 0:
                Gs = self.energy(boundary_hist, num_n)
            else:
                Gs = self.boundary_energy[h,w,d]
            
            Hs = self.energy(color_hist,num_p)
            Es = self.alpha*Gs + Hs
            #print(Hs, Gs)

            for expo_cluster in expo_clusters:
                expo_cluster =  int(expo_cluster)
                if expo_cluster ==0 or expo_cluster == label:
                    continue
                # calculate current energy value
                cluster_index = expo_cluster-1
                c_hist2 = self.clusters[cluster_index].color_hist
                c_bins2 = self.clusters[cluster_index].color_bins
                num_p2 = self.clusters[cluster_index].num_pixels
                Hs2 = self.energy(c_hist2, num_p2)
                Es += Hs2

                #energy value after current pixel moved to expo_cluster
                
                c_hist_ = color_hist.copy()
                #print(color, c_hist_,color_bins)
                c_hist_[color_bins.index(color)]-= 1 

                c_hist2_ = c_hist2.copy()
                #print(c_hist2,c_bins2)

                if color not in c_bins2:
                    c_hist2_.append(1)
                    #print("hahahha",c_hist2_,c_bins2)
                else:
                    c_hist2_[c_bins2.index(color)] += 1
                
                
                Hs_ = self.energy(c_hist_,num_p-1)
                Hs2_ = self.energy(c_hist2_,num_p2+1)
                

                b_hist_ = np.copy(boundary_hist)
                b_hist_[np.where(expo_clusters==expo_cluster)] +=1
                b_hist_[np.where(expo_clusters==label)] -=1
                Gs_ = self.energy(b_hist_,num_n)

                #print(label,expo_cluster,[Hs,Hs2,Gs],[ Hs_ ,Hs2_, Gs_ ] )

                if Hs_ + Hs2_ + self.alpha*Gs_ > Es:
                    #print("updated")
                    self.Segment[h,w,d] = expo_cluster
                    self.boundary_energy[h,w,d] = Gs_

                    self.clusters[label-1].num_pixels -= 1
                    #self.clusters[label].pixels.remove([h,w,d])
                    self.clusters[label-1].color_hist = c_hist_

                    self.clusters[cluster_index].num_pixels += 1
                    #self.clusters[expo_cluster].pixels.append([h,w,d])
                    self.clusters[cluster_index].color_hist = c_hist2_
                    if color not in c_bins2:
                        self.clusters[cluster_index].color_bins.append(color) 
                
                    #print("updated",c_hist_,color_bins,expo_cluster,self.clusters[cluster_index].color_hist,self.clusters[cluster_index].color_bins)

        self.boundary = find_boundaries(self.Segment)
        #plt.figure()
        #plt.imshow(self.boundary[:,:,0])

    def save_lab_image(self, path, lab_arr,slice):
        """
        save the gray image
        :param path:
        :param lab_arr:
        :return:
        """
        abs_path = os.path.join("SEEDS_outputs",path)
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
        self.init_clusters()
        print("init cluster number",len(self.clusters))
        print("init seg max",np.max(self.Segment.flatten()))
        slice = 75
        name = 'ceus3d_init.png'
        self.save_lab_image(name, self.Segment, 25)
        print("SEEDS initilization completed")

        for i in trange(iter):

            self.assignment()
            #print('iteration{} finished'.format(i))
            if i%100 ==0:
                name = 'ceus3d_N{N}_K{k}_loop{loop}.png'.format(loop=i, N=self.N, k=self.K)
                self.save_lab_image(name, self.Segment, 25)
            #name = 'ceus3d_N{N}_K{k}_loop{loop}.png'.format(loop=i, N=self.N, k=self.K)
            #self.save_current_image(name,slice)

        
        
#name = 'lenna_M{m}_K{k}_loop{loop}_slice{slice}.png'.format(loop=0, m=40, k=2000, slice =slice)
#p.save_current_image(name, slice)
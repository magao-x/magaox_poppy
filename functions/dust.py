import numpy as np

class dust:
    def __init__(self, name):
        self.name = name
        
    def set_mask(self, size, x_loc, y_loc, rad):
        self.size=size
        self.mask = [] # start off empty
        for nx in range(0, len(x_loc)):
            mask_step = np.ones((size, size)).astype(float)
            dmc = np.zeros((size, size))
            dm_coord = draw.circle(r=y_loc, c=x_loc[nx], radius=rad)
            dmc[dm_coord] = True
            mask_step[dmc==True] = np.nan
            self.mask.append(mask_step)
        self.n_steps = len(self.mask)
            
    def load_mask(self, mask_set):
        self.mask=[]
        for nx in range(0, np.shape(mask_set)[0]):
            self.mask.append(mask_set[nx])
        self.size=mask_set[nx].shape[0]
        self.n_steps=len(self.mask)
    
    def calc_median_bias(self, data_flat):
        med_bias = np.zeros((self.size*self.size))
        for k in range(0, self.size*self.size):
            med_bias[k] = np.median(data_flat[:,k].value)
        self.med_bias = med_bias.reshape((self.size, self.size))
        
    def apply_mask_data_flat(self, data_flat):
        fd = np.zeros((len(self.mask), self.size*self.size))
        for ns in range(0, self.n_steps):
            fd[ns] = data_flat[ns]*(self.mask[ns].reshape(self.size*self.size))
        self.flat_data_mask = fd
    
    def calc_mean_mask(self):
        mean_flat = np.zeros((self.size*self.size))
        #med_flat = np.zeros_like(mean_flat)
        for k in range(0, self.size*self.size):
            pix_data = self.flat_data_mask[:,k]
            pix_bin = pix_data[~np.isnan(pix_data)]
            if len(pix_bin) != 0: # if not all nan values
                mean_flat[k] = np.mean(pix_bin)
            else:
                mean_flat[k] = np.nan # not sure if this will work
            #med_flat[k] = np.median(pix_bin)
        self.mean_surf = mean_flat.reshape((self.size, self.size))
        #self.med_surf = med_flat.reshape((self.size, self.size))

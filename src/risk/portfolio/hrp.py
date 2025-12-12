
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# v39 Hierarchical Risk Parity (HRP)
# Robust portfolio allocation that treats correlation matrix as a hierarchy.

class HRP:
    def __init__(self):
        pass
        
    def get_quasi_diag(self, link):
        # Sort clustered items by order
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]) #.sort_index()
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    def get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]
        w_ = self.get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    def get_ivp(self, cov):
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def optimize(self, returns_df):
        corr = returns_df.corr()
        cov = returns_df.cov()
        
        # 1. Clustering
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist), 'single')
        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        # 2. Recruitment
        hrp = self.get_rec_bipart(cov, sort_ix)
        return hrp.sort_index()

if __name__ == "__main__":
    # Test
    hrp = HRP()
    dummy = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
    w = hrp.optimize(dummy)
    print("HRP Weights:", w)

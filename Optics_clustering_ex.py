'''
nguồn:
https://scikit-learn.org/dev/modules/generated/sklearn.cluster.OPTICS.html
https://github.com/scikit-learn/scikit-learn/issues/11677
'''

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu

np.random.seed(0) # Random với giá trị khởi tạo ở phía trước
n_diem = 250

X1 = [-5, -2] + .8 * np.random.randn(n_diem, 2)
X2 = [4, -1] + .1 * np.random.randn(n_diem, 2)
X3 = [1, -2] + .2 * np.random.randn(n_diem, 2)
X4 = [-2, 3] + .3 * np.random.randn(n_diem, 2)
X5 = [3, -2] + 1.6 * np.random.randn(n_diem, 2)
X6 = [5, 6] + 2 * np.random.randn(n_diem, 2)
X = np.vstack((X1, X2, X3, X4, X5, X6))  #Theo thứ tự theo chiều dọc

clust_optics = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05) #Truyền tham số có hàm
# OPTICS với MinPts:50 , e=0,05


# Run the fit
clust_optics.fit(X)   # Run OPTICS

labels_050 = cluster_optics_dbscan(reachability=clust_optics.reachability_,
                                   core_distances=clust_optics.core_distances_,
                                   ordering=clust_optics.ordering_, eps=0.5)#RUN DBSCAN VỚI EPS=0,5
labels_200 = cluster_optics_dbscan(reachability=clust_optics.reachability_,
                                   core_distances=clust_optics.core_distances_,
                                   ordering=clust_optics.ordering_, eps=2)#RUN DBSCAN VỚI EPS=2

space = np.arange(len(X)) #Độ dài mãng dư liệu
reachability = clust_optics.reachability_[clust_optics.ordering_]
labels = clust_optics.labels_[clust_optics.ordering_]

plt.figure(figsize=(10, 7))#Độ lớn figure
G = gridspec.GridSpec(2, 3)#Tạo các vị trí cho ax gồm có 2 hàng 3 cột
ax1 = plt.subplot(G[0, :]) #ax1 hiển thị ở hàng 0 cột 1 2 3
ax2 = plt.subplot(G[1, 0])#ax2 hiển thị ở hàng 1 cột 0
ax3 = plt.subplot(G[1, 1])#ax3 hiển thị ở hàng 1 cột 1
ax4 = plt.subplot(G[1, 2])#ax2 hiển thị ở hàng 1 cột 2

# Reachability plot  # Đồ thị biển diễn cho phân nhóm được hỗ trợ của thư viện sklearn
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Epsilon distance)')
ax1.set_title('Reachability Plot')


# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust_optics.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust_optics.labels_ == -1, 0], X[clust_optics.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon at \nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon at\nDBSCAN')

plt.tight_layout()
plt.show()
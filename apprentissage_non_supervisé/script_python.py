#import libraries
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import time



import hdbscan

#Data choosen
rings_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/rings.arff','r'))
rings =rings_dataset[0]

diamond_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/diamond9.arff','r'))
diamond=diamond_dataset[0]

spiral_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/3-spiral.arff','r'))
spiral=spiral_dataset[0]

convexe_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/cure-T0-2000n-2D.arff','r'))
convexe=convexe_dataset[0]

convexe1_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/2d-3c-no123.arff','r'))
convexe1=convexe1_dataset[0]

banana_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/banana.arff','r'))
banana=banana_dataset[0]
k=0
for g in banana:
    if(g[2]==b'Class 1'):
        banana[k][2]=1
    else:
        banana[k][2]=2
    k=k+1
    
    
square_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/square5.arff','r'))
square=square_dataset[0]

complex_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/complex8.arff','r'))
complexe=complex_dataset[0]



convexe2_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/2d-4c-no9.arff','r'))
convexe2=convexe2_dataset[0]

diamond_like_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/2d-20c-no0.arff','r'))
diamond_like=diamond_like_dataset[0]


#pour les data avec du bruits, remplacer b'noise' par -1
complex_noise_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/cluto-t8-8k.arff','r'))
complexe_noise=complex_noise_dataset[0]
k=0
for g in complexe_noise:
    if(g[2]==b'noise'):
        complexe_noise[k][2]=-1
        
complex2_noise_dataset = arff.loadarff(open('clustering-benchmark-master/src/main/resources/datasets/artificial/cluto-t7-10k.arff','r'))
complexe2_noise=complex2_noise_dataset[0]
k=0
for g in complexe2_noise:
    if(g[2]==b'noise'):
        complexe2_noise[k][2]=-1
#list
list_dataset=[rings, diamond, spiral, convexe1, banana, square, complexe, convexe2, diamond_like, convexe]
name_dataset=["rings", "diamond", "spiral", "convexe1", "banana", "square", "complexe", "convexe2", "diamond like", "convexe"]
nbr_cluster=[]


#Plot data choosen

k=0
fig = plt.figure(figsize=(20,15))
for b in list_dataset:
    y=[]
    x=[]
    color=[]
    for i in b:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.add_subplot(4,3,k+1)   #
    plt.scatter(x,y,c=color)
    plt.title(name_dataset[k])
    k=k+1



#K-Means bonne data
#etude avec bon nombre de clusters


data=[diamond, diamond_like, convexe, convexe1]  #TODO Trouver des bonnes figures
name=["diamond", "diamond_like", "convexe", "convexe1"]  #TODO
nbrs_cluster=[9,20,3,3]
j=0
fig = plt.figure(figsize=(20,10))
for c in data:
    x=[]
    y=[]
    color=[]
    x_center=[]
    y_center=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])

    time1 = time.time()
    kmeans =  KMeans(n_clusters=nbrs_cluster[j], random_state=0).fit(np.column_stack((x,y))) #kmeans++
    predict = kmeans.predict(np.column_stack((x,y)))
    time2 = time.time()
    center = kmeans.cluster_centers_
    for v in center:
        x_center.append(v[0])
        y_center.append(v[1])
    fig.add_subplot(2,2,j+1)
    plt.scatter(x,y, c=predict)
    plt.scatter(x_center,y_center,c='red',s=5)
    plt.title(name[j])
    print("time for", name[j], ":", time2-time1, "seconds")
    j=j+1
    
    #K-Means
#etude comparaison sur nombre de clusters
data=[diamond, diamond_like, convexe, convexe2]  #TODO Trouver des bonnes figures
name=["diamond", "diamond_like", "convexe", "convexe2"]  #TODO
nbrs_cluster=[9,20,3,3]
nbrs_clusters=[[7,8,9,10,11,12],[18,19,20,21,22,23],[2,3,4,5],[2,3,4,5]]
j=0
fig = plt.figure(figsize=(20,10))

fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.append(plt.figure(num=j, figsize=(20,15)))
    total_plot=1

    
    for nbr_cluster in nbrs_clusters[j]: #TODO changer la range
        x_center=[]
        y_center=[]
        time1 = time.time()
        kmeans =  KMeans(n_clusters=nbr_cluster, random_state=0).fit(np.column_stack((x,y))) #kmeans++
        predict = kmeans.predict(np.column_stack((x,y)))
        time2 = time.time()
        center = kmeans.cluster_centers_
        for v in center:
            x_center.append(v[0])
            y_center.append(v[1])
        fig[j].add_subplot(5,4,total_plot)    #TODO changer la taille
        plt.scatter(x,y, c=predict)
        plt.scatter(x_center,y_center,c='red',s=5)
        plt.title(name[j]+"+"+str(nbr_cluster)+"clusters")
        print("time for", name[j], " with ", nbr_cluster, "cluster(s): ", time2-time1, "seconds")
        total_plot=total_plot+1
    j=j+1


#K-Means mauvaise data
#etude comparaison sur nombre de clusters


data=[rings, convexe1, spiral]  #TODO Trouver des bonnes figures
name=["rings", "convexe1", "spiral"]  #TODO
nbrs_cluster=[3,3,3]
j=0
fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
        
    fig.append(plt.figure(num=j, figsize=(20,15)))
    total_plot=1

    for a in range(2,6):    
        x_center=[]
        y_center=[]
        time1 = time.time()
        kmeans =  KMeans(n_clusters=a, random_state=0).fit(np.column_stack((x,y))) # kmeans++
        predict = kmeans.predict(np.column_stack((x,y)))
        time2 = time.time()
        center = kmeans.cluster_centers_
        for v in center:
            x_center.append(v[0])
            y_center.append(v[1])
        fig[j].add_subplot(3,4,total_plot)    
        plt.scatter(x,y, c=predict)
        plt.scatter(x_center,y_center, c='red',s=5)
        plt.title(name[j]+"+"+str(a)+"clusters")
        print("time for ", name[j], a ,"clusters:", time2-time1, "seconds")
        total_plot=total_plot+1
    j=j+1
    
    
 #clustering agglomératif bonne data
#etude avec bon nombre de clusters
#étude des paramètres


data=[diamond, diamond_like, convexe, convexe2]  #TODO Trouver des bonnes figures
name=["diamond", "diamond_like", "convexe", "convexe2"]  #TODO
nbrs_cluster=[9,20,3,4]
linkage=["single","average", "complete", "ward"]
j=0
total_plot=1
fig = plt.figure(figsize=(20,15))
for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
        z=0
    for e in linkage:
        time1 = time.time()
        agglo =  AgglomerativeClustering(n_clusters=nbrs_cluster[j], linkage=e).fit_predict(np.column_stack((x,y)))
        #predict = agglo.predict(np.column_stack((x,y)))
        time2 = time.time()
        fig.add_subplot(4,4,total_plot)
        plt.scatter(x,y, c=agglo)
        plt.title(name[j]+"+"+linkage[z])
        print("time for ", name[j], linkage[z], ":", time2-time1, "seconds")
        total_plot=total_plot+1
        z=z+1
    j=j+1  
    
    
    
 #clustering agglomératif bonne data
#etude comparaison sur nombre de clusters
#étude des paramètres


data=[diamond, diamond_like, convexe, convexe2]  #TODO Trouver des bonnes figures
name=["diamond", "diamond_like", "convexe", "convexe2"]  #TODO
nbrs_cluster=[9,20,3,4]
nbrs_clusters=[[7,8,9,10,11,12],[18,19,20,21,22,23],[2,3,4,5],[2,3,4,5]]
linkage=["single","average", "complete", "ward"]
j=0

fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.append(plt.figure(num=j, figsize=(20,15)))
    total_plot=1
    for nbr_cluster in nbrs_clusters[j]:     #TODO changer la range
        z=0
        for e in linkage:
            time1 = time.time()
            agglo =  AgglomerativeClustering(n_clusters=nbr_cluster, linkage=e).fit_predict(np.column_stack((x,y)))
            time2 = time.time()
            fig[j].add_subplot(6,4,total_plot)    #TODO changer la taille
            plt.scatter(x,y, c=agglo)
            plt.title(name[j]+"+"+linkage[z]+"+"+str(nbr_cluster))
            print("time for ", name[j], linkage[z], nbr_cluster, " cluster(s):", time2-time1, "seconds")
            total_plot=total_plot+1
            z=z+1
    j=j+1
    
    
    
#clustering agglomératif mauvaise data
#etude comparaison sur nombre de clusters
#étude des paramètres


data=[rings, convexe1, spiral]  #TODO Trouver des bonnes figures
name=["rings", "convexe1", "spiral"]  #TODO
nbrs_cluster=[3,3,3]
linkage=["single","average", "complete", "ward"]
j=0

fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.append(plt.figure(num=j, figsize=(20,15)))
    total_plot=1
    for a in range (2,6):      
        z=0
        for e in linkage:
            time1 = time.time()
            agglo =  AgglomerativeClustering(n_clusters=a, linkage=e).fit_predict(np.column_stack((x,y)))
            time2 = time.time()
            fig[j].add_subplot(4,4,total_plot)    
            plt.scatter(x,y, c=agglo)
            plt.title(name[j]+"+"+linkage[z]+"+"+str(a))
            print("time for ", name[j], linkage[z], a, " cluster(s):", time2-time1, "seconds")
            total_plot=total_plot+1
            z=z+1
    j=j+1
    
    
    
    
#DBSCAN bonne data
#étude des paramètres min_samples et eps


data=[diamond, diamond_like, convexe, convexe2]  #TODO Trouver des bonnes figures
name=["diamond", "diamond_like", "convexe", "convexe2"]  #TODO
nbrs_cluster=[9,20,4,3]
min_samples=[1,5,8,12]    #TODO change it
j=0

fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.append(plt.figure(num=j, figsize=(20,20)))
    total_plot=1
    for eps_value in range (1,5):     #TODO changer la range
        z=0
        for min_value in min_samples:
            time1 = time.time()
            predict =  DBSCAN(eps=eps_value, min_samples=min_value).fit_predict(np.column_stack((x,y)))
            time2 = time.time()
            fig[j].add_subplot(5,4,total_plot)    #TODO changer la taille
            plt.scatter(x,y, c=predict)
            plt.title(name[j]+"+min_value="+str(min_value)+"+eps_value="+str(eps_value))
            print("time for ", name[j],"with" ,min_value, "min samples and ",eps_value, " eps value:", time2-time1, "seconds")
            total_plot=total_plot+1
            z=z+1
    j=j+1
    
    
#DBSCAN mauvaise data
#étude des paramètres min_samples et eps


data=[rings, spiral, convexe1, complexe_noise, complexe2_noise]  #TODO Trouver des bonnes figures
name=["rings", "spiral", "convexe1", "complexe_noise","complexe2_noise"]  #TODO
nbrs_cluster=[3,3,8,8]
min_samples=[1,2,3,4]    #TODO change it
j=0

fig=[]

for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])
    fig.append(plt.figure(num=j, figsize=(20,20)))
    total_plot=1
    for eps_value in range (1,5):     #TODO changer la range
        z=0
        for min_value in min_samples:
            time1 = time.time()
            predict =  DBSCAN(eps=eps_value, min_samples=min_value).fit_predict(np.column_stack((x,y)))
            time2 = time.time()
            fig[j].add_subplot(5,4,total_plot)    #TODO changer la taille
            plt.scatter(x,y, c=predict)
            plt.title(name[j]+"+min_value="+str(min_value)+"+eps_value="+str(eps_value))
            print("time for ", name[j],"with" ,min_value, "min samples and ",eps_value, " eps value:", time2-time1, "seconds")
            total_plot=total_plot+1
            z=z+1
    j=j+1
    
    
#HDBSCAN 

data=[diamond, diamond_like, square, convexe2, complexe_noise, complexe2_noise]  #TODO Trouver des bonnes figures AVEC DU BRUITS !!!§§
name=["diamond", "diamond_like", "square", "convexe2", "complexe noise", "complexe2 noise"]  #TODO
nbrs_cluster=[9,20,4,3,8,9]
min_samples=[1,2,3,4]    #TODO change it
j=0
fig = plt.figure(figsize=(20,10))
total_plot=1
for c in data:
    x=[]
    y=[]
    color=[]
    for i in c:
        x.append(i[0])
        y.append(i[1])
        color.append(i[2])

    time1 = time.time()
    predict =  hdbscan.HDBSCAN(min_cluster_size=nbrs_cluster[j]).fit_predict(np.column_stack((x,y)))
    time2 = time.time()
    fig.add_subplot(3,2,total_plot)    #TODO changer la taille
    plt.scatter(x,y, c=predict)
    plt.title(name[j])
    print("time for ", name[j],":", time2-time1, "seconds")
    total_plot=total_plot+1
    j=j+1
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 06 14:10:30 2018

@author: Shyam
"""
import numpy as np
import os
import scipy
import math
os.chdir('C:\stuff\Studies\Spring 18\Data Mining\Assignment 4\Recommendation System\ml-10M100K')

#USers demographic info
file_name = 'u.user'
users = {}
f=open(file_name)
for line in f:
    x = line.split().pop()
    array = x.split('|')
    users[int(array[0])] = {}
    users[int(array[0])]['Age'] = int(array[1])
    users[int(array[0])]['Gender'] = (array[2])
    users[int(array[0])]['Occupation'] = (array[3])
    users[int(array[0])]['Zip'] = (array[4])    
f.close()

#Movies storage
file_name = 'u.item'
movies = {}
genre_dictionary = {}
for i in range(19):
    genre_dictionary[i] = []
    
f=open(file_name)
for line in f:
    x = line.split('|')
    #x[int(x[0])] = {}
    array = x[5:]
    array_2 = []
    j = -1
    for element in array:
        j += 1
        array_2.append(int(element))
        if int(element) == 1:
            genre_dictionary[j].append(int(x[0]))
    movies[int(x[0])] = array_2
f.close()    

ratings = np.zeros((len(users.keys()), len(movies.keys())))

file_name = 'u3.base'
f=open(file_name)
user_movie_rating = {}
for line in f:
    array = line.split()[:-1]
    try:
        user_movie_rating[int(array[0])][int(array[1])] = int(array[2])
    except:
        user_movie_rating[int(array[0])] = {}
    ratings[int(array[0]) - 1 , int(array[1]) - 1] = int(array[2])
f.close()

#To get first line into array.
#Check test data
k_nearest = 15
len_movies = ratings.shape[1]
file_name = 'u3.test'
f=open(file_name)
count = 0
error_euclidean = 0
error_cityblock = 0
error_cosine = 0
error_naive = 0
for line in f:
    count += 1
    temp_array = line.split()[:-1]
    user = int(temp_array[0])
    movie = int(temp_array[1])
    user_rating = int(temp_array[2])
    user_index = user-1
    movie_index = movie-1
    compare_users = ratings[ratings[:,movie_index] > 0]
    user_array = ratings[user_index,:].reshape(1,-1)
    compare_users_missing_movie_rating = compare_users[:,movie_index].reshape(-1,1)
    #Deleting the movie index from user and compare_users
    array_index = range(len_movies)
    del(array_index[movie_index])
    user_array =  user_array[:,array_index]
    compare_users = compare_users[:,array_index]
    len_compare_users = int(compare_users.shape[0])
    if len_compare_users > 0:
        #Calculate distance of user_array with compare_users, and then partition k nearest
        temporary = abs(compare_users - user_array)
        euclidean_distance = pow(np.sum(pow(temporary,2),axis = 1),0.5)
        cityblock_distance = np.sum(temporary,axis = 1)
        cosine_distance = 1.0 - (np.sum((compare_users*user_array),axis = 1)) / (pow(np.sum(pow(user_array,2),axis = 1) * np.sum(pow(compare_users,2),axis = 1),0.5))
        if k_nearest < len_compare_users:        
            index_euclidean = np.argpartition(euclidean_distance,k_nearest)[:k_nearest]
            index_cityblock = np.argpartition(cityblock_distance,k_nearest)[:k_nearest]
            index_cosine = np.argpartition(cosine_distance,k_nearest)[:k_nearest]
        else:
            index_euclidean = range(len_compare_users)
            index_cityblock = range(len_compare_users)
            index_cosine = range(len_compare_users)
        #Calculating the mean
        mean_euclidean = float(np.mean(compare_users_missing_movie_rating[index_euclidean,:]))
        mean_cityblock = float(np.mean(compare_users_missing_movie_rating[index_cityblock,:]))
        mean_cosine = float(np.mean(compare_users_missing_movie_rating[index_cosine,:]))
        mean_naive = float(np.mean(compare_users_missing_movie_rating))
        if math.isnan(mean_euclidean):
            mean_euclidean = 3
        if math.isnan(mean_cityblock):
            mean_cityblock = 3
        if math.isnan(mean_cosine):
            mean_cosine = 3        
        if math.isnan(mean_naive):
            mean_naive = 3                            
        
        error_euclidean += abs(mean_euclidean - user_rating)
        error_cityblock += abs(mean_cityblock - user_rating)
        error_cosine += abs(mean_cosine - user_rating)
        error_naive += abs(mean_naive - user_rating)  
    else:
        mean_naive = float(np.mean(user_array[user_array > 0].reshape(1,-1), axis = 1))
        if math.isnan(mean_naive):
            mean_naive = 3        
        error_euclidean += abs(mean_naive - user_rating)
        error_cityblock += abs(mean_naive - user_rating)
        error_cosine += abs(mean_naive - user_rating)
        error_naive += abs(mean_naive - user_rating)

f.close()
#Out of file
error_euclidean /= count
error_cityblock /= count
error_cosine /= count
error_naive /= count

#Sanity check
#    if distance_function == "euclidean":
#        centroid_dist = (pow(np.sum(pow(compare_users - user_array,2),axis = 1),0.5))
#    elif distance_function == "cityblock":
#        centroid_dist = (np.sum(abs(compare_users - user_array),axis = 1))
#    elif distance_function == "cosine":
#        centroid_dist = 1.0 - (np.sum((compare_users*user_array),axis = 1)) / (pow(np.sum(pow(user_array,2),axis = 1) * np.sum(pow(compare_users,2),axis = 1),0.5))    

#Part 2

ratings = np.zeros((len(users.keys()), len(movies.keys())))

#Adding Age
age_array = []
for i in range(len(users.keys())):
   age_array.append(users[i+1]['Age'] / 6.0)

age_array = np.array(age_array) 
age_array = age_array.reshape(-1,1)
ratings = np.append(ratings,age_array, axis = 1)    

file_name = 'u1.base'
f=open(file_name)
user_movie_rating = {}
for line in f:
    array = line.split()[:-1]
    try:
        user_movie_rating[int(array[0])][int(array[1])] = int(array[2])
    except:
        user_movie_rating[int(array[0])] = {}
    ratings[int(array[0]) - 1 , int(array[1]) - 1] = int(array[2])
f.close()

#To get first line into array.
#Check test data
k_nearest = 5
len_movies = ratings.shape[1]
file_name = 'u1.test'
f=open(file_name)
count = 0
error_euclidean = 0
error_cityblock = 0
error_cosine = 0
error_naive = 0
for line in f:
    count += 1
    temp_array = line.split()[:-1]
    user = int(temp_array[0])
    movie = int(temp_array[1])
    user_rating = int(temp_array[2])
    user_index = user-1
    movie_index = movie-1
    compare_users = ratings[ratings[:,movie_index] > 0]
    user_array = ratings[user_index,:].reshape(1,-1)
    compare_users_missing_movie_rating = compare_users[:,movie_index].reshape(-1,1)
    #Deleting the movie index from user and compare_users
    #array_index = range(len_movies)
    movie_genre_array = np.array(movies[movie])
    genres = np.where(movie_genre_array == 1)[0]
    movies_array = []
    for genre in list(genres):
        genre_array = list(np.array(genre_dictionary[genre]) - 1)
        movies_array = movies_array + genre_array
    index_to_keep = np.unique(movies_array)
    array_index = list(index_to_keep) + [len_movies - 1]
    del(array_index[array_index.index(movie_index)])
    #del(array_index[movie_index])
    user_array =  user_array[:,array_index]
    compare_users = compare_users[:,array_index]
    len_compare_users = int(compare_users.shape[0])
    if len_compare_users > 0:
        #Calculate distance of user_array with compare_users, and then partition k nearest
        temporary = abs(compare_users - user_array)
        euclidean_distance = pow(np.sum(pow(temporary,2),axis = 1),0.5)
        cityblock_distance = np.sum(temporary,axis = 1)
        cosine_distance = 1.0 - (np.sum((compare_users*user_array),axis = 1)) / (pow(np.sum(pow(user_array,2),axis = 1) * np.sum(pow(compare_users,2),axis = 1),0.5))
        if k_nearest < len_compare_users:        
            index_euclidean = np.argpartition(euclidean_distance,k_nearest)[:k_nearest]
            index_cityblock = np.argpartition(cityblock_distance,k_nearest)[:k_nearest]
            index_cosine = np.argpartition(cosine_distance,k_nearest)[:k_nearest]
        else:
            index_euclidean = range(len_compare_users)
            index_cityblock = range(len_compare_users)
            index_cosine = range(len_compare_users)
        #Calculating the mean
        mean_euclidean = float(np.mean(compare_users_missing_movie_rating[index_euclidean,:]))
        mean_cityblock = float(np.mean(compare_users_missing_movie_rating[index_cityblock,:]))
        mean_cosine = float(np.mean(compare_users_missing_movie_rating[index_cosine,:]))
        mean_naive = float(np.mean(compare_users_missing_movie_rating))
        if math.isnan(mean_euclidean):
            mean_euclidean = 3
        if math.isnan(mean_cityblock):
            mean_cityblock = 3
        if math.isnan(mean_cosine):
            mean_cosine = 3        
        if math.isnan(mean_naive):
            mean_naive = 3                    
        error_euclidean += abs(mean_euclidean - user_rating)
        error_cityblock += abs(mean_cityblock - user_rating)
        error_cosine += abs(mean_cosine - user_rating)
        error_naive += abs(mean_naive - user_rating)  
    else:
        user_array = user_array[:,:-1]
        mean_naive = float(np.mean(user_array[user_array > 0].reshape(1,-1), axis = 1))
        if math.isnan(mean_naive):
            mean_naive = 3        
        error_euclidean += abs(mean_naive - user_rating)
        error_cityblock += abs(mean_naive - user_rating)
        error_cosine += abs(mean_naive - user_rating)
        error_naive += abs(mean_naive - user_rating)

f.close()
#Out of file
error_euclidean /= count
error_cityblock /= count
error_cosine /= count
error_naive /= count

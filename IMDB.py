import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('movie_metadata.csv')



 

a=raw_input("To get the number of null values in each columns enter 'show' else 'no':")


if(a=='show'):
  print dataset.isnull().sum()

dataset.iloc[:,:].fillna(dataset.iloc[:,:].mean(),inplace=True)

a=raw_input("to get the highlights of the data enter 'show' else 'no':")

if(a=='show'):
 print dataset.head()

a=raw_input("to get the numerical description of the data enter 'show' else 'no':")

if(a=='show'):
 print dataset.describe()
 
 #the below method gives the maximum row of each column
a=raw_input("to get the details of the best performing movie in a definite domain type the domain name else enter 'no':")
if(a== 'num_critic_for_reviews'):
    print dataset.loc[dataset['num_critic_for_reviews'].idxmax()]
elif(a== 'duration' )             : 
    print dataset.loc[dataset[' duration'].idxmax()]
elif(a=='gross')                 : 
    print dataset.loc[dataset['gross'].idxmax()]
elif(a=='num_user_for_reviews' ):
   print dataset.loc[dataset['num_user_for_reviews'].idxmax()]
elif(a=='budget')              : 
    print dataset.loc[dataset['budget'].idxmax()]
elif(a=='imdb_score')            : 
    print dataset.loc[dataset['imdb_score'].idxmax()]
elif(a=='movie_facebook_likes' ) : 
            print dataset.loc[dataset['movie_facebook_likes'].idxmax()]
    
#to plot graph 

a=raw_input("to plot graph between two parameters enter the first:")
b=raw_input("enter the second:")
dataset.plot(kind='scatter',x=a,y=b)

#the below method gives the average imbd score for actors and directors
i=raw_input("to get the average imdb score of an actor enter 'actor', to get the average imdb of a director enter 'director' ")
if(i=='actor'):
 a=raw_input("enter the name of the actor")
 imdb=0;
 count=0
 for i in range (0,5043):
   if(str(dataset.iloc[i,10])==a):
       imdb=imdb+dataset.iloc[i,25]
       count=count+1
 print imdb/count

elif(i=='director'):
 a=raw_input("enter the name of the director:")
 imdb=0;
 count=0
 for i in range (0,5043):
   if(str(dataset.iloc[i,1])==a):
       imdb=imdb+dataset.iloc[i,25]
       count=count+1
 print imdb/count
 
 

       


    
    
    
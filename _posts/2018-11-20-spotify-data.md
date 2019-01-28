---
title: "Spotify: Music Data retrieve and EDA"
date: 2018-11-20
tags: [spotify, explanatory data analysis, music data]
header:
  image: "/air_rev_output/air_revenue.jpeg"
excerpt: "Spotify, Explanatory data analysis, Music data"
mathjax: "true"
---

## About Spotify

Originally found in Sweden, Spotify became one of the largest worldwide music streaming services. Since  2006 company has been growing its popularity and reached 87 million paying subscribers in November 2018. The large number of Spotify users allowed to create significant content database and retreive valuable patterns from tracks attributes. <br>

For current project we used developers documentation and Web API to access music data, provided by platform. 

**Data**:

We will go through official documentation, using "Search" endpoint reference to collect data, includes:<br>
 - tracks metadata;
 - artists' attributes;
 - audio features.
 
 
### Install libraries, API connection setting

Let's store data into panads data frame and import visualization libraries for explanatory analysis part.

<details><summary>Python code</summary> 
  
<p>
  
 ```python

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

%matplotlib inline
from pylab import rcParams

plt.style.use('bmh')

"""
- establish client credentials for authentification

"""
import spotipy 
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
 ```
 
 </p>
</details>

We access data with username and client id of application, registrated on developers page. Additionally we need to use token with scope, such as user library/playlist read/modify to retreive tracks and attributes.

<details><summary>Python code</summary> 
  
<p>
  
 ```python

"""
- we access data with username and client id of application, registrated
- import visualization libraries for explanatory analysis part

"""
 
username='XXXXXX' # Spotify username
scope = 'user-library-read playlist-modify-public playlist-read-private'
client_id='XXXXX' # app-redirect url
client_secret='XXXXXXXXX'

redirect_uri='http://localhost:8888/callback/' # Paste your Redirect URI here
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Token is not accessible", username)
 ```
 
 </p>
</details>

### Part 1 - retrieve the tracks' list

On this stage we will retrieve tracks metadata: artist name, album name, track name, popularity, track id and write into data frame.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- generate empty lists for related track's characteristics
- using sp.search() will iterate through 2018 tracks, appending items in lists

"""

artist_ids=[]
artist_names=[]
track_names=[]
track_ids=[]
popularity_score=[]

for i in range(0,10000,10):
    res=sp.search(q='year:2018', type='track', offset=i)
    for i, j in enumerate(res['tracks']['items']):
        artist_id = j['artists'][0]['id']
        artist_ids.append(artist_id)
        artist_name = j['artists'][0]['name']
        artist_names.append(artist_name)
        track_name = j['name']
        track_names.append(track_name)
        track_id = j['id']
        track_ids.append(track_id)
        track_popularity = j['popularity']
        popularity_score.append(track_popularity)
# create data frame with lists as columns 
track_attributes = pd.DataFrame({'artist_ids':artist_ids, 'artist_names':artist_names, 'track_names':track_names, 'track_ids':track_ids, 'popularity_score':popularity_score})
 ```
 
 </p>
</details>


### Part 2 - retrieve the artists' genres

However Spotify documentation does not provide the way to look up the genres tags for selected tracks from database. Hence, we will use the same approach to get artists data using Spotify.search().

<details><summary>Python code</summary> 
  
<p>
  
 ```python
# create the lists for artists ids and genres tags
artist_ids_genres=[]
genres_all=[]

for i in range(0,10000,10):
    res_genres=sp.search(q='year:2018', type='artist', offset=i)
    for i, j in enumerate(res_genres['artists']['items']):
        artist_id_genre=j['id']
        artist_ids_genres.append(artist_id_genre)
        artist_genre=j['genres']
        genres_all.append(artist_genre)
        
# separate tags with comma          
genres_list = [', '.join(x) for x in genres_all]
genres = pd.DataFrame({'artist_ids':artist_ids_genres, 'genres_all':genres_list})

genres.to_csv('genres.csv')
 ```
 
 </p>
</details>

### Part 3 - retrieve the tracks' features

Additionally we are going to retreive the attributes: energy, liveness, acousticness, instrumentalness, loudness, danceability and valence. Before writing into frame, its worth checking for 'None' values - otherwise, the error can be occured.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- start with empty list - it will include the sets of all features 
- check for None values 
- write a list into dataframe 

"""
features = []
for i in range(0,len(track_ids)):
    track=str(track_ids[i])
    audio_features = sp.audio_features(track)
    for track in audio_features:
        features.append(track)
# use list comprehension, leave elements that are not None
f = [feat for feat in features if feat is not None]
playlist_df = pd.DataFrame(f)
playlist_df.rename(columns={'id': 'track_ids'}, inplace=True)

playlist_df.head()
 ```
 
 </p>
</details>

### Part 4 - enitire dataframe collection

Finally we merge all collected dataframes in one to prepare for explanatory analysis.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- start with empty list - it will include the sets of all features 
- check for None values 
- write a list into dataframe 

"""
features = []
for i in range(0,len(track_ids)):
    track=str(track_ids[i])
    audio_features = sp.audio_features(track)
    for track in audio_features:
        features.append(track)
# use list comprehension, leave elements that are not None
f = [feat for feat in features if feat is not None]
playlist_df = pd.DataFrame(f)
playlist_df.rename(columns={'id': 'track_ids'}, inplace=True)

playlist_df.head()
 ```
 
 </p>
</details>


Delete columns we don't need for analysis and check NaN values.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- read from csv and and drop NaNs 
- display the shape of frame

"""
df = pd.read_csv('tracks_data.csv', sep='\t', encoding='utf-8')
df.drop(['artist_ids', 'track_ids', 'analysis_url', 'track_href', 'uri', 'key', 'type', 'Unnamed: 0'], axis=1, inplace=True)

df.isna().any()
df.dropna(inplace=True)
df.to_csv('tracks_assignment.csv', sep='\t', encoding='utf-8')
print df.shape
df.head()
 ```
 
 </p>
</details>

 ```python
 (1533, 16)
  ```
## Explanatory data analysis

### Audio fetaures exploration

We will start describing the dataframe and analysis of popularity score. <br>
As per documentation, the score value ranges from 0 (least popular) to 100 (most popular) and primarily based on number of plays the track achieved. 

How does popularity score relate to other audio features? Is there any correlation between all variables?<br>

Next we will visualize paired correlation between numerical variables.


<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- with seaborn library display pairplots; add density plots on main diagonal 

"""

scatter = sns.pairplot(df[['popularity_score', 'acousticness', 'danceability', 
                           'duration_ms', 'energy', 'instrumentalness']], diag_kind="kde")
 ```
 
 </p>
</details>

![LSTM]({{ 'spotify_test/scatter_1.png' | absolute_url }})

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- include popularity score in list to visulaize correlation between rest of variables 

"""
scatter2 = sns.pairplot(df[['popularity_score',
                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence']], diag_kind="kde")
 ```
 
 </p>
</details>

![LSTM]({{ 'spotify_test/scatter_2.png' | absolute_url }})


We see few features show high correlation on pairplots. Then, form a heatmap to summarize the correlation ratio.


#!/usr/bin/env python
# coding: utf-8

# In[1]:


Using TikTok API by David Teather `pip3 install TikTokApi`


# In[5]:


from TikTokApi import TikTokApi
api = TikTokApi.get_instance()
results = 10

# Since TikTok changed their API you need to use the custom_verifyFp option. 
# In your web browser you will need to go to TikTok, Log in and get the s_v_web_id value.
trending = api.by_trending(count=results, custom_verifyFp="")

for tiktok in trending:
    # Prints the id of the tiktok
    print(tiktok['id'])

print(len(trending))


# In[4]:


from TikTokApi import TikTokApi
api = TikTokApi()
n_videos = 100
username = 'washingtonpost'
user_videos = api.byUsername(username, count=n_videos)


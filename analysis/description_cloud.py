# Created by Victoria Zhang at 10/27/2022
# File: description_cloud.py
# Description: generate word cloud for each motif
# Scenario:
# Usage:
#%%

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#%%
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import glob2 as glob
#%%
project_name = 'BD20-Jun5-2022'
n_cluster = 10
model_name = 'VAME'
control_videos = [
    "subject1",
    "subject3",
    "subject4",
    "subject5",
    "subject6",
    "subject7",
    "subject8",
    "subject9",
    "subject10",
    "subject11",
    "subject13",
    "subject14",
    "subject15",
    "subject17",
    "subject18",
    "subject19",
    "subject21",
    "subject22",
    "subject23",
    "subject24",
    "subject25",
    "subject27",
    "subject28",
    "subject41",
    "subject42"
]

BD_videos      = [
'subject26',
'subject33',
'subject35',
'subject36',
'subject37',
'subject39',
'subject44',
'subject2',
'subject12',
'subject16',
'subject20',
'subject29',
'subject30',
'subject32',
'subject34',
'subject38',
'subject45',
'subject46',
'subject47',
'subject48',
'subject49',
'subject50',
'subject31',
'subject43',
'subject40'
]

#%%
pwd = ''
for k in range(n_cluster):
    texts = " "
    nouns = " "
    verbs = " "
    prepositions = " "

    count = 0
    count_semicolons = 0
    count_ands = 0
    count_then = 0
    for j, videos in enumerate([control_videos, BD_videos]):
        n = 0
        for i in range(len(videos)):
            v = videos[i]

            description_pth = r'C:\Users\zhanq\OneDrive - UC San Diego\SURF\{}'.format(v)
            if os.path.exists(description_pth):
                description = description_pth + '\*.csv'
                for file in glob.glob(description):
                    df = pd.read_csv(file, skiprows=lambda x: x%2 == 0, header=None, keep_default_na=False)
                    for text in df.iloc[:, 0]:
                        if len(text):
                            token = nltk.word_tokenize("people " + text)
                            pos_tagged = nltk.pos_tag(token)

                            noun = list(filter(lambda x: x[1] == 'NN' or x[1] == 'NNS', pos_tagged))
                            if noun: noun = list(zip(*noun))[0]
                            verb = list(filter(lambda x: x[1] == 'VB' or x[1] == 'VBP' or x[1] == 'JJ' or x[1] == 'VBZ', pos_tagged))
                            if verb: verb = list(zip(*verb))[0]
                            preposition = list(filter(lambda x: x[1] == 'ADP' or x[1] == 'TO' or x[1] == 'IN' or x[1] == 'PRT', pos_tagged))
                            if preposition: preposition = list(zip(*preposition))[0]

                            while_and = len(list(filter(lambda x: x[1] == 'CC', pos_tagged)))
                            semicolons = text.count(";")
                            thens = text.count("then")

                            noun = ' '.join(noun) + ' '
                            verb = ' '.join(verb) + ' '
                            preposition = ' '.join(preposition) + ' '

                            texts += text
                            nouns += noun
                            verbs += verb
                            prepositions += preposition
                            count_ands += while_and
                            count_semicolons += semicolons
                            count_then += thens
    to_remove = ['people', 'room', 'middle', 'left','right','upper', 'bottom','top','top_right', 'top_left', 'bottom_right', 'bottom_left']
    to_remove_2 = ['stand', 'wander','interact']
    for word in to_remove:
        if word in nouns: nouns = nouns.replace(word, '')
        if word in texts: texts = texts.replace(word, '')
        if word in verbs: verbs = verbs.replace(word, '')
        if word in prepositions: prepositions = prepositions.replace(word, '')

    texts2 = texts
    verbs2 = verbs
    nouns2 = nouns
    prepositions2 = prepositions
    for word in to_remove_2:
        if word in texts2: texts2 = texts2.replace(word, '')
        if word in verbs2: verbs2 = verbs2.replace(word, '')


    word_cloud_texts = WordCloud(collocations=False, background_color='white').generate(texts)
    word_cloud_nouns = WordCloud(collocations=False, background_color='white').generate(nouns)
    word_cloud_verbs = WordCloud(collocations=False, background_color='white').generate(verbs)
    word_cloud_prepositions = WordCloud(collocations=False, background_color='white').generate(prepositions)

    word_cloud_texts2 = WordCloud(collocations=False, background_color='white').generate(texts2)
    word_cloud_nouns2 = WordCloud(collocations=False, background_color='white').generate(nouns2)
    word_cloud_verbs2 = WordCloud(collocations=False, background_color='white').generate(verbs2)
    word_cloud_prepositions2 = WordCloud(collocations=False, background_color='white').generate(prepositions2)
    concurrent_pose_freq = count_ands + count_semicolons

    fig, axs = plt.subplots(1,4, figsize=(12,3))
    objective = ['texts', 'nouns', 'verbs', 'prepositions']
    for i, ax in enumerate(axs):
        word_cloud = eval('word_cloud_{}'.format(objective[i]))
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(objective[i])
    fig.suptitle("Motif {} with {} concurrent and {} transition poses".format(k, concurrent_pose_freq, count_then))
    fig.show()
    pwd = r'C:\Users\zhanq\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\wordcloud'
    fname = "Motif-{}-word-cloud.png".format(k)
    fig.savefig(os.path.join(pwd, fname))

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    objective = ['texts', 'nouns', 'verbs', 'prepositions']
    for i, ax in enumerate(axs):
        word_cloud = eval('word_cloud_{}2'.format(objective[i]))
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(objective[i] + " shorten")
    fig.suptitle("Motif {} with {} concurrent and {} transition poses".format(k, concurrent_pose_freq, count_then))
    fig.show()
    pwd = r'C:\Users\zhanq\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\wordcloud'
    fname = "Motif-{}-shorten-word-cloud.png".format(k)
    fig.savefig(os.path.join(pwd, fname))
plt.close('all')
#%%



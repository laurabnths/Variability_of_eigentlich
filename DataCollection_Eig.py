'Python file containing the code used for the data collection of "eigentlich"'

# A2 = word two before eigentlich
# A1 = word one before eigentlich
# C1 = word one after eigentlich
# C2 = word two after eigentlich


from praatio import tgio
import pandas as pd
import re
import os
import numpy as np
import nltk
from nltk.util import ngrams
from collections import Counter, defaultdict
import pandas
from ipapy import UNICODE_TO_IPA
import pyphen
from scipy.stats import entropy
from math import log2

filenames = os.listdir("/Users/laurabenthaus/Desktop/KEC/Korrigiert")
nl = []

for filename in filenames:
  try:
    tg = tgio.openTextgrid("/Users/laurabenthaus/Desktop/KEC/Korrigiert/{filename}".format(filename=filename))
    tierWord = tg.tierDict['words']
    tierSegm = tg.tierDict['segments']
    tierCanonIPA = tg.tierDict['canonIPA']
    tierMorph = tg.tierDict['morphTag']

    wordtier_array = np.array(tierWord.entryList)
    segmtier_array = np.array(tierSegm.entryList)
    canonIPAtier_array = np.array(tierCanonIPA.entryList)
    morphtier_array = np.array(tierMorph.entryList)

    word_dataframe = pd.DataFrame(wordtier_array, columns=['start', 'stop', 'word'])  # dataframe of word tier
    segment_dataframe = pd.DataFrame(segmtier_array, columns=['start', 'stop', 'segment'])  # dataframe of segment tier
    canonIPA_dataframe = pd.DataFrame(canonIPAtier_array, columns=['start', 'stop', 'canonIPA'])  # dataframe of canonIPA tier
    morphTag_dataframe = pd.DataFrame(morphtier_array, columns=['start', 'stop', 'morphTag'])  # dataframe of morphTag tier

    word_column = list(word_dataframe['word'])
    canon_column = canonIPA_dataframe['canonIPA']
    morphTag_column = morphTag_dataframe['morphTag']

    word_df1 = pd.concat([word_dataframe, canon_column], axis=1)
    word_df2 = pd.concat([word_df1, morphTag_column], axis=1)

    startIndices = []
    for start in word_df2['start']:
      startIndex = segment_dataframe.index[segment_dataframe["start"] == start].tolist()[0]
      startIndices.append(startIndex)

    segment = []
    segments = []
    for i, row in enumerate(segment_dataframe.values):
      startTime = row
      subsegment = row
      segment.append(subsegment)
      if (i + 1 in startIndices or i + 1 == len(segment_dataframe)):
        elem = [ele[-1] for ele in segment]
        segments.append("".join(elem))
        segment = []

    full_df_pro_file = word_df2.assign(segment=segments)

    full_df_pro_file['filename'] = filename

    segm_list = []
    segms_list = []
    segms_start_list = []
    segms_stop_list = []
    for p, r in enumerate(segment_dataframe.values):
        startTime_segm = r
        subsegment_segm = r
        segm_list.append(subsegment_segm)
        if (p + 1 in startIndices or p + 1 == len(segment_dataframe)):
            elem1 = [ele1[-1] for ele1 in segm_list]
            elem2 = [round(float(ele2[0]), 3) for ele2 in segm_list]  # check if this works?
            elem3 = [round(float(ele3[1]), 3) for ele3 in segm_list]  # check if this works?
            segms_list.append("_".join(elem1))
            segms_start_list.append("_".join([str(il2) for il2 in elem2]))
            segms_stop_list.append("_".join([str(il3) for il3 in elem3]))
            segm_list = []

    full_df_pro_file['phon_segm'] = segms_list
    full_df_pro_file['phon_start_segm'] = segms_start_list
    full_df_pro_file['phon_stop_segm'] = segms_stop_list

    word_number = []
    for i, e in enumerate(full_df_pro_file['word'], start=1):
        word_number.append(i)

    full_df_pro_file['Word_Number_in_TG'] = word_number

    nl.append(full_df_pro_file)

  except:
      pass

dtf = pd.concat(nl, ignore_index=True) # complete dataframe of KEC

dtf['word'] = dtf.word.str.replace('[', '', regex=True) # words in row without []
dtf['word'] = dtf.word.str.replace(']', '', regex=True)
dtf['segment'] = dtf.segment.str.replace('[', '', regex=True)
dtf['segment'] = dtf.segment.str.replace(']', '', regex=True)
dtf['segment'] = dtf.segment.str.replace('#', '', regex=True)
dtf['segment'] = dtf.segment.str.replace('_p:_', '', regex=True) # segment column without _p:_
dtf['segment'] = dtf.segment.str.replace('_NOISE_', '', regex=True) # segment column without _NOISE_

dtf.loc[dtf["segment"] == '','segment'] = dtf["canonIPA"] # empty segment rows (because of previous _p:_ deletion changed

########################################################################################################################
word_counts = dtf['word'].value_counts() # count of each word in df
word_list = dtf["word"].tolist()
all_counts = len(word_list)

prob_each_word = word_counts / all_counts
prob_eigentlich = prob_each_word['eigentlich'] # P(B)

bigrams_words = list(nltk.bigrams(word_list))
bigram_count_words = Counter(bigrams_words)
sum_bigrams = len(bigrams_words)

trigrams_words = list(nltk.trigrams(word_list))
trigram_count_words = Counter(trigrams_words)
########################################################################################################################
segments_list = dtf['segment'].tolist()
segments_count = dtf['segment'].value_counts()
prob_each_segments = segments_count / all_counts
bigrams_segments = list(nltk.bigrams(segments_list))
bigram_count_segments = Counter(bigrams_segments)
trigrams_segments = list(nltk.trigrams(segments_list))
trigram_count_segments = Counter(trigrams_segments)

all_wordsegment = list(enumerate(list(zip(word_list, segments_list))))
all_bigrams_wordssegments = list(zip(bigrams_words, bigrams_segments))
count_bi_wordssegm = Counter(all_bigrams_wordssegments)
all_trigrams_wordssegments = list(zip(trigrams_words, trigrams_segments))
count_tri_wordssegm = Counter(all_trigrams_wordssegments)

########################################################################################################################
davor_list = []
danach_list = []
for tupl in bigrams_words:
    if tupl[0] == "eigentlich":
      danach_list.append(tupl[1])
for tupl1 in bigrams_words:
    if tupl1[1] == "eigentlich":
        davor_list.append(tupl1[0])

Freq_Davor = Counter(davor_list) # // print(Freq_Davor.most_common)
Freq_Danach = Counter(danach_list)


davor2_list = []
danach2_list = []
for wx in trigrams_words:
    if wx[2] == "eigentlich":
        davor2_list.append(wx[0])
for wx1 in trigrams_words:
    if wx1[0] == "eigentlich":
        danach2_list.append(wx1[2])

Freq_Davor2 = Counter(davor2_list)
Freq_Danach2 = Counter(danach2_list)

############################################################
newbigr_davor = [] # list with bigrams of eigentlich variation
newbigr_danach = [] # list with bigrams of eigentlich variation
for ha in all_bigrams_wordssegments:
    if ha[0][0] == "eigentlich":
        newbigr_danach.append((ha[1][0], ha[0][1]))
    if ha[0][1] == "eigentlich":
        newbigr_davor.append((ha[0][0], ha[1][1]))

Count_bigr_davor = Counter(newbigr_davor)
Count_bigr_danach = Counter(newbigr_danach)

newtrigr_davor = [] # list with trigrams of eigentlich variation
newtrigr_davordanach = [] # list with trigrams of eigentlich variation
newtrigr_danach = [] # list with trigrams of eigentlich variation
for h in all_trigrams_wordssegments:
    if h[0][2] == "eigentlich":
        newtrigr_davor.append((h[0][0], h[0][1], h[1][2]))
    if h[0][1] == "eigentlich":
        newtrigr_davordanach.append((h[0][0], h[1][1], h[0][2]))
    if h[0][0] == "eigentlich":
        newtrigr_danach.append((h[1][0], h[0][1], h[0][2]))

Count_trigr_davor = Counter(newtrigr_davor)
Count_trigr_davordanach = Counter(newtrigr_davordanach)
Count_trigr_danach = Counter(newtrigr_danach)

############################################################
idx_list_eig = dtf.index[dtf['word']=='eigentlich'].tolist()  # list of eigentlich-index

# dataframes for all eigentlichs and two preceding and following words
dtf_davor_2 = dtf.iloc[[(idx - 2) for idx in idx_list_eig]]
dtf_davor_1 = dtf.iloc[[(idx - 1) for idx in idx_list_eig]]
dtf_eig = dtf.iloc[[idx for idx in idx_list_eig]]
dtf_danach_1 = dtf.iloc[[(idx + 1) for idx in idx_list_eig]]
dtf_danach_2 = dtf.iloc[[(idx + 2) for idx in idx_list_eig]]

############################################################
# counts of preceding and following words (eig-2 and eig+2)
dict_cminus2 = {}
for w in dtf_davor_2['word']:
    for x, y in word_counts.items():
        if w == x:
            dict_cminus2[w] = y


dict_cplus2 = {}
for w1 in dtf_danach_2['word']:
    for x1, y1 in word_counts.items():
        if w1 == x1:
            dict_cplus2[w1] = y1


######################################################################################################
# Lists for CSV file
TextGrid = dtf_eig['filename'].tolist()
word = dtf_eig['word'].tolist()
CanonIPA_list = dtf_eig['canonIPA'].tolist()
morpgTag_eig = dtf_eig['morphTag'].tolist()
segments = dtf_eig['segment'].tolist()
word_start = dtf_eig['start'].tolist()
word_stop = dtf_eig['stop'].tolist()
word_davor_2 = dtf_davor_2['word'].tolist()
IPA_davor2 = dtf_davor_2['canonIPA'].tolist()
morphTag_davor2 = dtf_davor_2['morphTag'].tolist()
word_davor_1 = dtf_davor_1['word'].tolist()
IPA_davor1 = dtf_davor_1['canonIPA'].tolist()
morphTag_davor1 = dtf_davor_1['morphTag'].tolist()
word_danach_1 = dtf_danach_1['word'].tolist()
IPA_danach1 = dtf_danach_1['canonIPA'].tolist()
morphTag_danach1 = dtf_danach_1['morphTag'].tolist()
word_danach_2 = dtf_danach_2['word'].tolist()
IPA_danach2 = dtf_danach_2['canonIPA'].tolist()
morphTag_danach2 = dtf_danach_2['morphTag'].tolist()

word_count_davor_2 = []
for wrt in word_davor_2:
    word_count_davor_2.append(dict_cminus2[wrt]) #Frequency of words two before eigentlich

word_count_davor_1 = [] #Frequency of words before eigentlich
for wrt in word_davor_1:
    word_count_davor_1.append(word_counts[wrt])

word_count_danach_1 = [] # Frequency of words after eigentlich
for wt in word_danach_1:
    word_count_danach_1.append(word_counts[wt])

word_count_danach_2 = []
for wt in word_danach_2:
    word_count_danach_2.append(dict_cplus2[wt]) #Frequency of words two after eigentlich


#################################### phoneme count and speech duration
Count_phonemes_eig = [len(s) for s in segments]
len_CanonIPA = [len(t) for t in CanonIPA_list]

len_difference = [] # difference between canonical form and variation
zip_object = zip(len_CanonIPA, Count_phonemes_eig)
for list1_i, list2_i in zip_object:
    len_difference.append(list1_i-list2_i)


speak_duration = [] # duration of saying "eigentlich"
zip_object_duration = zip(word_stop, word_start)
for list1_t, list2_t in zip_object_duration:
    speak_duration.append(float(list1_t)-float(list2_t))

######################################################################

dtf_segm_column = list(dtf_eig['segment'])
cnt_segm = Counter(dtf_segm_column) # in total 168 different variations
segm_count_list = []  # list with how often this type of segment occurs
for l in segments:
    for i, p in cnt_segm.items():
        if l == i:
            segm_count_list.append(p)

# Lists for csv file:
phon_segm = list(dtf_eig['phon_segm'])
phon_segm_start = list(dtf_eig['phon_start_segm'])
phon_segm_stop = list(dtf_eig['phon_stop_segm'])
WordNumber_in_TG = list(dtf_eig['Word_Number_in_TG'])

#################################### bigram counts with eig variation ############################################
cnt_bigr_davorlist = [] # list with bigram count (with eigentlich variation)
for davor in newbigr_davor:
    for big, cnt in Count_bigr_davor.items():
        if davor == big:
            cnt_bigr_davorlist.append(cnt)

cnt_bigr_danachlist = [] # list with bigram count (with eigentlich variation)
for danach in newbigr_danach:
    for big1, cnt1 in Count_bigr_danach.items():
        if danach == big1:
            cnt_bigr_danachlist.append(cnt1)

####################### cond probabilities of bigrams  ################################
eigentlich_bigr_AB = []
for tupl3 in bigrams_words:
    if tupl3[1] == "eigentlich":
        eigentlich_bigr_AB.append(tupl3)

cond_probs = {} # dict with condprobs: P(B|A)
for dm in eigentlich_bigr_AB:
    cond_probs[dm] = bigram_count_words[dm] / word_counts[dm[0]]

copro_AB = [] # list with cond probs of P(B|A)
for hs in newbigr_davor:
    for hs1, hs2 in cond_probs.items():
        if hs[0] == hs1[0]:
            copro_AB.append(hs2)


eigentlich_bigr_BC = []
for tupl4 in bigrams_words:
    if tupl4[0] == "eigentlich":
        eigentlich_bigr_BC.append(tupl4)

inv_cond_probs = {} # dict with inv condprobs: P(B|C)
for dm2 in eigentlich_bigr_BC:
    inv_cond_probs[dm2] = bigram_count_words[dm2] / word_counts[dm2[1]]

inv_copro_BC = [] # CHECK
for hs3 in newbigr_danach:
    for hs4, hs5 in inv_cond_probs.items():
        if hs3[1] == hs4[1]:
            inv_copro_BC.append(hs5)

####################### conditional probabilities of bigrams with eigentlich variations ################################

cond_probs_variation = {} # P(B|A)
for tupl5, tupl6 in Count_bigr_davor.items():
    cond_probs_variation[tupl5] = tupl6 / word_counts[tupl5[0]]

copro_AB_variation = [] # list with cond probs with variation
for bigi in newbigr_davor:
    for bigi1, bigi2 in cond_probs_variation.items():
        if bigi == bigi1:
            copro_AB_variation.append(bigi2)


inv_cond_probs_variation = {} # P(B|C)
for tupl7, tupl8 in Count_bigr_danach.items():
    inv_cond_probs_variation[tupl7] = tupl8 / word_counts[tupl7[1]]

inv_copro_BC_variation = [] # list with inv cond pros with variation
for bigi3 in newbigr_danach:
    for bigi4, bigi5 in inv_cond_probs_variation.items():
        if bigi3 == bigi4:
            inv_copro_BC_variation.append(bigi5)

################################### fivegrams ###########################################################

def find_ngrams(word_list, n):   # definition für n-grams
  return zip(*[word_list[i:] for i in range(n)])

fivegrams_words = list(find_ngrams(word_list, 5)) # list with all fivegrams with words
fivegrams_segments = list(find_ngrams(segments_list, 5))
all_fivegrams_wordssegments = list(zip(fivegrams_words, fivegrams_segments))


fg_eig = []   # list with eigentlich-fivegrams
for f in all_fivegrams_wordssegments:
    if f[0][2] == "eigentlich":
        fg_eig.append((f[0][0], f[0][1], f[1][2], f[0][3], f[0][4]))


################################## speech duration for fivegrams ##################################

start_wordA2 = dtf_davor_2['start'].tolist()
stop_wordC2 = dtf_danach_2['stop'].tolist()

speak_duration_A2toC2 = [] # duration of "eigentlich" fivegram
zip_A2C2_duration = zip(stop_wordC2, start_wordA2)
for liststop_t, liststart_t in zip_A2C2_duration:
    speak_duration_A2toC2.append(float(liststop_t)-float(liststart_t))


###################### speech rate ##################################################################
fg_eig_words = [] # List with eigentlich-fivegrams with normal words
for f1 in fivegrams_words:
    if f1[2] == "eigentlich":
        fg_eig_words.append(f1)


fivegram_syllable_count = dict(zip(fg_eig_words, speak_duration_A2toC2))
fi = pd.DataFrame([{"fivegram": fg, "duration": du} for fg, du in zip(fg_eig_words, speak_duration_A2toC2)])
fg_du_dtf = pd.DataFrame([{"fivegram": fg, "duration": du} for fg, du in fivegram_syllable_count.items()])

fg_du_dtf['syllable_count'] = ""
fg_du_dtf.to_csv("fivegram_syllables.csv", sep=',', index=True)

fiveg_dtf = pd.read_csv('fivegram_syllables_newnew.csv', sep=';')
fg_dur_list = fiveg_dtf['duration'].tolist()
fg_syl_cnt_list = fiveg_dtf['syllable_count'].tolist()
fiveg = fiveg_dtf['fivegram'].tolist()


speech_rate = [float(b) / float(m) for b, m in zip(fg_syl_cnt_list, fg_dur_list)] # list with fivegram speech rate

##################################### syllable count for variations ############################################

cnt_df = pd.DataFrame([{"variation": variation, "count": count} for variation, count in cnt_segm.items()])
cnt_df["syllable_count"] = ""

cnt_df.to_csv("variation_syllables.csv", sep=',', index=True)
cnt_dtf_new = pd.read_csv('variation_syllables_NEW.csv', sep=';') # dataframe with variation's syllable count
variation_list = cnt_dtf_new['variation'].tolist()
syllable_cnt_list = cnt_dtf_new['syllable_count'].tolist()

vari_syl_dict = dict(zip(variation_list, syllable_cnt_list)) # dict with variation and syllable count

syllable_count = []
for variant in segments:
    for t, p in vari_syl_dict.items():
        if variant == t:
            syllable_count.append(p)

############################## trigram counts lists ########################################################

Count_trigr_danach = Counter(newtrigr_danach)

tri_count_AAB = []
for bt in newtrigr_davor:
    for bt1, bt2 in Count_trigr_davor.items():
        if bt == bt1:
            tri_count_AAB.append(bt2)

tri_count_ABC = []
for bt3 in newtrigr_davordanach:
    for bt4, bt5 in Count_trigr_davordanach.items():
        if bt3 == bt4:
            tri_count_ABC.append(bt5)

tri_count_BCC = []
for bt6 in newtrigr_danach:
    for bt7, bt8 in Count_trigr_danach.items():
        if bt6 == bt7:
            tri_count_BCC.append(bt8)

###################################################
# Probability that word before and after eigentlich is <P>
count = word_davor_1.count('<P>')
count2 = word_danach_1.count('<P>')

prob = count / len(word_davor_1) # 0.09099350046425256
prob2 = count2 / len(word_danach_1) # 0.15598885793871867

################ Entropy calculations ###################################################################################
keyslist = list(cnt_segm.keys())
valueslist = list(cnt_segm.values())
word = dtf_eig['word'].tolist()

relfreq_list = []
for element in valueslist:
    relfreq_list.append(element / len(word))

# calculate amount of information in each variation probability
info = [-log2(p) for p in relfreq_list]
# entropy() takes two entries: the list of probabilities and the logarithmic base
variant_entropy = entropy(relfreq_list, qk=None, base=2)

eig_entropy = [variant_entropy for en in segments]

#### calculate information gain for wach word --> SURPRISAL
info_dict = prob_each_word.copy()
info_dict = info_dict.to_dict()
info_each_word = {} # dict with amount of information carried by each word in corpus, as done in Millin et al. (2009)
for key, value in info_dict.items():
    info_each_word[key] = -log2(value)

bef_info = [info_each_word[i] for i in word_davor_1]
aft_info = [info_each_word[i1] for i1 in word_danach_1]


###########################################################################################################
newbigr_davordict = {} # dict with word before eigentlich and eigentlich variation, and counts
newbigr_danachdict = {} # dict with word after eigentlich and eigentlich variation, and counts
for hallo, hallo1 in count_bi_wordssegm.items():
    if hallo[0][0] == "eigentlich":
        newbigr_danachdict[(hallo[1][0], hallo[0][1])] = hallo1
    if hallo[0][1] == "eigentlich":
        newbigr_davordict[(hallo[0][0], hallo[1][1])] = hallo1

for a, b in newbigr_danachdict.items(): # e.g., P(ainic|sowieso) = c(ainic, sowieso) / c(sowieso)
    newbigr_danachdict[a] = b / word_counts[a[1]]

for a1, b1 in newbigr_davordict.items(): # e.g., P(ainic | so) = c(so, ainic) / c(so)
    newbigr_davordict[a1] = b1 / word_counts[a1[0]]

######################## Calculating Entropy Paradigm of before word ################################################
word_davor_count_dict = {} # --> dict with absoluter Frequenz von Wörter vor eigentlich
for wort in word_davor_1:
    word_davor_count_dict[wort] = word_counts[wort]

davor = word_davor_count_dict.keys()

#List of bigrams with X (of "X, eigentlich, Y")
biliX = [x for x in bigrams_words if x[0] in word_davor_1]

a = {} # --> dict with key = davor words of eigentlich; value = list with all bigrams of davor word
for element in davor:
    a.setdefault(element, [])

for ab, bc in a.items():
    for bigi in biliX:
        if ab == bigi[0]:
            a[ab].append(bigi)

# Adding the bigram counts to dicts
for x, y in a.items():
    a[x] = Counter(y)

a1 = a.copy()
for ab1, bc1 in a1.items():
    a1[ab1] = [i for i in bc1.values()]

# adding probabilities of each bigram for each dict key (before word)
for elemento, elemento1 in a1.items():
    a1[elemento] = [s / word_counts[elemento] for s in elemento1]

a1probslist = [a1[f] for f in word_davor_1] # list with probs for before word

# dict with entropy paradigm of before word
davor_entropy = a1.copy()
for hf, hf1 in davor_entropy.items():
    davor_entropy[hf] = entropy(hf1, qk=None, base=2)

davor_entropy_list = [davor_entropy[st] for st in word_davor_1] # entropy list for csv


######################## Calculating Entropy Paradigm of after word
word_danach_count_dict = {} # --> dict with absolute Frequency of words after eigentlich
for wort1 in word_danach_1:
    word_danach_count_dict[wort1] = word_counts[wort1]

danach = word_danach_count_dict.keys()

# List of bigrams with Y (of "X, eigentlich, Y")
biliY = [y for y in bigrams_words if y[1] in word_danach_1]

b = {} # --> dict with key = danach words of eigentlich; value = list with all bigrams of danach word
for element2 in danach:
    b.setdefault(element2, [])

for ab2, bc2 in b.items():
    for bigi2 in biliY:
        if ab2 == bigi2[1]:
            b[ab2].append(bigi2)

# Adding the bigram counts to dicts
for x1, y1 in b.items():
    b[x1] = Counter(y1)

b1 = b.copy()
for bye, bye1 in b1.items():
    b1[bye] = [i1 for i1 in bye1.values()]

# adding probabilities of each bigram for each dict key (after word)
for elemento2, elemento3 in b1.items():
    b1[elemento2] = [s1 / word_counts[elemento2] for s1 in elemento3]

b1probslist = [b1[g] for g in word_danach_1] # list with cond probs for after word

# dict with entropy paradigm of after word
danach_entropy = b1.copy()
for hf2, hf3 in danach_entropy.items():
    danach_entropy[hf2] = entropy(hf3, qk=None, base=2)

after_entropy_list = [danach_entropy[st2] for st2 in word_danach_1] # entropy list for csv file

###################### IPA Last and First segment ###############################################
segment_davor_1 = dtf_davor_1['segment'].tolist()
segment_danach_1 = dtf_danach_1['segment'].tolist()

extra_exceptions = ['NA', 'NANA', 'NANANA', '<P>', '<HÄSITATION>', '<ATMEN>', '<UNVERSTÄNDLICH>', '<LACHEN>', '<RÄUSPER>']
suprasegmental = ['ː']

lp_davor = [] # last segment of davor word !!!
for segm1 in segment_davor_1:
    if segm1 in extra_exceptions:
        lp_davor.append(segm1)
    elif segm1[-1] in suprasegmental:
        lp_davor.append(segm1[-2])
    else:
        lp_davor.append(segm1[-1])


cnt_lastphon = Counter(lp_davor) # Count(last segment, eigentlich)

count_lastphon = [] # count of last segment of before-word depending on eigentlich (but not variation); list for csv
for segi in lp_davor:
    count_lastphon.append(cnt_lastphon[segi])

########
lp_danach = [] # first segment of danach word
for segm2 in segment_danach_1:
    if segm2 in extra_exceptions:
        lp_danach.append(segm2)
    elif segm2[0] in suprasegmental:
        lp_danach.append(segm2[1])
    else:
        lp_danach.append(segm2[0])

cnt_firstphon = Counter(lp_danach)

count_firstphon = [] # count of last segment of before-word depending on eigentlich (not variation); list for csv
for segi2 in lp_danach:
    count_firstphon.append(cnt_firstphon[segi2])


########################## last and first segments of all words ##########################
last_segm_all = [] # last segments for all words
for parola in segments_list:
    if parola in extra_exceptions:
        last_segm_all.append(parola)
    elif parola[-1] in suprasegmental:
        last_segm_all.append(parola[-2])
    else:
        last_segm_all.append(parola[-1])

first_segm_all = [] # first segments for all words
for parola2 in segments_list:
    if parola2 in extra_exceptions:
        first_segm_all.append(parola2)
    elif parola2[0] in suprasegmental:
        first_segm_all.append(parola2[1])
    else:
        first_segm_all.append(parola2[0])

cnt_all_last_segm = Counter(last_segm_all)
cnt_all_first_segm = Counter(first_segm_all)


################################################################################
################################################################################
### bigrams with segments
seg_bigram_davor = [] # list with word before eigentlich and eigentlich variation
seg_bigram_danach = [] # list with word after eigentlich and eigentlich variation
for hal1 in all_bigrams_wordssegments:
    if hal1[0][0] == "eigentlich":
        seg_bigram_danach.append((hal1[1][0], hal1[1][1]))
    if hal1[0][1] == "eigentlich":
        seg_bigram_davor.append((hal1[1][0], hal1[1][1]))


newlist1 = [] # list with last segments before eigentlich variation !!
for hy in seg_bigram_davor:
    if hy[0] in extra_exceptions:
        newlist1.append((hy[0], hy[1]))
    elif hy[0][-1] in suprasegmental:
        newlist1.append((hy[0][-2], hy[1]))
    else:
        newlist1.append((hy[0][-1], hy[1]))
cnt_bigr_lastphon_davor = Counter(newlist1)

if_beforesegm_in_Vari = [] # list TRUE/FALSE before segment in variation
for dv in newlist1:
    if dv[0] in dv[1]:
        if_beforesegm_in_Vari.append(True)
    else:
        if_beforesegm_in_Vari.append(False)


newlist2 = [] # list with first segment after eigentlich variation !!
for hysh in seg_bigram_danach:
    if hysh[1] in extra_exceptions:
        newlist2.append((hysh[0], hysh[1]))
    elif hysh[1][0] in suprasegmental:
        newlist2.append((hysh[0], hysh[1][1]))
    else:
        newlist2.append((hysh[0], hysh[1][0]))
cnt_bigr_firstphon_danach = Counter(newlist2)

if_aftersegm_in_Vari = [] # list TRUE/FALSE
for dv1 in newlist2:
    if dv1[1] in dv1[0]:
        if_aftersegm_in_Vari.append(True)
    else:
        if_aftersegm_in_Vari.append(False)

########################################################
'Lists for horizontal position of vowels'
front_vowel = ['a', 'i', 'e', 'ø', 'ɪ', 'ɛ', 'y', 'ʏ']
central_vowel = ['ɐ', 'ə']
back_vowel = ['o', 'u', 'ɔ', 'ʊ']

'Lists for phonation'
voiced = ['n', 'r', 'l', 'm', 'ŋ', 'd', 'b', 'g', 'v', 'z', 'j', 'w', 'a', 'i', 'e', 'ø', 'ɪ', 'ɛ', 'y', 'ʏ', 'ɐ', 'ə', 'o', 'u', 'ɔ', 'ʊ']
voiceless = ['s', 't', 'ç', 'x', 'ʃ', 'p', 'h', 'f', 'k', 'c']

'Lists for consonants'
plosives = ['t', 'p', 'd', 'k', 'b', 'g', 'c']
nasals = ['n', 'm', 'ŋ']
fricatives = ['s', 'ç', 'x', 'ʃ', 'h', 'f', 'v', 'z']
laterals = ['l']
trill = ['r']
approximants = ['j', 'w']

########################################################

########################################################
# 1. count: how many times does the preceding element occur in variation
count_davorsegm_in_vari = dict()
for a, b in cnt_bigr_lastphon_davor.items():
    if a[0] in a[1]:
        count_davorsegm_in_vari[a] = a[1].count(a[0])
    else:
        count_davorsegm_in_vari[a] = 0


newdict = dict() # dict with counts of how many times it occurs at least once
for a1, b1 in count_davorsegm_in_vari.items():
    if b1 >= 1:
        newdict[a1] = b1

# prob that before-segment occurs at least once in variation
prob_before_segm_in_vari = len(newdict) / len(count_davorsegm_in_vari) # 0.2425531914893617

###################
# 2. count: how many times does the following element occur in variation
count_danachsegm_in_vari = dict()
for a2, b2 in cnt_bigr_firstphon_danach.items():
    if a2[1] in a2[0]:
        count_danachsegm_in_vari[a2] = a2[0].count(a2[1])
    else:
        count_danachsegm_in_vari[a2] = 0


newdict1 = dict() # dict with counts of how many times it occurs at least once
for a3, b3 in count_danachsegm_in_vari.items():
    if b3 >= 1:
        newdict1[a3] = b3

newdict2 = dict() # dict with count more than once
for a4, b4 in count_danachsegm_in_vari.items():
    if b4 > 1:
        newdict2[a4] = b4


# prob that after-segment occurs at least once in variation
prob_after_segm_in_vari = len(newdict1) / len(count_danachsegm_in_vari) # 0.20977596741344195

# prob that after-segment occurs more than once in variation (2 times)
prob_after_segm_in_vari2 = len(newdict2) / len(count_danachsegm_in_vari) # 0.008146639511201629

########################### probabilities of preceding segment ############################################
# prob that preceding segment is a front_vowel and in variation
count_davorfrontvowel_in_vari = dict()
for c, d in cnt_bigr_lastphon_davor.items():
    if c[0] in c[1] and c[0] in front_vowel:
        count_davorfrontvowel_in_vari[c] = c[1].count(c[0])

prob_davorfrontvowel_in_vari = len(count_davorfrontvowel_in_vari) / len(count_davorsegm_in_vari) # 0.1

if_before_frontvowel = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in front_vowel:
        if_before_frontvowel.append(True)
    else:
        if_before_frontvowel.append(False)


# prob that preceding segment is a central_vowel and in variation
count_davorcentralvowel_in_vari = dict()
for c1, d1 in cnt_bigr_lastphon_davor.items():
    if c1[0] in c1[1] and c1[0] in central_vowel:
        count_davorcentralvowel_in_vari[c1] = c1[1].count(c1[0]) # NONE

if_before_centralvowel = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in central_vowel:
        if_before_centralvowel.append(True)
    else:
        if_before_centralvowel.append(False)

# prob that preceding segment is a back_vowel and in variation
count_davorbackvowel_in_vari = dict()
for c2, d2 in cnt_bigr_lastphon_davor.items():
    if c2[0] in c2[1] and c2[0] in back_vowel:
        count_davorbackvowel_in_vari[c2] = c2[1].count(c2[0]) # NONE

if_before_backvowel = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in back_vowel:
        if_before_backvowel.append(True)
    else:
        if_before_backvowel.append(False)

# prob that preceding segment is a voiced and in variation
count_davorvoiced_in_vari = dict()
for c3, d3 in cnt_bigr_lastphon_davor.items():
    if c3[0] in c3[1] and c3[0] in voiced:
        count_davorvoiced_in_vari[c3] = c3[1].count(c3[0])

prob_davorvoiced_in_vari = len(count_davorvoiced_in_vari) / len(count_davorsegm_in_vari) # 0.07234042553191489

if_before_voiced = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in voiced:
        if_before_voiced.append(True)
    else:
        if_before_voiced.append(False)

# prob that preceding segment is a voiceless and in variation
count_davorvoiceless_in_vari = dict()
for c4, d4 in cnt_bigr_lastphon_davor.items():
    if c4[0] in c4[1] and c4[0] in voiceless:
        count_davorvoiceless_in_vari[c4] = c4[1].count(c4[0])

prob_davorvoiceless_in_vari = len(count_davorvoiceless_in_vari) / len(count_davorsegm_in_vari) # 0.07021276595744681

if_before_voiceless = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in voiceless:
        if_before_voiceless.append(True)
    else:
        if_before_voiceless.append(False)

# prob that preceding segment is a plosive and in variation
count_davorplosive_in_vari = dict()
for c5, d5 in cnt_bigr_lastphon_davor.items():
    if c5[0] in c5[1] and c5[0] in plosives:
        count_davorplosive_in_vari[c5] = c5[1].count(c5[0])

prob_davorplosive_in_vari = len(count_davorplosive_in_vari) / len(count_davorsegm_in_vari) # 0.00851063829787234

if_before_plosive = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in plosives:
        if_before_plosive.append(True)
    else:
        if_before_plosive.append(False)

# prob that preceding segment is a nasal and in variation
count_davornasal_in_vari = dict()
for c6, d6 in cnt_bigr_lastphon_davor.items():
    if c6[0] in c6[1] and c6[0] in nasals:
        count_davornasal_in_vari[c6] = c6[1].count(c6[0])

prob_davornasal_in_vari = len(count_davornasal_in_vari) / len(count_davorsegm_in_vari) # 0.0574468085106383

if_before_nasal = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in nasals:
        if_before_nasal.append(True)
    else:
        if_before_nasal.append(False)

# prob that preceding segment is a fricative and in variation
count_davorfricative_in_vari = dict()
for c7, d7 in cnt_bigr_lastphon_davor.items():
    if c7[0] in c7[1] and c7[0] in fricatives:
        count_davorfricative_in_vari[c7] = c7[1].count(c7[0])

prob_davorfricative_in_vari = len(count_davorfricative_in_vari) / len(count_davorsegm_in_vari) # 0.06170212765957447

if_before_fricative = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in fricatives:
        if_before_fricative.append(True)
    else:
        if_before_fricative.append(False)

# prob that preceding segment is a lateral and in variation
count_davorlateral_in_vari = dict()
for c8, d8 in cnt_bigr_lastphon_davor.items():
    if c8[0] in c8[1] and c8[0] in laterals:
        count_davorlateral_in_vari[c8] = c8[1].count(c8[0])

prob_davorlateral_in_vari = len(count_davorlateral_in_vari) / len(count_davorsegm_in_vari) # 0.014893617021276596

if_before_lateral = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in laterals:
        if_before_lateral.append(True)
    else:
        if_before_lateral.append(False)

# prob that preceding segment is a trill and in variation
count_davortrill_in_vari = dict()
for c9, d9 in cnt_bigr_lastphon_davor.items():
    if c9[0] in c9[1] and c9[0] in trill:
        count_davortrill_in_vari[c9] = c9[1].count(c9[0]) # NONE

if_before_trill = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in trill:
        if_before_trill.append(True)
    else:
        if_before_trill.append(False)

# prob that preceding segment is an approximant and in variation
count_davorappro_in_vari = dict()
for c10, d10 in cnt_bigr_lastphon_davor.items():
    if c10[0] in c10[1] and c10[0] in approximants:
        count_davorappro_in_vari[c10] = c10[1].count(c10[0]) # NONE

if_before_approximant = [] # list TRUE/FALSE
for dv2 in newlist1:
    if dv2[0] in dv2[1] and dv2[0] in approximants:
        if_before_approximant.append(True)
    else:
        if_before_approximant.append(False)

########################### probabilities of following segment ############################################

# prob that following segment is a front_vowel and in variation
count_danachfrontvowel_in_vari = dict()
for e, f in cnt_bigr_firstphon_danach.items():
    if e[1] in e[0] and e[1] in front_vowel:
        count_danachfrontvowel_in_vari[e] = e[0].count(e[1])

prob_danachfrontvowel_in_vari = len(count_danachfrontvowel_in_vari) / len(count_danachsegm_in_vari) # 0.14052953156822812

if_after_frontvowel = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in front_vowel:
        if_after_frontvowel.append(True)
    else:
        if_after_frontvowel.append(False)


# prob that following segment is a central_vowel and in variation
count_danachcentralvowel_in_vari = dict()
for e1, f1 in cnt_bigr_firstphon_danach.items():
    if e1[1] in e1[0] and e1[1] in front_vowel:
        count_danachcentralvowel_in_vari[e1] = e1[0].count(e1[1])

prob_danachcentralvowel_in_vari = len(count_danachcentralvowel_in_vari) / len(count_danachsegm_in_vari) # 0.14052953156822812

if_after_centralvowel = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in central_vowel:
        if_after_centralvowel.append(True)
    else:
        if_after_centralvowel.append(False)

# prob that following segment is a back_vowel and in variation
count_danachbackvowel_in_vari = dict()
for e2, f2 in cnt_bigr_firstphon_danach.items():
    if e2[1] in e2[0] and e2[1] in back_vowel:
        count_danachbackvowel_in_vari[e2] = e2[0].count(e2[1]) # NONE

if_after_backvowel = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in back_vowel:
        if_after_backvowel.append(True)
    else:
        if_after_backvowel.append(False)

# prob that following segment is a voiced and in variation
count_danachvoiced_in_vari = dict()
for e3, f3 in cnt_bigr_firstphon_danach.items():
    if e3[1] in e3[0] and e3[1] in voiced:
        count_danachvoiced_in_vari[e3] = e3[0].count(e3[1])

prob_danachvoiced_in_vari = len(count_danachvoiced_in_vari) / len(count_danachsegm_in_vari) # 0.04480651731160896

if_after_voiced = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in voiced:
        if_after_voiced.append(True)
    else:
        if_after_voiced.append(False)


# prob that following segment is a voiceless and in variation
count_danachvoiceless_in_vari = dict()
for e4, f4 in cnt_bigr_firstphon_danach.items():
    if e4[1] in e4[0] and e4[1] in voiceless:
        count_danachvoiceless_in_vari[e4] = e4[0].count(e4[1])

prob_danachvoiceless_in_vari = len(count_danachvoiceless_in_vari) / len(count_danachsegm_in_vari) # 0.024439918533604887

if_after_voiceless = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in voiceless:
        if_after_voiceless.append(True)
    else:
        if_after_voiceless.append(False)

# prob that following segment is a plosives and in variation
count_danachplosive_in_vari = dict()
for e5, f5 in cnt_bigr_firstphon_danach.items():
    if e5[1] in e5[0] and e5[1] in plosives:
        count_danachplosive_in_vari[e5] = e5[0].count(e5[1])

prob_danachplosive_in_vari = len(count_danachplosive_in_vari) / len(count_danachsegm_in_vari) # 0.008146639511201629

if_after_plosive = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in plosives:
        if_after_plosive.append(True)
    else:
        if_after_plosive.append(False)

# prob that following segment is a nasal and in variation
count_danachnasal_in_vari = dict()
for e51, f51 in cnt_bigr_firstphon_danach.items():
    if e51[1] in e51[0] and e51[1] in nasals:
        count_danachnasal_in_vari[e51] = e51[0].count(e51[1])

prob_danachnasal_in_vari = len(count_danachnasal_in_vari) / len(count_danachsegm_in_vari) # 0.034623217922606926

if_after_nasal = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in nasals:
        if_after_nasal.append(True)
    else:
        if_after_nasal.append(False)

# prob that following segment is a fricative and in variation
count_danachfricative_in_vari = dict()
for e6, f6 in cnt_bigr_firstphon_danach.items():
    if e6[1] in e6[0] and e6[1] in fricatives:
        count_danachfricative_in_vari[e6] = e6[0].count(e6[1])

prob_danachfricative_in_vari = len(count_danachfricative_in_vari) / len(count_danachsegm_in_vari) # 0.024439918533604887

if_after_fricative = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in fricatives:
        if_after_fricative.append(True)
    else:
        if_after_fricative.append(False)

# prob that following segment is a lateral and in variation
count_danachlateral_in_vari = dict()
for e7, f7 in cnt_bigr_firstphon_danach.items():
    if e7[1] in e7[0] and e7[1] in laterals:
        count_danachlateral_in_vari[e7] = e7[0].count(e7[1]) # NONE

if_after_lateral = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in laterals:
        if_after_lateral.append(True)
    else:
        if_after_lateral.append(False)

# prob that following segment is a trill and in variation
count_danachtrill_in_vari = dict()
for e8, f8 in cnt_bigr_firstphon_danach.items():
    if e8[1] in e8[0] and e8[1] in trill:
        count_danachtrill_in_vari[e8] = e8[0].count(e8[1]) # NONE

if_after_trill = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in trill:
        if_after_trill.append(True)
    else:
        if_after_trill.append(False)

# prob that following segment is an approximant and in variation
count_danachapproximant_in_vari = dict()
for e9, f9 in cnt_bigr_firstphon_danach.items():
    if e9[1] in e9[0] and e9[1] in approximants:
        count_danachapproximant_in_vari[e9] = e9[0].count(e9[1])

prob_danachapproximant_in_vari = len(count_danachapproximant_in_vari) / len(count_danachsegm_in_vari) # 0.002036659877800407

if_after_approximant = [] # list TRUE/FALSE
for dv3 in newlist2:
    if dv3[1] in dv3[0] and dv3[1] in approximants:
        if_after_approximant.append(True)
    else:
        if_after_approximant.append(False)

########################################################################################################################
#Important to consider for the calculations of preceding and following segments:
#Many instances of variations not included here because preceding or following element is an exception.
#See list "extra_exceptions", which includes breaks, hesitations, breathing or anonymized elements.
#For preceding segments: <P>: 98, <HÄSITATION>: 6, NANA: 3, <LACHEN>: 1, NANANA: 1
#--> in total 109 excluded; the probability that a variation is preceded by an exception is 0.10120706
#For following segments: <P>: 168, <HÄSITATION>: 4, NANA: 3, <ATMEN>: 2, <LACHEN>: 1, <UNVERSTÄNDLICH>: 1, <RÄUSPER>: 1
#--> in total 180 excluded; the probability that a variation is followed by an exception is 0.16713092
########################################################################################################################
# Lists for csv file
# list with count that preceding last segment occurs in variation (probability: 0.2425531914893617 at least once)
count_list_davor = [count_davorsegm_in_vari[r] for r in newlist1]
# list with count that following first segment occurs in variation (probability: 0.20977596741344195 (at least once))
count_list_danach = [count_danachsegm_in_vari[r1] for r1 in newlist2]
########################################################################################################################
relfreq_A1 = [prob_each_word[a] for a in word_davor_1]
relfreq_C1 = [prob_each_word[c] for c in word_danach_1]

########################################################################################################################
# Question of assimilation at word boundaries: Following first segment is == esh [ʃ]

esh_dict = {}
for bi_fr, bi_fr_c in cnt_bigr_firstphon_danach.items():
    if bi_fr[1] == "ʃ":
        esh_dict[bi_fr] = bi_fr_c

sum(esh_dict.values()) # --> the first segment of the following word is esh 58 times



esh_dict2 = {}
for bi_fr1, bi_fr_c1 in cnt_bigr_firstphon_danach.items():
    if bi_fr1[0][-1] == "ʃ" and bi_fr1[1] == "ʃ":
        esh_dict2[bi_fr1] = bi_fr_c1

sum(esh_dict2.values()) # --> 14 times that the tokens last and the following words first segment is an esh


esh_dict3 = {}
for bi_fr2, bi_fr_c2 in cnt_bigr_firstphon_danach.items():
    if bi_fr2[0][-1] != "ç" and bi_fr2[1] == "ʃ":
        esh_dict3[bi_fr2] = bi_fr_c2

sum(esh_dict3.values()) # --> 38 times the last segment of the token is not the fricative ç and the following segment is esh


esh_list_token = []
for sem_esh in segments:
    if sem_esh[-1] == "ʃ":
        esh_list_token.append(sem_esh)

len(esh_list_token) # --> 65 tokens with esh as last segment
Counter(esh_list_token) # Count of variations with final esh

########################################################################################################################
df = pandas.DataFrame(data={"Textgrid": TextGrid, "WordNumber_inTG": WordNumber_in_TG,  "Word_B": word, 'Count_B': len(word),
                            'IPA_word': CanonIPA_list, "Variation": segments, 'Variation_Occurence':  segm_count_list, 'Variation_B_Entropy': eig_entropy ,
                            "syllable_count": syllable_count, "Word_start": word_start,
                            "Word_stop": word_stop, 'Speak_duration_of_B': speak_duration,
                            "Phon_Segments": phon_segm, "Phon_Segm_Start": phon_segm_start, "Phon_Segm_Stop": phon_segm_stop,
                            'Count_phonemes_eig': Count_phonemes_eig,'Difference_count': len_difference,
                            'Word_A2': word_davor_2, 'A2_IPA': IPA_davor2, 'A2_POS': morphTag_davor2, 'Count_A2': word_count_davor_2,
                            'Word_A1': word_davor_1, 'A1_IPA': IPA_davor1, 'A1_POS': morphTag_davor1,
                             'Count_A1': word_count_davor_1, 'RelFreq_A1': relfreq_A1 , 'A1_Entropy': davor_entropy_list, 'A1_Surprisal': bef_info, 'Word_C1': word_danach_1, 'C1_IPA': IPA_danach1,
                            'C1_POS': morphTag_danach1, 'Count_C1': word_count_danach_1, 'RelFreq_C1': relfreq_C1, 'C1_Entropy': after_entropy_list, 'C1_Surprisal': aft_info ,
                            'Word_C2': word_danach_2, 'C2_IPA': IPA_danach2, 'C2_POS': morphTag_danach2,
                            'Count_C2': word_count_danach_2,
                            'Bigram_A1B': newbigr_davor, 'Bigram_Count_A1B(variation)': cnt_bigr_davorlist, 'Pr_B(variation)_given_A1': copro_AB_variation,
                            'Pr_B_given_A1': copro_AB, 'Bigram_BC1': newbigr_danach, 'Bigram_Count_B(variation)C1': cnt_bigr_danachlist,
                            'Pr_B(variation)_given_C1': inv_copro_BC_variation, 'Pr_B_given_C1': inv_copro_BC,

                            'LastSegment_Variation': newlist1, 'LS_in_Vari?': if_beforesegm_in_Vari, 'Count_LastSegm_in_Variation': count_list_davor,
                            'LS_frontvowel': if_before_frontvowel, 'LS_centralvowel': if_before_centralvowel, 'LS_backvowel': if_before_backvowel,
                            'LS_voiced': if_before_voiced, 'LS_voiceless': if_before_voiceless,
                            'LS_plosive': if_before_plosive, 'LS_nasal': if_before_nasal, 'LS_fricative': if_before_fricative,
                            'LS_lateral': if_before_lateral, 'LS_trill': if_before_trill, 'LS_approximant': if_before_approximant,
                            'Variation_FirstSegment': newlist2, 'FS_in_Vari?': if_aftersegm_in_Vari,'Count_FirstSegm_in_Variation': count_list_danach,
                            'FS_frontvowel': if_after_frontvowel, 'FS_centralvowel': if_after_centralvowel, 'FS_backvowel': if_after_backvowel,
                            'FS_voiced': if_after_voiced, 'FS_voiceless': if_after_voiceless,
                            'FS_plosive': if_after_plosive, 'FS_nasal': if_after_nasal, 'FS_fricative': if_after_fricative,
                            'FS_lateral': if_after_lateral, 'FS_trill': if_after_trill, 'FS_approximant': if_after_approximant,

                            'Trigram_A2A1B': newtrigr_davor, 'Trigram_Occurence_A2A1B': tri_count_AAB, 'Trigram_A1BC1': newtrigr_davordanach,
                            'Trigram_Occurence_A1BC1': tri_count_ABC, 'Trigram_BC1C2': newtrigr_danach, 'Trigram_Occurence_BC1C2': tri_count_BCC,
                            'Fivegram_A2A2BC1C2': fg_eig, 'Start_A2': start_wordA2, 'Stop_C2': stop_wordC2,
                            'Speech_duration_fivegram': speak_duration_A2toC2, 'speech_rate': speech_rate
                            })
df.to_csv("A1_copy.csv", sep=',', index=True)


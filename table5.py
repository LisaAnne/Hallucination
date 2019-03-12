#!/usr/bin/env python
"""
Compute correlation between sentence scores and human scores.
"""

import glob
import json
import numpy as np

##########################################################################
def correctForNan(inputVec):
  # If all the values are identical, introduce a small perturbation
  outputVec = inputVec[:]
  item = inputVec[0]
  allEqual = True
  for it in inputVec:
    if it <> item:
      allEqual = False
      break

  if allEqual:
    outputVec[np.random.choice(range(len(inputVec)))] += 0.001

  return outputVec
##########################################################################

SCORES_PATH = 'output/hallucination/'

f_scores = [SCORES_PATH+'hallucinated_words_baseline-gan_beam5_test.json',
            SCORES_PATH+'hallucinated_words_nbt_beam5_test.json',
            SCORES_PATH+'hallucinated_words_gan_beam5_test.json',
            SCORES_PATH+'hallucinated_words_td_beam5_test.json',
            SCORES_PATH+'hallucinated_words_td-sc_beam5_test.json']

ids_file = 'data/human_scores/' + 'imageIDs.txt' # images used in the human evaluation
of = open(ids_file, 'r')
image_ids = of.read().split('\n')

HUMAN_SCORES_PATH = 'data/human_scores/0*.txt'
f_human_subj_scores = glob.glob(HUMAN_SCORES_PATH)
f_human_subj_scores.sort()

MODELS = ['MPI-CE', 'NBT-CE', 'MPI-GAN', 'TD-CE', 'TD-SC']
SCORES = ['B@1', 'B@2', 'B@3', 'B@4', 'R', 'M', 'C', 'S']
CHAIR = ['1-CHs', '1-CHi']

NIMAGES = len(image_ids)
NCAPTIONS = len(MODELS)

s_scores_m = {} # sentence scores
c_scores_m = {} # chair scores
h_s_scores_m = {} # human scores

for i, fn in enumerate(f_scores):
  s_scores_m[i] = [None] * NIMAGES
  c_scores_m[i] = [None] * NIMAGES
  of = open(fn, 'r')
  f_data = json.load(open(fn, 'r'))
  f_data = f_data['sentences']
  of.close()
  for item in f_data:
    im_id = item['image_id']
    if str(im_id) not in image_ids:
      continue
    metrics = item['metrics']
    b1 = metrics['Bleu_1']
    b2 = metrics['Bleu_2']
    b3 = metrics['Bleu_3']
    b4 = metrics['Bleu_4']
    rl = metrics['ROUGE_L']
    me = metrics['METEOR']
    ci = metrics['CIDEr']
    sp = metrics['SPICE']['All']['f']
    ind = image_ids.index(str(im_id))
    s_scores_m[i][ind] = [b1, b2, b3, b4, rl, me, ci, sp]
    #
    ch_s = metrics['CHAIRs']
    ch_i = metrics['CHAIRi']
    c_scores_m[i][ind] = [1-ch_s, 1-ch_i]

for i, fn in enumerate(f_human_subj_scores):
  h_s_scores_m[i] = []
  of = open(fn, 'r')
  f_data = of.read().split('\n')
  if f_data[-1] == '':
    f_data = f_data[0:-1]
  for line in f_data:
    items = line.split('\t')
    h_s_scores_m[i].append([float(x) for x in (items[1:])])

# PEARSON'S correlation across NCAPTIONS=5 captions per image (from each system), averaged over NIMAGES=500 images

corr_s_s = [0] * len(SCORES) # correlation between sentence scores and human scores
corr_s_cs_s = [0] * len(SCORES) # correlation between sentence scores+(1-CHs) and human scores
corr_s_ci_s = [0] * len(SCORES) # correlation between sentence scores+(1-CHi) and human scores

for im in range(NIMAGES):
  s_m = []
  c_m = []
  h_s_m = []
  for i in range(NCAPTIONS):
    s_m.append(s_scores_m[i][im])
    c_m.append(c_scores_m[i][im])
    h_s_m.append(h_s_scores_m[i][im])
  for metric in range(len(SCORES)):
    corr = np.corrcoef(correctForNan([x[metric] for x in s_m]), correctForNan([x[0] for x in h_s_m]))[0][1]
    corr_s_s[metric] += corr

  for metric in range(len(SCORES)):
    #ch_s
    corr = np.corrcoef(correctForNan([x[metric] + c_m[i][0] for i, x in enumerate(s_m)]), correctForNan([x[0] for x in h_s_m]))[0][1]
    corr_s_cs_s[metric] += corr
    #ch_i
    corr = np.corrcoef(correctForNan([x[metric] + c_m[i][1] for i, x in enumerate(s_m)]), correctForNan([x[0] for x in h_s_m]))[0][1]
    corr_s_ci_s[metric] += corr

print "Metric\tCorrelation"

for metric in [5,6,7]: # focus on 'M', 'C', 'S'
  print('%s\t%.04f' % (SCORES[metric], corr_s_s[metric]/float(NIMAGES)))

for metric in [5,6,7]: # focus on 'M', 'C', 'S'
  print('%s\t%.04f' % (SCORES[metric]+'+'+CHAIR[0], corr_s_cs_s[metric]/float(NIMAGES)))

for metric in [5,6,7]: # focus on 'M', 'C', 'S'
  print('%s\t%.04f' % (SCORES[metric]+'+'+CHAIR[1], corr_s_ci_s[metric]/float(NIMAGES)))


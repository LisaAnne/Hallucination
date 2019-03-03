import numpy as np
import sys
import json
import os
import pickle as pkl
from nltk import word_tokenize
from collections import defaultdict
from pattern.en import singularize
from make_plural import *
import pdb
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
parser.add_argument("--tag", type=str, default='td-fc_beam1_test')
parser.add_argument('--robust', dest='robust', action='store_true')
parser.set_defaults(robust=False)
args = parser.parse_args()

blank_lm_predictions = './output/language_model_blank_input/%s/%%d.npy' %(args.tag)
caps = json.load(open('./generated_sentences/%s.json' %args.tag))

hallucinated_json = './output/hallucination/hallucinated_words_%s.json' %args.tag
hallucination_data = json.load(open(hallucinated_json))

#read vocab
#TODO create single vocab file (double check they are the same)
if args.robust:
    infos_file = 'data/infos_language_bias_robust.pkl'
else:
    infos_file = 'data/infos_language_bias.pkl'
infos = pkl.load(open(infos_file, 'rb'))
vocab = infos['vocab'] # ix -> word mapping
unk_idx = 9487
assert vocab[str(unk_idx)] == 'UNK'
word_to_idx = defaultdict(lambda: unk_idx)  # word -> ix
for key, value in zip(vocab.keys(), vocab.values()):
    word_to_idx[value] = int(key)

#get mscoco objects and such
synonyms = open('./data/synonyms.txt').readlines()
synonyms = [s.strip().split(', ') for s in synonyms]
synonym_dict = {}
synonym_node_dict = {}
all_synonyms = []
for line in synonyms:
    for synonym in line:
        synonym_dict[synonym] = line
        synonym_node_dict[synonym] = line[0]
    all_synonyms.extend(line + [pluralize(l) for l in line])

def softmax(array):
    shift = array - np.max(array)
    return np.exp(shift)/np.sum(np.exp(shift))

hallucination_by_imid = {h['image_id']: h for h in hallucination_data['sentences']}

word_hallucinated_idxs_no_mask = 0.
word_hallucinated_total = 0.

for i, imid in enumerate(sorted(hallucination_by_imid.keys())):
    sys.stdout.write("\r%d/%d" %(i, len(hallucination_by_imid.keys())))
    probs = np.load(blank_lm_predictions %int(imid))
    item = hallucination_by_imid[imid]
    caption = item['caption'] 

    mscoco_words_orig = [(idx, word) for idx, word in enumerate(word_tokenize(caption.lower())) 
                if singularize(word) in set(all_synonyms)]
    
    hallucinated_words = [i[0].split(' ')[0] for i in item['mscoco_hallucinated_words']] #hallucinated words stored as (word, node_word) 
    caption_words = word_tokenize(caption.lower())
    mscoco_words = zip(item['hallucination_idxs'], \
                       [caption_words[i] for i in item['hallucination_idxs']])
#    import pdb; pdb.set_trace()   
 
#    for mscoco_word in mscoco_words:
#        idx, word = mscoco_word
#        word_probs_no_mask = softmax(probs[idx,:])
#        sorted_objects_no_mask = np.argsort(word_probs_no_mask)[::-1]
#        word_idx_no_mask = np.where(sorted_objects_no_mask == word_to_idx[word])[0][0] + 1
#        if word in hallucinated_word:
#            word_hallucinated_idxs_no_mask += word_idx_no_mask
#            word_hallucinated_total += 1
    for mscoco_word in mscoco_words:
        idx, word = mscoco_word
        word = word.split(' ')[0]
        word_probs_no_mask = softmax(probs[idx,:])
        sorted_objects_no_mask = np.argsort(word_probs_no_mask)[::-1]
        word_idx_no_mask = np.where(sorted_objects_no_mask == word_to_idx[word])[0][0] + 1
        word_hallucinated_idxs_no_mask += word_idx_no_mask
        word_hallucinated_total += 1

consistency = word_hallucinated_total/word_hallucinated_idxs_no_mask
print "\nConsistency: %0.04f" %consistency
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--cap_file", type=str, default='')
#    parser.add_argument("--annotation_path", type=str, default='coco/annotations')
#    args = parser.parse_args()
#
#    _, imids, _ = load_generated_captions(args.cap_file)
#
#    evaluator = CHAIR(imids, args.coco_path) 
#    evaluator.get_annotations()
#    cap_dict = evaluator.compute_chair(args.cap_file) 
#    
#    print_metrics(cap_dict)
#    save_hallucinated_words(args.cap_file, cap_dict)
#

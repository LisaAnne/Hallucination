import sys
from nltk.stem import *
from nltk.corpus import wordnet as wn
import nltk
import json
from pattern.en import singularize
from pattern.en import tag 
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cap_file", type=str, default='')
parser.add_argument("--coco_path", type=str, default='coco')
args = parser.parse_args()

lemma = nltk.wordnet.WordNetLemmatizer()

#Read in captions

caps = json.load(open(args.cap_file))
try:
    all_caps = caps['imgToEval'].values()
except:
    raise Exception("Expect caption file to consist of a dectionary with sentences correspdonding to the key 'imgToEval'")

imids = set([cap['image_id'] for cap in all_caps])

#Get list of image ids and MSCOCO objects in the images. 

coco_segments = json.load(open(args.coco_path + '/instances_all2014.json'))
coco_caps = json.load(open(args.coco_path + '/captions_all2014.json'))

#make dict linking object name to ids
id_to_name = {} #dict with id to synsets 
for cat in coco_segments['categories']:
    id_to_name[cat['id']] = cat['name']

#read in synonyms
synonyms = open('data/synonyms.txt').readlines()
synonyms = [s.strip().split(', ') for s in synonyms]
mscoco_objects = [] #mscoco objects and *all* synonyms
synonym_dict = {}
inverse_synonym_dict = {}
for synonym in synonyms:
    mscoco_objects.extend(synonym)
    for s in synonym:
        synonym_dict[s] = synonym
        inverse_synonym_dict[s] = synonym[0]

#Hard code some rules for special cases in MSCOCO
animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
vehicle_words = ['jet', 'train']

def caption_to_words(caption):

    #standard preprocessing
    words = nltk.word_tokenize(caption.lower())
    words = [singularize(w) for w in words]
    double_words = []
    for i in range(len(words)-1):
        double_words.append('%s %s' %(words[i], words[i+1]))
    words += double_words

    #hard coded rules for "double" rules: 'hot dog' is not a dog, 'teddy bear' is not a bear, 'baby giraffe' is not a (human) baby
    if 'hot dog' in words: words = [word for word in words if word != 'dog'] 
    if 'teddy bear' in words: words = [word for word in words if word != 'bear'] 
    if 'home plate' in words: words = [word for word in words if word != 'plate'] 
    if 'motor bike' in words: words = [word for word in words if word != 'bike'] 

    for vehicle in vehicle_words:
        if 'passenger %s' %vehicle in words: words = [word for word in words if word != 'passenger'] 
    if 'bow tie' in words: words = [word for word in words if word != 'bow']
    for animal in animal_words:
        if 'baby %s' %animal in words: words = [word for word in words if word != 'baby'] 
        if 'adult %s' %animal in words: words = [word for word in words if word != 'adult'] 
        if 'baby cub' in words: words = [word for word in words if word != 'baby'] 
        if 'baby animal' in words: words = [word for word in words if word != 'baby'] 
        if 'adult animal' in words: words = [word for word in words if word != 'adult'] 
    #toilet seat is not chair
    if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

    #get synonyms for all words in the caption
    words = list(set(words) & set(mscoco_objects))
    synonym_words = []
    for word in words:
        synonym_words.extend(synonym_dict[word])
    return synonym_words


#match image ids to objects in image (from coco segmentations)
imid_to_objects = {imid: [] for imid in imids}
for i, annotation in enumerate(coco_segments['annotations']):
    sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" 
                      %(i, len(coco_segments['annotations'])))
    imid = annotation['image_id']
    if imid in imid_to_objects:
        imid_to_objects[imid].extend(synonym_dict[id_to_name[annotation['category_id']]])
print "\n"

#match image ids to objects in image (from coco captions)
for i, annotation in enumerate(coco_caps['annotations']):
    sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' 
                      %(i, len(coco_caps['annotations'])))
    imid = annotation['image_id']
    if imid in imid_to_objects:
        words = caption_to_words(annotation['caption'].lower())
        imid_to_objects[imid].extend(words)
print "\n"


def find_hallucinated_sentences(all_caps, imid_to_objects):

    '''
    Given ground truth objects and generated captions, determine which sentences have hallucinated words.
    '''

    non_hallucinated_sentences = []
    hallucinated_sentences = []

    num_coco_caps = 0.
    num_coco_words = 0.
    sentence_length = 0.
    hallucinated_word_count = 0.
    coco_words_count = 0.

    for i, cap_eval in enumerate(all_caps):
        sys.stdout.write('\r%d/%d' %(i, len(all_caps)))

        cap = cap_eval['caption']

        #get all words in the caption, as well as all synonyms
        words = caption_to_words(cap) 

        gt_objects = set(imid_to_objects[cap_eval['image_id']])
        cap_dict = {'image_id': cap_eval['image_id'], 
                    'caption': cap,
                    'hallucinated_words': [],
                    'nouns': words,
                    'gt_objects': list(gt_objects),
                    'sentence_info': []}
 

        check_words = set(words) 
        if len(check_words) > 0: num_coco_caps += 1 
        hallucinated = False 

        for check_word in check_words:
            if check_word not in gt_objects:
                cap_dict['hallucinated_words'].append(check_word) 
                hallucinated = True 
        gt_words = set([inverse_synonym_dict[word] for word in cap_dict['gt_objects']])
        hallucinated_words = set([inverse_synonym_dict[word] for word in cap_dict['hallucinated_words']])


        sentence_info = []
        double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven']
        double_cap = cap
        double_words_track = []
        for word in double_words:
            if word in double_cap: 
                double_cap = double_cap.replace(word, ''.join(word.split(' ')))
                double_words_track.append(word)
        raw_words = nltk.word_tokenize(double_cap.lower())

        for word in double_words_track:
            index = [idx for idx, raw_word in enumerate(raw_words) if ''.join(word.split(' ')) in raw_word]
            raw_words[index[0]] = word
        raw_words = [singularize(w) for w in raw_words]
        for wi, w in enumerate(raw_words): 
            if w == 'wine glas':
                raw_words[wi] = 'wine glass'

        for raw_word in raw_words:
            item = {'word': raw_word, 'mscoco': 0, 'hallucinated': 0, 'mscoco_synonym': 0}
            if raw_word in mscoco_objects:
                item['mscoco'] = 1
                item['mscoco_synonym'] = inverse_synonym_dict[raw_word]
                coco_words_count += 1
            if raw_word in set(cap_dict['hallucinated_words']):
                item['hallucinated'] = 1
                hallucinated_word_count += 1
             
            sentence_info.append(item)

        cap_words = set([inverse_synonym_dict[word] for word in words])
        cap_dict['hallucinated_words_node'] = list(hallucinated_words)   
        cap_dict['gt_objects_node'] = list(gt_words)   
        cap_dict['generated_words'] = list(cap_words)
        cap_dict['sentence_info'] = sentence_info

        num_coco_words += len(gt_words)

        if hallucinated:
            hallucinated_sentences.append(cap_dict)
        else:
            non_hallucinated_sentences.append(cap_dict)

    print "\n"
    print "Percentatge of hallucinated sentences: %0.03f" %(len(hallucinated_sentences)/float(len(all_caps)))
    print "Percentage of hallucinated words: %f" %(hallucinated_word_count/coco_words_count)
    print "%0.03f\t%0.03f" %(len(hallucinated_sentences)/float(len(all_caps)), hallucinated_word_count/coco_words_count)

    print "Average number coco words: %f" %(coco_words_count/len(all_caps))
    print "Average sentence_length: %f" %(sentence_length/len(all_caps))
    return {'not_hallucinated': non_hallucinated_sentences, 'hallucinated': hallucinated_sentences}

hallucinated_sentences = find_hallucinated_sentences(all_caps, imid_to_objects)

tag = args.cap_file.split('/')[-1] 
with open('output/hallucinated_words_%s' %tag, 'w') as f:
    json.dump(hallucinated_sentences, f)

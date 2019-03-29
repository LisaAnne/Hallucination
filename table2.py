from utils import chair 
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
args = parser.parse_args()

sentence_template = 'generated_sentences/%s.json'
table2_tags = [('FC', 'fc-robust_beam5_test'),
               ('att2in', 'att2in-robust_beam5_test'),
               ('TD', 'td-robust_beam5_test'), 
               ('NBT', 'nbt-robust_beam5_test')] 

_, imids, _ = chair.load_generated_captions(sentence_template %table2_tags[0][1])

evaluator = chair.CHAIR(imids, args.annotation_path) 
evaluator.get_annotations()

print "\t\tCross Entropy\t\t\t"
print "Model\tSPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi"

for tag in table2_tags:

    cap_dict = evaluator.compute_chair(sentence_template %tag[1]) 
    metric_string = chair.print_metrics(cap_dict, True)
    chair.save_hallucinated_words(sentence_template %tag[1], cap_dict)
    print "%s\t%s\t" %(tag[0], metric_string) 

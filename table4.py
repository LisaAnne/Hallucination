from utils import misc
import os
from utils import chair
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
args = parser.parse_args()

output_template = "output/hallucination/hallucinated_words_%s.json" 
sentence_template = "generated_sentences/%s.json" 

table4_tags = [('FC', 'fc_beam5_test'),
               ('att2in', 'att2in_beam5_test'),
               ('td', 'td_beam5_test')]


_, imids, _ = chair.load_generated_captions(sentence_template %table4_tags[0][1])
evaluator = chair.CHAIR(imids, args.annotation_path)
evaluator.get_annotations()

print "Model\tCIDEr\tMETEOR\tSPICE"

for tag in table4_tags:

    if not os.path.exists(output_template %tag[1]):
        cap_dict = evaluator.compute_chair(sentence_template %tag[1])
        chair.save_hallucinated_words(sentence_template %tag[1], cap_dict)
        
    cider, meteor, spice = misc.score_correlation(output_template %tag[1],
                                                  quiet=True)
    print "%s\t%0.03f\t%0.03f\t%0.03f" %(tag[0], cider, meteor, spice)

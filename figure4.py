from utils import misc 
from utils import chair 
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
args = parser.parse_args()
    
figure4_tags_karpathy = [('TD', 'td_beam1_test'),
               ('No Att', 'td-noatt_beam1_test'),
               ('No Conv', 'td-noconv_beam1_test'), 
               ('Single', 'td-single_beam1_test'),
               ('FC', 'td-fc_beam1_test')] 

print "=================Karpathy Split================="
print "Model\tCHAIRi\tLM Consistency\tIM Consistency"

for tag in figure4_tags_karpathy:

    chair_i, lm_consistency, im_consistency = misc.get_consistency(tag[1],
                                                                  args.annotation_path, 
                                                                  robust=False)
    
    print "%s\t%0.04f\t%0.04f\t\t%0.04f" %(tag[0], 
                                         chair_i, 
                                         lm_consistency,
                                         im_consistency)

print "=================Robust Split================="
print "Model\tCHAIRi\tLM Consistency\tIM Consistency"

figure4_tags_robust = [('TD', 'td-robust_beam1_test'),
                       ('No Att', 'td-noatt-robust_beam1_test'),
                       ('No Conv', 'td-noconv-robust_beam1_test'), 
                       ('Single', 'td-single-robust_beam1_test'),
                       ('FC', 'td-fc-robust_beam1_test')] 

#generate hallucination files for robust split for fig 4
evaluator = None
output_template = "output/hallucination/hallucinated_words_%s.json" 
sentence_template = "generated_sentences/%s.json" 
for tag in figure4_tags_robust:
    if not os.path.exists(output_template %tag[1]):
        if not evaluator:
            _, imids, _ = chair.load_generated_captions(sentence_template %figure4_tags_robust[0][1])
            evaluator = chair.CHAIR(imids, args.annotation_path)
            evaluator.get_annotations()
        cap_dict = evaluator.compute_chair(sentence_template %tag[1])
        chair.save_hallucinated_words(sentence_template %tag[1], cap_dict)

for tag in figure4_tags_robust:

    chair_i, lm_consistency, im_consistency = misc.get_consistency(tag[1], 
                                                                  args.annotation_path, 
                                                                  robust=True)
    
    print "%s\t%0.04f\t%0.04f\t\t%0.04f" %(tag[0], 
                                         chair_i, 
                                         lm_consistency,
                                         im_consistency)

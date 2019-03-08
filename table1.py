from utils import chair 
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
args = parser.parse_args()

sentence_template = 'generated_sentences/%s.json'
table1_tags = [('LRCN', 'lrcn_beam5_test', 'lrcn-sc_beam5_test'),
               ('FC', 'fc_beam5_test', 'fc-sc_beam5_test'),
               ('att2in', 'att2in_beam5_test', 'att2in-sc_beam5_test'),
               ('TD', 'td_beam5_test', 'td-sc_beam5_test'),
               ('TD-BB', 'td-bb_beam5_test', 'td-bb-sc_beam5_test'),
               ('NBT', 'nbt_beam5_test'),
               ('GAN', 'baseline-gan_beam5_test', 'gan_beam5_test')]

_, imids, _ = chair.load_generated_captions(sentence_template %table1_tags[0][1])

evaluator = chair.CHAIR(imids, args.annotation_path) 
evaluator.get_annotations()

print "\t\tCross Entropy\t\t\t\tSelf-Critical\t\t"
print "Model\tSPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi\t|SPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi"

for tag in table1_tags:

    cap_dict = evaluator.compute_chair(sentence_template %tag[1]) 
    metric_string_ce = chair.print_metrics(cap_dict, True)
    chair.save_hallucinated_words(sentence_template %tag[1], cap_dict)
    if len(tag) > 2:
        cap_dict = evaluator.compute_chair(sentence_template %tag[2]) 
        metric_string_sc = chair.print_metrics(cap_dict, True)
        chair.save_hallucinated_words(sentence_template %tag[2], cap_dict)
    else:
        metric_string_sc = "-\t-\t-\t-\t-"
    print "%s\t%s\t|%s" %(tag[0], metric_string_ce, metric_string_sc) 

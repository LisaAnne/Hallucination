from utils import misc
import os

output_template = "output/hallucination/hallucinated_words_%s.json" 
sentence_template = "generated_sentences/%s.json" 

table4_tags = [('FC', 'fc_beam5_test'),
               ('att2in', 'att2in_beam5_test'),
               ('td', 'td_beam5_test')]

print "Model\tCIDEr\tMETEOR\tSPICE"

for tag in table4_tags:

    cider, meteor, spice = misc.score_correlation(output_template %tag[1],
                                                  quiet=True)
    print "%s\t%0.03f\t%0.03f\t%0.03f" %(tag[0], cider, meteor, spice)

from utils import misc 
from utils import chair 
import argparse
import json
import os
    
figure4_tags_karpathy = [('TD', 'td_beam1_test'),
               ('No Att', 'td-noatt_beam1_test'),
               ('No Conv', 'td-noconv_beam1_test'), 
               ('Single', 'td-single_beam1_test'),
               ('FC', 'td-fc_beam1_test')] 

print "=================Karpathy Split================="
print "Model\tCHAIRi\tLM Consistency\tIM Consistency"

for tag in figure4_tags_karpathy:

    chair, lm_consistency, im_consistency = misc.get_consistency(tag[1], 
                                                                 robust=False)
    
    print "%s\t%0.02f\t%0.04f\t\t%0.04f" %(tag[0], 
                                         chair*100, 
                                         lm_consistency,
                                         im_consistency)

print "=================Robust Split================="
print "Model\tCHAIRi\tLM Consistency\tIM Consistency"

figure4_tags_robust = [('TD', 'td-robust_beam1_test'),
                       ('No Att', 'td-noatt-robust_beam1_test'),
                       ('No Conv', 'td-noconv-robust_beam1_test'), 
                       ('Single', 'td-single-robust_beam1_test'),
                       ('FC', 'td-fc-robust_beam1_test')] 

for tag in figure4_tags_robust:

    chair, lm_consistency, im_consistency = misc.get_consistency(tag[1], 
                                                                 robust=True)
    
    print "%s\t%0.02f\t%0.04f\t\t%0.04f" %(tag[0], 
                                         chair*100, 
                                         lm_consistency,
                                         im_consistency)

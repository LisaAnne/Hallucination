from utils import misc 

template = './output/hallucination/hallucinated_words_%s.json'
fc_hallucination = template %'fc_beam5_test'
td_hallucination = template %'td_beam5_test'

diffs = misc.predictive_metrics(fc_hallucination, td_hallucination)

print "Differences in Hallucination for sentences with similar SPICE score:"
print "\t(caomparing fc and td models)"

for i in range(0, 100, 10):
    print "Between %d-%d:\t%0.04f" %(i, i+10, diffs[i/10])

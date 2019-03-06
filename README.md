# Object Hallucination in Image Captioning

Rohrbach*, Anna and Hendricks*, Lisa Anne, et al. "Object Hallucination in Image Captioning." EMNLP (2018).

Find the paper [here](https://arxiv.org/pdf/1809.02156.pdf).
```
@inproceedings{objectHallucination, 
        title = {Object Hallucination in Image Captioning.}, 
        author = {Rohrbach, Anna and Hendricks, Lisa Anne and Burns, Kaylee, and Darrell, Trevor, and Saenko, Kate}, 
        booktitle = {Empirical Methods in Natural Language Processing (EMNLP)}, 
        year = {2018} 
}
```

License: BSD 2-Clause license

## Running the Code

**Getting Started**

Run [setup.sh](setup.sh) to download generated sentences used for our analysis.
Additionally you will need MSCOCO annotations (both the instance segmentations and ground truth captions).
If you do not already have them, they can be downloaded [here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).
Our code also relies on the python [natural language toolkit (nltk)](https://www.nltk.org/).

**Replicating Results**

After running ```setup.sh``` you should be able to replicate results in our paper by running ```table1.py```, ```table2.py```, ```table3.py```, ```table4.py``` and ```figure6.py``` (example usage ```python table1.py --annotation_path PATH_TO_COCO_ANNOTATIONS``` where --annotation_path default is ```coco/annotations```).
If you would like to run ```figure4.py``` you will need to download some intermediate features for computing the image consistency [here](https://drive.google.com/drive/u/1/folders/1dnci1Kv6ez-hsFOqZt_gwiAv2FTAjDP4).
Place this under ```outputs``` and unzip the file.
These scripts call on ```utils/chair.py``` to compute the CHAIR metric.  See below for more details on ```utils/chair.py```.

**Evaluating CHAIR**

See ```utils/chair.py``` to understand how we compute the CHAIRs and CHAIRi metrics.  
Evaluate generated sentences by inputting a path to the generated sentences as well as the path which includes coco annotations.

Example usage is:

```python utils/chair.py --cap_file generated_sentences/fc_beam5_test.json --coco_path coco```

where ```cap_file``` corresponds to a json file with your generated captions and ```coco_path``` points to where MSCOCO annotations are stored.

We expect generated sentences to be stored as a decitionary with the following keys:

* overall:  metrics from the COCO evaluation toolkit computed over the entire dataset.
* imgToEval: a dictionary with keys corresponding to image ids and values with a caption, image_id, and sentence metrics for the particular caption.

Note that this is the format of the captions output by the open sourced code [here](https://github.com/ruotianluo/self-critical.pytorch), 
which we used to replicate most of the models presented in the paper.

**Language and Image Model Consistency**

To compute language and image consistency, we trained an classifier to predict class labels given an image and a language model to predict the next word in a sentence given all previous words in a sentence.
You can access the labels predicted by our language model in ```outputs/image_classifier``` and the words predicted by our language model [here](https://drive.google.com/drive/u/1/folders/1dnci1Kv6ez-hsFOqZt_gwiAv2FTAjDP4).
Once you have these intermediate features you can look at ```utils/lm_consistency.py``` and ```utils/im_consistency.py``` to understand how these metrics are computed.
Running ```figure4.py``` will outputs the results from our paper (constructing the actual bar plot is left as an exercise to the reader).

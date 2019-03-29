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
You can see other python requirements in [requirements.txt](requirements.txt).

**Replicating Results**

After running ```setup.sh``` you should be able to replicate results in our paper by running ```table1.py```, ```table2.py```, ```table3.py```, ```table4.py``` and ```figure6.py``` (example usage ```python table1.py --annotation_path PATH_TO_COCO_ANNOTATIONS``` where ```coco/annotations``` is the default for ```--annotation_path```).
Our scripts call on ```utils/chair.py``` to compute the CHAIR metric.  See below for more details on ```utils/chair.py```.

If you would like to run ```figure4.py``` (language and image model consistency) you will need to download some intermediate features. Please see the *Language and Image Model Consistency* section below.

For reproducing our results on correlation with human scores, run ```python table5.py```. The file with images IDs used in the human evaluation, as well as the average human scores for each of the compared models, will be found in ```data/human_scores```, after running the ```setup.sh```.

**Evaluating CHAIR**

See ```utils/chair.py``` to understand how we compute the CHAIRs and CHAIRi metrics.  Evaluate generated sentences by inputting a path to the generated sentences as well as the path which includes coco annotations.

Example usage is:

```python utils/chair.py --cap_file generated_sentences/fc_beam5_test.json --annotation_path coco```

where ```cap_file``` corresponds to a json file with your generated captions and ```annotation_path``` points to where MSCOCO annotations are stored.

We expect generated sentences to be stored as a dictionary with the following keys:

* overall:  metrics from the COCO evaluation toolkit computed over the entire dataset.
* imgToEval: a dictionary with keys corresponding to image ids and values with a caption, image_id, and sentence metrics for the particular caption.

Note that this is the format of the captions output by the open sourced code [here](https://github.com/ruotianluo/self-critical.pytorch), 
which we used to replicate most of the models presented in the paper.

**Language and Image Model Consistency**

To compute language and image consistency, we trained a classifier to predict class labels given an image and a language model to predict the next word in a sentence given all previous words in a sentence.
You can access the labels predicted by our language model in ```output/image_classifier``` and the words predicted by our language model [here](https://drive.google.com/drive/u/1/folders/1dnci1Kv6ez-hsFOqZt_gwiAv2FTAjDP4).
To run our code, you ned to first download the [zip file](https://drive.google.com/drive/u/1/folders/1dnci1Kv6ez-hsFOqZt_gwiAv2FTAjDP4) into the main directory and unzip.
Once you have these intermediate features you can look at ```utils/lm_consistency.py``` and ```utils/im_consistency.py``` to understand how these metrics are computed.
Running ```figure4.py``` will output the results from our paper (constructing the actual bar plot is left as an exercise to the reader).

**Human Eval**

Replicate the results from our human evaluation by running ```python table5.py```.  Raw human evaluation scores can be found in ```data/human_scores``` after running ```setup.sh```.

**Captioning Models**

We generated sentences for the majority of models by training open source models available [here](https://github.com/ruotianluo/self-critical.pytorch).
Within this framework, we wrote code for the LRCN model as well as the topdown deconstructed models (Table 3 in the paper).
This code is available upon request.
For the top down model with bounding boxes, we used the code [here](https://github.com/peteanderson80/Up-Down-Captioner).
For the Neural Baby Talk model, we used the code [here](https://github.com/jiasenlu/NeuralBabyTalk).
For the GAN based model, we used the sentences from the paper [here](https://arxiv.org/abs/1703.10476).  Sentences were obtained directly from the author (we did not train the GAN model).

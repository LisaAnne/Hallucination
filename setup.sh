#!/bin/bash

#Download generated sentences
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/generated_sentences.zip
unzip generated_sentences.zip
rm -r generated_sentences.zip

#Download 
mkdir output
mkdir output/hallucination
mkdir output/language_model_blank_input
cd output
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/intermediate_image.zip
unzip intermediate_image.zip
rm -r intermediate_image.zip
mv intermediate_image image_classifier

cd ../data
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/gt_labels.p
wget https://people.eecs.berkeley.edu/~lisa_anne/hallucination/vocab.p

cd ..

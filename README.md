# att-fusion
Attention-based multimodal fusion for sentiment analysis

Code for the paper

[Context-Dependent Sentiment Analysis in User-Generated Videos](http://sentic.net/context-dependent-sentiment-analysis-in-user-generated-videos.pdf) (ACL 2017).
[Context-Dependent Sentiment Analysis in User-Generated Videos](http://sentic.net/context-dependent-sentiment-analysis-in-user-generated-videos.pdf) (ACL 2017).
[Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis](https://ieeexplore.ieee.org/abstract/document/8215597/)(ICDM 2017).

### Dataset
We provide results on the [MOSI dataset](https://arxiv.org/pdf/1606.06259.pdf)  
Please cite the creators 


### Preprocessing
As data is typically present in utterance format, we combine all the utterances belonging to a video using the following code

```
python create_data.py
```

Note: This will create speaker independent train and test splits 

### Running sc-lstm

Sample command:

```
python lstm_tf.py --unimodal True
python lstm_tf.py --unimodal False
```

Note: Keeping the unimodal flag as True (default False) shall train all unimodal lstms first (level 1 of the network mentioned in the paper)

### Citation 

If using this code, please cite our work using : 
```
@inproceedings{soujanyaacl17,
  title={Context-dependent sentiment analysis in user-generated videos},
  author={Poria, Soujanya  and Cambria, Erik and Hazarika, Devamanyu and Mazumder, Navonil and Zadeh, Amir and Morency, Louis-Philippe},
  booktitle={Association for Computational Linguistics},
  year={2017}
}

@inproceedings{poriaicdm17, 
author={S. Poria and E. Cambria and D. Hazarika and N. Mazumder and A. Zadeh and L. P. Morency}, 
booktitle={2017 IEEE International Conference on Data Mining (ICDM)}, 
title={Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis}, 
year={2017},  
pages={1033-1038}, 
keywords={data mining;feature extraction;image classification;image fusion;learning (artificial intelligence);sentiment analysis;attention-based networks;context learning;contextual information;contextual multimodal sentiment;dynamic feature fusion;multilevel multiple attentions;multimodal sentiment analysis;recurrent model;utterances;videos;Context modeling;Feature extraction;Fuses;Sentiment analysis;Social network services;Videos;Visualization}, 
doi={10.1109/ICDM.2017.134}, 
month={Nov},}

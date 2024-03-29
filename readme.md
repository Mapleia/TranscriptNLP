# TranscriptNLP

Project for comparing the caption of popular youtuber's apology videos.
Scores for similarity. 

## Get Started
Structure this project:
```
PARENTFOLDER/
    TRANSCRIPTS/
        TEXT/
        JSON/
        GRAPHS/
        ratio_list.json
        youtube_vids.json
        
    TranscriptNLP/
        graph.py
        main.py
```

1. Clone the repo.
2. Run `pip install -r requirements.txt`
3. Go to `main.py` and change `score_word` to `True`.
4. Run `main.py`.
5. Profit!


## Overview
### main.py
Using the files in the `TEXT/` folder, it will train a gensim model with `"wiki-gigaword-100"` 
and all the sentences from the transcripts. It will generate a score and save it to **similarity_score.json**.

Using the ratio_list.json generated by TranscriptCollect, it will create a .csv file and save it as **ratios.csv**.

### graph.py
Using **similarity_score.json** it will create a heatmap png named **similarity_graph.png**.
Requires /TEXTS/ to match the file.

Using ratios.csv, it will create a grouped bar graph named **ratio_graph.png**.

## Need to get the captions?
Try [this repo](https://github.com/Mapleia/TranscriptCollect) out!
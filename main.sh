#! /bin/bash
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip

pip install -r requirements.txt

python ./Process/getgraph_twitter.py
#Generate graph data and store in /data/Twitter15graph

python ./model/Twitter/model_Twitter.py

#save results in output_dir //
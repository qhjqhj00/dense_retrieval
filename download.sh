# download MSMARCO passage
mkdir -p $1/MSMARCO-Passage
cd $1/MSMARCO-Passage
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./
rm collectionandqueries.tar.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
# tar xvfz triples.train.small.tar.gz
# rm triples.train.small.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar xvfz top1000.dev.tar.gz
rm top1000.dev.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
gunzip qidpidtriples.train.full.2.tsv.gz
wget https://vault.cs.uwaterloo.ca/s/NYibRJ9bXs5PspH/download -O queries.dev.small.deepimpact.tsv.gz
gunzip queries.dev.small.deepimpact.tsv.gz
# download distillation-necessary files
# wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz
# gunzip msmarco-hard-negatives.jsonl.gz
# wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz
# gunzip cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz

# mkdir -p $1/MSMARCO-Document
# cd $1/MSMARCO-Document
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz -O ./collection.tsv.gz
# gunzip collection.tsv.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz -O ./queries.train.tsv.gz
# gunzip queries.train.tsv.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz -O ./qrels.train.tsv.gz
# gunzip qrels.train.tsv.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz -O ./queries.dev.small.tsv.gz
# gunzip queries.dev.small.tsv.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz -O ./qrels.dev.small.tsv.gz
# gunzip qrels.dev.small.tsv.gz
# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz
# gunzip msmarco-doctrain-top100.gz
# download java-11 required by anserini
# cd $2
# wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
# tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz
# rm openjdk-11.0.2_linux-x64_bin.tar.gz
# JAVA_HOME="$2/jdk-11.0.2"
# PATH="$JAVA_HOME/bin:$PATH"
# export PATH
# source ~/.profile
# source ~/.bashrc
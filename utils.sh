echo "list files in zip"
unzip -l ../datasets/wiki/wiki.en.zip

echo "wordcounts"
unzip -c ../datasets/wiki/wiki.en.zip wiki.en.bin | wc -l

echo "words vectors out-oof-vocabulary"
MODEL = "../datasets/wiki/wiki.en.bin"
echo "w1 w2 w3" | ./fasttext print-word-vectors MODEL

#zcat datasets/en-it/cc.it.300.vec.gz | less
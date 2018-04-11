echo "list files in zip"
unzip -l ../datasets/wiki/wiki.en.zip

echo "wordcounts"
unzip -c ../datasets/wiki/wiki.en.zip wiki.en.bin | wc -l

echo "words vectors out-oof-vocabulary"
MODEL = "../datasets/wiki/wiki.en.bin"
echo "w1 w2 w3" | ./fasttext print-word-vectors MODEL

zcat datasets/en-it/cc.it.300.vec.gz | less


# buscar palabras w2v en fasttext 
python do-fasttext.py datasets/en-it/it.200k.300d.w2v it datasets/en-it/it.200k.300d.tmp.fst > datasets/en-it/not_found.it
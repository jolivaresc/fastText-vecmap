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



python train_tm.py -o tm ../fastText-vecmap/datasets/dictionaries/es-na.train.txt ../fastText-vecmap/models/es-na/encoding_n2v/es-na/es.norm.n2v.encoded ../fastText-vecmap/models/es-na/encoding_n2v/na-es/na.norm.n2v.encoded > ../fastText-vecmap/models/es-na/encoding_n2v/es-na.norm.train.log
python test_tm.py tm.txt ../fastText-vecmap/datasets/dictionaries/es-na.test.true.txt ../fastText-vecmap/models/es-na/encoding_n2v/es-na/es.n2v.encoded ../fastText-vecmap/models/es-na/encoding_n2v/na-es/na.n2v.encoded > ../fastText-vecmap/models/es-na/encoding_n2v/es-na.norm.encoded.test.log



########################################################
########################################################
########################################################
########################################################
#echo "Training EN-DE..."
#python train_tm.py -o tm dictionaries/en-de.train.txt en.200k.300d.embeddings en-de/de.200k.300d.embeddings > logs/en-de.eval.train.log

#echo "Testing EN-DE..."
#python test_tm.py tm.txt dictionaries/en-de.test.txt  en.200k.300d.embeddings en-de/de.200k.300d.embeddings > logs/en-de.eval.test.log

echo "model 3: 2h-370d train no norm"
echo "Training ES-NA"

python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/3/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/3/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/2h-370d/en-it.encoded.train.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/3/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/3/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/2h-370d/en-it.encoded.test.log

####
echo "model 3: 2h-370d train norm"
echo "Training ES-NA"
python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/3/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/3/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/2h-370d/en-it.encoded.norm.train.log

echo "Testing ES-NA test norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/3/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/3/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/2h-370d/en-it.encoded.norm.test.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/3/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/3/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/2h-370d/en-it.encoded.test1.log


################################################

echo "model 5 1h-370d train no norm"
echo "Training ES-NA"

python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/5/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/5/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-370d/en-it.encoded.train.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/5/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/5/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-370d/en-it.encoded.test.log

####
echo "model 5: 1h-370d train norm"
echo "Training ES-NA"
python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/5/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/5/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-370d/en-it.encoded.norm.train.log

echo "Testing ES-NA test norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/5/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/5/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-370d/en-it.encoded.norm.test.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/5/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/5/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-370d/en-it.encoded.test1.log



################################################

echo "model 6 1h-128d train no norm"
echo "Training ES-NA"

python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/6/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/6/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-128d/en-it.encoded.train.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/6/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/6/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-128d/en-it.encoded.test.log

####
echo "model 6: 1h-128d train norm"
echo "Training ES-NA"
python train_tm.py -o tm /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.train.drive.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/6/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/6/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-128d/en-it.encoded.norm.train.log

echo "Testing ES-NA test norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/6/en.norm.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/6/it.norm.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-128d/en-it.encoded.norm.test.log

echo "Testing ES-NA test no norm"
python test_tm.py tm.txt /home/olivares/Documents/fastText-vecmap/datasets/dictionaries/en-it.test.txt /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/en-it/6/en.fst.encoded /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/it-en/6/it.fst.encoded > /home/olivares/Documents/fastText-vecmap/models/en-it/encoder/1h-128d/en-it.encoded.test1.log
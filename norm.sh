echo "normalizing en"
python normalize/normalize_embeddings.py unit center -i en.200k.300d.embeddings -o normalize/unit-center/en.emb.txt
echo "normalizing es"
python normalize/normalize_embeddings.py unit center -i en-es/es.200k.300d.embeddings -o normalize/unit-center/es.emb.txt
echo "normalizing it"
python normalize/normalize_embeddings.py unit center -i en-it/it.200k.300d.embeddings -o normalize/unit-center/it.emb.txt
echo "normalizing de"
python normalize/normalize_embeddings.py unit center -i en-de/de.200k.300d.embeddings -o normalize/unit-center/de.emb.txt
echo "normalizing fi"
python normalize/normalize_embeddings.py unit center -i en-fi/fi.200k.300d.embeddings -o normalize/unit-center/fi.emb.txt
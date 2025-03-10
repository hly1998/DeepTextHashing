# python Doc2Hash.py --dataset ng20.tfidf --bit 8
# python Doc2Hash.py --dataset ng20.tfidf --bit 16
# python Doc2Hash.py --dataset ng20.tfidf --bit 32
# python Doc2Hash.py --dataset ng20.tfidf --bit 64
# python Doc2Hash.py --dataset agnews.tfidf --bit 8
# python Doc2Hash.py --dataset agnews.tfidf --bit 16
# python Doc2Hash.py --dataset agnews.tfidf --bit 32
# python Doc2Hash.py --dataset agnews.tfidf --bit 64
# python Doc2Hash.py --dataset reuters.tfidf --bit 8
# python Doc2Hash.py --dataset reuters.tfidf --bit 16
# python Doc2Hash.py --dataset reuters.tfidf --bit 32
# python Doc2Hash.py --dataset reuters.tfidf --bit 64

# python Doc2Hash.py --dataset ng20.tfidf --bit 16
# python Doc2Hash.py --dataset ng20.tfidf --bit 32
# python Doc2Hash.py --dataset ng20.tfidf --bit 64
# python Doc2Hash.py --dataset agnews.tfidf --bit 16
# python Doc2Hash.py --dataset agnews.tfidf --bit 32
# python Doc2Hash.py --dataset agnews.tfidf --bit 64
# python Doc2Hash.py --dataset dbpedia.tfidf --bit 16 --epoch 10
# python Doc2Hash.py --dataset dbpedia.tfidf --bit 32 --epoch 10
# python Doc2Hash.py --dataset dbpedia.tfidf --bit 64 --epoch 10

python BVAE.py --dataset ng20.tfidf --bit 64
python BVAE.py --dataset agnews.tfidf --bit 16
python BVAE.py --dataset agnews.tfidf --bit 32
python BVAE.py --dataset agnews.tfidf --bit 64
python BVAE.py --dataset dbpedia.tfidf --bit 16 --epoch 10
python BVAE.py --dataset dbpedia.tfidf --bit 32 --epoch 10
python BVAE.py --dataset dbpedia.tfidf --bit 64 --epoch 10
python BVAE.py --dataset reuters.tfidf --bit 16
python BVAE.py --dataset reuters.tfidf --bit 32
python BVAE.py --dataset reuters.tfidf --bit 64
python BVAE.py --dataset tmc.tfidf --bit 64
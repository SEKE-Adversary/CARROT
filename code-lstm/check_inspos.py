from dataset import OJ104
import sys, gzip, pickle
sys.path.append("../preprocess-lstm")
import pattern

if __name__=="__main__":

	poj = OJ104(path="../data/oj.pkl.gz", max_len=50, vocab_size=10000)
	train, valid, test = poj.train, poj.dev, poj.test
	with gzip.open('../data/oj_inspos.pkl.gz', "rb") as f:
		instab = pickle.load(f)

	for _ in range(2):
		b = test.next_batch(1)
		stmt_ins_poses = instab['stmt_te'][b['id'][0]]
		x_raw = b['raw'][0]
		pattern._InsVis(x_raw, stmt_ins_poses)

	for _ in range(2):
		b = valid.next_batch(1)
		stmt_ins_poses = instab['stmt_tr'][b['id'][0]]
		x_raw = b['raw'][0]
		pattern._InsVis(x_raw, stmt_ins_poses)

	for _ in range(2):
		b = train.next_batch(1)
		stmt_ins_poses = instab['stmt_tr'][b['id'][0]]
		x_raw = b['raw'][0]
		pattern._InsVis(x_raw, stmt_ins_poses)

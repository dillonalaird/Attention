from data import data_iterator
from data import read_vocabulary


input_path  = "data/train.small.en"
output_path = "data/train.small.vi"

in_vocab = read_vocabulary("data/vocab.small.en")
out_vocab = read_vocabulary("data/vocab.small.vi")


i = 0
max_size = 20
batch_size = 10
for din, dout in data_iterator(input_path, output_path, in_vocab, out_vocab, max_size, batch_size):
    assert dout[0][0] == out_vocab["<s>"]
    assert len(dout[0]) == max_size
    assert len(dout) == batch_size
    i += 1
    if i == 10000: break

from __future__ import division
from __future__ import print_function


def pre_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[(max_len - len(lst)):] = lst
    return nlst


def post_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[:len(lst)] = lst
    return nlst


def read_vocabulary(data_path):
    return {w:i for i,w in enumerate(open(data_path).read().splitlines())}


def data_iterator(data_path_input, data_path_output, in_vocab, out_vocab, max_size, batch_size):
    with open(data_path_input, "rb") as f_in, open(data_path_output) as f_out:
        prev_batch = 0
        next_batch = 0
        data_in    = []
        data_out   = []
        for i, (lin, lout) in enumerate(zip(f_in, f_out)):
            if next_batch - prev_batch < batch_size:
                in_text = [in_vocab[w] if w in in_vocab else in_vocab["<unk>"]
                           for w in lin.replace("\n", "").split(" ")][:max_size][::-1]
                out_text = [out_vocab[w] if w in out_vocab else out_vocab["<unk>"]
                            for w in lout.replace("\n", " " + str(out_vocab["</s>"]))
                            .split(" ")][:max_size-1]
                out_text = [out_vocab["<s>"]] + out_text
                data_in.append(pre_pad(in_text, in_vocab["<pad>"], max_size))
                data_out.append(post_pad(out_text, out_vocab["<pad>"], max_size))
                next_batch += 1
            else:
                prev_batch = next_batch
                yield data_in, data_out
                data_in  = []
                data_out = []

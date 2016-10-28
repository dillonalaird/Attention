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


def data_iterator(source_data_path, target_data_path, source_vocab, target_vocab, max_size, batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path) as f_out:
        prev_batch = 0
        next_batch = 0
        data_in    = []
        data_out   = []
        for i, (lin, lout) in enumerate(zip(f_in, f_out)):
            if next_batch - prev_batch < batch_size:
                in_text = [source_vocab[w] if w in source_vocab else source_vocab["<unk>"]
                           for w in lin.replace("\n", "").split(" ")][:max_size][::-1]
                out_text = [target_vocab[w] if w in target_vocab else target_vocab["<unk>"]
                            for w in lout.replace("\n", " " + str(target_vocab["</s>"]))
                            .split(" ")][:max_size-1]
                out_text = [target_vocab["<s>"]] + out_text
                data_in.append(pre_pad(in_text, source_vocab["<pad>"], max_size))
                data_out.append(post_pad(out_text, target_vocab["<pad>"], max_size))
                next_batch += 1
            else:
                prev_batch = next_batch
                yield data_in, data_out
                data_in  = []
                data_out = []

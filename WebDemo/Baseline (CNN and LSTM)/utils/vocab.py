class Vocab:
    def __init__(self):
        self.sign2id = {
            "<s>": 0,
            "</s>": 1,
            "<pad>": 2,
            "<unk>": 3,
        }
        self.id2sign = dict((idx, token) for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self) -> int:
        return self.length
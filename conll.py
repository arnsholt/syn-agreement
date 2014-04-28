from codecs import open

# From string to <type>, with CoNLL null handling
def toint(s):
    if s == u"_": return None
    else:        return int(s)

def tostr(s):
    if s == u"_": return None
    else:        return s

def tolist(s):
    if s == u"_": return []
    else:        return s.split(u"|")

# To <type>, with null handling
def fromint(i):
    if i is None: return u"_"
    else:         return unicode(i)

def fromstr(s):
    if s is None: return "_"
    else:         return s

def fromlist(l):
    if l == []: return u"_"
    else:       return u"|".join(l)

class ConllError(Exception):
    def __init__(self, str):
        self.msg = str

    def __str__(self): return self.msg


class ConllToken:
    def __init__(self, line = u""):
        # CoNLLX data format members:
        self.id = 0
        self.word = None
        self.lemma = None
        self.cpos = None
        self.pos = None
        self.feats = []
        self.head = None
        self.deprel = None
        self.phead = None
        self.prel = None

        # Tree members:
        self.children = []

        if line != u"":
            parts = line.split()
            if len(parts) != 10:
                raise ConllError(u"Wrong number of parts in token: %d"%len(parts))

            self.id     = toint(parts[0])
            self.word   = tostr(parts[1])
            self.lemma  = tostr(parts[2])
            self.cpos   = tostr(parts[3])
            self.pos    = tostr(parts[4])
            self.feats  = tolist(parts[5])
            self.head   = toint(parts[6])
            self.deprel = tostr(parts[7])
            self.phead  = toint(parts[8])
            self.prel   = tostr(parts[9])

    def push(self, t):
        self.children.append(t)

    def __str__(self):
        if self.id == 0:
            return u"__ROOT__"
        else:
            return u"\t".join([fromint(self.id),
                              fromstr(self.word),
                              fromstr(self.lemma),
                              fromstr(self.cpos),
                              fromstr(self.pos),
                              fromlist(self.feats),
                              fromint(self.head),
                              fromstr(self.deprel),
                              fromint(self.phead),
                              fromstr(self.prel)
                             ])

class ConllSentence:
    def __init__(self):
        root = ConllToken()
        self.tokens = [root]
        self.comments = []

    def push_comment(self, c):
        self.comments.append(c)

    def push(self, t):
        self.tokens.append(t)

    def finish(self):
        for t in self.tokens[1:]:
            self.tokens[t.head].push(t)

    def __str__(self):
        l = []
        for c in self.comments:
            l.append(c)
        for t in self.tokens[1:]:
            l.append(unicode(t))
        l.append(u"")

        return u"\n".join(l)

class ConllCorpus:
    def __init__(self, input, name=None):
        if isinstance(input, str) or isinstance(input, unicode):
            def build_sentence(string):
                s = ConllSentence()

                for line in string.splitlines():
                    if line[0] == u'#':
                        s.push_comment(line)
                    else:
                        s.push(ConllToken(line))

                s.finish()
                return s

            self.sentences = [build_sentence(string) for string in
                    [sent for sent in open(input, u'r', u'UTF-8').read().split(u"\n\n") if len(sent) > 0]]
            self.name = name or input
        else:
            self.sentences = input
            self.name = name

    def __len__(self):               return len(self.sentences)
    def __getitem__(self, i):        return self.sentences[i]
    def __setitem__(self, i, value): self.sentences[i] = value
    def __delitem__(self, i):        del self.sentences[i]
    def __iter__(self):              return self.sentences.__iter__()
    def __reversed__(self):          return ConllCorpus(reversed(self.sentences), self.name)
    def __contains__(self, i):       return i in self.sentences

def read_corpus(file): return ConllCorpus(file)

def compare(A, B):
    # Let's make sure the corpora are the same lengths:
    if len(A) != len(B):
        raise ConllError("Can't compare corpora of different lengths (%d vs. %d)" % (len(A), len(B)))

    tokens = 0
    UAS = 0
    LAS = 0
    lbl = 0
    for a, b in zip(A, B):
        cmp = sentence_compare(a, b)
        tokens += cmp['tokens']
        UAS += cmp['UAS']
        LAS += cmp['LAS']
        lbl += cmp['lbl']

    return {'UAS': float(UAS)/tokens, 'LAS': float(LAS)/tokens, 'lbl': float(lbl)/tokens}

def pairwise_compare(*corpora):
    def do_pairs(*sentences):
        UAS = 0.0
        LAS = 0.0
        lbl = 0.0
        for i in xrange(len(sentences)):
            for j in xrange(i+1, len(sentences)):
                cmp = sentence_compare(sentences[i], sentences[j])
                UAS += cmp['UAS']
                LAS += cmp['LAS']
                lbl += cmp['lbl']
        n = len(sentences)
        n = n*(n-1)/2
        return {'UAS': UAS/n, 'LAS': LAS/n, 'lbl': lbl/n, 'tokens': cmp['tokens']}

    n = 0
    UAS = 0.0
    LAS = 0.0
    lbl = 0.0
    tokens = 0
    ignored = 0
    for sentences in zip(*corpora):
        sentences = [sent for sent in sentences if sent]
        if len(sentences) == 1:
            continue

        # Ignore sentences with tokenisation discrepancies
        try: cmp = do_pairs(*sentences)
        except: ignored += 1; continue
        n += 1
        UAS += cmp['UAS']
        LAS += cmp['LAS']
        lbl += cmp['lbl']
        tokens += cmp['tokens']

    return {'UAS': UAS/tokens, 'LAS': LAS/tokens, 'lbl': lbl/tokens, 'ignored': ignored}

def sentence_compare(a, b):
    if len(a.tokens) != len(b.tokens):
        raise ConllError("Can't compare sentences of different lengths (%d vs. %d)" % (len(a.tokens), len(b.tokens)))

    tokens = 0
    UAS = 0
    LAS = 0
    lbl = 0
    for ta, tb in zip(a.tokens[1:], b.tokens[1:]):
        tokens += 1
        if ta.head == tb.head:
            UAS += 1
            if ta.deprel == tb.deprel: LAS += 1
        if ta.deprel == tb.deprel: lbl += 1

    return {'tokens': tokens, 'UAS': UAS, 'LAS': LAS, 'lbl': lbl}

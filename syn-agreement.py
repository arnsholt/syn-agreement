#!/usr/bin/env pypy

try:
    import numpypy
except:
    pass
from alpha import krippendorff_alpha as alpha
import codecs
import conll
from conll import ConllCorpus
from getopt import getopt
import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.tree import Tree
from numpy import empty
import os
import sys
from zss.compare import simple_distance as distance

sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)

class Error(Exception):
    def __init__(self, str):
        self.msg = str

    def __str__(self): return self.msg

def die(message): raise Error(message)

def strdist(x, y):
    if x == y:
        return 0
    else:
        return 1

def delta_conll(a, b):
    return float(distance(a.tokens[0], b.tokens[0],
                          get_label=lambda t: t.deprel or "",
                          get_children=lambda t: t.children,
                          label_dist=strdist))

def delta_tree(a, b):
    return float(distance(a, b,
                          get_label=lambda t: t.label(),
                          get_children=lambda t: t,
                          label_dist=strdist))

def read_conll(file):
    return conll.read_corpus(file)

# This is a dummy subclass of list. We need to return one of these from
# read_tree because aggregate_tree needs to receive an object which it can set
# new attributes on, and you can't do that on instances of built-in classes
# (but you can on user-defined subclasses of them. Python internals
# weirdness.).
class FiddlyList(list): pass

def read_tree(file):
    def munge(t):
        if type(t) == Tree:
            toks = t.leaves()
            t = Tree(t.label(), [munge(child) for child in t])
            setattr(t, "tokens", toks)
            return t
        else:
            return Tree(t, [])

    return FiddlyList(munge(t) for t in BracketParseCorpusReader(".", file).parsed_sents())

def aggregate_conll(*dirnames):
    dirs = {dir: conlls(dir) for dir in dirnames}
    filenames = set()
    for dir in dirs.values():
        filenames = filenames | dir
    uniques = sorted(list(set(filenames)))

    def reader(dir, file):
        try:
            #file = file+os.path.basename(dir)+".conll"
            corpus = read_corpus(os.path.join(dir, file+os.path.basename(dir)+".conll"))
            corpus.name = file
            return corpus
        except IOError:
            return None

    def same_lengths(*corpora):
        n = 0
        for sents in zip(*corpora):
            l = len(sents[0].tokens)
            for s in sents[1:]:
                if len(s.tokens) != l: return n
            n += 1
        return n # This should never happen.

    # 1) Create map of dir => ConllCorpus|None
    data = {dir: [reader(dir, file) for file in uniques] for dir in dirs.keys()}

    # Basic sanity check and build dict of corpus lengths
    lengths = {}
    for tuple in zip(*data.values()):
        has_data = [c for c in tuple if c]
        first = has_data[0]
        length = len(first)
        name = first.name
        for c in has_data[1:]:
            if len(c) != length:
                die("Differing lengths for %s (first %d sentences are same length)" % (name, same_lengths(*has_data)))
        lengths[name] = length

    def conjoin(annotator, *corpora):
        catted = ConllCorpus([], annotator)
        for name, corpus in zip(uniques, corpora):
            if corpus is None:
                catted.sentences += [None] * lengths[name]
            else:
                catted.sentences += corpus.sentences

        return catted

    return [conjoin(name, *corpora) for name, corpora in data.items()]

def aggregate_tree(*dirnames):
    dirs = {dir: trees(dir) for dir in dirnames}
    filenames = set()
    for dir in dirs.values():
        filenames = filenames | dir
    uniques = sorted(list(set(filenames)))

    def reader(dir, file):
        if os.path.exists(os.path.join(dir, file+os.path.basename(dir)+".tree")):
            corpus = read_corpus(os.path.join(dir, file+os.path.basename(dir)+".tree"))
            corpus.name = file
            return corpus
        else:
            return None

    def same_lengths(*corpora):
        n = 0
        for sents in zip(*corpora):
            l = len(sents[0].tokens)
            for s in sents[1:]:
                if len(s.tokens) != l: return n
            n += 1
        return n # This should never happen.

    # 1) Create map of dir => BracketParseCorpusReader|None
    data = {dir: [reader(dir, file) for file in uniques] for dir in dirs.keys()}

    # Basic sanity check and build dict of corpus lengths
    lengths = {}
    for tuple in zip(*data.values()):
        has_data = [c for c in tuple if c]
        first = has_data[0]
        length = len(first)
        name = first.name
        for c in has_data[1:]:
            if len(c) != length:
                die("Differing lengths for %s (first %d sentences are same length)" % (name, same_lengths(*has_data)))
        lengths[name] = length

    def conjoin(annotator, *corpora):
        catted = []
        for name, corpus in zip(uniques, corpora):
            if corpus is None:
                catted += [None] * lengths[name]
            else:
                catted += corpus

        return catted

    return [conjoin(name, *corpora) for name, corpora in data.items()]

def conlls(dir):
    return set(file[:-(len(os.path.basename(dir))+6)] for file in sorted(os.listdir(dir)) if file.endswith('.conll'))

def trees(dir):
    return set(file[:-(len(os.path.basename(dir))+5)] for file in os.listdir(dir) if file.endswith('.tree'))

def jaccard(a, b):
    def number(tree):
        if len(tree) > 0:
            return Tree(tree.label(), [number(child) for child in tree])
        else:
            r = i[0]
            i[0] += 1
            return Tree(tree.label(), [r])

    def bracket_set(tree):
        brackets = set()
        for position in tree.treepositions():
            t = tree[position]
            if type(t) != Tree: continue
            (l, r) = (t.leaves()[0], t.leaves()[-1])
            brackets.add("%d,%d,%s"%(l, r, t.label()))
        return brackets

    # XXX: Silly hack to share state with number(), since I don't have time to
    # refactor it properly.
    i = [1] # List because Python 2 doesn't have nonlocal
    a = number(a)
    i = [1] # List because Python 2 doesn't have nonlocal
    b = number(b)

    bracket_a = bracket_set(a)
    bracket_b = bracket_set(b)
    intersection = bracket_a & bracket_b
    union = bracket_a | bracket_b

    return {'jaccard': float(len(intersection))/len(union), 'tokens': len(a.leaves())}

def pairwise_jaccard(*sentence_lists):
    def same_length(*sentences):
        l = len(sentences[0].leaves())
        for s in sentences[1:]:
            if l != len(s.leaves()):
                return False
        return True

    def do_pairs(*sentences):
        jacc = 0.0
        for i in xrange(len(sentences)):
            for j in xrange(i+1, len(sentences)):
                cmp = jaccard(sentences[i], sentences[j])
                jacc += cmp['jaccard']
        n = len(sentences)
        n = n*(n-1)/2
        return {'jaccard': jacc/n, 'tokens': cmp['tokens']}

    jacc = 0.0
    ignored = 0
    tokens = 0

    for sentences in zip(*sentence_lists):
        sentences = [sent for sent in sentences if sent]
        if len(sentences) == 1: continue
        if not same_length(*sentences):
            ignored += 1
            continue

        cmp = do_pairs(*sentences)
        l = cmp['tokens']
        jacc += l*cmp['jaccard']
        tokens += l

    score = jacc/tokens
    return {'UAS': score, 'LAS': score, 'lbl': score, 'ignored': ignored}

metrics = {'plain': lambda a, b: delta(a,b)**2,
           'diff':  lambda a, b: (delta(a,b)-abs(len(a.tokens)-len(b.tokens)))**2,
           'norm':  lambda a, b: (delta(a,b)/(len(a.tokens) + len(b.tokens)))**2}

# TODO: Labelled vs. unlabelled alpha
options, args = getopt(sys.argv[1:], 'x:', ['metric=', 'acc', 'dirs', 'tree', 'conll', 'help'])
options = {key: value for key, value in options}

if '--help' in options:
    print "Usage: %s [--tree|--conll] [--acc] [--metric=plain|diff|norm|all] fileA fileB" % sys.argv[0]
    print "       %s [--tree|--conll] [--acc] [--metric=plain|diff|norm|all] --dirs dir..." % sys.argv[0]
    sys.exit()

if '--tree' in options and '--conll' in options:
    raise Error("--tree and --conll' are mutually exclusive")
elif '--tree' in options:
    read_corpus = read_tree
    delta       = delta_tree
    aggregate   = aggregate_tree
    # Because NLTK's trees aren't normally hashable and I'm too lazy to
    # convert everything into an ImmutableTree (which is).
    Tree.__hash__ = lambda t: hash( (t.label(), tuple(t)) )
else:
    read_corpus = read_conll
    delta       = delta_conll
    aggregate   = aggregate_conll

metric = options.get('--metric', 'plain')
if metric != 'all' and metric not in metrics:
    raise Error("Unknown metric: %s" % metric)

# TODO: Reading both CoNLL graphs and bracketed trees. Just rely on file
# extensions, or do something more clever to detect file types?
if '--dirs' not in options:
    corpusA = read_corpus(args[0])
    corpusB = read_corpus(args[1])

    if len(corpusA) != len(corpusB):
        print >> sys.stderr, "Corpora have different lengths"
        sys.exit(1)

    corpora = [corpusA, corpusB]
else:
    corpora = aggregate(*[os.path.normpath(dir) for dir in args])

if '--acc' in options:
    if '--tree' in options:
        cmp = pairwise_jaccard(*corpora)
    else:
        cmp = conll.pairwise_compare(*corpora)

if metric == 'all':
    for name, d in metrics.items():
        print "\\alpha_%s=%f" % (name, alpha(corpora, metric=d, convert_items=lambda s: s))

    if '--acc' in options:
        print "UAS: ", cmp['UAS']
        print "LAS: ", cmp['LAS']
        print "lbl: ", cmp['lbl']
        if cmp.get('ignored', 0) > 0:
            print "Ignored %d sentences in accuracy computation" % cmp['ignored']
else:
    print alpha(corpora, metric=metrics[metric], convert_items=lambda s: s),

    if '--acc' in options:
        print cmp['UAS'], cmp['LAS'], cmp['lbl'],

    print ''

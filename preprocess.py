#该文件的目标是获取三个数据，1、词向量 2、字向量 ==>3、输入数据的向量化表示，这些内容都存放在一个json文件中
import numpy as np
import hyperparams as hp
import msgpack
import json
import jieba

#step1
def build_CharVocab(sourcePath):
    vocab = {}
    fw = open(sourcePath,'r',encoding='utf-8')
    content = fw.read()
    for c in content:
        if c == ' ':
            continue
        if c in vocab.keys():
            vocab[c] += 1
        else:
            vocab[c] = 1

    vocabs = []
    for k, v in vocab.items():
        if v > 5:
            vocabs.append(k)

    vocabs.append(' ')
    vocabs.append('<UNK>')

    #store vocabs of char
    # fr = open('resource/char_vocab.txt','w',encoding='utf-8')
    # fr.write('\n'.join(vocabs))

    #store char embedding
    setUp_charEmbedding(vocabs)

def setUp_charEmbedding(vocabs):
    embedding_table = np.reshape(np.random.rand(hp.Hyperparams.char_dimension * len(vocabs)),
                                 newshape=[len(vocabs),hp.Hyperparams.char_dimension])
    vocabs_dict = {}
    for i, c in enumerate(vocabs):
        vocabs_dict[c] = ' '.join(map(str,embedding_table[i]))

    with open('resource/char_embedding.msgpack','w',encoding='utf-8') as fw:
        json.dump(vocabs_dict,fw)

    with open('resource/char_embedding.msgpack', 'r',encoding='utf-8') as fr:
        vocab_ = json.load(fr)
        assert len(vocab_.keys()) == len(vocabs), ValueError("The number of read msgpack is not equal to inital number, that is :{0}".format(len((vocab_.keys()))))

# sourcePath = 'resource/corpus.txt'
# build_CharVocab(sourcePath)

#step2 prepare char_input
def setUp_inputs(trainPath = None, valPath = None, testPath = None):
    #read word and char info
    f_char = open('resource/char_embedding.msgpack','rb')
    charEmbedding = msgpack.load(f_char,encoding='utf-8')
    if '<UNK>' not in charEmbedding.keys():
        charEmbedding['<UNK>'] = '\t'.join(map(str,np.random.rand((hp).Hyperparams.char_dimension)))
    charVocab = charEmbedding.keys()
    assert len(charVocab) == hp.Hyperparams.char_vocab_size, ValueError('the number of char vocab is wrong, {0}'.format(len(charVocab)))

    f_word = open('resource/word_embedding.json', 'r', encoding='utf-8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(hp.Hyperparams.word_dimension)])
    wordVocab = wordEmbedding.keys()

    assert '<UNK>' in wordEmbedding.keys(), ValueError('space and unk not in word dict')
    assert len(wordVocab) == hp.Hyperparams.word_vocab_size, ValueError('the number of char vocab is wrong, {0}'.format(len(wordVocab)))


    # fw = open('resource/inputs_simpleRun.json', 'w',encoding='utf-8')
    train = ""
    test = ""
    val = ""
    if trainPath:
       train = _setUp_inputs_(trainPath,wordEmbedding, wordVocab, charEmbedding,charVocab)
    if testPath:
       test = _setUp_inputs_(testPath, wordEmbedding, wordVocab, charEmbedding, charVocab)
    if valPath:
       val = _setUp_inputs_(valPath, wordEmbedding, wordVocab, charEmbedding, charVocab)
    env = {'train': train, 'test': test, 'val': val}
    return env
    # json.dump(env, fw)


def _setUp_inputs_(sourcePath, wordEmbedding, wordVocab, charEmbedding,charVocab):
    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    for line in lines:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4")
            fact_input = processText(items[1],wordEmbedding, wordVocab, charEmbedding,charVocab)
            law_input = processText(items[2],wordEmbedding, wordVocab, charEmbedding,charVocab)
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input, label])
    return result



def processText(line,wordEmbedding, wordVocab, charEmbedding,charVocab):
    initContent = line.strip()
    if initContent != "":
        content = jieba.cut(initContent)
        lines = list(map(lambda x: str(x).strip(), content))
        contentcut = list(filter(lambda x: x != "", lines))
        wordEmbs = []
        charEmbs = []
        for word in contentcut:
            wordEmb = processWord(word,wordEmbedding,wordVocab)
            charEmb = processChars(word, charEmbedding, charVocab)
            wordEmbs.append(wordEmb)
            charEmbs.append(charEmb)
        return {'word_input': wordEmbs, 'char_input': charEmbs}
    return []


def processChars(word, char_embedding, vocabs):
    embeddings = []
    for c in word:
        if c not in vocabs:
            embeddings.append(getVector(char_embedding['<UNK>']))
        else:
            embeddings.append(getVector(char_embedding[c]))

    embeddings = '\t'.join(map(str,np.amax(embeddings,axis=0)))
    return embeddings

def getVector(str_vector):
    vectors = str_vector.split('\t')
    vectors = list(map(float, map(lambda x:x.strip(),filter(lambda x: x.strip() != '', vectors))))
    return vectors

def processWord(word, word_embedding, vocabs):
    if word not in vocabs:
        return word_embedding['<UNK>']
    else:
        return word_embedding[word]


def buildWordEembeddingFile():
    fw = open('resource/word_embedding.json', 'w',encoding='utf-8')

    with open('resource/vectors_w2v.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        w2v_dict = {}
        for line in lines:
            line = line.strip()
            if line != "":
                units = line.split()
                word = units[0]
                vector = units[1:]
                assert len(vector) == hp.Hyperparams.word_dimension, ValueError('vector dimension is wrong, that is {0}'.format(len(vector)))
                w2v_dict[word] = '\t'.join(vector)

        if '<UNK>' not in w2v_dict.keys():
            w2v_dict['<UNK>'] = '\t'.join(['0' for _ in range(hp.Hyperparams.word_dimension)])

        assert '<UNK>' in w2v_dict.keys(), ValueError('space and unk not in word dict')
        json.dump(w2v_dict, fw)

# # buildWordEembeddingFile()
# trainPath = 'resource/train-原始.txt'
# valPath = 'resource/val-原始.txt'
# testPath = 'resource/test-原始.txt'
# setUp_inputs(trainPath=trainPath, valPath=valPath, testPath=testPath)




















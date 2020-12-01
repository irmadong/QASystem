import nltk
import re
from gensim import corpora
from gensim import similarities
from gensim import models
from ordered_set import OrderedSet
#new location  features

stop_words = set(nltk.corpus.stopwords.words('english'))
kept_words = ['when','what','why','who','where','which','whom','how']
for ele in kept_words:
    stop_words.remove(ele)
kept_words.append("name")

def preprocessing_question(filename):
    """
    Preprocessing the question file
    @param filename: the path to the question file
    @type filename: string
    @return: questions separated to words; remove stopwords and punctuation
    @rtype: dictionary with keys of question number and values of list
            of words in each question
    """
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    question_training = {}
    q_num = 0
    with open(filename, 'rt',encoding = 'utf-8-sig') as file:
        try:
            while True:
                current_line = next(file)
                if "Number" in current_line:
                    word_list = current_line.split(": ")
                    q_num = int(word_list[1])
                else:
                    temp=[]
                    word_tokens = tokenizer.tokenize(current_line)
                    filter_sent = [word for word in word_tokens if not word in stop_words]
                    temp.append(filter_sent)
                    question_training[q_num] = temp
                    next(file)
        except StopIteration:
            return question_training


new_added_stop_words = ['DATE','SECTION','P','LENGTH','HEADLINE','BYLINE','TEXT','DNO','TYPE','SUBJECT','DOC','DOCID','COUNTRY','EDITION','NAME','PUBDATE',
               'DAY','MONTH','PG','COL','PUBYEAR','REGION','FEATURE','STATE','WORD','CT','DATELINE','COPYRGHT','LIMLEN','LANGUAGE','FILEID','FIRST','SECOND',
                        'HEAD','BYLINE','HL','DOCNO',"PHRASE","F","HT","TP","AU","TI","H3","CN"]
for ele in new_added_stop_words:
    stop_words.add(ele)


def chunks(list, n):
    final_list =[]

    for i in range(0, len(list), n):
        temp_list = list[i:i+n]
        final_list.append(temp_list)

    return final_list

def document_sep(filename,question,id,n=20,num_chunks=20):
    '''
    This function separate all related documents of a given question into
    n-tokens block and compare the similarity between the question and the
    n-token block. Then, this functions returns the top n most similar blocks.

    :param filename: The path to the document file of a given question
    :param question: the question
    :param id: question id
    :param n: the number of tokens in each candidate passage
    :param num_chunks: number of blocks that are returned
    :return: return a list of list with top-n-scored candidate passages

    '''
    doc = []
    # dict_document_tokenblocks ={}
    with open(filename, 'r', encoding='latin-1') as f:
        large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+\smillion|\d+,?\d+|\w+')
        start_tag = ("<DOCNO>", "<DOCID>", "<FILEID>", "<FIRST>", "<SECOND>", "<HEAD>",
                     "<NOTE>", "<BYLINE>", "<DATELINE>", "<DATE>", "<SECTION>",
                     "<LENGTH>", "<HEADLINE>", "<ACCESS>", "<CAPTION>", "<DESCRIPT>",
                     "<LEADPARA>", "<MEMO>", "<COUNTRY>", "<CITY>", "<EDITION>", "<CODE>",
                     "<NAME>", "<PUBDATE>", "<DAY>", "<MONTH>", "<PG.COL>", "<PUBYEAR>",
                     "<REGION>", "<FEATURE>", "<STATE>", "<WORD.CT>", "<COPYRGHT>",
                     "<LIMLEN>", "<LANGUAGE>", "<EDITION>", "<DNO>", "<TYPE>", "<SUBJECT>", "<HL>","<GRAPHIC>",
                     "<XX>","<F P=103>")
        end_tag = ("</DOCNO>", "</DOCID>", "</FILEID>", "</FIRST>", "</SECOND>", "</HEAD>",
                   "</NOTE>", "</BYLINE>", "</DATELINE>", "</DATE>", "</SECTION>",
                   "</LENGTH>", "</HEADLINE>", "</ACCESS>", "</CAPTION>", "</DESCRIPT>",
                   "</LEADPARA>", "</MEMO>", "</COUNTRY>", "</CITY>", "</EDITION>", "</CODE>",
                   "</NAME>", "</PUBDATE>", "</DAY>", "</MONTH>", "</PG.COL>", "</PUBYEAR>",
                   "</REGION>", "</FEATURE>", "</STATE>", "</WORD.CT>", "</COPYRGHT>",
                   "</LIMLEN>", "</LANGUAGE>", "</EDITION>", "</DNO>", "</TYPE>", "</SUBJECT>",
                   "</HL>","</GRAPHIC>","</XX>","</F>")


        try:
            # current_line = next(f)
            while True:
                current_line = next(f)
                #print(current_line)
                if "Rank" in current_line:

                    next(f)

                else:

                    if current_line.startswith(start_tag):

                        if any(s in current_line for s in end_tag):
                            continue
                        else:
                            while True:
                                current_line = next(f)

                                if any(s in current_line for s in end_tag):
                                    break
                                else:
                                    continue
                    else:

                        current_tokens = large_scale_tokenizer.tokenize(current_line)
                        filter_token = [word for word in current_tokens if not word in stop_words]

                        doc.extend(filter_token)

        except StopIteration:

            filtered_sentences = chunks(doc, num_chunks)
            dictionary = corpora.Dictionary(filtered_sentences)
            corpus = [dictionary.doc2bow(text)for text in filtered_sentences]
            lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
            current_question = question[id][0]
            vec_bow = dictionary.doc2bow(current_question)
            vec_lsi = lsi[vec_bow]
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save('/tmp/deerwester.index')
            index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            i = 0
            top_n_passages = []
            for i, s in enumerate(sims):
                if i<n:
                    top_n_passages.append(filtered_sentences[i])
                else:
                    break
                i+=1
            return top_n_passages

import copy
#add pos tag for questions
def get_question_with_tag(question):
    '''
    This funtion add pos tags for questions
    :param question a dictionary of questions; keys are question numbers;
                values are question contents.
    :return: a dictionary of questions with tags; keys are question numbers;
            values are list of tuples, with words and pos tags in tuple
    '''
    question_with_tag = {}
    for ques_number in question.keys():
        question_with_tag[ques_number] = nltk.pos_tag(question[ques_number][0])
    return question_with_tag

#todo lower case
#if keys found, find the keys
easy_question_type={"where":(["ORGANIZATION", "GPE"],False),"when":(["CD"],True),"who":(["PERSON"],False)}
gep_what = ["continent","nationality","city","province","state","tourist","area","cities","mountain"]
org_what = ["Cruise","university","airport","radio"]
cd_what=["population","year","zip","salary"]
#todo all abbreviation
def question_extraction(question, question_with_tag):
    '''
       This function determines what type of answers each question will have.

       :param question: original dictionary of questions; keys are question id;
               values are list of words in the question
       :param question_with_tag: questions with added pos tag
       :return: a dictionary with keys of question number; values of list of three
               elements: first, a list of words in questions; second, answer type,
               a list of answer tags we are looking for; third, a flag indicating
               whether it will use a noun phrase pattern
       '''

    answer_key = []
    np=True
    for num in question_with_tag.keys():
        quest = question_with_tag[num]

        for i in range(0,len(quest)):

            if quest[i][0].lower() in easy_question_type.keys():

                answer_key = easy_question_type[quest[i][0].lower()][0]
                np= easy_question_type[quest[i][0].lower()][1]
                break

            elif re.search('name',quest[i][0].lower(),flags=re.IGNORECASE):
                if quest[i+1][0] == "city":
                    np = False
                    answer_key = ["GPE"]
                    break
                answer_key = ['NNP_Pattern']
                np = True
                break

            elif re.search('what',quest[i][0].lower(), flags=re.IGNORECASE):

                if quest[i+1][0] in gep_what:
                    np = False
                    answer_key = ["GPE"]
                    break

                elif (quest[i+1][0] in cd_what ):
                    answer_key = ["CD"]
                    np = True
                    break
                elif quest[i+1][1] == "JJ":
                    if quest[i+2][0] == "model":
                        np = False
                        answer_key = ["PERSON"]
                        break
                    if quest[i+2][0] == "name":
                        np=True
                        answer_key = ['NNP_Pattern']
                        break
                    elif quest[i+2][0] == "tourist":
                        np = False
                        answer_key = ["GPE"]
                        break
                elif quest[i + 1][1] == "DT":
                    if quest[i+2][0] == "name":
                        np = True
                        answer_key = ['NNP_Pattern']
                        break

                elif quest[i+1][0] in org_what:
                    np = False
                    answer_key = ["ORGANIZATION"]
                    break

                elif quest[i+1][0] == "king":
                    np = False
                    answer_key = ["PERSON"]
                    break
                elif quest[i+1][0] == "name":
                    answer_key = ['NNP_Pattern']
                    np = True
                    break

                else:
                    answer_key = ["NNP","NN","NNS"]
                    np = True
                    break
            elif quest[0][0] == "How":
                if quest[1][0] == ("many" or "much"):
                    np = True
                    answer_key = ["CD"]
                    break
                np = True
                answer_key = ["NNP", 'NN','NNS']
                break
            else:
                np = True
                answer_key = ["NNP", 'NN','NNS']
                break

        question[num].append(answer_key)
        question[num].append(np)

    return question

def clean_question(question):
    '''
    This functions removes what, when, why, who, how, whom, where, name from
    contents of questions.

    :param question: a dictionary with keys of question id and values of list
            of words in question.
    :return: a dictionary of question after removing the words indicated above
    '''
    new_ques = copy.deepcopy(question)
    for quest in new_ques.values():
        for ele in quest[0]:
            if ele.lower() in kept_words:
                quest[0].remove(ele)
    return new_ques
import math

def find_location (list_index):
    '''
    This function computes the averaged location of a list of indices of words
    :param list_index: a list of indices of words
    :return: an averaged location
    '''

    return sum(list_index)/len(list_index)

def find_closest_distance_to_any_keyword(loc_ans, list_index):
    '''
    This function finds the distance of an answer to the closest keywords.

    :param loc_ans: location of the answer
    :param list_index: list of index of all keywords
    :return: the distance between the the answer with the closest keywords
    '''

    distance = math.inf
    for ele in list_index:
        current_distance = abs(loc_ans - ele)
        if distance > current_distance:
            distance = current_distance
    return distance



def ne_pattern_extraction(ele,tag_looking_for,keyword_list,list_index,answer_list,list_answer_type,list_of_locality):
    '''
    This function finds the answer which matches the ne pattern
    in the candidate passage .

    :param ele: one block in top n candidate blocks
    :param tag_looking_for: the answer tag we are looking for
    :param keyword_list: list of keywords in questions
    :param list_index: list of indices of keywords
    :param answer_list: list of answers
    :param list_answer_type: list of answer tags
    :param list_of_locality: list of location of answers
    :return: if we find the answer, return true; else return false
    '''
    find_tag = False
    token_pos_list = nltk.pos_tag(ele)
    passage_with_tag = nltk.ne_chunk(token_pos_list)

    tracker = 0
    for child in passage_with_tag:
        if type(child) == nltk.tree.Tree and child.label() in tag_looking_for:

            current_list= []
            for x in child.leaves():
                if x[0] in keyword_list:
                    current_list = []
                    break
                else:
                    current_list.append(x[0])

            if len(current_list) > 0:
                x = ' '.join(current_list)
                answer_list.append(x)
                num_token = len(child.leaves())
                list_answer_type.append(1.0)
                loc_answer = (tracker + tracker + num_token - 1) / 2
                distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                list_of_locality.append(distance)
                tracker += num_token
                find_tag = True
        else:
            tracker += 1

    return find_tag


def find_cd_answer (passage_with_tag,keyword_list,answer_list,tracker,list_answer_type,list_index,list_of_locality):
    '''
    This function finds the answer which matches the cd tag,
    in the candidate passage .

    :param passage_with_tag: one block of top n blocks with tags
    :param keyword_list: list of keywords in answers
    :param answer_list: list of answers
    :param tracker: tracker which tracks the location of the answer
    :param list_answer_type: list of answer tags
    :param list_index: list of indices of keywords
    :param list_of_locality: list of location of answers
    :return: if we find the answer with the tag of CD , return true; else return false
    '''
    find_tag = False
    #find token with tag CD
    for token in passage_with_tag:
        if token[1] =="CD":
            #eliminate the key word
            if token[0] in keyword_list:
                continue
            #append the answer_list
            answer_list.append(token[0])
            list_answer_type.append(1.0)

            loc_answer = tracker
            distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
            list_of_locality.append(distance)
            find_tag = True
        tracker += 1
    return find_tag

def different_pattern_np_extractor(ele,keyword_list,list_index,list_answer_type,answer_list,list_of_locality,type_match,pattern_id):
    '''
      This function finds the answer which matches the ne pattern
    in the candidate passage .

    :param ele: one block in top n candidate blocks
    :param keyword_list: list of keywords in questions
    :param list_index: list of indices of keywords
    :param list_answer_type: list of answer tags
    :param answer_list: list of answers
    :param list_of_locality: list of locations of answers
    :param type_match: if the answer type matches the type we are looking for,
            type_match is 1.0; else type_match is 0.0
    :param pattern_id: the specific pattern we are looking for
    :return: if we find the answer with that pattern , return true; else return false
    '''
    np_parser_list = []
    t_list = []
    find_tag = False
    if pattern_id == 1:
        pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NNP>+}']

    elif pattern_id ==2:
        pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>+}', 'NP: {<DT>?<NN|NNP|NNS>+}']

    else:
        pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>}']


    for pattern in pattern_list:
        np_parser_list.append(nltk.RegexpParser(pattern))


    for np_parser in np_parser_list:
        t_list.append(np_parser.parse(nltk.pos_tag(ele)))

    for t in t_list:
        tracker = 0
        for child in t:
            if type(child) == nltk.tree.Tree:
                current_list = []
                for x in child.leaves():
                    if x[0] in keyword_list:
                        current_list = []
                        break
                    else:
                        current_list.append(x[0])

                if len(current_list) > 0:
                    x = ' '.join(current_list)
                    answer_list.append(x)
                    num_token = len(child.leaves())
                    list_answer_type.append(type_match)
                    loc_answer = (tracker + tracker + num_token - 1) / 2
                    distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                    list_of_locality.append(distance)
                    tracker += num_token
                    find_tag = True
            else:
                tracker+=1
    return find_tag

def answer_extract(top_n_passages, single_question):
    """
    This function extracts answers from all candidate blocks.

    @param top_n_passages: top n most similary blocks to question
    @type top_n_passages: list of list
    @param single_question: the question we are looking for the answer
    @type single_question:list of lists element 1: all key words,
     element two: answer type, element three: nl or not
    @return: answer_list, list of answer_type(0/1), list of locality
    @rtype: dictionary
    """

    answer_list = []
    list_answer_type =[]
    list_of_locality = []
    np = single_question[2]
    tag_looking_for = single_question[1]
    keyword_list = single_question[0]
    find_tag = False
    for ele in top_n_passages:

        #calculate location
        i = 0
        list_index = []

        for word in ele:

            if word in keyword_list:
                list_index.append(i)
            i+=1

        if np:

            passage_with_tag = nltk.pos_tag(ele)
            tracker = 0

            if tag_looking_for[0] == "CD":

                find_tag=find_cd_answer(passage_with_tag,keyword_list,answer_list,tracker,list_answer_type,list_index,list_of_locality)

            elif tag_looking_for[0] == "NNP_Pattern":

                find_tag = different_pattern_np_extractor(ele,keyword_list,list_index,list_answer_type,answer_list,list_of_locality,1.0,1)
            else:
                find_tag = different_pattern_np_extractor(ele,keyword_list, list_index, list_answer_type, answer_list,
                                                          list_of_locality, 1.0, 2)

        else:
            #find ne tag

            find_tag = ne_pattern_extraction(ele,tag_looking_for,keyword_list,list_index,answer_list,list_answer_type,list_of_locality)

        if find_tag==False:

             different_pattern_np_extractor(ele,keyword_list, list_index, list_answer_type, answer_list,
                                                      list_of_locality, 0.0, 3)

    return answer_list,list_answer_type,list_of_locality


import numpy as np
def ranking(list_answer_type, list_of_locality,N):
    '''
    This function ranks all the candidate answers. The ranking is determined
    with 0.9 weight of whether the type matches the type we want; and 0.1
    weight of how close the answer is to the keywords

    :param list_answer_type: list of 0.0 and 1.0; 1.0 indicates the answer matches
            the type we want; 0.0 indictes it does not match.
    :param list_of_locality: list of locations of answers
    :param N: top n answers we want
    :return: list of top n answers
    '''
    weighted_ans_type = [ele*0.9 for ele in list_answer_type]
    weighted_locality = [ele*0.1 for ele in list_of_locality]
    weighted_value = np.subtract(weighted_ans_type,weighted_locality)
    top_n_indices = np.argsort(weighted_value.tolist())[-N:]
    result = top_n_indices.tolist()
    result.reverse()
    return result

#todo :evaluation_results = 0.9*list_answer_type + 0.1* list_of_locality
# add tags using spacy

def write_file(answer,filename):
    '''
    This function writes the results to a file.
    :param answer: list of final answer
    :return: None
    '''
    f = open(filename, "w")
    for elem in answer:
        f.write("qid " + str(elem))
        f.write("\n")
        for ans in answer[elem]:
            f.write(ans)
            f.write("\n")
    f.close()
def predict(question, topn, chunk,training):
    '''
    This functions runs the whole QA system.

    :param question: dictionary of all questions
    :param topn: top n most similar blocks to question
    :param chunk: number of tokens in a block
    :return: None
    '''
    answer = {}
    n_list = list(question.keys())

    for n in n_list:
        if training:
            filename = './training/topdocs/top_docs.' + str(n)
            prediction_file_name = "prediction_training_file.txt"
        else:
            filename = './test/topdocs/top_docs.' + str(n)
            prediction_file_name = "prediction_test_file.txt"

        filter_question = clean_question(question)
        top_n_passages = document_sep(filename, filter_question, n, topn, chunk)



        question_with_tag = get_question_with_tag(question)



        full_question_info = question_extraction(filter_question, question_with_tag)

        answer_list_0, list_answer_type_0, list_locality_0 = answer_extract(top_n_passages, full_question_info[n])

        ranking_top_n_indices = ranking(list_answer_type_0, list_locality_0, 40)

        #remove duplicate
        ordered_final_answer_n = OrderedSet()
        for ele in ranking_top_n_indices:
            ordered_final_answer_n.add(answer_list_0[ele])
        temp_answer = list(ordered_final_answer_n)
        final_answer = [temp_answer[i]for i in range(10)]
        answer[n] = final_answer
    write_file(answer,prediction_file_name)


if __name__ == "__main__":

    path_training = './training/qadata/questions.txt'
    question_training = preprocessing_question(path_training)
    predict(question_training,27,40,1)
    path_test = './test/qadata/questions.txt'
    question_test = preprocessing_question(path_test)
    predict(question_test, 27,40, 0)




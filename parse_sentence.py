from __future__ import unicode_literals, print_function
from lxml import etree
import operator as op
#import of nlputils

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from protolib.python import document_pb2, rpc_pb2
from utils.rpc import grpcapi

def process_one_document(request):
    # Use biotm2 as server.
    interface = grpcapi.GrpcInterface(host='128.4.20.169')
    # interface = grpcapi.GrpcInterface(host='localhost')
    response = interface.process_document(request)
    assert len(response.document) == 1
    return response.document[0]


def parse_using_bllip(doc):
    request = rpc_pb2.Request()
    request.request_type = rpc_pb2.Request.PARSE_BLLIP
    request.document.extend([doc])
    return process_one_document(request)


def parse_using_stanford(doc):
    request = rpc_pb2.Request()
    request.request_type = rpc_pb2.Request.PARSE_STANFORD
    request.document.extend([doc])
    return process_one_document(request)


def split_using_stanford(doc):
    request = rpc_pb2.Request()
    request.request_type = rpc_pb2.Request.SPLIT
    request.document.extend([doc])
    return process_one_document(request)


def run(text):

    raw_doc = document_pb2.Document()
    raw_doc.doc_id = '26815768'
    raw_doc.text = text

    # Parse using Bllip parser.
    result = parse_using_bllip(raw_doc)
    #print(result)
    return result
    # Parse Using Stanford CoreNLP parser.
    #result = parse_using_stanford(raw_doc)
    #print(result)

    # Only split sentences using Stanford CoreNLP.
    #for i in range(100):
    #    result = split_using_stanford(raw_doc)
    #    print('Split {} documents'.format(i))

def extract_sentence_info(inputfile,outputfile):
    tree = etree.parse(inputfile)
    root = tree.getroot()
    file_object=open(outputfile,'w+')
    for docu in root.iter("document"):
        #iterate every document
        for sent in docu.iter("sentence"):
            #iterate all the sentence
            #print(sent.get("text"))
            ids_all=[]
            protein_names=[]
            protein_start=[]
            protein_end=[]
            for enti in sent.iter("entity"):
                ids_all.append(enti.get("id"))
                offset_str=enti.get("charOffset")
                ind=offset_str.find("-")
                comma_index=offset_str.find(",")
                protein_names.append(enti.get("text"))
                               
                if comma_index==-1:
                    len_offset_str=len(offset_str)  
                    protein_start.append(int(offset_str[0:ind]))                  
                    protein_end.append(int(offset_str[ind+1:len_offset_str]))
                else:
                    len_offset_str=comma_index
                    protein_start.append(int(offset_str[0:ind]))  
                    protein_end.append(int(offset_str[ind+1:len_offset_str]))


            #print("protein_offset:",protein_offset)
            #parse the sentence here
            parse_result=run(sent.get("text"))
            #get the incoming dependency information
            incoming_dependency={}
            head_word={}
            for sente in parse_result.sentence:
                 for dep in sente.dependency:
                     d_index=getattr(dep,'dep_index')
                     # print(type(d_index))
                     if d_index not in incoming_dependency.keys():
                         incoming_dependency[d_index]=dep.relation
                     if d_index not in head_word.keys():
                         head_word[d_index]=dep.gov_index
                     #if d_index==dep.gov_index:
                     #    head_word[d_index]='ROOT'
                     #else:
                     #    head_word[d_index]=parse_result.token[dep.gov_index].word
                 #incoming_dependency[len(incoming_dependency)]='None'
            #print(incoming_dependency)
            #file object
            
            for i in range(len(ids_all)):
                 for j in range(i+1,len(ids_all)):
                    ids=[ids_all[i],ids_all[j]]
                    ids.sort()
                    #print(ids)
                    interaction_relation =False
                    for inter in sent.iter("interaction"):
                        id_inter=[inter.get("e1"),inter.get("e2")]
                        id_inter.sort()
                        if op.eq(ids,id_inter):
                            interaction_relation=True

                    #get the index of two proteins
                    #in some cases, there are two proteins having the same name and in different locations
                    protein_index_start_1=len(parse_result.token)
                    protein_index_end_1=0
                    protein_index_start_2=len(parse_result.token)
                    protein_index_end_2=0

                    for tok in parse_result.token:
                        #print(type(tok.index))
                        
                        if tok.word in protein_names[i] and int(tok.char_start)>=protein_start[i] and int(tok.char_end)<=protein_end[i]:
                            protein_index_start_1=min(tok.index,protein_index_start_1)
                            protein_index_end_1=max(tok.index,protein_index_end_1)
                        elif tok.word in protein_names[j] and int(tok.char_start)>=protein_start[j] and int(tok.char_end)<=protein_end[j]:
                            protein_index_start_2=min(tok.index,protein_index_start_2)
                            protein_index_end_2=max(tok.index,protein_index_end_2)
                    #print("Index info:")
                    #print(protein_index_start_1,"-",protein_index_end_1,",",protein_index_start_2,"-",protein_index_end_2)
                    #print("Protein names:",protein_names)        
                    #print("Protein start:",protein_start)
                    #print("Protein end:",protein_end)
                    #generate the sentence information
                    token_list=[]
                    for ii in range(len(parse_result.token)):
                        toke=parse_result.token[ii]
                        #judge a token is a protein or not
                        token_is_protein=False
                        for kk in range(len(protein_names)):
                            if toke.word in protein_names[kk] and int(toke.char_start)>=protein_start[kk] and int(toke.char_end)<=protein_end[kk]:
                                #print("Word:",toke.word)
                                #print(int(toke.char_start),"-",protein_start[kk],",",int(toke.char_end),"-",protein_end[kk])
                                token_is_protein=True

                        if ii not in incoming_dependency.keys():
                            incoming_dependency[ii]='None'
                        if ii not in head_word.keys():
                            head_word[ii]='None'
                        # deicide the index of relative position
                        if toke.index-protein_index_start_1<0:
                            index_1=toke.index-protein_index_start_1
                        else:
                            index_1=toke.index-protein_index_end_1
                        if toke.index-protein_index_start_2<0:
                            index_2=toke.index-protein_index_start_2
                        else:
                            index_2=toke.index-protein_index_end_2

                        if toke.word in protein_names[i] and int(toke.char_start)>=protein_start[i] and int(toke.char_end)<=protein_end[i]:

                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT1"+"|"+str(0)+"|"+str(index_2)+"|"+incoming_dependency[ii]+"|"+str(head_word[ii]))
                        elif toke.word in protein_names[j] and int(toke.char_start)>=protein_start[j] and int(toke.char_end)<=protein_end[j]:

                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT2"+"|"+str(index_1)+"|"+str(0)+"|"+incoming_dependency[ii]+"|"+str(head_word[ii]))
                        elif token_is_protein:
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT"+"|"+str(index_1)+"|"+str(index_2)+"|"+incoming_dependency[ii]+"|"+str(head_word[ii]))
                        else:
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"O"+"|"+str(index_1)+"|"+str(index_2)+"|"+incoming_dependency[ii]+"|"+str(head_word[ii]))
                    if interaction_relation is True:
                        file_object.write('Positive ')
                        print(token_list)
                        for t in token_list:
                            file_object.write(t)
                            file_object.write(' ')
                        file_object.write('\n')
                    else:
                        file_object.write('Negative ')
                        print(token_list)
                        for tt in token_list:
                            file_object.write(tt)
                            file_object.write(' ')
                        file_object.write('\n')

if __name__ == '__main__':

    #python parse_sentence.py ./corpus/bioinfer/bioinfer-1.2.0b-unified-format.xml bioinfer.txt
    #python parse_sentence.py ./corpus/aimed/aimed.xml aimed.txt    
    inputfile=sys.argv[1]
    outputfile=sys.argv[2]
    #print(inputfile)
    extract_sentence_info(inputfile,outputfile)

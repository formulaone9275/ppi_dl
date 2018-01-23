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
if __name__ == '__main__':
    tree = etree.parse("./corpus/bioinfer/bioinfer-1.2.0b-unified-format.xml")
    root = tree.getroot()
    file_object=open('bioinfer.txt','w+')
    for docu in root.iter("document"):
        #iterate every document
        for sent in docu.iter("sentence"):
            #iterate all the sentence
            #print(sent.get("text"))
            ids_all=[]
            protein_names=[]
            protein_offset=[]
            for enti in sent.iter("entity"):
                ids_all.append(enti.get("id"))
                offset_str=enti.get("charOffset")
                ind=offset_str.find("-")
                comma_index=offset_str.find(",")
                if comma_index==-1:
                    len_offset_str=len(offset_str)
                else:
                    len_offset_str=comma_index
                #there are spaces in the protein names in some case
                space_index=enti.get("text").find(" ")
                if space_index>0:
                    protein_names.append(enti.get("text")[0:space_index])
                    protein_offset.append(str(int(offset_str[ind+1:len_offset_str])-(len(enti.get("text"))-space_index)))
                else:
                    protein_names.append(enti.get("text"))
                    protein_offset.append(offset_str[ind+1:len_offset_str])


            #print("protein_offset:",protein_offset)
            #parse the sentence here
            parse_result=run(sent.get("text"))
            #get the incoming dependency information
            incoming_dependency={}
            for sente in parse_result.sentence:
                 for dep in sente.dependency:
                     d_index=getattr(dep,'dep_index')
                     # print(type(d_index))
                     incoming_dependency[d_index]=dep.relation
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
                    #print(protein_names[i]+"-"+protein_offset[i]+"-"+protein_names[j]+"-"+protein_offset[j]+"-")
                    for tok in parse_result.token:
                        # print(tok.word+"-"+str(tok.char_end))
                        
                        if tok.word == protein_names[i] and protein_offset[i]==str(tok.char_end):
                            protein_index_1=tok.index
                        elif tok.word == protein_names[j] and protein_offset[j]==str(tok.char_end):
                            protein_index_2=tok.index

                    #generate the sentence information
                    token_list=[]
                    for ii in range(len(parse_result.token)):
                        toke=parse_result.token[ii]
                        if ii not in incoming_dependency.keys():
                            incoming_dependency[ii]='None'
                        if toke.word == protein_names[i] and protein_offset[i]==str(toke.char_end):
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT1"+"|"+str(toke.index-protein_index_1)+"|"+str(toke.index-protein_index_2)+"|"+incoming_dependency[ii])
                        elif toke.word == protein_names[j] and protein_offset[j]==str(toke.char_end):
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT2"+"|"+str(toke.index-protein_index_1)+"|"+str(toke.index-protein_index_2)+"|"+incoming_dependency[ii])
                        elif toke.word in protein_names:
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"PROT"+"|"+str(toke.index-protein_index_1)+"|"+str(toke.index-protein_index_2)+"|"+incoming_dependency[ii])
                        else:
                            token_list.append("token:"+toke.word+"|"+toke.pos+"|"+"O"+"|"+str(toke.index-protein_index_1)+"|"+str(toke.index-protein_index_2)+"|"+incoming_dependency[ii])
                    if interaction_relation is True:
                        file_object.write('True ')
                        print(token_list)
                        for t in token_list:
                            file_object.write(t)
                            file_object.write(' ')
                        file_object.write('\n')
                    else:
                        file_object.write('False ')
                        print(token_list)
                        for tt in token_list:
                            file_object.write(tt)
                            file_object.write(' ')
                        file_object.write('\n')

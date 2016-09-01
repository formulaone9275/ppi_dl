# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import os
import sys

# nlputis codes.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from protolib.python import document_pb2, rpc_pb2
from utils.rpc.iterator import request_iter_docs
from utils.rpc import grpcapi


# Given a request iterator, send requests in parallel and return response.
def get_response(request_iter, server_ip='128.4.20.169', port=8900,
                 thread_size=5):
    queue = grpcapi.get_queue(server_ip, port, thread_size, request_iter)
    try:
        for response in queue:
            yield response
    except KeyboardInterrupt:
        return


def run():
    text = u'MicroRNAs (miRNAs) are small non-coding RNAs of ∼19-24 ' \
           'nucleotides (nt) in length and considered as potent ' \
           'regulators of gene expression at transcriptional and ' \
           'post-transcriptional levels. Here we report the identification ' \
           'and characterization of 15 conserved miRNAs belonging to 13 ' \
           'families from Rauvolfia serpentina through in silico analysis ' \
           'of available nucleotide dataset. The identified mature R. ' \
           'serpentina miRNAs (rse-miRNAs) ranged between 20 and 22nt in ' \
           'length, and the average minimal folding free energy index (MFEI) ' \
           'value of rse-miRNA precursor sequences was found to be ' \
           '-0.815kcal/mol. Using the identified rse-miRNAs as query, their ' \
           'potential targets were predicted in R. serpentina and other plant ' \
           'species. Gene Ontology (GO) annotation showed that predicted ' \
           'targets of rse-miRNAs include transcription factors as well as ' \
           'genes involved in diverse biological processes such as primary ' \
           'and secondary metabolism, stress response, disease resistance, ' \
           'growth, and development. Few rse-miRNAs were predicted to target ' \
           'genes of pharmaceutically important secondary metabolic pathways ' \
           'such as alkaloids and anthocyanin biosynthesis. Phylogenetic ' \
           'analysis showed the evolutionary relationship of rse-miRNAs and ' \
           'their precursor sequences to homologous pre-miRNA sequences from ' \
           'other plant species. The findings under present study besides giving ' \
           'first hand information about R. serpentina miRNAs and their targets, ' \
           'also contributes towards the better understanding of miRNA-mediated ' \
           'gene regulatory processes in plants.'

    raw_doc = document_pb2.Document()
    raw_doc.doc_id = '26815768'
    raw_doc.text = text

    one_hundred_docs = [raw_doc] * 100

    requests = request_iter_docs(one_hundred_docs, request_type=rpc_pb2.Request.PARSE_BLLIP)
    responses = get_response(requests, thread_size=10)
    for response in responses:
        print(len(response.document))


if __name__ == '__main__':
    run()

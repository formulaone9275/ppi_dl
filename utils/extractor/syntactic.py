from __future__ import unicode_literals, print_function
import sys
from ..dependency.path import undirected_shortest_dep_path
from ..dependency.graph import DependencyGraph
from ..dependency.path import PathDirection
from ..constituent.path import undirected_shortest_const_path
from ..helper import RangeHelper


def extract_dep_lemma_path(helper, graph, src_entity, dst_entity):
    src_head = helper.entity_head_token(graph, src_entity)
    dst_head = helper.entity_head_token(graph, dst_entity)

    path = undirected_shortest_dep_path(graph, src_head.index, dst_head.index)
    if path is None:
        return None
    
    result = []
    prev_direction = None
    # The first element stores the relation from the src_entity head
    # to a previous token, which is not on the path. The last element
    # stores the relation from the token to the dst_entity head.
    for e in path[1:-1]:
        if prev_direction is not None and prev_direction != e.direction_from_prev:
            # Add a separator when the direction changes.
            result.append('__')
            
        if e.direction_from_prev == PathDirection.dep_to_gov:
            result.append('_'+e.relation_with_prev+'_')
            result.append('<-')
            result.append(helper.doc.token[e.token_index].lemma)
        else:
            result.append('->')
            result.append('_'+e.relation_with_prev+'_')
            result.append(helper.doc.token[e.token_index].lemma)
        
        prev_direction = e.direction_from_prev
        
    if path[-1].direction_from_prev == PathDirection.dep_to_gov:
        result.append('_' + path[-1].relation_with_prev + '_')
        result.append('<-')
    else:
        result.append('->')
        result.append('_' + path[-1].relation_with_prev + '_')
            
    return ''.join(result)


def extract_dep_path(helper, graph, src_entity, dst_entity):
    # Note that either an arrow or a relation tag can connect to tokens,
    # so that the left or right ending could be either of them.
    src_head = helper.entity_head_token(graph, src_entity)
    dst_head = helper.entity_head_token(graph, dst_entity)

    path = undirected_shortest_dep_path(graph, src_head.index, dst_head.index)
    if path is None:
        return None
    
    result = []
    prev_direction = None
    
    for e in path[1:]:
        if prev_direction is not None and prev_direction != e.direction_from_prev:
            # Add a separator when the direction changes.
            result.append('__')
        if e.direction_from_prev == PathDirection.dep_to_gov:
            result.append(e.relation_with_prev)
            result.append('<-')
        else:
            result.append('->')
            result.append(e.relation_with_prev)
        prev_direction = e.direction_from_prev
        
    return ''.join(result)


def extract_word_on_dep_path(helper, graph, src_entity, dst_entity):
    src_head = helper.entity_head_token(graph, src_entity)
    dst_head = helper.entity_head_token(graph, dst_entity)

    path = undirected_shortest_dep_path(graph, src_head.index, dst_head.index)
    if path is None:
        return None
    
    result = []
    for e in path[1:-1]:
        result.append(helper.doc.token[e.token_index].word)
    #result.append(path[-1].relation)

    return result


def extract_relation_dep_lemma_path(helper, relation, role_1, role_2):
    assert role_1 != role_2
    arg_1 = []
    arg_2 = []
    for arg in relation.argument:
        if arg.role == role_1:
            arg_1.append(helper.doc.entity[arg.entity_duid])

        if arg.role == role_2:
            arg_2.append(helper.doc.entity[arg.entity_duid])
            
    assert len(arg_1) == 1
    assert len(arg_2) == 1
    if arg_1[0].sentence_index != arg_2[0].sentence_index:
        print(helper.doc.doc_id, 'multiple sentences', file=sys.stderr)
        return None
    
    sentence = helper.doc.sentence[arg_1[0].sentence_index]
    graph = DependencyGraph.build_from_proto(sentence.dependency)

    src_head = helper.entity_head_token(graph, arg_1[0])
    dst_head = helper.entity_head_token(graph, arg_2[0])

    path = undirected_shortest_dep_path(graph, src_head.index, dst_head.index)
    if path is None:
        return None

    result = []
    for e in path[1:-1]:
        result.append(e.relation)
        result.append(helper.doc.token[e.token_index].lemma)
    result.append(path[-1].relation)

    return '/'.join(result)


def get_mininal_cover_constituent(constituents, entity):
    target = None

    for constituent in constituents:
        if RangeHelper.char_range_include(constituent, entity):
            if target is None:
                target = constituent
            else:
                if target.char_end - target.char_start > \
                                constituent.char_end - constituent.char_start:
                    target = constituent
    return target


def get_all_cover_constituent(constituents, entity):
    targets = []

    for constituent in constituents:
        if RangeHelper.char_range_include(constituent, entity):
            targets.append(constituent)
    return targets


def extract_constituent_path(constituents, src_entity, dst_entity):
    src_cst = get_mininal_cover_constituent(constituents, src_entity)
    dst_cst = get_mininal_cover_constituent(constituents, dst_entity)

    if src_cst is None or dst_cst is None:
        return None

    path = undirected_shortest_const_path(constituents, src_cst, dst_cst)

    results = []
    prev = None
    for cst in path:
        if prev is not None:
            if prev.index in cst.children:
                results.append('<-')
            elif cst.index in prev.children:
                results.append('->')
            else:
                raise ValueError(''.join([str(e) for e in path]) + str(prev) + str(cst))
        results.append(cst.label)
        prev = cst

    return ''.join(results)


def extract_largest_constituent_path(constituents, src_entity, dst_entity):
    pass

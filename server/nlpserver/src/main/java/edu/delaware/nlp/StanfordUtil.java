package edu.delaware.nlp;

import edu.delaware.nlp.DocumentProto;
import edu.delaware.nlp.BllipUtil;

import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.ling.HasOffset;

//import java.util.*;
import java.util.List;
import java.util.LinkedList;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.Collection;
import java.util.Queue;
import java.util.Properties;


public class StanfordUtil {

    private final String annotators;
    private final int maxParseSeconds;
    private StanfordCoreNLP pipeline;
    private CollinsHeadFinder headFinder;

    public StanfordUtil(String annotators, int maxParseSeconds) {
        this.annotators = annotators;
        this.maxParseSeconds = maxParseSeconds;
        loadPipeline();
    }

    public StanfordUtil(String annotators) {
        this.annotators = annotators;
        this.maxParseSeconds = 0;
        loadPipeline();
    }

    private void loadPipeline() {
        // Initialize StanfordNLP pipeline.
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        // 300 seconds for the whole parsing process for a document.
        if (maxParseSeconds > 0) {
            props.setProperty("parse.maxtime", Integer.toString(maxParseSeconds * 1000));
        }
        props.setProperty("annotators", annotators);
        pipeline = new StanfordCoreNLP(props);
        headFinder = new CollinsHeadFinder();
    }

    public Map<String, String> splitSentence(String text) {
        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);
        // run all Annotators on this text
        pipeline.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        int sentIndex = 0;
        HashMap<String, String> sentence_map = new HashMap<String, String>();
        for (CoreMap sentence : sentences) {
            sentence_map.put(Integer.toString(sentIndex), sentence.toString());
            sentIndex++;
        }
        return sentence_map;
    }

    public DocumentProto.Document parseUsingBllip(DocumentProto.Document protoDoc,
                                                  Map<String, String> sentences,
                                                  Map<String, String> parses) {
        String text = protoDoc.getText();
        DocumentProto.Document.Builder dbuilder = protoDoc.toBuilder();

        int tokenIndex = 0;
        int sentIndex = 0;
        int charIndex = 0;

        for (int i = 0; i < sentences.size(); i++) {
            // We use sentence order as sentence id.
            String sentence_id = Integer.toString(i);
            String sentence_text = sentences.get(sentence_id);
            String parse = parses.get(sentence_id);

            // Starting character offset of current sentence.
            int currIndex = text.indexOf(sentence_text, charIndex);
            // Starting point for next search.
            charIndex += sentence_text.length();

            if (parse == null) {
                System.err.println("Sentence parse error");
                continue;
            }

            // this is the parse tree of the current sentence
            // obtained from Charniak Bllip parser.
            Tree tree = Tree.valueOf(parse);
            List<Tree> leaves = tree.getLeaves();

            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            int sentenceBoundary = -1;
            int sentenceTokenIndex = 1;
            HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();
            DocumentProto.Sentence.Builder sbuilder = DocumentProto.Sentence.newBuilder();

            for (Tree leaf : leaves) {
                if (sentenceBoundary == -1) {
                    sbuilder.setTokenStart(tokenIndex);
                }

                Tree preTerminal = leaf.parent(tree);

                HasOffset leafLabel = (HasOffset) leaf.label();
                HasOffset preTerminalLabel = (HasOffset) preTerminal.label();

                String word = leaf.label().value();
                String pos = preTerminal.label().value();
                String unescaped = BllipUtil.unescape(word);
                int wordCharStart = text.indexOf(unescaped, currIndex);

                assert wordCharStart >= 0 : sentence_text;

                int wordCharEnd = wordCharStart + unescaped.length() - 1;
                currIndex = wordCharEnd + 1;

                // Note that if any of the field is the default value, it woun't be printed.
                // For example, if the first token is "I", the its char_start, char_end and
                // token_id won't be printed by TextFormat.printToString().
                DocumentProto.Token.Builder tbuilder = DocumentProto.Token.newBuilder();
                tbuilder.setWord(unescaped);
                tbuilder.setPos(pos);
                // TODO: add lemma for bllip parser results.
                // tbuilder.setLemma(token.lemma());
                tbuilder.setCharStart(wordCharStart);
                tbuilder.setCharEnd(wordCharEnd);

                // Set the token offset as it is not set by reading the penn tree
                // bank parse tree produced by Bllip parser.
                // Here we add 1 to the end offsets to be consistent with the
                // Stanford parser output. In Stanford parser, the end offset is the
                // position of the last character + 1. Doing so we can use one
                // buildConstituent() function to build constituents for both Stanford
                // and Bllip outputs.
                leafLabel.setBeginPosition(wordCharStart);
                leafLabel.setEndPosition(wordCharEnd + 1);
                preTerminalLabel.setBeginPosition(wordCharStart);
                preTerminalLabel.setEndPosition(wordCharEnd + 1);

                tbuilder.setIndex(tokenIndex);
                dbuilder.addToken(tbuilder);
                indexMap.put(sentenceTokenIndex, tokenIndex);
                sentenceBoundary = tokenIndex;
                sentenceTokenIndex++;
                tokenIndex++;
            }
            sbuilder.setTokenEnd(tokenIndex - 1);
            sbuilder.setIndex(sentIndex);
            sentIndex++;


            buildConstituent(sbuilder, tree);

            // this is the Stanford dependency graph of the current sentence
            // Generated from Billip parser.
            EnglishGrammaticalStructure sd = new EnglishGrammaticalStructure(tree);
            SemanticGraph dependencies = new SemanticGraph(sd.typedDependenciesCCprocessed());
            buildDependency(sbuilder, dependencies, indexMap);

            dbuilder.addSentence(sbuilder);
        }
        return dbuilder.build();
    }

    public DocumentProto.Document parseUsingStanford(DocumentProto.Document protoDoc) {
        String text = protoDoc.getText();
        DocumentProto.Document.Builder dbuilder = protoDoc.toBuilder();

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        int tokenIndex = 0;
        int sentIndex = 0;
        for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            int sentenceBoundary = -1;
            HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();
            DocumentProto.Sentence.Builder sbuilder = DocumentProto.Sentence.newBuilder();

            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                if (sentenceBoundary == -1) {
                    sbuilder.setTokenStart(tokenIndex);
                }
                // this is the POS tag of the token
                String pos = token.get(PartOfSpeechAnnotation.class);

                // Note that if any of the field is the default value, it woun't be printed.
                // For example, if the first token is "I", the its char_start, char_end and
                // token_id won't be printed by TextFormat.printToString().
                DocumentProto.Token.Builder tbuilder = DocumentProto.Token.newBuilder();
                tbuilder.setWord(token.originalText());
                tbuilder.setPos(pos);
                tbuilder.setLemma(token.lemma());
                tbuilder.setCharStart(token.beginPosition());
                // token.endPosition() is the position after the last character.
                tbuilder.setCharEnd(token.endPosition() - 1);
                tbuilder.setIndex(tokenIndex);
                dbuilder.addToken(tbuilder);
                indexMap.put(token.index(), tokenIndex);
                sentenceBoundary = tokenIndex;
                tokenIndex++;
            }

            sbuilder.setTokenEnd(tokenIndex - 1);
            sbuilder.setIndex(sentIndex);
            sentIndex++;

            // this is the parse tree of the current sentence
            Tree tree = sentence.get(TreeAnnotation.class);
            buildConstituent(sbuilder, tree);

            // this is the Stanford dependency graph of the current sentence
            SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
            buildDependency(sbuilder, dependencies, indexMap);
            dbuilder.addSentence(sbuilder);
        }

        return dbuilder.build();
    }

    public DocumentProto.Document splitSentence(DocumentProto.Document protoDoc) {
        String text = protoDoc.getText();
        DocumentProto.Document.Builder dbuilder = protoDoc.toBuilder();

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        int tokenIndex = 0;
        int sentIndex = 0;
        for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            int sentenceBoundary = -1;
            HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();
            DocumentProto.Sentence.Builder sbuilder = DocumentProto.Sentence.newBuilder();

            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                if (sentenceBoundary == -1) {
                    sbuilder.setTokenStart(tokenIndex);
                }
                String pos = token.get(PartOfSpeechAnnotation.class);

                // Note that if any of the field is the default value, it won't be printed.
                // For example, if the first token is "I", the its char_start, char_end and
                // token_id won't be printed by TextFormat.printToString().
                DocumentProto.Token.Builder tbuilder = DocumentProto.Token.newBuilder();
                tbuilder.setWord(token.originalText());
                tbuilder.setLemma(token.lemma());
                tbuilder.setPos(pos);
                tbuilder.setCharStart(token.beginPosition());
                // token.endPosition() is the position after the last character.
                tbuilder.setCharEnd(token.endPosition() - 1);
                tbuilder.setIndex(tokenIndex);
                dbuilder.addToken(tbuilder);
                indexMap.put(token.index(), tokenIndex);
                sentenceBoundary = tokenIndex;
                tokenIndex++;
            }

            sbuilder.setTokenEnd(tokenIndex - 1);
            sbuilder.setIndex(sentIndex);
            dbuilder.addSentence(sbuilder);

            sentIndex++;

        }
        // dbuilder.clearText();
        return dbuilder.build();
    }

    private void buildDependency(DocumentProto.Sentence.Builder sbuilder,
                                 SemanticGraph dependencies,
                                 HashMap<Integer, Integer> indexMap) {
        // Add root relations. The root links to itself with the relation "root".
        Collection<IndexedWord> roots = dependencies.getRoots();
        for (IndexedWord root : roots) {
            int rootIndex = indexMap.get(root.index());
            DocumentProto.Sentence.Dependency.Builder depBuilder = DocumentProto.Sentence.Dependency.newBuilder();
            depBuilder.setDepIndex(rootIndex);
            depBuilder.setGovIndex(rootIndex);
            depBuilder.setRelation("root");
            sbuilder.addDependency(depBuilder);
        }
        HashMap<Integer, TreeSet<Integer>> childrenMap = new HashMap<Integer, TreeSet<Integer>>();

        // This only gets basic dependencies.
        // Collection<TypedDependency> typedDeps = dependencies.typedDependencies();
        // for (TypedDependency typedDep : typedDeps) {

        // Correct way to get collapsed and ccprocessed relations.
        for (SemanticGraphEdge edge : dependencies.edgeIterable()) {
            IndexedWord gov = edge.getGovernor();
            IndexedWord dep = edge.getDependent();

            int depIndex = indexMap.get(dep.index());
            int govIndex = indexMap.get(gov.index());

            // Only toString can get the collapsed and ccprocessed relations.
            // Neither getShortName() and getLongName() can. Don't know why.
            String depTag = edge.getRelation().toString();
            // String depTag = edge.getRelation().getShortName();

            DocumentProto.Sentence.Dependency.Builder depBuilder = DocumentProto.Sentence.Dependency.newBuilder();
            depBuilder.setDepIndex(depIndex);
            depBuilder.setGovIndex(govIndex);
            depBuilder.setRelation(depTag);
            sbuilder.addDependency(depBuilder);

        }
    }

    private void buildConstituent(DocumentProto.Sentence.Builder sbuilder, Tree tree) {
        Tree nextTree = tree;
        Queue<Tree> treeQueue = new LinkedList<Tree>();
        Queue<Integer> parentQueue = new LinkedList<Integer>();
        int treeIndex = 0;
        parentQueue.add(0);
        while (nextTree != null) {
            int parentIndex = parentQueue.poll();
            // Get the head leaf.
            Tree head = nextTree.headTerminal(headFinder);
            List<Tree> headLeaves = head.getLeaves();
            assert headLeaves.size() == 1;
            Tree only_leaf = headLeaves.get(0);

            // Get char start and end for the head token.
            int head_start = ((HasOffset) only_leaf.label()).beginPosition();
            // It looks the end position is the last char index + 1.
            int head_end = ((HasOffset) only_leaf.label()).endPosition() - 1;

            // Get char start and end for the phrase.
            List<Tree> treeLeaves = nextTree.getLeaves();
            Tree first_leaf = treeLeaves.get(0);
            Tree last_leaf = treeLeaves.get(treeLeaves.size() - 1);
            int phrase_start = ((HasOffset) first_leaf.label()).beginPosition();
            // It looks the end position is the last char index + 1.
            int phrase_end = ((HasOffset) last_leaf.label()).endPosition() - 1;

            assert phrase_end >= phrase_start;
            assert phrase_start >= 0;

            DocumentProto.Sentence.Constituent.Builder cbuilder = DocumentProto.Sentence.Constituent.newBuilder();
            cbuilder.setLabel(nextTree.label().value());
            cbuilder.setCharStart(phrase_start);
            cbuilder.setCharEnd(phrase_end);
            cbuilder.setHeadCharStart(head_start);
            cbuilder.setHeadCharEnd(head_end);
            cbuilder.setIndex(treeIndex);
            cbuilder.setParent(parentIndex);
            // Add children index to its parent.
            if (parentIndex < treeIndex) {
                sbuilder.getConstituentBuilder(parentIndex).addChildren(treeIndex);
            }
            for (Tree child : nextTree.children()) {
                treeQueue.add(child);
                parentQueue.add(treeIndex);
            }
            sbuilder.addConstituent(cbuilder);
            treeIndex++;
            nextTree = treeQueue.poll();
        }
    }
}

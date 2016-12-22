/*
 * Copyright (c) 2010 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.chimpler.example.bayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.vectorizer.TFIDF;
import java.net.URL;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

/**
 * http://www.chimpler.com
 */
public class Classifier {

	public static Map<String, Integer> readDictionnary(Configuration conf, Path dictionnaryPath) {
		Map<String, Integer> dictionnary = new HashMap<String, Integer>();
		for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(dictionnaryPath, true, conf)) {
			dictionnary.put(pair.getFirst().toString(), pair.getSecond().get());
		}
		return dictionnary;
	}

	public static Map<Integer, Long> readDocumentFrequency(Configuration conf, Path documentFrequencyPath) {
		Map<Integer, Long> documentFrequency = new HashMap<Integer, Long>();
		for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(documentFrequencyPath, true, conf)) {
			documentFrequency.put(pair.getFirst().get(), pair.getSecond().get());
		}
		return documentFrequency;
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 6) {//added user name/id
			System.out.println("Arguments: [model] [label index] [dictionnary] [document frequency] [tweet file] [userid]");
			return;
		}
		String modelPath = args[0];
		String labelIndexPath = args[1];
		String dictionaryPath = args[2];
		String documentFrequencyPath = args[3];
		String tweetsPath = args[4];
		String userid = args[5];
		int yes =0;
		int no =0;

		Configuration configuration = new Configuration();

		// model is a matrix (wordId, labelId) => probability score
		NaiveBayesModel model = NaiveBayesModel.materialize(new Path(modelPath), configuration);

		StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);

		// labels is a map label => classId
		Map<Integer, String> labels = BayesUtils.readLabelIndex(configuration, new Path(labelIndexPath));
		Map<String, Integer> dictionary = readDictionnary(configuration, new Path(dictionaryPath));
		Map<Integer, Long> documentFrequency = readDocumentFrequency(configuration, new Path(documentFrequencyPath));


		// analyzer used to extract word from tweet
		Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_43);

		int labelCount = labels.size();
		int documentCount = documentFrequency.get(-1).intValue();

		//System.out.println("Number of labels: " + labelCount);
		//	System.out.println("Number of documents in training set: " + documentCount);
		BufferedReader reader = new BufferedReader(new FileReader(tweetsPath));
		while(true) {
			String line = reader.readLine();
			if (line == null) {
				break;
			}

			String[] tokens = line.split("\t", 2);
			String tweetId = tokens[0];
			String tweet = tokens[1];

			//	System.out.println("User Record: " + tweetId + "\t" + tweet);

			Multiset<String> words = ConcurrentHashMultiset.create();

			// extract words from tweet
			TokenStream ts = analyzer.tokenStream("text", new StringReader(tweet));
			CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
			ts.reset();
			int wordCount = 0;
			while (ts.incrementToken()) {
				if (termAtt.length() > 0) {
					String word = ts.getAttribute(CharTermAttribute.class).toString();
					Integer wordId = dictionary.get(word);
					// if the word is not in the dictionary, skip it
					if (wordId != null) {
						words.add(word);
						wordCount++;
					}
				}
			}

			// create vector wordId => weight using tfidf
			Vector vector = new RandomAccessSparseVector(10000);
			TFIDF tfidf = new TFIDF();
			for (Multiset.Entry<String> entry:words.entrySet()) {
				String word = entry.getElement();
				int count = entry.getCount();
				Integer wordId = dictionary.get(word);
				Long freq = documentFrequency.get(wordId);
				double tfIdfValue = tfidf.calculate(count, freq.intValue(), wordCount, documentCount);
				vector.setQuick(wordId, tfIdfValue);
			}
			// With the classifier, we get one score for each label 
			// The label with the highest score is the one the tweet is more likely to
			// be associated to
			Vector resultVector = classifier.classifyFull(vector);
			double bestScore = -Double.MAX_VALUE;
			int bestCategoryId = -1;
			for(Element element: resultVector.all()) {
				int categoryId = element.index();
				double score = element.get();
				if (score > bestScore) {
					bestScore = score;
					bestCategoryId = categoryId;
				}
				//			System.out.print("  " + labels.get(categoryId) + ": " + score);
			}
			//	System.out.println(" => " + labels.get(bestCategoryId));
			System.out.println( labels.get(bestCategoryId));

			if (labels.get(bestCategoryId).equalsIgnoreCase("yes"))
				yes++;
			else 
				no++;
		}
		System.out.println( "yes ==="+yes +"      no ==="+no);
String url = "curl http://134.193.136.127:8983/solr/new_core/update/json?overwrite=true -H Content-type:application/json --data [{\"id\":\""+userid+"\",\"stomp_s\":\""+yes+"\",\"kick_s\":\""+no+"\"}]"; 
System.out.println("URL is : "+url);
System.out.println(executeCommand(url));
String op = "curl http://134.193.136.127:8983/solr/new_core/select?q=id%3A"+userid+"&wt=json";
System.out.println("URL  OP is : "+op);
System.out.println(executeCommand(op));

					
	analyzer.close();
	reader.close();
}

public static String executeCommand(String command) {
   	 
		StringBuffer output = new StringBuffer();
 
		Process p;
		try {
			p = Runtime.getRuntime().exec(command);
			p.waitFor();
			BufferedReader reader = 
                            new BufferedReader(new InputStreamReader(p.getInputStream()));
                         String line = "";			
			while ((line = reader.readLine())!= null) {
				output.append(line + "\n");
			}
 
		} catch (Exception e) {
			e.printStackTrace();
		}
 
		return output.toString();
 
	}
}

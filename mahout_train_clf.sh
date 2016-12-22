#start hadoop
start-all.sh

#go into work directory
cd /home/candace/Documents/

# compile the java programs
mvn clean package assembly:single

#convert the training set to the hadoop sequence file format
java -cp target/twitter-naive-bayes-example-1.0-jar-with-dependencies.jar com.chimpler.example.bayes.TweetTSVToSeq data/train.tsv tweets-seq

#upload this file to HDFS
hadoop fs -put tweets-seq tweets-seq

#run mahout to transform the training sets into vectors using tfidf weights
/usr/local/mahout/bin/mahout seq2sparse -i tweets-seq -o tweets-vectors

#splits the set into two sets: a training set and a testing set
/usr/local/mahout/bin/mahout split -i tweets-vectors/tfidf-vectors --trainingOutput train-vectors --testOutput test-vectors --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential

#train the classifier
/usr/local/mahout/bin/mahout trainnb -i train-vectors -el -li labelindex -o model -ow -c

# test that the classifier
/usr/local/mahout/bin/mahout testnb -i train-vectors -m model -l labelindex -ow -o tweets-testing -c


# words

Test: Files IO, processing. TF-IDF with Spark. 

Oracle VM VirtualBox, guest OS Ubuntu 17.10, 64-bit.

Tech stack: Spark Notebook (SN v 0.9.0, Scala 2.11.8, Spark 2.1.0, Hadoop 2.7.2 with Hive), Scala, OpenNLP, Jackson Json.

Code organized to NLP, OpenNLP classes.
Accompanying objects: NLP, OpenNLP do IO, store consts. (TODO: IO monad)
Code in classes built in immutable way without side-effects.

TODO: replace java.io with java.nio for concurrent IO.

Spark Notebook feedback: SN makes it hard to debug internal Spark framework errors with objects serialization, lacks lines numbers, code navigation, etc; also SN has internal JSON structure of .snb files. Which prevents SN from being the first choice in development. IDE with deployed Spark must be more convenient, have better developer productivity.

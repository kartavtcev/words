import scala.collection.JavaConversions._
import scala.language.postfixOps
import scala.collection.mutable.WrappedArray

import java.io._ // TODO: replace with java.nio <= concurrent IO
import java.nio.file.{Files, Path, Paths}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{Row, SparkSession, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}

//import opennlp.tools.langdetect._ // custom implementation
import opennlp.tools.lemmatizer.DictionaryLemmatizer
import opennlp.tools.postag.{POSModel, POSTaggerME}
import opennlp.tools.tokenize.{TokenizerME, TokenizerModel}

import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.databind.{ObjectMapper, DeserializationFeature}
import com.fasterxml.jackson.annotation.JsonProperty

// using .close() is not strictly required for sripts
def using[A <: { def close(): Unit }, B](resource: A)(f: A => B): B =
  try {
      f(resource)
  } /* catch { case e: Exception => throw e  } */
  finally {
      resource.close()
  }

type MapLangToStrings = scala.collection.immutable.Map[String, Seq[String]]
   /*
   since sparkContext is unavailable as var here, I can't just use it like in "Spark Dataset 101", "Spark 101"
   could be related to customDepts bug, because I added opennlp dependency:
   https://github.com/spark-notebook/spark-notebook/issues/563
   */
lazy val spark = SparkSession
    .builder()
    .appName("words")
    .config("spark.driver.allowMultipleContexts", "true")
    .master("local")
    .getOrCreate()
import spark.implicits._

object NLP { 
  
  case class Keyword(
    keyword: String,
    value: Double
  )
  case class TfIdfFile(
    filename: String,
    language:String,
    @JsonProperty("keywords") features : Seq[Keyword]
  )
  case class ProcessedFile(
    filename: String,
    words: Seq[String]
  )
  
  val numOfTopWords = 30  
  
  val vocabsRelativePath = "notebooks/words/vocabs/"
  val textsRelativePath = "notebooks/words/text-data/"
  val outputPath = "notebooks/words/output/"
  val langNotDetected = "lang-not-detected"
  
  def getLangs : Seq[String] = {
    val de : String = "de"
    val en : String = "en"
    val fr : String = "fr"
    Seq(de, en, fr)
  }
  
  def getStopwordsPerLang(langs : Seq[String]) : MapLangToStrings = {
    langs map { lang => (lang, using(scala.io.Source.fromFile(s"${vocabsRelativePath}${lang}-stopwords.txt")) { 
                                 source => source.getLines.toList })
    } toMap  
  }
  
  def getFilesPaths : Seq[String] = {
      Files.newDirectoryStream(Paths.get(textsRelativePath))
           .filter(_.getFileName.toString.endsWith(".txt"))
           .map(_.toString)
           .toSeq.sorted
  }
  
  def writeProcessedFiles(pfs: Seq[ProcessedFile]) = {
    pfs.foreach { pf =>
      using (new BufferedWriter(new FileWriter(new File(s"${outputPath}/${pf.filename}")))) { bw =>
          bw.write(pf.words.mkString("\t"))
      }      
    }
  }
  def writeTfIdfsJson(tfidfs: Seq[TfIdfFile]) = {
      using (new BufferedWriter(new FileWriter(new File(s"${outputPath}/tfidf_files.json")))) { bw =>
        bw.write(getJsonMapper().writeValueAsString(tfidfs))
      }    
  }
  
  def getJsonMapper() : ObjectMapper = {
    val mapper = new ObjectMapper with ScalaObjectMapper
    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
    mapper.registerModule(DefaultScalaModule)
    mapper
  }
}

class NLP(val stopwordsPerLang: MapLangToStrings, val textfilesPaths: Seq[String]) {
  def this() = this(NLP.getStopwordsPerLang(NLP.getLangs), NLP.getFilesPaths)

  def process(spark: SparkSession) : Seq[Either[(String, Seq[String]),(Seq[NLP.ProcessedFile], Seq[NLP.TfIdfFile])]] = {
    // no concurrent: https://stackoverflow.com/questions/41426576/is-using-parallel-collections-encouraged-in-spark
    getFilePathsPerLang(textfilesPaths) map { case (lang, textPaths) =>

        if(lang == NLP.langNotDetected) Left((lang, textPaths))
                                             
        val onlp = new OpenNLP(lang)
  
        val ls : Seq[(String,String,Array[String])] = textPaths map { path =>
          using(scala.io.Source.fromFile(path)) { source =>
            val text = source.getLines.mkString
            val unnoisedText = removeTextNoise(text)
                                               
            val tokens = onlp.tokenize(unnoisedText)
            val tokensExcludeStopWords = removeStopWords(lang, tokens, stopwordsPerLang)

            val lemmas = onlp.lemmatize(tokensExcludeStopWords)
            val lemmd = (tokensExcludeStopWords zip lemmas) map { case (t,l) => if(l != "O" && l != "--") l else t } // if no lemma => original
            (lang,path.split("/").takeRight(1).head,lemmd.toArray)
          }}
        val df = spark.createDataFrame(ls).toDF("language", "filename", "tokens") 

        val tf = new CountVectorizer()
              .setInputCol("tokens")
              .setOutputCol("tf")        

        val idf = new IDF()
              .setInputCol("tf")
              .setOutputCol("tfidf")
        
        val va = new VectorAssembler()
              .setInputCols(Array("tfidf"))
              .setOutputCol("features")
                                             
        val tfidfPipeline = new Pipeline().setStages(Array(tf, idf, va))
        val pipelineModel = tfidfPipeline.fit(df)
        val tfidf =  pipelineModel.transform(df)
        
        val cvModel = pipelineModel.stages(0).asInstanceOf[CountVectorizerModel]      
        val vocabulary = cvModel.vocabulary
                                             
        val tis = tfidf.select("filename", "language", "features")
                       .map { case Row(filename: String, language: String, features: Vector) => ((filename, language), features.toArray.toSeq)}        
                       .collect().toSeq
        val pfs = df.select("filename", "tokens")                                             
                    .map { case Row(filename: String, tokens: WrappedArray[String]) => (filename, tokens.toSeq)}
                    .collect().toSeq
                               
      Right((pfs.map { case (filename, tokens) => NLP.ProcessedFile(filename, tokens)}),
            tis.map { case ((filename, language), features) =>
                 NLP.TfIdfFile(filename, language, features.zipWithIndex
                                                              .sortBy(- _._1)
                                                              .take(NLP.numOfTopWords)
                                                              .map{ case (value, index) => NLP.Keyword(vocabulary(index), value)}) })
           
    } toSeq
  }
   
  def removeTextNoise(text:String) : String = {
    val removedNumbers = text.filter(!_.isDigit)
    // https://stackoverflow.com/questions/30074109/removing-punctuation-marks-form-text-in-scala-spark
    val removedWordsOfSizeLessEqual2AndPunctuation = removedNumbers.replaceAll("""([\p{Punct}]|\b\p{IsLetter}{1,2}\b)\s*""", " ")
    // https://stackoverflow.com/questions/6198986/how-can-i-replace-non-printable-unicode-characters-in-java
    val removedUnicodes = removedWordsOfSizeLessEqual2AndPunctuation.replaceAll("""[\p{C}]""", " ")
    val replacedEscapeSeqWithSpace =  removedUnicodes.replaceAll("""[\t\n\r\f\v]""", " ")
    // "Clean it all": replaceAll("""[^A-Za-z]""", " ")
    replacedEscapeSeqWithSpace
  }

  def removeStopWords(lang: String, tokens:Seq[String], stopwordsPerLang : MapLangToStrings) : Seq[String] = {
     tokens.filter(!stopwordsPerLang(lang).contains(_))
  }
  
  def getFilePathsPerLang(textfilePaths : Seq[String]) : MapLangToStrings = {
    textfilePaths map { file => 
      using(scala.io.Source.fromFile(file)) { source => 
        val firstLine = source.getLines.next() // detect language with first line, TODO: use a few random lines in the middle of the text
        detectLang(firstLine, stopwordsPerLang) match  {
          case Some(lang) => (lang, file)    
          case None => (NLP.langNotDetected, file)
        }                                              
      }    
    } groupBy(_._1) map { case (lang, group) => (lang, group.map(_._2)) }
  } 
  
  /*
    Before I googled Apache OpenNLP, I implemented custom language recognizer based on -stopwords.txt.
    Since some external libs are using dictionary approach anyway (https://github.com/optimaize/language-detector):
    stopwords are commonly found in the speech,
    stopwords dictionary is relatively small and stopwords of 3 langs provided differ a lot.
  */
  def detectLang(line : String, stopwordsPerLang : MapLangToStrings) : Option[String] = {
    val langs = line.split(" ").flatMap(item => stopwordsPerLang.filter(_._2.exists(_.equalsIgnoreCase(item))).map(_._1))
                    .groupBy(f => f)
                    .map{ case (l,ls) => (l, ls.size)}
    if(langs.isEmpty) None
    else Some(langs.maxBy(_._2)._1)
  } 
}

class OpenNLP(val tokenizerModel: TokenizerModel, val posModel : POSModel, val lemmatizer : DictionaryLemmatizer) {
  def this(lang:String) = this(OpenNLP.loadTokenizerModel(lang), OpenNLP.loadPOSModel(lang), OpenNLP.loadLemmatizer(lang))

  val tokenizer = new TokenizerME(tokenizerModel)
  val posTagger = new POSTaggerME(posModel)

  def tokenize(text: String): Seq[String] = {
    val positions = tokenizer.tokenizePos(text)
    val strings = positions.map {
      pos => text.substring(pos.getStart, pos.getEnd)
    }
    strings.filter(_.length > 1).map(s => s.toLowerCase) // additional cleanup after regexps & to lower case
  }
  
  def lemmatize(tokens:Seq[String]): Seq[String] = {
    val tags = posTagger.tag(tokens.toArray)
    lemmatizer.lemmatize(tokens.toArray, tags)
  }
} 

object OpenNLP {
  def loadTokenizerModel(lang:String): TokenizerModel = {
    using(new FileInputStream(s"${NLP.vocabsRelativePath}${lang}-token.bin")) { stream =>
      new TokenizerModel(stream)
    }
  }
  
  def loadPOSModel(lang:String): POSModel = {
    using(new FileInputStream(s"${NLP.vocabsRelativePath}${lang}-pos-maxent.bin")) { stream =>
      new POSModel(stream)
    }
  }
  
  def loadLemmatizer(lang:String): DictionaryLemmatizer = {
    using(new FileInputStream(s"${NLP.vocabsRelativePath}${lang}-lemmatizer-columns-reordered.txt")) { stream =>
      new DictionaryLemmatizer(stream)
    }
  }
}

try { // for those who're looking for "error handling"
  new NLP().process(spark) 
    .foldRight(
        Right((Seq.empty[NLP.ProcessedFile], Seq.empty[NLP.TfIdfFile])): 
          Either[(String, Seq[String]),(Seq[NLP.ProcessedFile], Seq[NLP.TfIdfFile])]) { (elem, acc) =>
      for {
        t <- acc.right
        h <- elem.right    
      } yield (h._1++t._1, h._2++t._2)
    }
    match {
      case Right((file, tfidfjson)) => 
        { NLP writeProcessedFiles(file); NLP writeTfIdfsJson(tfidfjson); } 
      case Left((lang, files)) => 
        files foreach (f => println("Lang: $lang is not detected for file: $f"))}
} catch {
  case e: Exception => println(e)
}

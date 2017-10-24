//=============================================================================
//CS5234 -- Distributed WordCount MapReduce Program -- The Mapper
//=============================================================================

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

//import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;


public class SubRedditCountMapper extends Mapper<Object, Text, Text, IntWritable> {

	/* *****************************************************************************************************
	 *  See: 
	 *   - http://www.valleyprogramming.com/blog/simple-hadoop-mapreduce-tutorial-example-boulder-colorado 
	 *   - https://it332ksu.files.wordpress.com/2013/09/hadoop-wordcount-explained-students-ver.pdf
	 * *****************************************************************************************************/
	
	private Text word = new Text();						
	private final IntWritable one = new IntWritable(1); 

	// You can put instance variables here to store state between iterations of
	// the map task.

	// The setup method. Anything in here will be run exactly once before the
	// beginning of the map task.
	public void setup(Context context) throws IOException, InterruptedException {
		//System.out.println("Mapping...");
	}

	// The map method
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

		// The FileInputFormat will supply input elements to the mapper
		// line-by-line using the line number as a key.
		
		// try to parse json
		String jsonString = value.toString();
		HashMap<String,String> comment = new Gson().fromJson(jsonString, new TypeToken<HashMap<String, String>>() {}.getType());
		String subReddit = comment.get("subreddit"); 
		
		// get the name of the input file
		FileSplit fileSplit = (FileSplit)context.getInputSplit();
		String fileName = fileSplit.getPath().getName();
		
		word.set(fileName + " " + subReddit);
		//System.out.println("<word> class: " + word.getClass() + ", <one> class: " + one.getClass());
		//System.out.println("key: " + word.toString());
		context.write(this.word, this.one);											   		
	}

	// The cleanup method. Anything in here will be run exactly once after the
	// end of the map task.
	public void cleanup(Context context) throws IOException,InterruptedException {
		// Note: you can use Context methods to emit records in the setup and cleanup as well.
		//System.out.println("Map complete");
	}

}

//=============================================================================
//CS5234 -- Distributed WordCount MapReduce Program -- The Mapper
//=============================================================================

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

//CS5234 -- Distributed WordCount MapReduce Program -- The Mapper

public class CommentHierarchyMapper extends Mapper<Object, Text, Text, Text> {
	
	private Text subreddit = new Text();	
	private Text output = new Text();
		
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
		
		// (try to) parse json
		String jsonString = value.toString();
		HashMap<String,String> comment = new Gson().fromJson(jsonString, new TypeToken<HashMap<String, String>>() {}.getType());
		
		// get info from the json string
		/*parent.set(comment.get("parent_id"));		// id of the parent link / comment
		child.set(comment.get("name"));				// name of the comment
		timestamp.set(comment.get("created_utc"));	// timestamp at which the comment was created*/
		subreddit.set(comment.get("subreddit"));	// subreddit name		
		output.set(comment.get("parent_id") + "\t" + comment.get("name") + "\t" + comment.get("created_utc"));
		
		context.write(subreddit, output);											   		
	}

	// The cleanup method. Anything in here will be run exactly once after the
	// end of the map task.
	public void cleanup(Context context) throws IOException,InterruptedException {
		// Note: you can use Context methods to emit records in the setup and cleanup as well.
		//System.out.println("Map complete");
	}

}

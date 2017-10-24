
//=============================================================================
// Distributed Subreddit Comment Hierarchy MapReduce Program -- The Reducer
//=============================================================================

// Libraries
import java.io.IOException;
import java.util.HashMap;

import com.google.gson.Gson;	

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class CommentHierarchyReducer extends Reducer<Text, Text, Text, Text> {

	// Instance variables. Note that their state is stored *between* instances of reduce tasks!
	private Text output = new Text();
	//private HashMap<String, HashMap<String,String>> commentMap = new HashMap<String, HashMap<String,String>>();
	
	// Setup method. Anything in here will be run exactly once before the
	// beginning of the reduce task.
	public void setup(Context context) throws IOException, InterruptedException {
		//System.out.println("Reducing...");
	}
	
	// The reducer method
	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		// Clear the commentMap (since its state will be stored between tasks; maybe better as a local variable?)
		//commentMap.clear();
		
		
		// This is a little confusing...! The input is of the form:
		// key = <subreddit_name>, value = list of "<parent_id \t child_name \t timestamp>" strings.
		// I need to extract the parent_ids, child_ids and timestamps, to create a json-type structure.
		
		// Look through each value in the list
		for(Text val : values) {
			HashMap<String, HashMap<String,String>> commentMap = new HashMap<String, HashMap<String,String>>();
			// Split the value into the individual parts
			String[] details = val.toString().split("\t");
			// Don't strictly need these variables, but aids readability (IMO)
			String parent = details[0];
			String child = details[1];
			String timestamp = details[2];
			
			// Next, create a 'HashMap of HashMaps' structure (the commentMap), with 'parent' as the 
			// key to a series  of {'child' : 'timestamp' } values; i.e. 
			// { parent : { child1 : timestamp1, child2 : timestamp2, ...etc. } } 
									
			// If parent currently isn't a key in the map; put it in, initiate the HashMap and add
			// the current <child_id : timestamp> pair
			if(!commentMap.containsKey(parent)){									
				commentMap.put(parent, new HashMap<String, String>());
				commentMap.get(parent).put(child, timestamp);
			// Otherwise, just add the current {timestamp : child_id} HashMap to the ArrayList.
			} else {				
				commentMap.get(parent).put(child, timestamp);
			}
			
			// Serialise the HashMap to json, and write to file
			output.set(new Gson().toJson(commentMap));
			context.write(key, output);
		}
		
		// Serialise the HashMap to json, and write to file
		//output.set(new Gson().toJson(commentMap));
		//context.write(key, output);
	}
	
	// The cleanup method. Anything in here will be run exactly once after the
	// end of the reduce task.
	public void cleanup(Context context) throws IOException, InterruptedException {
		//System.out.println("Reduce Complete");
	}
}